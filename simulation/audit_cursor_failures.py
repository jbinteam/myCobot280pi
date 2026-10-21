"""Replay saved cursor failures and retain endpoint evidence; simulation only.

This audit does not change the production search, tolerances, checkpoint,
roadmap, or collision policy. Failed finite searches do not prove infeasibility.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np

from . import ik_recovery, planning, pose5, service
from .cursor_geometry_audit import geometry_certificate
from .kinematics import ARM
from .prm import planning_space


def cursor_metrics(q, cursor_mm):
    transform = ARM.transform(q)
    delta = np.asarray(cursor_mm) - transform[:3, 3]*1000
    distance = float(np.linalg.norm(delta))
    angle = pose5.axis_error(transform[:3, 2], pose5.unit_axis(delta))[2] if distance > 1e-6 else 180.
    return {'gap_error_mm': abs(distance-50.), 'direction_error_deg': angle,
            'accepted': abs(distance-50.) <= 1. and angle <= 1.}


def endpoint(q, target, axis, cursor, space, source, method=None):
    q = np.asarray(q)
    error = planning.goal_error(q, target, axis)
    valid = space.valid(q)
    cursor_error = cursor_metrics(q, cursor)
    near_lower = np.flatnonzero(q-ARM.lower <= np.deg2rad(.01))
    near_upper = np.flatnonzero(ARM.upper-q <= np.deg2rad(.01))
    _, residual, jacobian = ik_recovery.residual_jacobian(q, target, axis)
    gradient = jacobian.T @ residual
    projected = gradient.copy()
    projected[near_lower] = np.minimum(projected[near_lower], 0.)
    projected[near_upper] = np.maximum(projected[near_upper], 0.)
    evidence = {'source': source, 'method': method, 'q_deg': np.rad2deg(q).tolist(),
                **error, 'valid_configuration': valid, 'cursor': cursor_error,
                'nominal_tolerance_ratio': max(error['error_mm'], error['orientation_error_deg']),
                'all_tolerance_ratio': max(error['error_mm'], error['orientation_error_deg'],
                                           cursor_error['gap_error_mm'], cursor_error['direction_error_deg']),
                'near_lower_joint_numbers': (near_lower+1).tolist(),
                'near_upper_joint_numbers': (near_upper+1).tolist(),
                'projected_gradient_inf': float(np.linalg.norm(projected, ord=np.inf))}
    if not valid:
        evidence['configuration_diagnostic'] = space.diagnostic(q)
    return evidence


def audit(benchmark_path, fixtures_path, output):
    benchmark_path, fixtures_path, output = map(Path, (benchmark_path, fixtures_path, output))
    benchmark = json.loads(benchmark_path.read_text())
    if benchmark['tolerance_mm'] != 1. or benchmark['orientation_tolerance_deg'] != 1.:
        raise ValueError('This cursor audit expects the saved 1 mm / 1 degree benchmark')
    fixtures = json.loads(fixtures_path.read_text())
    cases = {case['id']: case for case in fixtures['cases']}
    saved_cases = {case['id']: case for case in benchmark['cursor']['tasks']}
    failures = [row for row in benchmark['cursor']['rows']['upgraded'] if not row['planned']]
    for row in failures:
        case, saved = cases[row['task_id']], saved_cases[row['task_id']]
        if case['start_deg'] != saved['start_deg'] or case['cursor_mm'] != saved['cursor_mm']:
            raise ValueError('Fixture differs from saved benchmark: '+row['task_id'])
    space = planning_space('self', 1.)
    print('Validating existing roadmap for exact production replay', flush=True)
    began = perf_counter()
    road = service.load_roadmap(service.ARTIFACTS/'roadmap.json', space)
    cold_ms = (perf_counter()-began)*1000
    asset_paths = [service.ARTIFACTS/'roadmap.json', service.ARTIFACTS/'seed5_v1/models.json',
                   service.ARTIFACTS/'seed5_v1/seed5_v1.pt']
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in asset_paths}
    if hashes[str(asset_paths[1])] != benchmark['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from saved benchmark')
    rows = []
    report = {'simulation_only': True, 'benchmark': str(benchmark_path), 'fixtures': str(fixtures_path),
              'input_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (benchmark_path, fixtures_path)},
              'asset_sha256': hashes, 'roadmap_load_ms': cold_ms,
              'tolerance_mm': 1., 'orientation_tolerance_deg': 1., 'clearance_mm': 1.,
              'best_valid_endpoint_ranking': 'Minimum largest error/tolerance across nominal '
                                            'position, nominal direction, cursor gap and cursor aim.',
              'limitations': 'Exact failed fixtures only; finite production search is not a reachability '
                             'proof. Endpoint checks do not establish a route. Joint-bound proximity '
                             'and projected gradients describe candidates, not global infeasibility.',
              'rows': rows}

    def save():
        report['summary'] = {'tasks': len(rows),
            'reproduced_failures': sum(row['status'] == row['saved_status'] for row in rows),
            'failure_classes': dict(Counter(row['classification'] for row in rows)),
            'routing_started': sum(row['planning_outcome'] != 'not_started' for row in rows),
            'pose_accurate_invalid_tasks': sum(row['diagnostic']['pose_accurate_invalid_candidates'] > 0 for row in rows),
            'cursor_constraint_rejection_tasks': sum(row['diagnostic']['goal_constraint_rejections'] > 0 for row in rows),
            'geometry_excluded_tasks': sum(row['geometry_certificate']['geometric_infeasible_with_tolerances'] for row in rows),
            'geometry_unresolved_tasks': [row['id'] for row in rows
                                         if not row['geometry_certificate']['geometric_infeasible_with_tolerances']],
            'best_valid_endpoints_near_joint_bound': sum(bool(row['best_valid_endpoint'] and
                (row['best_valid_endpoint']['near_lower_joint_numbers'] or
                 row['best_valid_endpoint']['near_upper_joint_numbers'])) for row in rows)}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')

    for index, saved_row in enumerate(failures):
        case = cases[saved_row['task_id']]
        captured = []
        original = ik_recovery.optimize

        def capture(*args, **kwargs):
            solved = original(*args, **kwargs)
            captured.append(np.asarray(solved['q_rad']).tolist())
            return solved

        began = perf_counter()
        with patch.object(ik_recovery, 'optimize', side_effect=capture):
            result = service.mouse_aim_plan({key: case[key] for key in ('start_deg', 'cursor_mm')}, road)
        elapsed_ms = (perf_counter()-began)*1000
        generated = result['ik']
        attempts = generated['attempts']
        trust_attempts = [attempt for attempt in attempts if attempt.get('refinement_method') == 'trust_region']
        if len(captured) != len(trust_attempts):
            raise ValueError('Recovery endpoints could not be paired with recorded attempts')
        for attempt, q in zip(trust_attempts, captured):
            attempt['end_q_rad'] = q
        target = np.asarray(result['target_mm'])/1000
        axis = np.asarray(result['target_axis'])
        endpoints = [endpoint(attempt['end_q_rad'], target, axis, case['cursor_mm'], space,
                              attempt['source'], attempt.get('refinement_method'))
                     for attempt in attempts if 'end_q_rad' in attempt]
        valid = [entry for entry in endpoints if entry['valid_configuration']]
        best = min(valid, key=lambda entry: entry['all_tolerance_ratio'], default=None)
        accurate_invalid = [entry for entry in endpoints if not entry['valid_configuration']
                            and entry['nominal_tolerance_ratio'] <= 1.]
        pairs = Counter(tuple(conflict['links']) for entry in accurate_invalid
                        for conflict in entry['configuration_diagnostic'].get('self_collision', {}).get('conflicts', []))
        classification = ('recovered_on_replay' if result['converged'] else
                          'routing_failure' if generated['goals'] else
                          'pose_accurate_candidates_collide' if accurate_invalid else
                          'cursor_constraint_rejections' if generated['diagnostic']['goal_constraint_rejections'] else
                          'no_pose_accurate_endpoint')
        rows.append({'id': case['id'], 'start_deg': case['start_deg'], 'cursor_mm': case['cursor_mm'],
                     'plane_offset_mm': case['plane_offset_mm'], 'target_mm': result['target_mm'],
                     'target_axis': result['target_axis'], 'saved_status': saved_row['status'],
                     'status': result['status'], 'reason': result.get('reason'),
                     'ik_outcome': result['ik_outcome'], 'planning_outcome': result['planning_outcome'],
                     'classification': classification, 'elapsed_ms': elapsed_ms,
                     'geometry_certificate': geometry_certificate(target, axis),
                     'diagnostic': generated['diagnostic'], 'best_valid_endpoint': best,
                     'pose_accurate_collision_pairs': [{'links': list(pair), 'endpoints': count}
                                                       for pair, count in pairs.items()],
                     'recovery_stop_reasons': dict(Counter(attempt['stop_reason'] for attempt in trust_attempts)),
                     'recovery': generated.get('recovery'), 'attempts': attempts, 'endpoints': endpoints})
        save()
        print(f"{index+1}/{len(failures)} {case['id']}: {classification}; "
              f"best valid tolerance ratio {best['all_tolerance_ratio'] if best else None}", flush=True)
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path)] for path in asset_paths):
        raise ValueError('Audit assets changed during replay')
    report['assets_unchanged'] = True
    save()
    print(json.dumps(report['summary'], indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, default=Path('output/ik_recovery/benchmark.json'))
    parser.add_argument('--fixtures', type=Path, default=Path(__file__).parent/'fixtures/mouse_aim_recovery_cases.json')
    parser.add_argument('--output', type=Path, default=Path('output/ik_recovery/cursor_failure_audit.json'))
    args = parser.parse_args()
    audit(args.benchmark, args.fixtures, args.output)
