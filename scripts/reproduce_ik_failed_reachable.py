"""Reproduce a real bounded-IK failure and independently verify reachability.

Runs local simulation only. No HTTP server, ROS node, or hardware is used.
The normal planner never receives the witness or additional candidates.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.camera_model import CameraModel
from simulation.diffusion import classical_tracking
from simulation.diffusion.env import ArmEnv
from simulation.diffusion.planner import RRTConfig, plan as rrt_plan
from simulation.diffusion.service import errors
from simulation.diffusion.visibility import VisibilityEnv
from simulation.kinematics import ARM
from simulation.tracking_geometry import candidate_poses, region_for, region_metrics

FIXTURE = ROOT / 'simulation/diffusion/tests/fixtures/ik_failed_reachable.json'


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def load_case(path=FIXTURE):
    case = json.loads(Path(path).read_text())
    require(case['schema'] == 1 and case['physical_motion'] is False,
            'Expected a simulation-only reachability fixture')
    # These are historical, self-contained proofs. Replay the recorded optical
    # model explicitly, independently of a later runtime FOV configuration.
    CameraModel(**case['camera'])
    return case


def reproduce_failure(case):
    """Exercise the unmodified public planner with its normal candidate budget."""
    body = deepcopy(case['request'])
    body['_visibility_camera'] = CameraModel(**case['camera'])
    result = classical_tracking.plan(body)
    expected = case['expected_baseline']
    require(result['checkpoint_sha256'] == case['checkpoint_sha256'], 'IK checkpoint changed')
    require(result['environment'] == case['environment'], 'Collision geometry changed')
    require(result['visibility_model']['camera'] == case['camera'], 'Planner camera changed')
    require(result['status'] == expected['status'],
            f"Expected the recorded limitation, got {result['status']}; "
            'if the planner improved, retain this case as a reachable-target regression')
    require(result['playable'] is False and result['frames'] == [], 'Failed IK must hold')
    require(result['physical_reach_precheck']['possible'], 'Target was rejected by the reach bound')
    require(result['configuration']['region_candidate_budget'] == expected['candidate_limit'],
            'Candidate budget changed')
    require(result['region_candidates_filtered'] == expected['candidates_filtered'],
            'Candidate filtering changed')
    require(result['region_candidates_tested'] == expected['candidates_tested'],
            'Candidate coverage changed')
    require(result['endpoint_source']['jobs_started'] == expected['jobs_started'],
            'IK jobs changed')
    require(result['endpoint_source']['goals'] == [], 'An endpoint was found')
    require(len(result['region_route_attempts']) == expected['route_attempts'],
            'This should fail before route search')
    require(all(a['status'] == 'search_exhausted'
                for a in result['endpoint_source']['attempts']),
            'Expected real solver exhaustion for each unfiltered candidate job')
    return result


def verify_witness(case, replan=False):
    """Check a saved solution without relying on the failing planner's verdict."""
    body, witness = case['request'], case['witness']
    start = np.deg2rad(body['start_deg'])
    goal = np.deg2rad(witness['q_deg'])
    region = region_for(body['cursor_mm'], body['target_axis'], body['tracking_region'])
    candidates = list(candidate_poses(region, start))
    index = witness['candidate_index']
    require(index >= case['expected_baseline']['candidate_limit'],
            'Witness candidate must be outside the baseline candidate set')
    target, axis = candidates[index]
    acceptance = region_metrics(goal, region, 1., 1.)
    require(acceptance['accepted'], 'Witness is outside the disk or pointing tolerance')
    require(not region_metrics(start, region, 1., 1.)['accepted'],
            'Start already satisfies the goal')
    endpoint_error = errors(goal, target, axis)
    require(endpoint_error['error_mm'] <= 1. and endpoint_error['orientation_error_deg'] <= 1.,
            'Witness does not match its candidate pose')

    # Real IK recovery on the additional point; no solver mocks or injected goals.
    with ArmEnv() as collision:
        collision.set_obstacle(None)
        require(collision.metadata() == case['environment'], 'Witness geometry changed')
        recovered = collision.inverse_kinematics(target, axis, current=start,
                                               **witness['recovery_ik'])
        require(collision.valid(recovered), 'Recovered IK pose is in collision')
        require(region_metrics(recovered, region, 1., 1.)['accepted'],
                'Recovered IK pose does not satisfy the region')
        recovery_error = errors(recovered, target, axis)
        require(recovery_error['error_mm'] <= 1. and recovery_error['orientation_error_deg'] <= 1.,
                'Recovered pose misses the additional candidate')

    route = np.asarray(witness['route_rad'])
    require(route.ndim == 2 and route.shape[1] == 6 and len(route) >= 2,
            'Expected a complete six-joint geometric route')
    require(np.allclose(route[0], start, atol=1e-12, rtol=0), 'Route starts at another pose')
    require(np.allclose(route[-1], goal, atol=1e-12, rtol=0), 'Route misses the witness')
    with VisibilityEnv(np.asarray(body['cursor_mm'])/1000,
                       camera=CameraModel(**case['camera'])) as view:
        view.set_obstacle(None)
        require(view.metadata() == case['environment'], 'Replay geometry changed')
        require(view.valid(start) and view.valid(goal) and view.valid(recovered),
                'Start, witness, or recovered endpoint is invalid or invisible')
        require(view.path_valid(route, .005), 'Saved route fails dense collision/visibility replay')
        audit = view.audit(route, .005)
        require(audit['passed'], 'Saved route fails point visibility')
        regenerated = None
        if replan:
            spec = witness['rrt']
            path, diagnostics = rrt_plan(view, start, goal, seed=spec['seed'],
                                        config=RRTConfig(**spec['configuration']))
            require(path is not None and view.path_valid(path, .005),
                    'Fresh witness route search failed')
            regenerated = {'diagnostics': diagnostics, 'audit': view.audit(path, .005),
                           'route_rad': path.tolist()}

    transform = ARM.transform(goal)
    return {'passed': True, 'candidate_index': index, 'candidate_number': index+1,
            'position_mm': (transform[:3, 3]*1000).tolist(),
            'tool_axis': transform[:3, 2].tolist(), 'acceptance': acceptance,
            'candidate_error': endpoint_error, 'recovered_q_deg': np.rad2deg(recovered).tolist(),
            'recovery_error': recovery_error, 'route_audit': audit,
            'regenerated_route': regenerated}


def verify_final_timed_route(case):
    """Independently reconstruct every stored reference to the 150 mm finish."""
    from simulation import profile_tracking, continuous
    body = case['request']
    proof = case['final_stage_proof']['timed_route']
    require(proof['complete_route'] and body['tracking_region']['distance_mm'] == 150.,
            'Expected a complete final-stage route')
    zero = np.zeros(6)
    current = np.deg2rad(body['start_deg'])
    final = np.deg2rad(case['witness']['q_deg'])
    combined = [current]
    duration = 0.
    with VisibilityEnv(np.asarray(body['cursor_mm'])/1000, camera=CameraModel(**case['camera'])) as view:
        view.set_obstacle(None)
        require(view.metadata() == case['environment'], 'Timed witness geometry changed')
        for curve in proof['references']:
            times = np.arange(curve['frames'])*curve['dt']
            states = np.asarray([profile_tracking.sample_motion(curve, t)[0] for t in times])
            frames = continuous.frames_for(states, times, ARM.fk(final), ARM.transform(final)[:3, 2])
            profile_tracking.validate(curve, frames, (current, zero, zero), body['speed_rad_s'], view)
            require(view.path_valid(states, .005), 'Timed reference fails dense scene/visibility checks')
            require(curve['reference_shape'] == body['velocity_shape'], 'Timed witness changed velocity shape')
            combined.extend(states[1:]); current = states[-1]
            duration += float(times[-1])
        require(np.allclose(current, final, atol=1e-12, rtol=0), 'Timed references stop before the final witness')
        require(view.path_valid(combined, .005), 'Complete timed replay fails scene checks')
        audit = view.audit(combined, .005)
        require(audit['passed'], 'Complete timed replay loses the object')
    region = region_for(body['cursor_mm'], body['target_axis'], body['tracking_region'])
    acceptance = region_metrics(current, region, 1., 1.)
    require(acceptance['accepted'], 'Complete timed replay misses the 150 mm disk')
    return {'passed': True, 'complete_route': True, 'references': len(proof['references']),
            'frames': len(combined), 'duration_s': duration, 'acceptance': acceptance, 'audit': audit,
            'hardware_qualified': False, 'encoder_rounding_checked': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, default=FIXTURE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeat', type=int, default=2)
    parser.add_argument('--replan-witness', action='store_true')
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error('--repeat must be positive')
    if args.output.exists():
        parser.error('Output already exists; choose a new path to preserve earlier evidence')
    case = load_case(args.fixture)
    results = []
    for index in range(args.repeat):
        result = reproduce_failure(case)
        results.append(result)
        print(f"Baseline {index+1}/{args.repeat}: {result['status']}", flush=True)
    witness = verify_witness(case, replan=args.replan_witness)
    timed = verify_final_timed_route(case) if case.get('final_stage_proof', {}).get('timed_route') else None
    report = {'passed': True, 'physical_motion': False, 'fixture': str(args.fixture.resolve()),
              'fixture_sha256': hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
              'baseline_results': results, 'witness': witness, 'final_timed_route': timed,
              'limitations': case['limitations']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        output.write(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'passed': True, 'baseline_runs': len(results),
                      'witness_candidate_number': witness['candidate_number'],
                      'route_audit': witness['route_audit']}, indent=2))


if __name__ == '__main__':
    main()
