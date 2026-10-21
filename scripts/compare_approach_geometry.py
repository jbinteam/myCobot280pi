"""Offline paired cone versus exact-center ablation of the production planner.

No robot transport, controller, detector, or ROS imports. Run with:
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/compare_approach_geometry.py
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulation.kinematics import ARM
from simulation import pose5
from simulation.tracking_geometry import region_for, region_metrics
from simulation.diffusion import classical_tracking as tracker, classical_service
from simulation.diffusion.env import ArmEnv

MODES = ('cone30', 'center4', 'center32')
DISTANCES = (350, 320, 290, 260, 230, 200, 170, 150)
SPEED = min(1., np.deg2rad(20.*70/30.))


def center_metrics(q, cursor, position_tol, angle_tol, region=None):
    """The same D/cursor, but only the frozen disk-center pose is accepted."""
    if region is None:
        raise AssertionError('Ablation must preserve region_for and staged distance')
    transform = ARM.transform(q)
    base = region_metrics(q, region, position_tol, angle_tol)
    error = float(np.linalg.norm(transform[:3, 3]*1000-np.asarray(region['center_mm'])))
    axis_error = pose5.axis_error(transform[:3, 2], region['axis'])[2]
    base.update(center_error_mm=error, fixed_axis_error_deg=axis_error,
                accepted=bool(error <= position_tol and axis_error <= angle_tol
                              and base['direction_error_deg'] <= angle_tol))
    base['tracking_accepted'] = base['accepted']
    return base


def center_arrival_metrics(q, cursor, region):
    """Use the same actual object-gaze gate for both native arrival modes."""
    base = center_metrics(q, cursor, 15., 3., region)
    base['accepted'] = bool(base['center_error_mm'] <= 15. and base['direction_error_deg'] <= 3.)
    base['tracking_accepted'] = base['accepted']
    return base


@contextmanager
def geometry_mode(mode):
    """Restore all offline-only monkeypatches, including after timeout/failure."""
    if mode == 'cone30':
        yield
        return
    def center_only(region, start):
        yield np.asarray(region['center_mm'])/1000, np.asarray(region['axis'])
    original_ik = ArmEnv.inverse_kinematics
    def budgeted_ik(env, *args, **kwargs):
        kwargs['attempts'] = 32 if mode == 'center32' else 4
        return original_ik(env, *args, **kwargs)
    with patch.object(tracker, 'candidate_poses', center_only), \
         patch.object(tracker, 'cursor_metrics', center_metrics), \
         patch.object(ArmEnv, 'inverse_kinematics', budgeted_ik):
        yield


def body_for(q, cursor, axis, distance, seed):
    return {'start_deg': np.rad2deg(q).tolist(), 'cursor_mm': cursor.tolist(),
            'target_axis': axis.tolist(), 'obstacle_mm': None, 'seed': seed,
            'tracking_region': {'mode': 'cone_disk', 'distance_mm': distance, 'half_angle_deg': 30.},
            'speed_rad_s': float(SPEED), 'tolerance_mm': 1., 'orientation_tolerance_deg': 1.,
            'continuous_motion': True,
            'velocity_shape': {'kind': 'smooth_trapezoid', 'rise_percent': 5., 'fall_percent': 10.}}


def adapter_validation():
    q = np.deg2rad([0, -40, -130, 80, 0, 50])
    transform = ARM.transform(q)
    cursor = transform[:3, 3]*1000+350*transform[:3, 2]
    axis = transform[:3, 2]
    request = body_for(q, cursor, axis, 320, 42)
    original_candidates, original_metrics, original_ik = tracker.candidate_poses, tracker.cursor_metrics, ArmEnv.inverse_kinematics
    with geometry_mode('cone30'):
        cone = tracker.plan(request)
    with geometry_mode('center4'):
        center = tracker.plan(request)
    assert cone['playable'] and center['playable'], (cone['status'], center['status'])
    assert cone['region_candidates_tested'] == center['region_candidates_tested'] == 1
    np.testing.assert_allclose(cone['goal_deg'], center['goal_deg'], atol=1e-9)
    np.testing.assert_allclose([f['q'] for f in cone['frames']], [f['q'] for f in center['frames']], atol=1e-9)
    basis = np.cross(axis, np.eye(3)[np.argmin(abs(axis))]); basis /= np.linalg.norm(basis)
    tilted = np.cos(np.deg2rad(2))*axis+np.sin(np.deg2rad(2))*basis
    offcenter = region_for(cursor, tilted, {'distance_mm': 350., 'half_angle_deg': 30.})
    assert region_metrics(q, offcenter, 1., 1.)['accepted']
    assert not center_metrics(q, cursor, 1., 1., offcenter)['accepted']
    assert tracker.candidate_poses is original_candidates and tracker.cursor_metrics is original_metrics and ArmEnv.inverse_kinematics is original_ik
    return {'nominal_cone_center4_same_goal_and_frames': True, 'nominal_candidate_count': 1,
            'offcenter_cone_accepted_center_rejected': True, 'patches_restored': True,
            'nominal_request': request, 'nominal_goal_deg': cone['goal_deg']}


def cases():
    """Fixed, disclosed FK rays; no selection based on mode comparison outcomes."""
    starts = [
        [-73.295822,-95.275402,-50.682456,99.827044,125.692472,26.28218],
        [8.612338,-95.15118,16.137795,66.33714,129.720423,-138.708577],
        [-23.511648,-70.76599,73.304053,-96.198269,77.276192,-120.526962],
        [-130.527011,-84.949798,-38.71742,132.124535,99.160099,-151.587104],
        [135.228204,-40.156116,146.318314,-97.782367,142.045458,-10.123181],
        [-136.303686,70.242771,57.124369,106.248463,122.963325,-120.620321],
    ]
    result = []
    for index, start in enumerate(starts):
        q = np.deg2rad(start); transform = ARM.transform(q)
        cursor = transform[:3, 3]*1000+350*transform[:3, 2]
        result.append({'id': f'ray_{index+1}', 'kind': 'endpoint150_locally_solved' if index < 4 else 'near_limit_or_difficult_endpoint', 'start_deg': start,
                       'cursor_mm': cursor.tolist()})
        offset = np.array([12., -8., 6.]) if index < 4 else np.array([40., 25., 15.])
        result.append({'id': f'retarget_{index+1}', 'kind': 'mild_retarget' if index < 4 else 'outer_workspace_challenge',
                       'start_deg': start, 'cursor_mm': (cursor+offset).tolist(), 'retarget_offset_mm': offset.tolist()})
    # Deliberate farther-object challenges, identified before paired results.
    # Their final center is outside the ordinary arm workspace; a cone still
    # has lateral freedom. Bounded search failure is not a proof of infeasibility.
    for index in (0, 4):
        original = result[index*2]
        cursor = np.asarray(original['cursor_mm']); radial = cursor.copy(); radial[2] = 0.
        cursor += 250*radial/np.linalg.norm(radial)
        result.append({'id': f'far_object_{index+1}', 'kind': 'farther_object_endpoint_challenge',
                       'start_deg': original['start_deg'], 'cursor_mm': cursor.tolist(),
                       'retarget_offset_mm': (250*radial/np.linalg.norm(radial)).tolist()})
    return result


class RunDeadline(Exception):
    pass


def episode(case, mode, seed, output, deadline, max_replans):
    # Fresh scene per episode prevents cross-mode valid-cache warming bias.
    if tracker._env is not None:
        tracker._env.close()
        tracker._env = None
    q = np.deg2rad(case['start_deg']); cursor = np.asarray(case['cursor_mm'])
    report = {'case_id': case['id'], 'kind': case['kind'], 'seed': seed, 'mode': mode,
              'target_z_mm': float(cursor[2]), 'cursor_mm': cursor.tolist(), 'start_deg': case['start_deg'],
              'stages': [], 'native_final_stage_success': False, 'failure_reason': None,
              'planning_wall_ms': 0., 'nominal_executed_duration_s': 0., 'joint_travel_rad': 0.,
              'total_candidates_tested': 0, 'max_gaze_deg': 0., 'mean_gaze_deg': None}
    all_gaze = []
    with ArmEnv() as env:
        env.set_obstacle(None)
        report['valid_start'] = bool(env.valid(q))
    if not report['valid_start']:
        report['failure_reason'] = 'invalid_start'
        return report
    try:
        for distance in DISTANCES:
            axis = pose5.unit_axis(cursor-ARM.fk(q)*1000)
            region = region_for(cursor, axis, {'distance_mm': distance, 'half_angle_deg': 30.})
            metric = lambda state: (region_metrics(state, region, 15., 3.) if mode == 'cone30'
                else center_arrival_metrics(state, cursor, region))
            initial_range = float(np.linalg.norm(cursor-ARM.fk(q)*1000))
            stage = {'distance_mm': distance, 'frozen_region': region, 'initial_range_mm': initial_range,
                     'success': False, 'plans': [], 'retreat_mm': 0., 'max_step_retreat_mm': 0.}
            report['stages'].append(stage)
            range_trace = [initial_range]
            for chunk in range(max_replans):
                if metric(q)['accepted']:
                    stage['success'] = True
                    break
                if time.monotonic() >= deadline:
                    raise RunDeadline('global_runtime_bound')
                request = body_for(q, cursor, axis, distance, seed)
                began = time.perf_counter()
                with geometry_mode(mode):
                    result = tracker.plan(request)
                elapsed = (time.perf_counter()-began)*1000
                raw_path = output/'raw'/f"{case['id']}_seed{seed}_{mode}_D{distance}_chunk{chunk}.json"
                raw_path.write_text(json.dumps({'request': request, 'result': result}, indent=2))
                record = {'chunk': chunk, 'status': result['status'], 'reason': result['reason'],
                          'playable': result['playable'], 'planning_wall_ms': elapsed,
                          'region_candidates_tested': result.get('region_candidates_tested', 0),
                          'raw_result': str(raw_path.relative_to(output)), 'executed_frames': 0,
                          'planned_goal_radial_distance_mm': result.get('goal_cursor_acceptance', {}).get('radial_distance_mm'),
                          'selected_candidate_index_zero_based': result.get('region_candidates_tested', 0)-1 if result.get('goal_deg') and result.get('region_candidates_tested', 0) else None,
                          'planned_goal_deg': result.get('goal_deg'), 'planned_target_mm': result.get('target_mm')}
                stage['plans'].append(record)
                report['planning_wall_ms'] += elapsed
                report['total_candidates_tested'] += record['region_candidates_tested']
                if not result['playable']:
                    # A deterministic failed plan holds this stage. Retry seed
                    # would only repeat it, so terminate this bounded episode.
                    report['failure_reason'] = result['status']
                    break
                frames = result['frames']; chosen = len(frames)-1
                for i, frame in enumerate(frames):
                    if metric(np.deg2rad(frame['q']))['accepted']:
                        chosen = i
                        break
                executed = frames[:chosen+1]
                states = np.deg2rad([f['q'] for f in executed])
                travel = float(np.linalg.norm(np.diff(states, axis=0), axis=1).sum())
                duration = float(executed[-1]['time'])
                report['joint_travel_rad'] += travel
                report['nominal_executed_duration_s'] += duration
                record.update(executed_frames=len(executed), planned_frames=len(frames),
                              nominal_executed_duration_s=duration, joint_travel_rad=travel,
                              first_accepted_frame_stop=chosen < len(frames)-1)
                for frame in executed:
                    range_trace.append(float(np.linalg.norm(cursor-np.asarray(frame['position']))))
                    all_gaze.append(float(frame['cursor_direction_error_deg']))
                old_q = q; q = states[-1]
                if metric(q)['accepted']:
                    stage['success'] = True
                    break
                if np.linalg.norm(q-old_q) < 1e-7:
                    report['failure_reason'] = 'stall'
                    break
            stage['final_metrics'] = metric(q)
            stage['retreat_mm'] = max(0., max(range_trace)-initial_range)
            stage['max_step_retreat_mm'] = max(0., float(np.diff(range_trace).max(initial=0.)))
            if not stage['success']:
                report['failure_reason'] = report['failure_reason'] or 'max_replans'
                break
        report['native_final_stage_success'] = bool(report['stages'][-1]['distance_mm'] == 150 and report['stages'][-1]['success'])
    except RunDeadline as exc:
        report['failure_reason'] = str(exc)
    transform = ARM.transform(q); delta = cursor-transform[:3, 3]*1000
    distance = float(np.linalg.norm(delta)); gaze = pose5.axis_error(transform[:3, 2], delta)[2]
    report.update(final_deg=np.rad2deg(q).tolist(), final_range_mm=distance, final_gaze_deg=gaze,
                  common_150pm15_gaze3_success=bool(abs(distance-150) <= 15 and gaze <= 3),
                  common_150pm15_gaze1_success=bool(abs(distance-150) <= 15 and gaze <= 1),
                  max_gaze_deg=max(all_gaze, default=gaze), mean_gaze_deg=float(np.mean(all_gaze)) if all_gaze else gaze,
                  stages_completed=sum(s['success'] for s in report['stages']))
    return report


def summary(reports):
    result = {}
    for mode in MODES:
        rows = [r for r in reports if r['mode'] == mode]
        successes = [r for r in rows if r['native_final_stage_success'] and r.get('common_150pm15_gaze3_success')]
        result[mode] = {'episodes': len(rows), 'native_final_stage_successes': sum(r['native_final_stage_success'] for r in rows),
            'common_150pm15_gaze3_successes': len(successes),
            'common_150pm15_gaze1_successes': sum(r.get('common_150pm15_gaze1_success', False) for r in rows),
            'stage_successes': sum(sum(s['success'] for s in r['stages']) for r in rows),
            'stages_attempted': sum(len(r['stages']) for r in rows),
            'planning_wall_ms_total': sum(r['planning_wall_ms'] for r in rows),
            'planning_wall_ms_median_all_episodes': float(np.median([r['planning_wall_ms'] for r in rows])) if rows else None,
            'failure_counts': {key: sum(r['failure_reason'] == key for r in rows) for key in sorted(set(r['failure_reason'] for r in rows if r['failure_reason']))},
            'successful_duration_s_median': float(np.median([r['nominal_executed_duration_s'] for r in successes])) if successes else None,
            'successful_joint_travel_rad_median': float(np.median([r['joint_travel_rad'] for r in successes])) if successes else None}
    shared = []
    identifiers = sorted(set((r['case_id'], r['seed']) for r in reports))
    for case_id, seed in identifiers:
        paired = {r['mode']: r for r in reports if r['case_id'] == case_id and r['seed'] == seed}
        if len(paired) == len(MODES) and all(r['native_final_stage_success'] and r.get('common_150pm15_gaze3_success') for r in paired.values()):
            shared.append({'case_id': case_id, 'seed': seed,
                'nominal_duration_s': {m: r['nominal_executed_duration_s'] for m,r in paired.items()},
                'joint_travel_rad': {m: r['joint_travel_rad'] for m,r in paired.items()},
                'planning_wall_ms': {m: r['planning_wall_ms'] for m,r in paired.items()},
                'max_gaze_deg': {m: r['max_gaze_deg'] for m,r in paired.items()}})
    result['shared_successful_triplets'] = shared
    result['paired_outcomes'] = {}
    for other in ('center4', 'center32'):
        pairs = []
        for case_id, seed in identifiers:
            matched = {r['mode']: r for r in reports if r['case_id'] == case_id and r['seed'] == seed}
            if 'cone30' not in matched or other not in matched: continue
            if any(matched[m]['failure_reason'] == 'global_runtime_bound' for m in ('cone30', other)): continue
            good = lambda r: r['native_final_stage_success'] and r.get('common_150pm15_gaze3_success')
            pairs.append({'case_id': case_id, 'seed': seed, 'cone_success': good(matched['cone30']),
                          'center_success': good(matched[other])})
        result['paired_outcomes'][other] = {'complete_uncensored_pairs': len(pairs),
            'both_success': sum(p['cone_success'] and p['center_success'] for p in pairs),
            'cone_only_success': [p for p in pairs if p['cone_success'] and not p['center_success']],
            'center_only_success': [p for p in pairs if p['center_success'] and not p['cone_success']],
            'both_failure': [p for p in pairs if not p['cone_success'] and not p['center_success']]}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=300.)
    parser.add_argument('--max-replans', type=int, default=8)
    parser.add_argument('--output', type=Path, default=ROOT/'output/approach_geometry_comparison_20261012')
    args = parser.parse_args(); output = args.output; (output/'raw').mkdir(parents=True, exist_ok=True)
    started = time.monotonic(); deadline = started+args.seconds
    def timeout(signum, frame):
        raise RunDeadline('global_runtime_bound')
    signal.signal(signal.SIGALRM, timeout); signal.setitimer(signal.ITIMER_REAL, args.seconds)
    validation = adapter_validation(); status = classical_service.info()
    source_files = ('scripts/compare_approach_geometry.py', 'simulation/diffusion/classical_tracking.py',
                    'simulation/tracking_geometry.py', 'simulation/diffusion/endpoint_ik.py',
                    'simulation/diffusion/planner.py', 'simulation/continuous.py', 'webui/v2_mouse.py', 'webui/encoder_profile.py')
    metadata = {'simulation_only': True, 'physical_motion_sent': False, 'planner_status': status,
        'adapter_validation': validation, 'modes': {'cone30': 'production up to8 candidates x4 IK attempts; robust first candidate',
            'center4': 'exact frozen center and fixed axis; one candidate x4 robust IK attempts',
            'center32': 'exact frozen center and fixed axis; one candidate x32 robust IK attempts'},
        'search_allocation_confound': 'cone30 vs center32 differs geometry AND IK attempt allocation; center4 controls the first-candidate budget',
        'distance_stages_mm': DISTANCES, 'arrival_tolerance_mm': 15, 'native_arrival_angle_deg': 3,
        'center_arrival_gate': 'Euclidean frozen-center position error <=15mm AND actual object gaze<=3deg; fixed-axis diagnostic does not gate arrival',
        'cone_sampler_note': 'Current production [:8] limit explores center +7 half-radius alternatives at a fresh centered stage, excluding rim samples.',
        'planning_tolerance_mm': 1, 'planning_angle_deg': 1, 'requested_speed_rad_s': 1,
        'sdk_speed': 70, 'reference_speed_rad_s': float(SPEED), 'no_obstacle_block': True,
        'velocity_shape': {'kind': 'smooth_trapezoid', 'rise_percent': 5., 'fall_percent': 10.},
        'simulation_assumptions': ['Perfect kinematic frame feedback; first native accepted20ms frame stops immediately; zero coasting and settling cost.',
            'Fresh measured axis per new stage; frozen across chunks. Actual FK axis serves uncalibrated camera-origin/tool-Z proxy.',
            'Planner compute wall time measured, nominal reference duration is NOT measured hardware duration.',
            'Failed deterministic plan holds then terminates episode; no automatic next-distance shrink. No enforced monotonic endpoint rejection.',
            'No target loss in static target cases; no controller or LOST logic is changed; no detector visibility claims.'],
        'seeds': [42, 43], 'max_replans_per_stage': args.max_replans,
        'source_sha256': {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in source_files}, 'cases': cases()}
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2))
    reports = []
    for seed in (42, 43):
        for case_index, case in enumerate(metadata['cases']):
            rotation = (case_index+seed-42) % len(MODES)
            for mode in MODES[rotation:]+MODES[:rotation]:
                if time.monotonic() >= deadline:
                    break
                row = episode(case, mode, seed, output, deadline, args.max_replans); reports.append(row)
                (output/'episodes.json').write_text(json.dumps(reports, indent=2))
                print(json.dumps({k: row.get(k) for k in ('case_id','seed','mode','stages_completed','final_range_mm','final_gaze_deg','native_final_stage_success','common_150pm15_gaze3_success','failure_reason','planning_wall_ms')}), flush=True)
            if time.monotonic() >= deadline: break
        if time.monotonic() >= deadline: break
    signal.setitimer(signal.ITIMER_REAL, 0)
    aggregate = {'elapsed_wall_s': time.monotonic()-started, 'planned_episodes': len(metadata['cases'])*2*len(MODES),
                 'completed_episodes': len(reports), 'summary': summary(reports)}
    (output/'summary.json').write_text(json.dumps(aggregate, indent=2)); print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
