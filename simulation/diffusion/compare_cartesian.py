"""Paired Cartesian v3/v4 versus v2 with joint endpoint capture disabled.

Only v2 receives goal joints. Both policies use the same Cartesian ranking,
independent collision gates and C2 trajectory controller. The v3 environment's
IK method raises so a hidden IK fallback aborts the evaluation.
"""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ..continuous import prepare
from ..kinematics import ARM
from .common import file_sha256, write_json
from .compare_motion import tasks as legacy_tasks
from .env import ArmEnv
from .generate_motion import sample_curve
from .policy import DiffusionPolicy, checked_chunk
from .service import errors

SOURCE = Path('simulation/artifacts/diffusion_100k_v1/demonstrations.h5')
AUGMENTATION = Path('simulation/artifacts/diffusion_motion_v2/augmentation.h5')
BASELINE = Path('simulation/artifacts/diffusion_motion_v2/model/diffusion_policy_6dof.pth')
CANDIDATE = Path('simulation/artifacts/diffusion_cartesian_v3/model/diffusion_policy_6dof.pth')
OUTPUT = Path('output/diffusion/cartesian_v3_comparison.json')
SEED = 20261012


def _cartesian_task(task):
    """FK converts held-out task labels, before either policy is invoked."""
    fixed = 'goal' in task
    goals = np.repeat(task['goal'][None], 2, axis=0) if fixed else task['goals']
    transforms = np.asarray([ARM.transform(q) for q in goals])
    return {**task, 'goals': goals, 'targets': transforms[:, :3, 3],
            'axes': transforms[:, :3, 2],
            'times': np.array([0., .1]) if fixed else task['times'],
            'velocities': np.zeros((2, 6)) if fixed else task['velocities'],
            'accelerations': np.zeros((2, 6)) if fixed else task['accelerations'],
            'target_velocities': np.zeros((2, 3)) if fixed else task['target_velocities']}


def tasks(source=SOURCE, augmentation=AUGMENTATION, full=False):
    """Pilot: 4 blocked + 4 direct block + 8 absent fixed, 2 of each motion kind."""
    fixed, moving = legacy_tasks(source, augmentation, seed=SEED)
    if not full:
        blocked = [t for t in fixed if t.get('blocked')][:4]
        direct = [t for t in fixed if t['kind'] == 'block' and not t.get('blocked')][:4]
        absent = [t for t in fixed if t['kind'] == 'static_absent'][:8]
        fixed = blocked + direct + absent
        moving = [t for kind in ('moving_absent', 'moving_block', 'retarget_absent', 'retarget_block')
                  for t in [r for r in moving if r['kind'] == kind][:2]]
    return [_cartesian_task(t) for t in fixed], [_cartesian_task(t) for t in moving]


def forbid_ik(*args, **kwargs):
    raise AssertionError('Cartesian policy evaluation must never call inverse kinematics')


def policy_task(task, cartesian):
    """Make goal joint labels unavailable in the v3 runtime task dictionary."""
    return {k: v for k, v in task.items() if not cartesian or k not in ('goal', 'goals')}


def task_index(task, elapsed):
    return min(max(0, int(np.searchsorted(task['times'], elapsed, side='right') - 1)),
               len(task['targets']) - 1)


def continuous_suite(policy, input_tasks, max_decisions=100, dt=.4, samples=4, inference_steps=32,
                     prefix_capture=True):
    from .cartesian_conditioning import supports
    from .service import condition, cartesian_score, static_capture_prefix
    cartesian = supports(policy)
    task_rows = [policy_task(t, cartesian) for t in input_tasks]
    envs = []; qs = []; velocities = []; accelerations = []; curves = []; curve_times = []; rows = []
    began = perf_counter()
    try:
        for task in task_rows:
            env = ArmEnv(); envs.append(env); env.set_obstacle(task['box'])
            if cartesian: env.inverse_kinematics = forbid_ik
            q = task['q'][0].copy(); env.reset(q)
            if not env.path_valid(task['q']): raise ValueError('Held-out route failed collision replay')
            v = task['velocities'][0].copy(); a = task['accelerations'][0].copy()
            qs.append(q); velocities.append(v); accelerations.append(a)
            try:
                _, _, brake = prepare(env, np.vstack((q, ARM.clip(q + v * .25))), v, a)
            except ValueError: brake = None
            curves.append(brake); curve_times.append(0.)
            rows.append({'id': task['id'], 'kind': task['kind'], 'blocked': bool(task.get('blocked')),
                         'status': 'running' if brake is not None else 'initial_continuation_rejected',
                         'success': False, 'decisions': 0, 'proposals': 0, 'stalls': 0,
                         'rejections': Counter(), 'join_rejections': 0, 'execution_collisions': 0,
                         'tracking_error_mm': [], 'tracking_axis_error_deg': [],
                         'max_velocity_rad_s': 0., 'max_acceleration_rad_s2': 0., 'max_jerk_rad_s3': 0.,
                         'max_boundary_velocity_error': 0., 'max_boundary_acceleration_error': 0.,
                         'joint_goal_capture': 0, 'neural_prefix_capture': 0,
                         'neural_prefix_candidates': 0, 'static_target_brakes': 0,
                         'ik_disabled': cartesian})

        def advance(i, metadata, start_time):
            if metadata is None: return
            duration = min(dt, max(0., metadata['duration_s'] - start_time))
            if duration < 1e-10:
                velocities[i] = np.zeros(6); accelerations[i] = np.zeros(6); return
            times = np.linspace(start_time, start_time + duration, max(2, int(np.ceil(duration / .01)) + 1))
            states, v, a = sample_curve(metadata, times)
            env = envs[i]; previous = qs[i]; previous_time = start_time
            if not env.path_valid(states): raise RuntimeError('Continuous execution collision')
            for state, timestamp in zip(states[1:], times[1:]):
                if not env.execute_step(previous, state, timestamp - previous_time, 1.):
                    raise RuntimeError('Continuous execution safety gate rejected checked curve')
                previous = state; previous_time = timestamp
            row = rows[i]
            row['max_boundary_velocity_error'] = max(row['max_boundary_velocity_error'], float(np.linalg.norm(v[0] - velocities[i])))
            row['max_boundary_acceleration_error'] = max(row['max_boundary_acceleration_error'], float(np.linalg.norm(a[0] - accelerations[i])))
            row['max_velocity_rad_s'] = max(row['max_velocity_rad_s'], float(np.linalg.norm(v, axis=1).max()))
            row['max_acceleration_rad_s2'] = max(row['max_acceleration_rad_s2'], float(np.linalg.norm(a, axis=1).max()))
            row['max_jerk_rad_s3'] = max(row['max_jerk_rad_s3'], float(np.linalg.norm(np.diff(a, axis=0) / np.diff(times)[:, None], axis=1).max()))
            qs[i] = states[-1]; velocities[i] = v[-1]; accelerations[i] = a[-1]; curve_times[i] = start_time + duration

        for decision in range(max_decisions):
            active = []; indices = {}; stop_only = set()
            for i, (row, task) in enumerate(zip(rows, task_rows)):
                index = task_index(task, decision * dt); indices[i] = index
                error = errors(qs[i], task['targets'][index], task['axes'][index])
                row['tracking_error_mm'].append(error['error_mm'])
                row['tracking_axis_error_deg'].append(error['orientation_error_deg'])
                if row['status'] != 'running': continue
                if (index == len(task['targets']) - 1 and error['error_mm'] <= 1
                        and error['orientation_error_deg'] <= 1 and np.linalg.norm(velocities[i]) < 1e-10
                        and np.linalg.norm(accelerations[i]) < 1e-10):
                    row.update(success=True, status='success', success_time_s=decision * dt); continue
                active.append(i)
                if (prefix_capture and np.linalg.norm(task['target_velocities'][index]) <= 1e-6
                        and error['error_mm'] <= 1 and error['orientation_error_deg'] <= 1):
                    stop_only.add(i)
            if not active: continue  # retain equal-duration error measurements
            observations = []
            sampled = [i for i in active if i not in stop_only]
            for i in sampled:
                task = task_rows[i]; index = indices[i]
                goal = None if cartesian else task['goals'][index]
                observations.append(condition(policy, qs[i], goal, task['targets'][index], task['axes'][index],
                                              task['box'], velocities[i], accelerations[i], task['target_velocities'][index]))
            chunks = policy.sample(np.stack(observations), samples, inference_steps) if sampled else []
            by_scene = dict(zip(sampled, chunks))
            for i in active:
                row = rows[i]; task = task_rows[i]; index = indices[i]; q = qs[i]
                target = task['targets'][index]; axis = task['axes'][index]
                row['decisions'] += 1; candidates = []
                if i in stop_only:
                    candidates.append((-1, cartesian_score(q, target, axis), q[None], False))
                else: row['proposals'] += samples
                for raw in by_scene.get(i, []):
                    chunk, reason = checked_chunk(envs[i], q, raw)
                    if chunk is None: row['rejections'][reason] += 1; continue
                    captured = False
                    if prefix_capture:
                        chunk, captured = static_capture_prefix(chunk, target, axis, task['target_velocities'][index])
                    score = cartesian_score(chunk[-1], target, axis) + .25 * cartesian_score(chunk[min(1, len(chunk)-1)], target, axis)
                    if captured: row['neural_prefix_candidates'] += 1
                    candidates.append((0 if captured else 1, score, chunk, captured))
                accepted = None
                for priority, _, chunk, captured in sorted(candidates, key=lambda pair: pair[:2]):
                    try:
                        _, _, accepted = prepare(envs[i], np.vstack((q, chunk)), velocities[i], accelerations[i])
                        if captured: row['neural_prefix_capture'] += 1
                        if priority == -1: row['static_target_brakes'] += 1
                        break
                    except ValueError: row['join_rejections'] += 1
                if accepted is not None: curves[i] = accepted; curve_times[i] = 0.
                else: row['stalls'] += 1
                advance(i, curves[i], curve_times[i])
                if accepted is not None and np.linalg.norm(qs[i] - q) < .004: row['stalls'] += 1
            if decision % 20 == 0:
                print(f'{"cartesian" if cartesian else "joint"} {decision}: {sum(r["success"] for r in rows)}/{len(rows)}', flush=True)

        for row, q, v, a, task in zip(rows, qs, velocities, accelerations, task_rows):
            error = errors(q, task['targets'][-1], task['axes'][-1])
            row.update(final_error_mm=error['error_mm'], final_angle_deg=error['orientation_error_deg'],
                       final_speed_rad_s=float(np.linalg.norm(v)), rejections=dict(row['rejections']),
                       tracking_rmse_mm=float(np.sqrt(np.mean(np.square(row['tracking_error_mm'])))),
                       axis_tracking_rmse_deg=float(np.sqrt(np.mean(np.square(row['tracking_axis_error_deg'])))),
                       reached_5mm_5deg=error['error_mm'] <= 5 and error['orientation_error_deg'] <= 5,
                       reached_10mm_5deg=error['error_mm'] <= 10 and error['orientation_error_deg'] <= 5)
            if (row['status'] == 'running' and error['error_mm'] <= 1 and error['orientation_error_deg'] <= 1
                    and np.linalg.norm(v) < 1e-10 and np.linalg.norm(a) < 1e-10):
                row.update(success=True, status='success', success_time_s=max_decisions * dt)
            if row['status'] == 'running': row['status'] = 'decision_limit'
        summary = {'episodes': len(rows), 'successes': sum(r['success'] for r in rows),
                   'within_5mm_5deg': sum(r['reached_5mm_5deg'] for r in rows),
                   'within_10mm_5deg': sum(r['reached_10mm_5deg'] for r in rows),
                   'mean_final_error_mm': float(np.mean([r['final_error_mm'] for r in rows])),
                   'median_final_error_mm': float(np.median([r['final_error_mm'] for r in rows])),
                   'mean_final_angle_deg': float(np.mean([r['final_angle_deg'] for r in rows])),
                   'mean_tracking_rmse_mm': float(np.mean([r['tracking_rmse_mm'] for r in rows])),
                   'mean_axis_tracking_rmse_deg': float(np.mean([r['axis_tracking_rmse_deg'] for r in rows])),
                   'tracking_measurement_duration_s': max_decisions * dt,
                   'by_kind': {kind: {'episodes': sum(r['kind'] == kind for r in rows),
                                      'successes': sum(r['kind'] == kind and r['success'] for r in rows)}
                               for kind in sorted({r['kind'] for r in rows})},
                   'initial_continuation_rejections': sum(r['status'] == 'initial_continuation_rejected' for r in rows),
                   'seconds': perf_counter() - began}
        for key in ('execution_collisions', 'stalls', 'join_rejections', 'proposals', 'joint_goal_capture',
                    'neural_prefix_capture', 'neural_prefix_candidates', 'static_target_brakes'):
            summary[key] = sum(r[key] for r in rows)
        for key in ('max_velocity_rad_s', 'max_acceleration_rad_s2', 'max_jerk_rad_s3',
                    'max_boundary_velocity_error', 'max_boundary_acceleration_error'):
            summary[key] = max(r[key] for r in rows)
        summary['raw_collision_rejections'] = sum(r['rejections'].get('raw_collision', 0) for r in rows)
        return {'summary': summary, 'rows': rows, 'ik_disabled': cartesian, 'goal_joint_labels_removed': cartesian}
    finally:
        for env in envs: env.close()


def cpu_latency(checkpoint, task):
    from .cartesian_conditioning import supports
    from .service import condition
    policy = DiffusionPolicy(checkpoint, 'cpu', SEED)
    goal = None if supports(policy) else task['goals'][0]
    obs = condition(policy, task['q'][0], goal, task['targets'][0], task['axes'][0], task['box'])[None]
    policy.sample(obs, 4, 32); timings = []
    for _ in range(10):
        began = perf_counter(); policy.sample(obs, 4, 32); timings.append((perf_counter() - began) * 1000)
    return {'samples': 10, 'median_ms': float(np.median(timings)), 'p95_ms': float(np.percentile(timings, 95)),
            'scope': 'Single input; four candidates, 32 denoising steps, CPU four threads; inference only'}


def configuration(full=False, max_decisions=100, motion_max_decisions=120):
    return {'full_suite': full, 'samples': 4, 'inference_steps': 32, 'max_decisions': max_decisions,
            'motion_max_decisions': motion_max_decisions,
            'execute_seconds': .4, 'seed': SEED, 'motion_controller': 'collision_checked_bezier_c2',
            'ranking': 'Cartesian endpoint plus 0.25 second reference for both policies',
            'joint_endpoint_capture': False, 'static_neural_prefix_capture': True,
            'goal_rest_tolerance': 1e-10, 'candidate_ik': 'disabled with a raising stub'}


def run(name, baseline=BASELINE, candidate=CANDIDATE, source=SOURCE, augmentation=AUGMENTATION,
        output=OUTPUT, full=False, max_decisions=100, motion_max_decisions=120, device='cuda'):
    fixed, moving = tasks(source, augmentation, full); checkpoint = baseline if name == 'baseline' else candidate
    from .cartesian_conditioning import supports
    policy = DiffusionPolicy(checkpoint, device, SEED)
    if name == 'candidate' and not supports(policy): raise ValueError('Candidate does not use supported Cartesian conditioning')
    result = {'checkpoint_sha256': policy.sha256, 'observation_dim': policy.observation_dim,
              'conditioning': policy.conditioning,
              'source_sha256': file_sha256(source), 'augmentation_sha256': file_sha256(augmentation),
              'configuration': configuration(full, max_decisions, motion_max_decisions)}
    policy.generator.manual_seed(SEED); result['fixed'] = continuous_suite(policy, fixed, max_decisions)
    policy.generator.manual_seed(SEED + 1); result['motion'] = continuous_suite(policy, moving, motion_max_decisions)
    del policy; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    result['cpu_inference_latency'] = cpu_latency(checkpoint, fixed[0])
    write_json(Path(output).with_name(f'cartesian_{name}_comparison.json'), result)
    print(json.dumps({kind: result[kind]['summary'] for kind in ('fixed', 'motion')}), flush=True)
    return result


def merge(baseline=BASELINE, candidate=CANDIDATE, source=SOURCE, augmentation=AUGMENTATION,
          output=OUTPUT, full=False, max_decisions=100, motion_max_decisions=120):
    fixed, moving = tasks(source, augmentation, full); results = {}
    for name, checkpoint in (('baseline', baseline), ('candidate', candidate)):
        row = json.loads(Path(output).with_name(f'cartesian_{name}_comparison.json').read_text())
        if (row['checkpoint_sha256'] != file_sha256(checkpoint) or row['source_sha256'] != file_sha256(source)
                or row['augmentation_sha256'] != file_sha256(augmentation)
                or row['configuration'] != configuration(full, max_decisions, motion_max_decisions)
                or [r['id'] for r in row['fixed']['rows']] != [t['id'] for t in fixed]
                or [r['id'] for r in row['motion']['rows']] != [t['id'] for t in moving]):
            raise ValueError('Comparison task/configuration/checkpoint provenance differs')
        results[name] = row
    old = results['baseline']; new = results['candidate']
    checks = {
        'fixed_success_not_lower': new['fixed']['summary']['successes'] >= old['fixed']['summary']['successes'],
        'motion_success_not_lower': new['motion']['summary']['successes'] >= old['motion']['summary']['successes'],
        'fixed_position_error_not_higher': new['fixed']['summary']['mean_final_error_mm'] <= old['fixed']['summary']['mean_final_error_mm'],
        'motion_tracking_error_not_higher': new['motion']['summary']['mean_tracking_rmse_mm'] <= old['motion']['summary']['mean_tracking_rmse_mm'],
        'usable_fixed_precision': new['fixed']['summary']['within_5mm_5deg'] >= .8 * len(fixed),
        'usable_motion_precision': new['motion']['summary']['within_5mm_5deg'] >= .75 * len(moving),
        'executed_collision_free': all(new[k]['summary']['execution_collisions'] == 0 for k in ('fixed', 'motion')),
        'continuous_motion_limits': all(new[k]['summary']['max_velocity_rad_s'] <= 1 + 1e-6 and new[k]['summary']['max_acceleration_rad_s2'] <= 3 + 1e-6 for k in ('fixed', 'motion')),
        'continuous_boundaries_match': all(new[k]['summary']['max_boundary_velocity_error'] < 1e-6 and new[k]['summary']['max_boundary_acceleration_error'] < 1e-6 for k in ('fixed', 'motion')),
        'candidate_ik_disabled': all(new[k]['ik_disabled'] and new[k]['goal_joint_labels_removed'] for k in ('fixed', 'motion')),
        'no_joint_goal_capture': all(result[k]['summary']['joint_goal_capture'] == 0 for result in results.values() for k in ('fixed', 'motion')),
    }
    report = {'kind': 'paired_cartesian_c2_comparison', 'source_sha256': file_sha256(source),
              'augmentation_sha256': file_sha256(augmentation), 'configuration': configuration(full, max_decisions, motion_max_decisions),
              'results': results, 'deployment_checks': checks, 'recommended_for_deployment': all(checks.values()),
              'limitations': 'One training seed. Held-out reachable targets are derived by FK from expert labels; v3 receives no goal joints. This demonstrates runtime IK independence, not guaranteed feasibility of unreachable inputs. Moving trajectories follow expert targets. Compute delay excluded from tracking; CPU denoising latency reported separately. No hardware tests.'}
    write_json(output, report); print(json.dumps({'checks': checks}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', choices=('baseline', 'candidate'))
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    parser.add_argument('--candidate', type=Path, default=CANDIDATE)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--augmentation', type=Path, default=AUGMENTATION)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--max-decisions', type=int, default=100)
    parser.add_argument('--motion-max-decisions', type=int, default=120)
    parser.add_argument('--device', choices=('cpu', 'cuda', 'auto'), default='cuda')
    args = vars(parser.parse_args()); name = args.pop('only')
    if name: run(name, **args)
    else: args.pop('device'); merge(**args)
