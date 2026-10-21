"""Frozen paired classical RRT* versus production diffusion-v2 motion planning.

Both methods receive exactly the same independently validated precomputed IK
goal. This measures complete checked path success and planning compute time,
excluding common precomputed IK, checkpoint loading, and result-file writing.
No training, runtime deployment, server calls, or hardware IO occurs.
"""
import argparse
import atexit
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import multiprocessing as mp
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from ..kinematics import ARM
from ..pose5 import axis_error
from .benchmark_ik_refinement import sha256, stats, wilson, write_json
from .continuous_rollout import rollout
from .env import ArmEnv
from .planner import RRTConfig, plan as rrt_plan
from .service import condition, info
from .smoothing import prepare, motion_bounds

PIN = '6582cbf4e1d2575007cfead8f99ed5f15a7b7021083d0eb200c61b224401bddc'
CONFIG = {'seed': 42, 'samples': 4, 'inference_steps': 32, 'max_decisions': 200}
SOURCES = ('simulation/diffusion/continuous_rollout.py', 'simulation/diffusion/policy.py',
    'simulation/diffusion/model.py', 'simulation/diffusion/env.py',
    'simulation/diffusion/planner.py', 'simulation/diffusion/smoothing.py',
    'simulation/continuous.py', 'simulation/kinematics.py',
    'simulation/diffusion/service.py')
_ENVS = None
_POLICY = None
_OUTPUT = None
_TIMEOUT = None


def inputs(output, corpus, ik_rows):
    corpus_meta = json.loads(corpus.with_suffix('.json').read_text())
    if sha256(corpus) != corpus_meta['dataset_sha256']:
        raise ValueError('Frozen input corpus changed')
    with np.load(corpus, allow_pickle=False) as archive:
        data = {k: archive[k].copy() for k in archive.files}
    count = len(data['case_id'])
    rows = [json.loads(line) for line in ik_rows.read_text().splitlines()]
    rows.sort(key=lambda r: r['case_id'])
    if [r['case_id'] for r in rows] != list(range(count)):
        raise ValueError('Missing/duplicate saved IK goals')
    if not all(r['with_learned']['success'] for r in rows):
        raise ValueError('A shared validated IK goal is required for every case')
    data['q_goal_rad'] = np.array([r['with_learned']['joints_rad'] for r in rows])
    path = output/'inputs.npz'
    if not path.exists():
        np.savez_compressed(path, **data)
    else:
        with np.load(path, allow_pickle=False) as archive:
            if any(not np.array_equal(archive[k], v) for k, v in data.items()):
                raise ValueError('Existing planner inputs differ')
    metadata = {'count': count, 'corpus': str(corpus.resolve()),
                'corpus_sha256': sha256(corpus), 'source_ik_rows': str(ik_rows.resolve()),
                'source_ik_rows_sha256': sha256(ik_rows), 'inputs_sha256': sha256(path),
                'goal_source': 'Accepted with-learned numerical IK endpoints from frozen 10000-case comparison; exact same joints for both planners',
                'scene': corpus_meta['scene'], 'external_obstacle': None}
    write_json(output/'inputs.json', metadata)
    return data, metadata


def initialize(checkpoint, output, timeout, first):
    global _ENVS, _POLICY, _OUTPUT, _TIMEOUT
    import torch
    from .policy import DiffusionPolicy
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    _OUTPUT = Path(output)
    _TIMEOUT = timeout
    _POLICY = DiffusionPolicy(checkpoint, device='cuda', seed=42,
                              cuda_graph=True, scheduler_cache=True)
    torch.set_num_threads(1)
    if _POLICY.sha256 != PIN or _POLICY.observation_dim != 34:
        raise ValueError('Production v2 policy differs')
    _ENVS = {name: ArmEnv() for name in ('classical', 'diffusion')}
    for env in _ENVS.values():
        env.set_obstacle(None)
        atexit.register(env.close)
    # Warm GPU allocations/capture outside measurements, then reset the RNG.
    _, current, goal, target, axis = first
    obs = condition(_POLICY, current, goal, target, axis, None)[None]
    _POLICY.sample(obs, 4, 32)
    torch.cuda.synchronize()
    _POLICY.generator.manual_seed(42)


def endpoint_errors(q, target, axis):
    fk = ARM.transform(q)
    return float(np.linalg.norm(fk[:3, 3]-target)*1000), float(axis_error(fk[:3, 2], axis)[2])


def evaluate_method(mode, current, goal, target, axis, case_id):
    import torch
    env = _ENVS[mode]
    env._valid_cache.clear()
    env.reset(current)
    if not env.valid(current) or not env.valid(goal):
        raise AssertionError('Shared current/IK goal fails the runtime scene')
    goal_pos, goal_angle = endpoint_errors(goal, target, axis)
    if goal_pos > 1. or goal_angle > 1.:
        raise AssertionError('Shared IK goal misses the original target')
    env.reset(current)
    if mode == 'diffusion':
        _POLICY.generator.manual_seed(42)
        torch.cuda.synchronize()
    began = perf_counter()
    result = {'success': False, 'goal_position_error_mm': goal_pos,
              'goal_direction_error_deg': goal_angle}
    if mode == 'classical':
        points, diagnostics = rrt_plan(env, current, goal, seed=42, config=RRTConfig())
        result['diagnostics'] = diagnostics
        status = diagnostics['status']
        candidate_success = points is not None
        if points is None:
            points = current[None].copy()
    else:
        points, status, decisions, proposals, rejects, capture = rollout(
            env, _POLICY, current, goal, target, axis, None, CONFIG, began+_TIMEOUT)
        points = np.asarray(points)
        result['diagnostics'] = {'decisions': decisions, 'proposals': proposals,
                                 'rejections': rejects, 'endpoint_capture_steps': capture}
        candidate_success = status == 'success'
        if candidate_success and np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) > .1+1e-10):
            status = 'final_step_validation_failed'
            candidate_success = False
    result['planner_status'] = status
    before_smoothing = perf_counter()
    result['search_ms'] = (before_smoothing-began)*1000
    result['raw_vertices'] = len(points)
    final_position, final_direction = endpoint_errors(points[-1], target, axis)
    states = times = smoothing = None
    if candidate_success:
        if final_position > 1. or final_direction > 1. or not env.path_valid(points):
            status = 'final_path_validation_failed'
        else:
            try:
                states, times, smoothing = prepare(env, points, speed=1., acceleration=3., dt=.02)
            except ValueError as error:
                status = 'smoothing_validation_failed'
                result['failure_reason'] = str(error)
            else:
                final_position, final_direction = endpoint_errors(states[-1], target, axis)
                maximum_speed, maximum_acceleration = motion_bounds(states, times)
                if (not np.array_equal(states[0], current) or final_position > 1.
                        or final_direction > 1. or not env.path_valid(states)
                        or maximum_speed > 1.+1e-8 or maximum_acceleration > 3.+1e-8):
                    raise AssertionError('Final trajectory violated common validation gates')
                result['success'] = True
                status = 'success'
                result.update(duration_s=float(times[-1]), frames=len(states),
                              max_velocity_rad_s=maximum_speed,
                              max_acceleration_rad_s2=maximum_acceleration,
                              joint_travel_rad=float(np.linalg.norm(np.diff(states, axis=0), axis=1).sum()),
                              path_collision_rechecked=True,
                              smoothing={k: smoothing[k] for k in ('method', 'path_vertices',
                                  'duration_s', 'preserved_corner_fallback', 'specification_checked')})
    if mode == 'diffusion':
        torch.cuda.synchronize()
    result['planning_ms'] = (perf_counter()-began)*1000
    result['status'] = status
    result['position_error_mm'] = final_position
    result['direction_error_deg'] = final_direction
    result['final_joints_rad'] = points[-1].tolist()
    # Persist every geometric prefix and complete successful timed path. Result
    # IO is excluded from planning measurements.
    folder = _OUTPUT/'trajectories'/mode/f'{case_id//100:03d}'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/f'{case_id:05d}.npz'
    arrays = {'raw_q_rad': points}
    if result['success']:
        arrays.update(q_rad=states, time_s=times)
    np.savez_compressed(path, **arrays)
    result['trajectory_file'] = str(path.relative_to(_OUTPUT))
    result['trajectory_sha256'] = sha256(path)
    return result


def evaluate_case(task):
    case_id, current, goal, target, axis = task
    row = {'case_id': int(case_id)}
    for mode in (('classical', 'diffusion') if case_id % 2 == 0 else ('diffusion', 'classical')):
        row[mode] = evaluate_method(mode, current, goal, target, axis, int(case_id))
    return row


def summarize(rows):
    result = {'cases': len(rows)}
    for mode in ('classical', 'diffusion'):
        values = [r[mode] for r in rows]
        passed = [r for r in values if r['success']]
        result[mode] = {'successes': len(passed), 'failures': len(values)-len(passed),
            'success_rate_percent': len(passed)/len(values)*100,
            'success_rate_95pct_wilson_percent': wilson(len(passed), len(values)),
            'status_counts': dict(Counter(r['status'] for r in values)),
            'planning_ms_all_cases': stats([r['planning_ms'] for r in values]),
            'planning_ms_successes_only': stats([r['planning_ms'] for r in passed]),
            'position_error_mm_successes_only': stats([r['position_error_mm'] for r in passed]),
            'direction_error_deg_successes_only': stats([r['direction_error_deg'] for r in passed]),
            'motion_duration_s_successes_only': stats([r['duration_s'] for r in passed]),
            'joint_travel_rad_successes_only': stats([r['joint_travel_rad'] for r in passed]),
            'max_velocity_rad_s': max((r['max_velocity_rad_s'] for r in passed), default=0.),
            'max_acceleration_rad_s2': max((r['max_acceleration_rad_s2'] for r in passed), default=0.)}
    result['paired_outcomes'] = {
        'both_success': sum(r['classical']['success'] and r['diffusion']['success'] for r in rows),
        'classical_only_success': sum(r['classical']['success'] and not r['diffusion']['success'] for r in rows),
        'diffusion_only_success': sum(not r['classical']['success'] and r['diffusion']['success'] for r in rows),
        'both_fail': sum(not r['classical']['success'] and not r['diffusion']['success'] for r in rows)}
    result['classical_search_statuses'] = dict(Counter(r['classical']['planner_status'] for r in rows))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, default=Path('output/ik_refinement_20261008/full/cases.npz'))
    parser.add_argument('--ik-rows', type=Path, default=Path('output/ik_refinement_20261008/learned_enabled_comparison/rows.jsonl'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--limit', type=int, default=10000)
    parser.add_argument('--timeout', type=float, default=90.)
    args = parser.parse_args()
    if args.workers < 1 or args.limit < 1 or args.timeout <= 0:
        raise ValueError('Positive worker/count/time limits required')
    status = info()
    if status['status'] != 'ready' or status['checkpoint_sha256'] != PIN:
        raise ValueError('Pinned active diffusion v2 checkpoint is required')
    checkpoint = status['checkpoint_path']
    args.output.mkdir(parents=True, exist_ok=True)
    data, metadata = inputs(args.output, args.corpus, args.ik_rows)
    indices = np.sort(np.random.default_rng(202610082).permutation(metadata['count'])[:args.limit])
    metadata['selected_case_ids'] = indices.tolist()
    row_file = args.output/'rows.jsonl'
    rows = []
    if row_file.exists():
        rows = [json.loads(line) for line in row_file.read_text().splitlines()]
    done = {r['case_id'] for r in rows}
    if len(done) != len(rows) or not done <= set(indices.tolist()):
        raise ValueError('Duplicate or unexpected existing cases')
    source_hashes = {p: sha256(p) for p in SOURCES}
    experiment = {'inputs': metadata, 'checkpoint_sha256': PIN, 'source_hashes': source_hashes,
                  'benchmark_source_sha256': sha256(__file__), 'workers': args.workers,
                  'diffusion_config': CONFIG, 'diffusion_timeout_s': args.timeout,
                  'rrt_configuration': asdict(RRTConfig()), 'selected_cases': indices.tolist()}
    experiment_path = args.output/'experiment.json'
    if experiment_path.exists() and json.loads(experiment_path.read_text()) != experiment:
        raise ValueError('Existing experiment differs; choose a fresh output directory')
    write_json(experiment_path, experiment)
    tasks = [(int(i), data['q_current_rad'][i], data['q_goal_rad'][i],
              data['target_xyz_m'][i], data['target_axis_base'][i]) for i in indices if int(i) not in done]
    tick = perf_counter()
    if tasks:
        with row_file.open('a') as stream, ProcessPoolExecutor(
                max_workers=args.workers, mp_context=mp.get_context('spawn'),
                initializer=initialize, initargs=(checkpoint, str(args.output.resolve()), args.timeout, tasks[0])) as executor:
            futures = [executor.submit(evaluate_case, task) for task in tasks]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                stream.flush()
                progress = {'completed': len(rows), 'total': len(indices),
                            'classical_successes': sum(r['classical']['success'] for r in rows),
                            'diffusion_successes': sum(r['diffusion']['success'] for r in rows),
                            'seconds_this_run': perf_counter()-tick,
                            'last_case': row['case_id'],
                            'last_status': {m: row[m]['status'] for m in ('classical', 'diffusion')}}
                write_json(args.output/'progress.json', progress)
                if len(rows) % 10 == 0 or len(rows) == len(indices) or len(indices) <= 20:
                    print(json.dumps(progress), flush=True)
    if sha256(checkpoint) != PIN or source_hashes != {p: sha256(p) for p in SOURCES}:
        raise RuntimeError('Model or planner sources changed during evaluation')
    if metadata['inputs_sha256'] != sha256(args.output/'inputs.npz'):
        raise RuntimeError('Frozen inputs changed')
    rows.sort(key=lambda r: r['case_id'])
    if [r['case_id'] for r in rows] != indices.tolist():
        raise AssertionError('Missing case result')
    summary = summarize(rows)
    report = {'summary': summary, 'experiment': experiment,
              'scope': 'Same validated numerical IK endpoints for both; full path success including final checked smoothing, pose tolerance, scene validity and motion limits. Common precomputed IK, model loading and result-file IO excluded from measured planning time.',
              'timing': 'Per-case wall-clock timings; methods alternate order; workers share CPU/GPU. Parallel timings are load-dependent and are not isolated latency estimates.',
              'limits': {'position_error_mm': 1., 'directed_tool_z_error_deg': 1.,
                         'vector_speed_rad_s': 1., 'vector_acceleration_rad_s2': 3.,
                         'collision_clearance_mm': 1., 'path_sample_rad': .04},
              'limitations': 'Reachable full-range FK endpoint corpus, no external box; IK reachability does not prove path connectivity. Algorithms have their respective fixed production search budgets. Diffusion timeout is wall-clock and may be affected by worker/GPU contention. No hardware execution.',
              'per_case_results': str(row_file.resolve()),
              'seconds_this_run': perf_counter()-tick}
    write_json(args.output/'report.json', report)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
