"""Matched-budget guided/uniform routing repair comparison; simulation only.

Known joint goals isolate routing from IK. Every obstructed task has a checked
full-roadmap witness, retained for audit and never supplied to repair. Sparse
profiles are explicit coverage stress tests, not the production roadmap.
"""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np

from . import guided_repair, planning, service
from .kinematics import ARM
from .prm import Roadmap, planning_space


def sparse_roadmap(road, count, seed):
    """Induced subgraph: keep validated existing edges, choose nodes independently."""
    indices = np.sort(np.random.default_rng(seed).choice(len(road.q), count, replace=False))
    mapping = {int(old): new for new, old in enumerate(indices)}
    sparse = Roadmap(road.space, road.config)
    sparse.q = road.q[indices].copy()
    sparse.poses = road.poses[indices].copy()
    sparse.edges = {new: {mapping[other]: cost for other, cost in road.edges[int(old)].items()
                          if other in mapping} for new, old in enumerate(indices)}
    return sparse, indices.tolist()


def cases(road, count, direct_controls, seed):
    rng = np.random.default_rng(seed)
    tasks = []
    selection = Counter()
    fixture = json.loads((Path(__file__).parent/'fixtures/self_collision_path.json').read_text())
    fixed = (np.deg2rad(fixture['start_deg']), np.deg2rad(fixture['goal_deg']))

    def add(start, goal, group, name):
        witness = road.plan(start, [goal], shortcut=True)
        if witness['status'] != 'success':
            selection['no_full_roadmap_witness'] += 1
            return False
        tasks.append({'id': name, 'group': group, 'start_deg': np.rad2deg(start).tolist(),
                      'goal_deg': np.rad2deg(goal).tolist(),
                      'target_mm': (ARM.fk(goal)*1000).tolist(), 'target_axis': ARM.transform(goal)[:3, 2].tolist(),
                      'witness_waypoints_rad': witness['waypoints_rad'],
                      'witness_path_cost': witness['path_cost']})
        return True

    if count:
        if not road.space.valid(fixed[0]) or not road.space.valid(fixed[1]) or road.space.edge_valid(*fixed, road.config.max_step_rad):
            raise ValueError('Fixed self-collision fixture no longer has valid obstructed endpoints')
        if not add(*fixed, 'obstructed', 'fixed-self-collision'):
            raise ValueError('Fixed fixture lost its full-roadmap witness')
    for _ in range(max(1, count)*500):
        if len(tasks) >= count:
            break
        selection['candidate_pairs'] += 1
        start, goal = rng.uniform(ARM.lower, ARM.upper, (2, 6))
        # Avoid irrelevant extra free-spin travel in the known endpoint task.
        goal[5] = start[5]
        if not road.space.valid(start) or not road.space.valid(goal):
            selection['invalid_endpoints'] += 1
            continue
        if road.space.edge_valid(start, goal, road.config.max_step_rad):
            selection['unobstructed_pairs'] += 1
            continue
        if add(start, goal, 'obstructed', f'obstructed-{len(tasks)}'):
            print(f'Selected {len(tasks)}/{count} independently sampled obstructed tasks', flush=True)
    if len(tasks) < count:
        raise RuntimeError('Not enough checked obstructed tasks within selection budget')
    for _ in range(max(1, direct_controls)*500):
        if len(tasks) >= count+direct_controls:
            break
        start = rng.uniform(ARM.lower, ARM.upper)
        goal = ARM.clip(start+rng.normal(0., .1, 6))
        goal[5] = start[5]
        if (road.space.valid(start) and road.space.valid(goal)
                and road.space.edge_valid(start, goal, road.config.max_step_rad)):
            add(start, goal, 'direct_control', f'direct-{len(tasks)-count}')
    if len(tasks) != count+direct_controls:
        raise RuntimeError('Not enough direct controls within selection budget')
    return tasks, dict(selection)


def measure(road, task, repair_config=None):
    counts = {'state_validity_calls': 0, 'collision_queries': 0, 'edge_checks': 0}
    original_valid = road.space.valid
    original_collision = road.space.validity
    original_edge = road.space.edge_valid

    def valid(q):
        counts['state_validity_calls'] += 1
        return original_valid(q)

    def collision(q):
        counts['collision_queries'] += 1
        return original_collision(q)

    def edge(a, b, step):
        counts['edge_checks'] += 1
        return original_edge(a, b, step)

    start = np.deg2rad(task['start_deg'])
    goal = np.deg2rad(task['goal_deg'])
    began = perf_counter()
    with patch.object(road.space, 'valid', side_effect=valid), \
         patch.object(road.space, 'validity', side_effect=collision), \
         patch.object(road.space, 'edge_valid', side_effect=edge):
        result = (road.plan(start, [goal], shortcut=True) if repair_config is None else
                  guided_repair.plan(road, start, [goal], repair_config, shortcut=True))
        route_ms = (perf_counter()-began)*1000
        frames = []
        trajectory_ms = 0.
        if result['status'] == 'success':
            np.testing.assert_allclose(result['waypoints_rad'][0], start, rtol=0, atol=1e-12)
            np.testing.assert_allclose(result['waypoints_rad'][-1], goal, rtol=0, atol=1e-12)
            validation_began = perf_counter()
            frames = planning.trajectory(result['waypoints_rad'], np.asarray(task['target_mm'])/1000,
                                         road.space, road.config.max_step_rad, target_axis=task['target_axis'])
            trajectory_ms = (perf_counter()-validation_began)*1000
            assert frames[-1]['error_mm'] <= 1. and frames[-1]['orientation_error_deg'] <= 1.
            joints = np.deg2rad([frame['q'] for frame in frames])
            if len(joints) > 1:
                assert np.max(np.linalg.norm(np.diff(joints, axis=0), axis=1)) <= 1.2*.05+1e-9
    return {'status': result['status'], 'reason': result['reason'],
            'elapsed_ms': (perf_counter()-began)*1000, 'route_ms': route_ms, 'trajectory_ms': trajectory_ms,
            **counts, 'path_cost': result['path_cost'] if frames else None,
            'path_length_rad': result['path_length_rad'] if frames else None,
            'waypoint_count': len(result['waypoints_rad']), 'frame_count': len(frames),
            'simulated_duration_s': frames[-1]['time'] if frames else None,
            'eef_path_length_mm': float(np.linalg.norm(np.diff([frame['position'] for frame in frames], axis=0), axis=1).sum()) if frames else None,
            'position_error_mm': frames[-1]['error_mm'] if frames else None,
            'direction_error_deg': frames[-1]['orientation_error_deg'] if frames else None,
            'waypoints_rad': result['waypoints_rad'], 'repair': result.get('guided_repair')}


def summarize(tasks, rows, group='all'):
    selected = [(task, rows[task['id']]) for task in tasks if group == 'all' or task['group'] == group]
    successes = [(task, runs) for task, runs in selected if all(run['status'] == 'success' for run in runs)]
    for task, runs in selected:
        if len({run['status'] for run in runs}) != 1:
            raise ValueError('Deterministic route status changed across repetitions: '+task['id'])
    fields = ('elapsed_ms', 'route_ms', 'trajectory_ms', 'collision_queries', 'edge_checks')
    return {'tasks': len(selected), 'validated_successes': len(successes),
            'repair_triggered_tasks': sum(bool(runs[0]['repair'] and runs[0]['repair']['triggered']) for _, runs in selected),
            'recovered_tasks': [task['id'] for task, runs in successes if runs[0]['repair'] and runs[0]['repair']['recovered']],
            **{f'median_{field}': float(np.median([np.median([run[field] for run in runs]) for _, runs in selected])) if selected else None
               for field in fields},
            'median_success_path_cost': float(np.median([runs[0]['path_cost'] for _, runs in successes])) if successes else None,
            'median_success_waypoints': float(np.median([runs[0]['waypoint_count'] for _, runs in successes])) if successes else None}


def compare(tasks, baseline, after):
    good = lambda runs: all(run['status'] == 'success' for run in runs)
    common = [task['id'] for task in tasks if good(baseline[task['id']]) and good(after[task['id']])]
    return {'recovered': [task['id'] for task in tasks if not good(baseline[task['id']]) and good(after[task['id']])],
            'lost': [task['id'] for task in tasks if good(baseline[task['id']]) and not good(after[task['id']])],
            'common_successes': len(common),
            'common_success_median_path_cost_ratio': float(np.median([after[key][0]['path_cost']/baseline[key][0]['path_cost'] for key in common])) if common else None,
            'common_success_median_time_ratio': float(np.median([np.median([run['elapsed_ms'] for run in after[key]])/
                                                                np.median([run['elapsed_ms'] for run in baseline[key]]) for key in common])) if common else None}


def benchmark(output, count=20, direct_controls=4, sparse_nodes=64, repeats=3, seed=20261002,
              replay_from=None, repair_only=False, resume_from=None):
    output = Path(output)
    asset_paths = [service.ARTIFACTS/name for name in ('models.json', 'seed.pt', 'training.json', 'dataset.npz',
                                                    'roadmap.json', 'seed5_v1/models.json', 'seed5_v1/seed5_v1.pt')]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in asset_paths}
    print('Validating production roadmap (cold setup excluded from query timings)', flush=True)
    began = perf_counter()
    full = service.load_roadmap(service.ARTIFACTS/'roadmap.json', planning_space('self', 1.))
    cold_ms = (perf_counter()-began)*1000
    replay = None
    if replay_from is None:
        tasks, selection = cases(full, count, direct_controls, seed)
    else:
        replay_from = Path(replay_from)
        replay = json.loads(replay_from.read_text())
        if not replay.get('assets_unchanged') or replay['asset_sha256'] != hashes:
            raise ValueError('Replay source must be complete and use identical assets')
        tasks = replay['tasks']
        selection = {'replayed_exact_tasks': len(tasks)}
        for task in tasks:
            goal = np.deg2rad(task['goal_deg'])
            np.testing.assert_allclose(ARM.fk(goal)*1000, task['target_mm'], rtol=0, atol=1e-9)
            np.testing.assert_allclose(ARM.transform(goal)[:3, 2], task['target_axis'], rtol=0, atol=1e-12)
    sparse, indices = sparse_roadmap(full, sparse_nodes, seed ^ 0x280)
    variants = {'baseline': None,
                **{f'{family}_{budget}': guided_repair.RepairConfig(samples=budget,
                      guided_fraction=.75 if family == 'guided' else 0., seed=9001)
                   for budget in (32, 64) for family in ('uniform', 'guided')}}
    resume = json.loads(Path(resume_from).read_text()) if resume_from is not None else None
    configs = {name: asdict(config) if config is not None else None for name, config in variants.items()}
    if resume is not None and (resume['asset_sha256'] != hashes or resume['tasks'] != tasks or resume['configs'] != configs):
        raise ValueError('Resume source must use identical assets, tasks and repair configs')
    report = {'simulation_only': True, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'seed': seed, 'repeats': repeats, 'cold_roadmap_load_ms': cold_ms,
              'asset_sha256': hashes, 'tasks': tasks, 'selection': selection,
              'configs': configs,
              'timing': 'Warm routing plus final speed-limited trajectory validation; light validity/edge counters '
                        'are enabled for every variant. Cold setup and witness selection excluded. '
                        'Variant order randomized within each repeated task. OPENBLAS_NUM_THREADS=1.',
              'limitations': 'Development sample with full-roadmap route witnesses. Sparse induced subgraph '
                             'deliberately removes coverage; it is not the production roadmap. Known endpoints '
                             'isolate routing and do not measure IK success. Conservative convex self-collision '
                             'and discretized edge validation; no external obstacles or hardware motion.',
              'profiles': {}}
    if replay is not None:
        report['replay_source'] = {'path': str(replay_from),
                                   'sha256': hashlib.sha256(replay_from.read_bytes()).hexdigest()}
    if resume is not None:
        report['resume_source'] = {'path': str(resume_from),
                                   'sha256': hashlib.sha256(Path(resume_from).read_bytes()).hexdigest()}
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')

    rng = np.random.default_rng(seed+1)
    profiles = [(f'sparse_{sparse_nodes}' if sparse_nodes else 'empty_0', sparse)]
    if not repair_only:
        profiles.insert(0, ('full_400', full))
    for profile, road in profiles:
        rows = {name: {} for name in variants}
        prior = resume['profiles'].get(profile) if resume is not None else None
        if prior is not None:
            if prior['nodes'] != len(road.q) or prior['original_node_indices'] != (list(range(len(full.q))) if road is full else indices):
                raise ValueError('Resume graph profile differs')
            rows = prior['rows']
        report['profiles'][profile] = {'nodes': len(road.q), 'edges': sum(map(len, road.edges.values()))//2,
                                       'original_node_indices': list(range(len(full.q))) if road is full else indices,
                                       'rows': rows}
        original_q = road.q.copy()
        original_edges = json.dumps(road.edges, sort_keys=True)
        for index, task in enumerate(tasks):
            for name in variants:
                rows[name].setdefault(task['id'], [])
            for repeat in range(repeats):
                for name in rng.permutation(list(variants)):
                    if len(rows[name][task['id']]) > repeat:
                        continue
                    config = variants[name]
                    if config is not None:
                        config = guided_repair.RepairConfig(**{**asdict(config), 'seed': 9001+index*101})
                    rows[name][task['id']].append(measure(road, task, config))
            save()
            print(f"{profile} {index+1}/{len(tasks)} {task['id']}: "+', '.join(
                f"{name}={rows[name][task['id']][0]['status']} "
                f"({np.median([run['elapsed_ms'] for run in rows[name][task['id']]]):.0f} ms)" for name in variants), flush=True)
        np.testing.assert_array_equal(road.q, original_q)
        assert json.dumps(road.edges, sort_keys=True) == original_edges
        report['profiles'][profile].update(
            summary={name: {group: summarize(tasks, records, group) for group in ('all', 'obstructed', 'direct_control')}
                     for name, records in rows.items()},
            comparison={name: compare(tasks, rows['baseline'], records) for name, records in rows.items() if name != 'baseline'},
            matched_sampling_comparison={str(budget): compare(tasks, rows[f'uniform_{budget}'], rows[f'guided_{budget}'])
                                         for budget in (32, 64)}, graph_unchanged=True,
            actual_repetitions={task['id']: {name: len(rows[name][task['id']]) for name in variants} for task in tasks})
        save()
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path)] for path in asset_paths):
        raise ValueError('Existing assets changed during the experiment')
    report['assets_unchanged'] = True
    save()
    print(json.dumps({profile: data['summary'] for profile, data in report['profiles'].items()}, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('output/guided_repair/benchmark.json'))
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--direct-controls', type=int, default=4)
    parser.add_argument('--sparse-nodes', type=int, default=64)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20261002)
    parser.add_argument('--replay-from', type=Path, help='Use the exact tasks from a complete prior benchmark')
    parser.add_argument('--repair-only', action='store_true', help='Measure only the coverage-stress profile')
    parser.add_argument('--resume-from', type=Path, help='Retain completed same-input trials from a partial benchmark')
    args = parser.parse_args()
    if not 1 <= args.count <= 100 or not 0 <= args.direct_controls <= 20 or not 1 <= args.repeats <= 10:
        parser.error('Expected 1--100 obstructed tasks, 0--20 controls, 1--10 repeats')
    if not 0 <= args.sparse_nodes <= 400 or not 0 <= args.seed < 2**32-1:
        parser.error('Expected 0--400 sparse nodes and a supported RNG seed')
    benchmark(args.output, args.count, args.direct_controls, args.sparse_nodes, args.repeats, args.seed,
              args.replay_from, args.repair_only, args.resume_from)
