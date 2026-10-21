"""Reproducible dry-run IK/path ablations on identical held-out cases.

Real cases default to conservative URDF self-collision validity. Explicit limits
mode reproduces the original ablation. Synthetic constraints remain separately
labeled. Goal accuracy, endpoint validity and path feasibility are separate.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter
import numpy as np
from .kinematics import ARM
from . import solvers, pose5
from .planning import IKConfig, query
from .prm import JointSpace, PRMConfig, Roadmap, fk_samples, planning_space


def record(case, result):
    frames = result.get('frames', [])
    goals = result.get('ik', {}).get('goals', [])
    endpoint = np.deg2rad(frames[-1]['q']) if frames else (np.array(goals[0]['q_rad']) if goals else None)
    best = result.get('ik', {}).get('best_error') or {}
    return {'name': case['name'], 'status': result['status'], 'reason': result.get('reason'),
            'goal_error_mm': result.get('final_error_mm', best.get('error_mm')),
            'direction_error_deg': result.get('final_orientation_error_deg', best.get('orientation_error_deg')),
            'ik_success': bool(goals) if 'ik' in result else result.get('ik_success', False),
            'success': result['status'] == 'success', 'ik_ms': result.get('ik_ms', 0.),
            'planning_ms': result.get('planning_ms', 0.),
            'path_length_rad': result.get('path_length_rad', 0.),
            'scaled_path_cost': result.get('path_cost'),
            'endpoint_rad': endpoint.tolist() if endpoint is not None else None,
            'fallback_reasons': result.get('fallback_reasons', []),
            'attempts': result.get('ik', {}).get('attempts', [])}


def existing(case, method, space, steps, task='position', max_step_rad=.04):
    q = np.deg2rad(case['start_deg']); target = np.array(case['target_mm'])/1000
    if not space.valid(q):
        return {'status': 'invalid_start', 'reason': space.diagnostic(q)['reason'],
                'ik_success': False, 'ik_ms': 0., 'planning_ms': 0., 'fallback_reasons': ['invalid_start']}
    solved = (pose5.solve(q, target, case['orientation_xy_deg'], method, steps=steps)
              if task == 'pose5' else solvers.solve(q, target, method, steps=steps))
    endpoint = np.deg2rad(solved['frames'][-1]['q'])
    valid_goal = solved['converged'] and space.valid(endpoint)
    points = np.deg2rad([f['q'] for f in solved['frames']]); began = perf_counter()
    valid_path = all(space.edge_valid(a, b, max_step_rad) for a, b in zip(points, points[1:]))
    if not valid_goal: status = 'local_rollout_failed' if method == 'conditional' else 'ik_solution_not_found'
    elif not valid_path: status = 'local_rollout_constraint_violation'
    else: status = 'success'
    return {**solved, 'status': status, 'reason': solved['stop_reason'], 'ik_success': bool(valid_goal),
            'ik_ms': solved['elapsed_ms'], 'planning_ms': (perf_counter()-began)*1000,
            'path_length_rad': float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()),
            'fallback_reasons': [] if status == 'success' else [solved['stop_reason']]}


def candidate_plan(case, roadmap, config, via):
    start = np.deg2rad(case['start_deg']); target = np.array(case['target_mm'])/1000
    # Direct-only ablation still sees exactly the same roadmap IK seeds. It
    # removes graph routing only, keeping endpoint generation identical.
    result = query(start, target, roadmap, config, case.get('orientation_xy_deg'), use_roadmap=via)
    attempts = result.get('ik', {}).get('attempts', [])
    result['fallback_reasons'] = list(dict.fromkeys(
        [a['rollout_status'] for a in attempts if a.get('rollout_status') != 'success' and 'rollout_status' in a]
        + [a['source']+':'+a.get('stop_reason', 'inaccurate_or_invalid') for a in attempts if not a.get('accepted', False) and 'rollout_status' not in a]
        + ([result['reason']] if result['status'] != 'success' else [])))
    return result


def summarize(name, rows):
    errors = [r['goal_error_mm'] for r in rows if r['goal_error_mm'] is not None]
    successes = [r for r in rows if r['success']]
    angles = [r['direction_error_deg'] for r in rows if r['direction_error_deg'] is not None]
    eligible = sum(r['status'] != 'invalid_start' for r in rows)
    return {'method': name, 'success': len(successes), 'ik_success': sum(r['ik_success'] for r in rows),
            'total': len(rows), 'success_rate': len(successes)/len(rows),
            'eligible_starts': eligible, 'eligible_success_rate': len(successes)/eligible if eligible else None,
            'statuses': dict(Counter(r['status'] for r in rows)),
            'mean_goal_error_mm': float(np.mean(errors)) if errors else None,
            'mean_direction_error_deg': float(np.mean(angles)) if angles else None,
            'mean_ik_ms': float(np.mean([r['ik_ms'] for r in rows])),
            'mean_planning_ms': float(np.mean([r['planning_ms'] for r in rows])),
            'mean_success_path_length_rad': float(np.mean([r['path_length_rad'] for r in successes])) if successes else None,
            'fallback_reasons': dict(Counter(reason for r in rows for reason in r['fallback_reasons'])), 'cases': rows}


def routing_checks(samples=400, seed=9001):
    """Known q_goal isolates planning benefit without any IK confound."""
    def state(x, y=0):
        q = ARM.center.copy(); q[:2] = [x, y]; return q
    start = state(-1); goal = state(1)
    space = JointSpace(lambda q: np.linalg.norm(q[:2]) >= .35,
                       {'checker': 'synthetic_disk_v1', 'environment': 'q1_q2_radius_0.35'})
    road = Roadmap(space, PRMConfig(samples=samples, neighbors=12, max_step_rad=.04, seed=seed)).grow()
    rows = []
    for name, roadmap in [('direct_only', Roadmap(space, road.config)), ('prm', road)]:
        result = roadmap.plan(start, [goal], shortcut=True)
        rows.append({'method': name, **result})
    wall = JointSpace(lambda q: abs(q[0]) > .15,
                      {'checker': 'synthetic_wall_v1', 'environment': 'q1_wall_width_0.3'})
    disconnected = Roadmap(wall, road.config).grow().plan(start, [goal])
    return {'assumptions': space.metadata(), 'goal_error_mm': 0., 'roadmap': road.stats,
            'cases': rows, 'disconnected_wall_status': disconnected['status'],
            'interpretation': 'Known FK endpoint, synthetic q1/q2 constraints only; not physical collision validation.'}


def self_routing(roadmap):
    """A fixed valid endpoint with an actual URDF-proxy self-collision detour."""
    fixture = json.loads((Path(__file__).parent/'fixtures/self_collision_path.json').read_text())
    start, goal = np.deg2rad([fixture['start_deg'], fixture['goal_deg']])
    rows = []
    for name, road in [('direct_only', Roadmap(roadmap.space, roadmap.config)), ('prm', roadmap)]:
        rows.append({'method': name, **road.plan(start, [goal], shortcut=True)})
    return {'fixture': fixture, 'goal_error_mm': 0., 'rows': rows,
            'interpretation': 'Known valid goal configuration; URDF convex self-collision proxies, no environment checking.'}


def benchmark(count=24, samples=400, seed=9001, steps=240, task='position', collision='self', clearance_mm=1., max_step_rad=.04):
    if task == 'position': solvers.models()  # Warm load outside all timings.
    roadmap = Roadmap(planning_space(collision, clearance_mm),
                      PRMConfig(samples=samples, seed=seed, max_step_rad=max_step_rad)).grow(
        samples=fk_samples(solvers.ARTIFACTS/'dataset.npz'))
    fixtures = pose5.cases(seed=seed, count=count)
    configs = {
        'numerical_multiseed_direct': IKConfig(task=task, nn='none', steps=steps, seed=seed),
        'numerical_multiseed_prm': IKConfig(task=task, nn='none', steps=steps, seed=seed),
    }
    if task == 'position':
        configs.update({
            'conditional_refined_direct': IKConfig(nn='conditional', numerical_fallback=False, steps=steps, seed=seed),
            'nn_seed_refined_direct': IKConfig(nn='seed', numerical_fallback=False, steps=steps, seed=seed),
            'nn_prm': IKConfig(steps=steps, seed=seed),
            'nn_multiseed_direct': IKConfig(steps=steps, seed=seed),
            'numerical_matched_budget_prm': IKConfig(nn='none', random_seeds=8, steps=steps, seed=seed),
        })
    rows = []
    for method in (('conditional', 'seed_dls') if task == 'position' else ('bounded_dls',)):
        records = [record(case, existing(case, method, roadmap.space, steps, task, max_step_rad)) for case in fixtures]
        row = summarize('existing_'+method, records); rows.append(row)
        print(f'{row["method"]}: IK {row["ik_success"]}/{count}; path {row["success"]}/{count}', flush=True)
    for name, config in configs.items():
        records = [record(case, candidate_plan(case, roadmap, config, name.endswith('prm'))) for case in fixtures]
        row = summarize(name, records); row['config'] = asdict(config); rows.append(row)
        print(f'{name}: IK {row["ik_success"]}/{count}; path {row["success"]}/{count}; '
              f'IK {row["mean_ik_ms"]:.1f} ms; planning {row["mean_planning_ms"]:.1f} ms', flush=True)
    return {'task': task, 'seed': seed, 'cases': fixtures, 'rows': rows, 'roadmap': roadmap.stats,
            'prm_config': asdict(roadmap.config), 'metadata': roadmap.space.metadata(),
            'models_sha256': hashlib.sha256((solvers.ARTIFACTS/'models.json').read_bytes()).hexdigest(),
            'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'platform': platform.platform()},
            'synthetic_routing': routing_checks(samples, seed),
            'self_collision_routing': self_routing(roadmap) if collision == 'self' else None,
            'timing': 'CPU wall time, NumPy NN load excluded. Offline roadmap construction excluded from query timings; grow time recorded separately. Cache loading is not benchmarked. OPENBLAS_NUM_THREADS=1.',
            'interpretation': ('Self-collision mode: original held-out starts may be invalid; count them separately. Hull proxies are conservative and motion checks discretized. '
                              if collision == 'self' else 'Limit-only space is convex, so graph detours add no feasibility. ')
                              + 'Roadmap seeds may improve endpoint IK. '
                              'Numerical and NN multiseed variants share current/random/roadmap fallback seeds; NN adds four learned seeds (extra budget). '
                              'Direct/PRM pairs have identical goal generation; compare them to isolate graph routing. '
                              'Use attempts, success and timing to judge NN value, not only success on this small development set.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=24)
    parser.add_argument('--samples', type=int, default=400)
    parser.add_argument('--seed', type=int, default=9001)
    parser.add_argument('--steps', type=int, default=240)
    parser.add_argument('--task', choices=['position', 'pose5'], default='position')
    parser.add_argument('--collision', choices=['self', 'limits'], default='self')
    parser.add_argument('--self-clearance-mm', type=float, default=1.)
    parser.add_argument('--max-step-rad', type=float, default=.04)
    parser.add_argument('--output', type=Path, default=Path('simulation/artifacts/benchmark_prm_self.json'))
    args = parser.parse_args()
    if not 1 <= args.count <= 1000: parser.error('count must be between 1 and 1000')
    result = benchmark(args.count, args.samples, args.seed, args.steps, args.task,
                       args.collision, args.self_clearance_mm, args.max_step_rad)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f'Synthetic routing: {[r["status"] for r in result["synthetic_routing"]["cases"]]} → {args.output}', flush=True)


if __name__ == '__main__': main()
