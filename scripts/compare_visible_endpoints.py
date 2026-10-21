"""Offline endpoint-selection ablation; no robot transport or ROS imports.

All policies share a fully generated IK bank. First-visible and best-visible
share identical route results, isolating selection quality. Their actual cost
differs: first-visible uses the measured prefix, best-visible uses the full bank.
The collision-first reference approximates production's ordering, not its live
streaming IK, short tracking horizon, or stage executor.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulation.kinematics import ARM
from simulation.physical_reach import ENVELOPE, target_precheck
from simulation.tracking_geometry import region_for, region_metrics, candidate_poses
from simulation.diffusion.env import ArmEnv
from simulation.diffusion.visibility import VisibilityEnv
from simulation.diffusion.endpoint_ik import IKSearchExhausted
from simulation.diffusion.planner import RRTConfig, plan
from simulation.diffusion.smoothing import prepare
from simulation.diffusion.ik_seed import checkpoint_status

STEP = .005
POLICIES = ('first_collision_free', 'first_visible', 'best_visible')


def region(case):
    target = ARM.transform(case['goal_witness_rad'])
    return region_for(np.array(case['object_m'])*1000, target[:3, 2],
                      {'mode': 'cone_disk', 'distance_mm': 150., 'half_angle_deg': 30.})


def cases_for(env, view, generated):
    fixture = ROOT/'simulation/scenarios/reachable_failures.json'
    cases = json.loads(fixture.read_text())['cases']
    for c in cases:
        c['kind'] = 'previously_discovered_hard_case'
    rng = np.random.default_rng(20261012)
    home = np.deg2rad([0., -40., -130., 80., 0., 50.])
    trials = 0
    while len(cases) < generated + 2:
        trials += 1
        if trials > 10000:
            raise RuntimeError('Could not generate enough visible direct-path controls')
        goal = (np.clip(home+rng.uniform(-.35, .35, 6), ARM.lower, ARM.upper)
                if len(cases) % 2 == 0 else rng.uniform(ARM.lower, ARM.upper))
        start = np.clip(goal+rng.uniform(-.45, .45, 6), ARM.lower, ARM.upper)
        if not env.valid(start) or not env.valid(goal):
            continue
        t = ARM.transform(goal); obj = t[:3, 3]+.15*t[:3, 2]
        if obj[2] < 0:
            continue
        view.set_object(obj)
        if not view.edge_valid(start, goal, STEP):
            continue
        case = {'name': f'easy_fk_{len(cases)-2:02d}', 'kind': 'generated_direct_visible_control',
                'start_rad': start.tolist(), 'goal_witness_rad': goal.tolist(), 'object_m': obj.tolist()}
        if region_metrics(start, region(case), 1., 1.)['accepted']:
            continue
        cases.append(case)
    for case in cases:
        assert env.valid(case['goal_witness_rad'])
        assert region_metrics(case['goal_witness_rad'], region(case), 1., 1.)['accepted']
        view.set_object(case['object_m']); assert view.valid(case['start_rad'])
    return cases


def endpoint_bank(env, case, seed, count):
    began = perf_counter(); start = np.array(case['start_rad']); disk = region(case)
    precheck = target_precheck(np.array(disk['center_mm'])/1000, disk, 1.)
    assert precheck['possible']
    poses = list(candidate_poses(disk, start))[:count]
    goals, attempts = [], []
    rng = np.random.default_rng(seed+10000)
    for branch in range(2):
        for index, (position, axis) in enumerate(poses):
            if not ENVELOPE.point(position*1000, 1.)['possible']:
                continue
            # Virtual alternate starts are numerical guesses, never robot motion.
            current = start if branch == 0 else rng.uniform(ARM.lower, ARM.upper)
            settings = {'attempts': 4, 'robust': branch == 0 and index == 0,
                        'learned': branch == 0, 'seed': seed+100*branch+index}
            entry = {'point_index': index, 'branch_pass': branch, 'settings': settings,
                     'target_mm': (position*1000).tolist(), 'target_axis': axis.tolist()}
            try:
                goal = env.inverse_kinematics(position, axis, current=current, **settings)
                accepted = region_metrics(goal, disk, 1., 1.)['accepted']
                duplicate = any(np.linalg.norm(goal-np.array(g['q_rad'])) < .05 for g in goals)
                entry['status'] = 'accepted' if accepted and not duplicate else ('duplicate' if duplicate else 'region_rejected')
                if accepted and not duplicate:
                    assert env.valid(goal)
                    goals.append({'index': len(goals), 'q_rad': goal.tolist(),
                                  'point_index': index, 'branch_pass': branch})
            except IKSearchExhausted:
                entry['status'] = 'search_exhausted'
            entry['diagnostics'] = dict(env.ik_diagnostics)
            attempts.append(entry)
    return {'seconds': perf_counter()-began, 'goals': goals, 'attempts': attempts,
            'legacy_attempts_used': sum(a['diagnostics']['legacy_attempts'] for a in attempts),
            'fallback_attempts_used': sum(a['diagnostics']['fallback_attempts'] for a in attempts),
            'precheck': precheck}


def route_bank(env, view, case, bank, seed, budget, require_visibility, first_only):
    start = np.array(case['start_rad']); remaining = budget; began = perf_counter()
    routes, trajectories = [], {}
    for goal in bank['goals']:
        tick = perf_counter(); q = np.array(goal['q_rad'])
        row = {'goal_index': goal['index'], 'accepted': False}
        if remaining > 0:
            path, diagnostics = plan(env, start, q, seed=seed,
                config=RRTConfig(iterations=min(200, remaining), max_step_rad=.02))
            remaining -= diagnostics['iterations']
        elif env.edge_valid(start, q, .02):
            path = np.stack([start, q]); diagnostics = {'status': 'direct_optimal', 'iterations': 0}
        else:
            path = None; diagnostics = {'status': 'shared_iteration_budget_exhausted', 'iterations': 0}
        row['planner'] = diagnostics
        if path is not None:
            try:
                states, times, smoothing = prepare(env, path)
                if not env.path_valid(states, STEP):
                    raise ValueError('Dense final path validation failed')
                audit = view.audit(states, STEP)
                row.update(visibility=audit, joint_travel_rad=float(np.linalg.norm(np.diff(states, axis=0), axis=1).sum()),
                           duration_s=float(times[-1]), path_vertices_rad=path.tolist(),
                           goal_q_rad=q.tolist(), smoothing=smoothing)
                if require_visibility and not audit['passed']:
                    raise ValueError('Dense final visibility validation failed')
                assert region_metrics(states[-1], region(case), 1., 1.)['accepted']
                assert np.array_equal(states[0], start)
                row['accepted'] = True
                trajectories[goal['index']] = (states, times)
            except ValueError as exc:
                row['rejection'] = str(exc)
        row.update(seconds=perf_counter()-tick, cumulative_seconds=perf_counter()-began,
                   cumulative_rrt_iterations=budget-remaining)
        routes.append(row)
        if first_only and row['accepted']:
            break
    assert 0 <= remaining <= budget
    return {'routes': routes, 'seconds': perf_counter()-began,
            'rrt_iterations_used': budget-remaining}, trajectories


def select(policy, bank, routes):
    accepted = [r for r in routes['routes'] if r['accepted']]
    if not accepted:
        return {'policy': policy, 'success': False, 'route_found': False,
                'planning_seconds': bank['seconds']+routes['seconds'],
                'rrt_iterations_used': routes['rrt_iterations_used']}
    chosen = (min(accepted, key=lambda r: (r['joint_travel_rad'], r['duration_s'], r['goal_index']))
              if policy == 'best_visible' else accepted[0])
    full = policy == 'best_visible'
    return {'policy': policy, 'success': chosen['visibility']['passed'], 'route_found': True,
            'goal_index': chosen['goal_index'], 'joint_travel_rad': chosen['joint_travel_rad'],
            'duration_s': chosen['duration_s'], 'visibility': chosen['visibility'],
            'planning_seconds': bank['seconds']+(routes['seconds'] if full else chosen['cumulative_seconds']),
            'rrt_iterations_used': routes['rrt_iterations_used'] if full else chosen['cumulative_rrt_iterations']}


def summarize(rows):
    modes = {}
    common = [r for r in rows if all(v['success'] for v in r['policies'])]
    for policy in POLICIES:
        all_rows = [next(v for v in r['policies'] if v['policy'] == policy) for r in rows]
        shared = [next(v for v in r['policies'] if v['policy'] == policy) for r in common]
        modes[policy] = {'successful_visible_routes': sum(v['success'] for v in all_rows),
            'trials': len(rows), 'median_planning_seconds': float(np.median([v['planning_seconds'] for v in all_rows])),
            'median_joint_travel_shared_success_rad': float(np.median([v['joint_travel_rad'] for v in shared])) if shared else None,
            'median_duration_shared_success_s': float(np.median([v['duration_s'] for v in shared])) if shared else None}
    paired = [(r['policies'][1], r['policies'][2]) for r in rows
              if r['policies'][1]['success'] and r['policies'][2]['success']]
    return {'trials': len(rows), 'common_successes_all_policies': len(common), 'policies': modes,
            'visible_selection_pairs': len(paired),
            'best_visible_shorter_count': sum(b['joint_travel_rad'] < a['joint_travel_rad']-1e-6 for a,b in paired),
            'median_paired_travel_change_rad': float(np.median([b['joint_travel_rad']-a['joint_travel_rad'] for a,b in paired])) if paired else None,
            'median_paired_travel_change_percent': float(np.median([100*(b['joint_travel_rad']/a['joint_travel_rad']-1) for a,b in paired])) if paired else None,
            'median_paired_planning_increase_seconds': float(np.median([b['planning_seconds']-a['planning_seconds'] for a,b in paired])) if paired else None}


def plot(rows, output):
    # Keep the robot venv's NumPy ABI isolated from distro Matplotlib packages.
    interpreter = sys.executable if importlib.util.find_spec('matplotlib') else '/usr/bin/python3'
    subprocess.run([interpreter, str(ROOT/'scripts/plot_visible_endpoints.py'), str(output)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'output/visible_endpoint_comparison_20261012')
    parser.add_argument('--generated', type=int, default=6)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43])
    parser.add_argument('--candidates', type=int, default=8)
    parser.add_argument('--rrt-iterations', type=int, default=800)
    parser.add_argument('--report-only', action='store_true', help='Rebuild summary and chart from existing results')
    args = parser.parse_args()
    if args.generated < 0 or not 1 <= args.candidates <= 17 or args.rrt_iterations < 1:
        parser.error('Invalid experiment budget')
    output = args.output; output.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        rows = [json.loads(line) for line in (output/'results.jsonl').read_text().splitlines()]
        summary = summarize(rows)
        (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        plot(rows, output); print(json.dumps(summary, indent=2)); return
    checkpoint = checkpoint_status(); assert checkpoint['status'] == 'ready'
    rows = []
    with ArmEnv() as collision, VisibilityEnv() as visible:
        collision.set_obstacle(None); visible.set_obstacle(None)
        cases = cases_for(collision, visible, args.generated)
        (output/'cases.json').write_text(json.dumps(cases, indent=2)+'\n')
        metadata = {'hardware_motion': False, 'experimental_only': True,
            'endpoint_pool': {'disk_points': args.candidates, 'IK_calls_per_point': 2,
                             'legacy_attempts_cap': args.candidates*8, 'fallback_attempts_cap': 32},
            'route_budget_per_policy': args.rrt_iterations, 'per_goal_iteration_cap': 200,
            'equal_caps_not_equal_consumed_work': True,
            'selection_objective': 'minimum joint travel, then timed duration',
            'visibility': visible.visibility_metadata(), 'checkpoint': checkpoint,
            'scene': collision.metadata(), 'seeds': args.seeds,
            'slice_definition': 'fixed plane centered at known FK witness, 150 mm from object, half angle 30 deg',
            'witness_use': 'case generation and reachability validation only; never passed to IK or RRT',
            'limitations': ['Small targeted pilot; easy cases are selected to have a visible direct witness route.',
                'Policies share all IK cost; production instead streams IK and can stop earlier.',
                'First-visible and best-visible share the same route bank, so their success counts must match.',
                'First-visible time is a measured prefix estimate; best-visible includes the full route bank.',
                'Full trajectories from rest; no live stage executor or moving-boundary replacement.',
                'Synthetic point camera and sampled checks, not calibrated continuous hardware visibility.',
                'Timings include diagnostic visibility replay, even for the collision-only reference.']}
        with (output/'results.jsonl').open('w') as stream:
            for case in cases:
                for seed in args.seeds:
                    start = np.array(case['start_rad'])
                    collision.reset(start); visible.reset(start); visible.set_object(case['object_m'])
                    collision._valid_cache.clear(); visible._valid_cache.clear()
                    bank = endpoint_bank(collision, case, seed, args.candidates)
                    baseline, base_paths = route_bank(collision, visible, case, bank, seed,
                        args.rrt_iterations, require_visibility=False, first_only=True)
                    visible._valid_cache.clear(); visible._visibility_cache.clear()
                    alternatives, alt_paths = route_bank(visible, visible, case, bank, seed,
                        args.rrt_iterations, require_visibility=True, first_only=False)
                    selected = [select(POLICIES[0], bank, baseline),
                                select(POLICIES[1], bank, alternatives), select(POLICIES[2], bank, alternatives)]
                    for result, trajectories in zip(selected, (base_paths, alt_paths, alt_paths)):
                        if result['route_found']:
                            states, times = trajectories[result['goal_index']]
                            file = f'{case["name"]}_seed{seed}_{result["policy"]}.npz'
                            np.savez_compressed(output/file, joints_rad=states, times_s=times)
                            result['trajectory_file'] = file
                    row = {'case': case['name'], 'kind': case['kind'], 'seed': seed,
                           'endpoint_bank': bank, 'collision_routes': baseline,
                           'visible_routes': alternatives, 'policies': selected}
                    rows.append(row); stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
                    print(json.dumps({'case': case['name'], 'seed': seed, 'endpoints': len(bank['goals']),
                        'policies': [{k:v for k,v in r.items() if k in ('policy','success','joint_travel_rad','planning_seconds')} for r in selected]}), flush=True)
        sources = [Path(__file__), ROOT/'simulation/diffusion/visibility.py',
                   ROOT/'simulation/diffusion/planner.py', ROOT/'simulation/diffusion/endpoint_ik.py',
                   ROOT/'simulation/diffusion/env.py', ROOT/'simulation/diffusion/smoothing.py',
                   ROOT/'simulation/tracking_geometry.py', ROOT/'simulation/physical_reach.py',
                   ROOT/'simulation/pose5.py', ROOT/'simulation/self_collision.py',
                   ROOT/'simulation/scenarios/reachable_failures.json']
        metadata['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        (output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    summary = summarize(rows)
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    plot(rows, output)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
