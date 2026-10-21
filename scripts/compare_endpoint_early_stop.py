"""Compare first/full visible endpoint selection with budgeted improvement.

Reuses eight pilot scenarios and adds four unseen generated controls, with two
new seeds on the latter. The fixed policy is declared before evaluating either
group. Every policy shares a fresh IK bank; timings include that common cost.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_visible_endpoints import cases_for, endpoint_bank, route_bank, select, region
from simulation.diffusion.env import ArmEnv
from simulation.diffusion.visibility import VisibilityEnv
from simulation.diffusion.endpoint_selection import SelectionBudget, DeadlineEnv, choose_visible_endpoint
from simulation.diffusion.ik_seed import checkpoint_status
from simulation.tracking_geometry import region_metrics


def adaptive(view, case, bank, seed, budget):
    completed = []
    def evaluate(goal, cap, deadline):
        checked_env = view if deadline is None else DeadlineEnv(view, deadline)
        routes, trajectories = route_bank(checked_env, view, case, {'goals': [goal]}, seed,
                                          cap, require_visibility=True, first_only=True)
        row = routes['routes'][0]
        completed.append(row)
        return {'route': row, 'trajectory': trajectories.get(goal['index'])}
    chosen, stats = choose_visible_endpoint(np.array(case['start_rad']), bank['goals'], evaluate, budget)
    result = {'policy': 'bounded_visible', 'success': chosen is not None,
              'planning_seconds': bank['seconds']+stats['seconds'], 'route_search_seconds': stats['seconds'],
              'selection': stats, 'completed_routes': completed}
    if chosen is not None:
        route = chosen['route']
        result.update({key: route[key] for key in ('goal_index', 'joint_travel_rad', 'duration_s', 'visibility')})
        assert result['visibility']['passed']
        assert result['joint_travel_rad'] <= stats['first_joint_travel_rad']+1e-9
    return result, None if chosen is None else chosen['trajectory']


def summary_for(rows):
    result = {'trials': len(rows), 'policies': {}}
    for name in ('first_visible', 'bounded_visible', 'best_visible'):
        values = [r['policies'][name] for r in rows]
        successes = [v for v in values if v['success']]
        result['policies'][name] = {'visible_successes': len(successes),
            'median_planning_seconds': float(np.median([v['planning_seconds'] for v in values])),
            'median_route_search_seconds': float(np.median([v['route_search_seconds'] for v in values])),
            'median_joint_travel_rad': float(np.median([v['joint_travel_rad'] for v in successes])) if successes else None}
    common = [r['policies'] for r in rows if all(p['success'] for p in r['policies'].values())]
    if common:
        first = np.array([r['first_visible']['joint_travel_rad'] for r in common])
        bounded = np.array([r['bounded_visible']['joint_travel_rad'] for r in common])
        full = np.array([r['best_visible']['joint_travel_rad'] for r in common])
        available_savings = float((first-full).sum())
        result['paired'] = {'common_successes': len(common),
            'bounded_shorter_than_first': int(np.sum(bounded < first-1e-6)),
            'bounded_matches_full_travel': int(np.sum(np.abs(bounded-full) < 1e-6)),
            'median_bounded_travel_change_vs_first_percent': float(np.median(100*(bounded/first-1))),
            'aggregate_full_search_travel_savings_retained_percent': (100*float((first-bounded).sum())/available_savings
                                                                   if available_savings > 1e-9 else None),
            'median_total_planning_change_vs_full_percent': float(np.median([
                100*(r['bounded_visible']['planning_seconds']/r['best_visible']['planning_seconds']-1) for r in common])),
            'median_route_search_change_vs_full_percent': float(np.median([
                100*(r['bounded_visible']['route_search_seconds']/r['best_visible']['route_search_seconds']-1) for r in common])),
            'median_added_planning_vs_first_seconds': float(np.median([
                r['bounded_visible']['planning_seconds']-r['first_visible']['planning_seconds'] for r in common]))}
    stats = [r['policies']['bounded_visible']['selection'] for r in rows]
    reasons = {}
    for s in stats:
        reasons[s['stop_reason']] = reasons.get(s['stop_reason'], 0)+1
    result['stopping'] = {'reasons': reasons, 'maximum_deadline_overrun_seconds': max(s['deadline_overrun_seconds'] for s in stats),
        'median_improvement_seconds': float(np.median([s['improvement_seconds'] for s in stats])),
        'median_improvement_attempts': float(np.median([s['improvement_attempts'] for s in stats])),
        'maximum_iterations_charged': max(s['iterations_charged'] for s in stats)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'output/endpoint_early_stop_20261012')
    args = parser.parse_args(); output = args.output; output.mkdir(parents=True, exist_ok=True)
    budget = SelectionBudget()  # fixed before held-out evaluation; do not tune to results
    metadata = {'hardware_motion': False, 'policy': asdict(budget), 'improvement_order': 'ascending joint-distance lower bound',
        'shared_ik_primary_attempt_cap': 64, 'shared_ik_fallback_cap': 32,
        'first_and_full_route_cap': 800, 'first_and_full_per_goal_cap': 200,
        'timing': 'fresh shared IK cost plus measured route search; first-visible uses full-bank prefix time',
        'deadline': 'cooperative at validity samples; a native query or final audit may overrun slightly',
        'checkpoint': checkpoint_status(),
        'limitations': ['Synthetic flange camera and point target.', 'Dense sampled checking is not continuous proof.',
            'Whole trajectories from rest; no hardware stage executor.',
            'Shared IK bank remains fully generated before selection.',
            'First and full visible policies share routes; bounded search reruns independently.',
            'Held-out controls have a known direct visible witness route; they are not a representative deployment sample.']}
    source_paths = [Path(__file__), ROOT/'scripts/compare_visible_endpoints.py',
        ROOT/'simulation/diffusion/endpoint_selection.py', ROOT/'simulation/diffusion/visibility.py',
        ROOT/'simulation/diffusion/planner.py', ROOT/'simulation/diffusion/smoothing.py',
        ROOT/'simulation/diffusion/env.py', ROOT/'simulation/diffusion/endpoint_ik.py',
        ROOT/'simulation/tracking_geometry.py', ROOT/'simulation/pose5.py',
        ROOT/'simulation/self_collision.py', ROOT/'simulation/physical_reach.py']
    metadata['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    rows = []
    with ArmEnv() as collision, VisibilityEnv() as visible:
        collision.set_obstacle(None); visible.set_obstacle(None)
        cases = cases_for(collision, visible, 10)
        original = json.loads((ROOT/'output/visible_endpoint_comparison_20261012/cases.json').read_text())
        for case, old in zip(cases[:8], original):
            assert case['name'] == old['name']
            for key in ('start_rad', 'goal_witness_rad', 'object_m'):
                np.testing.assert_array_equal(case[key], old[key])
        for i,c in enumerate(cases):
            c['split'] = 'previous_pilot' if i < 8 else 'held_out_controls'
        (output/'cases.json').write_text(json.dumps(cases, indent=2)+'\n')
        metadata['visibility'] = visible.visibility_metadata(); metadata['scene'] = collision.metadata()
        (output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        with (output/'results.jsonl').open('w') as stream:
            for case in cases:
                for seed in ([42,43] if case['split'] == 'previous_pilot' else [44,45]):
                    start = np.array(case['start_rad'])
                    collision.reset(start); collision._valid_cache.clear()
                    bank = endpoint_bank(collision, case, seed, 8)
                    products = {}
                    # Alternate policy execution order to reduce systematic timing bias.
                    order = ('full', 'bounded') if len(rows)%2 == 0 else ('bounded', 'full')
                    for mode in order:
                        visible.reset(start); visible.set_object(case['object_m']); visible._valid_cache.clear()
                        if mode == 'full':
                            all_routes, trajectories = route_bank(visible, visible, case, bank, seed, 800, True, False)
                            first = select('first_visible', bank, all_routes)
                            full = select('best_visible', bank, all_routes)
                            for policy in (first, full):
                                policy['route_search_seconds'] = policy['planning_seconds']-bank['seconds']
                                products[policy['policy']] = (policy, trajectories.get(policy.get('goal_index')))
                        else:
                            products['bounded_visible'] = adaptive(visible, case, bank, seed, budget)
                    policies = {}
                    for name in ('first_visible', 'bounded_visible', 'best_visible'):
                        policy, trajectory = products[name]
                        if trajectory is not None:
                            states, times = trajectory
                            assert np.array_equal(states[0], start)
                            assert region_metrics(states[-1], region(case), 1., 1.)['accepted']
                            assert policy['visibility']['passed']
                            filename = f'{case["name"]}_{seed}_{name}.npz'
                            np.savez_compressed(output/filename, joints_rad=states, times_s=times)
                            policy['trajectory_file'] = filename
                        policies[name] = policy
                    row = {'case': case['name'], 'split': case['split'], 'seed': seed, 'execution_order': order,
                           'endpoint_bank': bank, 'full_routes': all_routes, 'policies': policies}
                    rows.append(row); stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
                    print(json.dumps({'case': case['name'], 'seed': seed, 'split': case['split'],
                        'policies': {n:{k:v for k,v in r.items() if k in ('success','joint_travel_rad','planning_seconds')}
                                     for n,r in policies.items()},
                        'stop': policies['bounded_visible']['selection']['stop_reason']}), flush=True)
    for p in source_paths:
        assert hashlib.sha256(p.read_bytes()).hexdigest() == metadata['source_sha256'][str(p.relative_to(ROOT))]
    summary = {'all': summary_for(rows)}
    for split in ('previous_pilot', 'held_out_controls'):
        summary[split] = summary_for([r for r in rows if r['split'] == split])
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
