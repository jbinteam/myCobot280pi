"""Measure on-demand IK against eager IK plus bounded visible selection.

Uses the same 24 case/seed pairs as the preceding early-stop experiment. Policies
run independently in alternating order with identical numerical seeds and caps.
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
from scripts.compare_visible_endpoints import endpoint_bank, route_bank, region
from scripts.compare_endpoint_early_stop import adaptive
from simulation.diffusion.env import ArmEnv
from simulation.diffusion.visibility import VisibilityEnv
from simulation.diffusion.endpoint_selection import SelectionBudget, DeadlineEnv
from simulation.diffusion.streaming_ik import EndpointSource, choose_streaming_route
from simulation.diffusion.ik_seed import checkpoint_status
from simulation.tracking_geometry import region_metrics


def streaming(collision, view, case, seed, budget, candidate_budget=None):
    began = perf_counter(); completed = []
    source = EndpointSource(collision, case['start_rad'], region(case), seed)
    setup_seconds = perf_counter()-began
    def evaluate(goal, cap, deadline):
        env = view if deadline is None else DeadlineEnv(view, deadline)
        routes, paths = route_bank(env, view, case, {'goals': [goal]}, seed, cap, True, True)
        row = routes['routes'][0]; completed.append(row)
        return {'route': row, 'trajectory': paths.get(goal['index'])}
    chosen, stats = choose_streaming_route(case['start_rad'], source, evaluate, budget,
                                          candidate_budget=candidate_budget)
    total = perf_counter()-began
    first_path = stats.pop('first_trajectory')
    first_seconds = None if stats['first_solution_seconds'] is None else setup_seconds+stats['first_solution_seconds']
    result = {'policy': 'streaming_bounded', 'success': chosen is not None,
        'planning_seconds': total, 'first_solution_seconds': first_seconds,
        'selection': stats, 'source': source.report(), 'completed_routes': completed}
    first_result = {'policy': 'streaming_first_prefix', 'success': chosen is not None,
                    'planning_seconds': first_seconds if chosen is not None else total,
                    'first_solution_seconds': first_seconds}
    if chosen is not None:
        for key in ('goal_index','joint_travel_rad','duration_s','visibility'):
            result[key] = chosen['route'][key]
        first_row = next(r for r in completed if r['goal_index'] == stats['first_goal_index'])
        for key in ('goal_index','joint_travel_rad','duration_s','visibility'):
            first_result[key] = first_row[key]
    return result, None if chosen is None else chosen['trajectory'], first_result, first_path


def summarize(rows):
    result = {'trials': len(rows), 'policies': {}}
    for mode in ('eager_bounded','streaming_bounded','streaming_first_prefix'):
        values = [r['policies'][mode] for r in rows]
        succeeded = [v for v in values if v['success']]
        result['policies'][mode] = {'visible_successes': len(succeeded),
            'median_planning_seconds': float(np.median([v['planning_seconds'] for v in values])),
            'median_time_to_first_visible_route_seconds': float(np.median([v['first_solution_seconds'] for v in succeeded])) if succeeded else None,
            'median_joint_travel_rad': float(np.median([v['joint_travel_rad'] for v in succeeded])) if succeeded else None}
    paired = [r for r in rows if r['policies']['eager_bounded']['success'] and r['policies']['streaming_bounded']['success']]
    if paired:
        eager = [r['policies']['eager_bounded'] for r in paired]
        stream = [r['policies']['streaming_bounded'] for r in paired]
        result['paired'] = {'count': len(paired),
            'median_total_planning_change_percent': float(np.median([100*(s['planning_seconds']/e['planning_seconds']-1) for e,s in zip(eager,stream)])),
            'median_time_to_first_change_percent': float(np.median([100*(s['first_solution_seconds']/e['first_solution_seconds']-1) for e,s in zip(eager,stream)])),
            'median_joint_travel_change_percent': float(np.median([100*(s['joint_travel_rad']/e['joint_travel_rad']-1) for e,s in zip(eager,stream)])),
            'maximum_joint_travel_increase_percent': max(100*(s['joint_travel_rad']/e['joint_travel_rad']-1) for e,s in zip(eager,stream)),
            'same_joint_travel_count': sum(abs(s['joint_travel_rad']-e['joint_travel_rad'])<1e-6 for e,s in zip(eager,stream)),
            'longer_route_count': sum(s['joint_travel_rad']>e['joint_travel_rad']+1e-6 for e,s in zip(eager,stream)),
            'shorter_than_stream_first_count': sum(s['joint_travel_rad']<r['policies']['streaming_first_prefix']['joint_travel_rad']-1e-6 for s,r in zip(stream,paired))}
    streams = [r['policies']['streaming_bounded'] for r in rows]
    reasons = {}
    for s in streams:
        reason = s['selection']['stop_reason']; reasons[reason] = reasons.get(reason,0)+1
    result['effort'] = {'eager_median_ik_jobs': float(np.median([len(r['endpoint_bank']['attempts']) for r in rows])),
        'stream_median_ik_jobs_started': float(np.median([s['source']['jobs_started'] for s in streams])),
        'stream_median_unstarted_jobs': float(np.median([s['source']['jobs_unstarted'] for s in streams])),
        'median_improvement_seconds': float(np.median([s['selection']['improvement_seconds'] for s in streams])),
        'maximum_deadline_overrun_seconds': max(s['selection']['deadline_overrun_seconds'] for s in streams),
        'stop_reasons': reasons}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'output/streaming_ik_20261012')
    args = parser.parse_args(); output = args.output; output.mkdir(parents=True, exist_ok=True)
    cases = json.loads((ROOT/'output/endpoint_early_stop_20261012/cases.json').read_text())
    (output/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    budget = SelectionBudget()
    paths = [Path(__file__), ROOT/'scripts/compare_visible_endpoints.py', ROOT/'scripts/compare_endpoint_early_stop.py',
        ROOT/'simulation/diffusion/streaming_ik.py', ROOT/'simulation/diffusion/endpoint_selection.py',
        ROOT/'simulation/diffusion/endpoint_ik.py', ROOT/'simulation/diffusion/env.py', ROOT/'simulation/pose5.py',
        ROOT/'simulation/diffusion/visibility.py', ROOT/'simulation/diffusion/planner.py', ROOT/'simulation/diffusion/smoothing.py',
        ROOT/'simulation/tracking_geometry.py', ROOT/'simulation/physical_reach.py', ROOT/'simulation/self_collision.py']
    metadata = {'hardware_motion': False, 'configuration': asdict(budget), 'checkpoint': checkpoint_status(),
        'candidate_order': 'same eight disk points, then same alternate-start pass; generated on demand',
        'caps': {'primary_ik_attempts': 64, 'fallback_ik_attempts': 32, 'rrt_iterations_charged': 800},
        'deadline_scope': 'after first fully verified visible route; covers additional numerical IK and route checks',
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'limitations': ['Same 24 scenarios/seeds reused; no new holdout in this experiment.',
            'Synthetic camera, point object and sampled collision/visibility checks.',
            'Whole trajectories from rest; not the hardware stage lifecycle.',
            'Ungenerated IK endpoints cannot be certified worse by joint-space bounds.',
            'Cooperative cancellation may slightly overrun inside a native query or final audit.',
            'Streaming-first is the measured prefix of the streaming run, not a separate execution.']}
    rows = []
    with ArmEnv() as collision, VisibilityEnv() as visible:
        collision.set_obstacle(None); visible.set_obstacle(None)
        metadata['visibility'] = visible.visibility_metadata(); metadata['scene'] = collision.metadata()
        (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        with (output/'results.jsonl').open('w') as stream:
            for case in cases:
                for seed in ([42,43] if case['split']=='previous_pilot' else [44,45]):
                    products = {}; bank = None
                    order = ['eager','stream'] if len(rows)%2==0 else ['stream','eager']
                    for mode in order:
                        collision.reset(case['start_rad']); collision._valid_cache.clear()
                        visible.reset(case['start_rad']); visible.set_object(case['object_m']); visible._valid_cache.clear()
                        if mode=='eager':
                            bank = endpoint_bank(collision, case, seed, 8)
                            result, trajectory = adaptive(visible, case, bank, seed, budget)
                            result['policy'] = 'eager_bounded'
                            result['first_solution_seconds'] = (bank['seconds']+result['selection']['first_solution_seconds']
                                                               if result['success'] else None)
                            products['eager_bounded'] = (result,trajectory)
                        else:
                            result,trajectory,first,first_path = streaming(collision,visible,case,seed,budget)
                            products['streaming_bounded'] = (result,trajectory)
                            products['streaming_first_prefix'] = (first,first_path)
                    produced = products['streaming_bounded'][0]['source']['goals']
                    for new,old in zip(produced,bank['goals']):
                        assert new['index']==old['index'] and new['point_index']==old['point_index']
                        np.testing.assert_allclose(new['q_rad'],old['q_rad'],atol=1e-12,rtol=0)
                    policies = {}
                    for name,(result,trajectory) in products.items():
                        if trajectory is not None:
                            q,t = trajectory
                            assert np.array_equal(q[0],case['start_rad'])
                            assert region_metrics(q[-1],region(case),1.,1.)['accepted']
                            assert result['visibility']['passed']
                            filename = f'{case["name"]}_{seed}_{name}.npz'
                            np.savez_compressed(output/filename,joints_rad=q,times_s=t)
                            result['trajectory_file'] = filename
                        policies[name] = result
                    row = {'case':case['name'],'split':case['split'],'seed':seed,'execution_order':order,
                           'endpoint_bank':bank,'policies':policies,'generated_endpoint_prefix_matches_eager':True}
                    rows.append(row); stream.write(json.dumps(row,allow_nan=False)+'\n'); stream.flush()
                    print(json.dumps({'case':case['name'],'seed':seed,
                        'policies':{n:{k:v for k,v in r.items() if k in ('success','planning_seconds','first_solution_seconds','joint_travel_rad')}
                                    for n,r in policies.items()},
                        'ik_jobs':policies['streaming_bounded']['source']['jobs_started'],
                        'stop':policies['streaming_bounded']['selection']['stop_reason']}),flush=True)
    for p in paths:
        assert hashlib.sha256(p.read_bytes()).hexdigest()==metadata['source_sha256'][str(p.relative_to(ROOT))]
    summary = {'all':summarize(rows)}
    for split in ('previous_pilot','held_out_controls'):
        summary[split] = summarize([r for r in rows if r['split']==split])
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
