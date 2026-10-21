"""Compare eager, original streaming and per-candidate bounded improvement.

Run a pilot on known cases before freezing policy for the complete validation.
The new controls extend the deterministic FK scenario generator; they are
unseen cases from the same simple family, not a deployment distribution.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_visible_endpoints import cases_for, endpoint_bank, region
from scripts.compare_endpoint_early_stop import adaptive
from scripts.compare_streaming_ik import streaming
from simulation.diffusion.env import ArmEnv
from simulation.diffusion.visibility import VisibilityEnv
from simulation.diffusion.endpoint_selection import SelectionBudget
from simulation.diffusion.streaming_ik import CandidateBudget
from simulation.tracking_geometry import region_metrics
from simulation.kinematics import ARM

MODES = ('eager', 'streaming', 'fair')


def summarize(rows):
    summary = {'trials':len(rows), 'policies':{}, 'paired':{}}
    for mode in MODES:
        values = [r['policies'][mode] for r in rows]
        ok = [v for v in values if v['success']]
        summary['policies'][mode] = {'successes':len(ok),
            'median_planning_seconds':float(np.median([v['planning_seconds'] for v in values])),
            'median_first_solution_seconds':float(np.median([v['first_solution_seconds'] for v in ok])) if ok else None,
            'median_joint_travel_rad':float(np.median([v['joint_travel_rad'] for v in ok])) if ok else None}
    for mode in ('streaming','eager'):
        pairs = [(r['policies'][mode],r['policies']['fair']) for r in rows
                 if r['policies'][mode]['success'] and r['policies']['fair']['success']]
        summary['paired']['fair_vs_'+mode] = {'count':len(pairs),
            'shorter':sum(b['joint_travel_rad']<a['joint_travel_rad']-1e-6 for a,b in pairs),
            'same':sum(abs(b['joint_travel_rad']-a['joint_travel_rad'])<=1e-6 for a,b in pairs),
            'longer':sum(b['joint_travel_rad']>a['joint_travel_rad']+1e-6 for a,b in pairs),
            'maximum_travel_ratio':max((b['joint_travel_rad']/a['joint_travel_rad'] for a,b in pairs),default=None),
            'median_planning_change_percent':float(np.median([100*(b['planning_seconds']/a['planning_seconds']-1) for a,b in pairs])) if pairs else None}
    summary['maximum_fair_deadline_overrun_seconds'] = max(r['policies']['fair']['selection']['deadline_overrun_seconds'] for r in rows)
    return summary


def verify(output, cases, rows):
    case_by_name = {c['name']:c for c in cases}
    count = 0; samples = 0
    with VisibilityEnv() as view:
        view.set_obstacle(None)
        for row in rows:
            case = case_by_name[row['case']]
            for mode, result in row['policies'].items():
                if not result['success']:
                    continue
                view._valid_cache.clear(); view.set_object(case['object_m'])
                path = np.load(output/result['trajectory_file'])
                q,t = path['joints_rad'],path['times_s']
                assert np.isfinite(q).all() and np.isfinite(t).all()
                assert len(q)==len(t) and np.all(np.diff(t)>0)
                assert np.all(q>=ARM.lower-1e-9) and np.all(q<=ARM.upper+1e-9)
                np.testing.assert_array_equal(q[0],case['start_rad'])
                assert region_metrics(q[-1],region(case),1.,1.)['accepted']
                assert view.path_valid(q,.005)
                audit = view.audit(q,.005); assert audit['passed']
                np.testing.assert_allclose(np.linalg.norm(np.diff(q,axis=0),axis=1).sum(),result['joint_travel_rad'],atol=1e-9)
                stats = result['selection']
                assert stats['iterations_charged']<=800
                attempts = stats.get('improvement_route_attempts',stats.get('improvement_attempts'))
                assert attempts<=3
                assert result['joint_travel_rad'] <= stats['first_joint_travel_rad']+1e-9
                if mode != 'eager':
                    source = result['source']
                    assert source['legacy_attempts_used']<=64 and source['fallback_attempts_used']<=32
                count += 1; samples += audit['samples']
    report = {'passed':True,'trajectories':count,'replay_samples':samples,'max_step_rad':.005,
              'fresh_environment':True,'hardware_motion':False}
    (output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--known-cases',type=Path,default=ROOT/'output/endpoint_early_stop_20261012/cases.json')
    parser.add_argument('--new-control-count',type=int,default=6)
    parser.add_argument('--new-seeds',type=int,nargs=2,default=[46,47])
    args = parser.parse_args(); output = args.output
    output.mkdir(parents=True,exist_ok=True)
    if (output/'results.jsonl').exists():
        raise RuntimeError('Use a fresh output directory to preserve prior evidence')
    budget = SelectionBudget(); candidate = CandidateBudget()
    paths = [Path(__file__),ROOT/'scripts/compare_streaming_ik.py',ROOT/'scripts/compare_endpoint_early_stop.py',
             ROOT/'scripts/compare_visible_endpoints.py', *sorted((ROOT/'simulation/diffusion').glob('*.py')),
             ROOT/'simulation/pose5.py',ROOT/'simulation/tracking_geometry.py',ROOT/'simulation/kinematics.py']
    metadata = {'hardware_motion':False,'budget':asdict(budget),'candidate_budget':asdict(candidate),
        'known_cases_file':str(args.known_cases),'known_cases_sha256':hashlib.sha256(args.known_cases.read_bytes()).hexdigest(),
        'new_control_count':args.new_control_count,'new_seeds':args.new_seeds,
        'pilot':args.pilot,'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'limitations':['Synthetic flange camera and point target; sampled checks are not continuous proof.',
            'New controls have known direct visible witness paths and share the original generation family.',
            'Candidate slices apply after the first verified route; first-solution time remains effort-bounded, not wall-time-bounded.',
            'Local interruption discards that job; no solver continuation or proof of infeasibility.',
            'Wall-clock cancellation can vary with machine load; one run per case/seed/policy.']}
    rows = []
    with ArmEnv() as collision, VisibilityEnv() as view:
        collision.set_obstacle(None);view.set_obstacle(None)
        cases = json.loads(args.known_cases.read_text())
        for c in cases:
            c['seeds'] = c.get('seeds', [42,43] if c['split']=='previous_pilot' else [44,45])
            c['split'] = 'known'
        if not args.pilot:
            generated = cases_for(collision,view,len(cases)-2+args.new_control_count)
            for a,b in zip(cases,generated):
                for key in ('name','start_rad','goal_witness_rad','object_m'):
                    assert a[key]==b[key]
            for c in generated[len(cases):]:
                c.update(split='new_controls',seeds=args.new_seeds);cases.append(c)
        (output/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
        metadata['visibility']=view.visibility_metadata()
        (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        with (output/'results.jsonl').open('w') as out:
            for case in cases:
                for seed in (case['seeds'][:1] if args.pilot else case['seeds']):
                    policies = {}; order = list(MODES)
                    shift = len(rows)%3;order=order[shift:]+order[:shift]
                    for mode in order:
                        collision.reset(case['start_rad']);collision._valid_cache.clear()
                        view.reset(case['start_rad']);view.set_object(case['object_m']);view._valid_cache.clear()
                        if mode=='eager':
                            bank=endpoint_bank(collision,case,seed,8)
                            result,path=adaptive(view,case,bank,seed,budget)
                            result['first_solution_seconds']=(bank['seconds']+result['selection']['first_solution_seconds'] if result['success'] else None)
                        else:
                            result,path,_,_=streaming(collision,view,case,seed,budget,candidate if mode=='fair' else None)
                        result['policy']=mode
                        if path is not None:
                            filename=f'{case["name"]}_{seed}_{mode}.npz'
                            np.savez_compressed(output/filename,joints_rad=path[0],times_s=path[1])
                            result['trajectory_file']=filename
                        policies[mode]=result
                    row={'case':case['name'],'seed':seed,'split':case['split'],'execution_order':order,'policies':policies}
                    rows.append(row);out.write(json.dumps(row,allow_nan=False)+'\n');out.flush()
                    print(json.dumps({'case':case['name'],'seed':seed,
                        'policies':{m:{k:r.get(k) for k in ('success','planning_seconds','joint_travel_rad')} for m,r in policies.items()},
                        'fair_jobs':policies['fair']['source']['jobs_started'],
                        'fair_stop':policies['fair']['selection']['stop_reason']}),flush=True)
    for p in paths:
        assert hashlib.sha256(p.read_bytes()).hexdigest()==metadata['source_sha256'][str(p.relative_to(ROOT))]
    summary={'all':summarize(rows)}
    for split in sorted({r['split'] for r in rows}):
        summary[split]=summarize([r for r in rows if r['split']==split])
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    print(json.dumps(verify(output,cases,rows)),flush=True)


if __name__=='__main__':
    main()
