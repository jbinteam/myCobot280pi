"""Reproducible old/new 5D search and routing comparison; no hardware imports."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
import numpy as np
from . import planning, pose5, service, seed5, ik_recovery
from .kinematics import ARM
from .prm import planning_space
from .benchmark_seed5 import held_out_tasks


def fk_tasks(count, seed, space):
    """Full-range FK witnesses, boundary goals, common starts and nearby tasks."""
    rng = np.random.default_rng(seed); tasks = []
    def valid_sample():
        for _ in range(10000):
            q = rng.uniform(ARM.lower, ARM.upper)
            if space.valid(q):return q
        raise RuntimeError('Could not sample a valid start')
    for i in range(count):
        current = valid_sample()
        if i % 8 == 0:current = np.zeros(6)
        elif i % 8 == 4:current = np.deg2rad([0,-40,-130,80,0,50])
        group = 'joint_limit' if i % 3 == 0 else ('nearby' if i % 4 == 1 else 'global')
        for _ in range(10000):
            goal = ARM.clip(current+rng.normal(0,.35,6)) if group == 'nearby' else valid_sample()
            if group == 'joint_limit':
                k = (i//3) % 5
                goal[k] = ARM.lower[k]+.001 if (i//15) % 2 == 0 else ARM.upper[k]-.001
            if space.valid(goal):break
        else:raise RuntimeError('Could not sample a valid boundary goal')
        t = ARM.transform(goal)
        tasks.append({'id':f'fk-{i}', 'group':group, 'current':current.tolist(),
                      'xyz':t[:3,3].tolist(), 'axis':t[:3,2].tolist(), 'witness':goal.tolist()})
    families, _ = held_out_tasks(16, seed+1, space)
    tasks.extend(t for t in families if t['group'].startswith('same_xyz_'))
    return tasks


def row(result, elapsed):
    ik = result.get('ik', {}); recovery = ik.get('recovery', {})
    return {'status':result['status'], 'ik_success':bool(ik.get('goals')), 'planned':bool(result['converged']),
            'ik_outcome':result['ik_outcome'], 'planning_outcome':result['planning_outcome'],
            'position_error_mm':result.get('final_error_mm'),
            'direction_error_deg':result.get('final_orientation_error_deg'),
            'cursor_acceptance':result.get('cursor_acceptance'),
            'ik_ms':result['ik_ms'], 'planning_ms':result['planning_ms'], 'elapsed_ms':elapsed,
            'goals':len(ik.get('goals', [])), 'goal_sources':sorted({g['source'] for g in ik.get('goals', [])}),
            'expanded_search_seeds':sum(p['used_seeds'] for p in recovery.get('phases', [])),
            'expanded_search_ms':recovery.get('elapsed_ms', 0.)}


def summary(rows):
    good = [r for r in rows if r['planned']]
    return {'tasks':len(rows), 'valid_ik':sum(r['ik_success'] for r in rows), 'validated_plans':len(good),
            'direct_plans':sum(r['planned'] and r['planning_outcome']=='direct_connection' for r in rows),
            'prm_plans':sum(r['planned'] and r['planning_outcome']=='roadmap_path' for r in rows),
            'median_ms':float(np.median([r['elapsed_ms'] for r in rows])),
            'p95_ms':float(np.percentile([r['elapsed_ms'] for r in rows],95)),
            'mean_ms':float(np.mean([r['elapsed_ms'] for r in rows])),
            'max_success_position_error_mm':max((r['position_error_mm'] for r in good),default=None),
            'max_success_direction_error_deg':max((r['direction_error_deg'] for r in good),default=None),
            'expanded_search_tasks':sum(r['expanded_search_seeds']>0 for r in rows)}


def compare(rows):
    before = {r['task_id']:r for r in rows['baseline']}; after = {r['task_id']:r for r in rows['upgraded']}
    return {'recovered_ik':[key for key in before if not before[key]['ik_success'] and after[key]['ik_success']],
            'lost_ik':[key for key in before if before[key]['ik_success'] and not after[key]['ik_success']],
            'recovered_plans':[key for key in before if not before[key]['planned'] and after[key]['planned']],
            'lost_plans':[key for key in before if before[key]['planned'] and not after[key]['planned']]}


def benchmark(output, count=64, seed=20261004, only='all'):
    loading = perf_counter(); space = planning_space('self',1.)
    road = service.load_roadmap(service.ARTIFACTS/'roadmap.json',space)
    cold_ms = (perf_counter()-loading)*1000
    # Warm model parsing and SciPy imports outside query timing.
    t = ARM.transform(np.zeros(6))
    seed5.propose(np.zeros(6),t[:3,3],t[:3,2])
    ik_recovery.optimize(np.zeros(6),t[:3,3],t[:3,2],lambda q:False,evaluations=1)
    report = {'simulation_only':True, 'seed':seed, 'roadmap_load_ms':cold_ms,
              'checkpoint_sha256':hashlib.sha256(seed5.DEFAULT_CHECKPOINT.read_bytes()).hexdigest(),
              'tolerance_mm':1., 'orientation_tolerance_deg':1.,
              'recovery_seeds':32, 'recovery_evaluations':96, 'initial_goal_budget':4, 'route_retry_goal_budget':8,
              'limitations':'Finite deterministic samples; one unprofiled query per variant/task. FK tasks have valid witnesses, '
              'but witnesses are not supplied as IK seeds. A valid goal does not guarantee a route. '
              'Cursor failures do not prove unreachability. Optimizer states are never executed.'}
    if only in ('all','fk'):
        tasks = fk_tasks(count,seed,space); rows = {'numerical_current':[], 'baseline':[], 'upgraded':[]}
        baseline = planning.IKConfig(task='pose5', nn='seed5', pose5_method='dls', pose5_recovery_methods=('dls',))
        variants = {'numerical_current':replace(baseline,nn='none',random_seeds=0,roadmap_seeds=0,max_candidates=1),
                    'baseline':baseline, 'upgraded':replace(baseline,pose5_global_recovery=True)}
        for i,task in enumerate(tasks):
            q=np.array(task['current']); xyz=np.array(task['xyz']); axis=np.array(task['axis'])
            assert space.valid(q) and space.valid(task['witness'])
            witness_error=planning.goal_error(task['witness'],xyz,axis)
            assert witness_error['error_mm']<=1 and witness_error['orientation_error_deg']<=1
            for name,cfg in variants.items():
                began=perf_counter()
                result=planning.query(q,xyz,road,cfg,target_axis=axis,shortcut=True)
                rows[name].append({'task_id':task['id'], 'group':task['group'], **row(result,(perf_counter()-began)*1000)})
                if result['converged']:
                    assert result['validation']=='joint_limits_and_self_collision'
                    assert result['final_error_mm']<=1 and result['final_orientation_error_deg']<=1
            print(f"FK {i+1}/{len(tasks)}: "+', '.join(f"{name}={rows[name][-1]['status']}" for name in variants),flush=True)
        report['fk']={'tasks':tasks, 'rows':rows, 'summary':{k:summary(v) for k,v in rows.items()}, 'comparison':compare(rows)}
    if only in ('all','cursor'):
        fixtures=json.loads((Path(__file__).parent/'fixtures/mouse_aim_recovery_cases.json').read_text())
        rows={'baseline':[], 'upgraded':[]}; original=service.plan
        for i,case in enumerate(fixtures['cases']):
            body={k:case[k] for k in ('start_deg','cursor_mm')}
            for name in rows:
                def plan(request, roadmap=None, goal_constraint=None):
                    return original({**request,'pose5_global_recovery':name=='upgraded'},roadmap,goal_constraint)
                began=perf_counter()
                with patch.object(service,'plan',side_effect=plan):
                    result=service.mouse_aim_plan(body,road)
                rows[name].append({'task_id':case['id'], **row(result,(perf_counter()-began)*1000)})
                if result['converged']:assert result['cursor_acceptance']['accepted']
            print(f"Cursor {i+1}/{len(fixtures['cases'])}: "+', '.join(f"{name}={rows[name][-1]['status']}" for name in rows),flush=True)
        report['cursor']={'tasks':fixtures['cases'], 'rows':rows, 'summary':{k:summary(v) for k,v in rows.items()}, 'comparison':compare(rows)}
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:{'summary':report[k]['summary'],'comparison':report[k]['comparison']} for k in ('fk','cursor') if k in report},indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('output/ik_recovery/benchmark.json'))
    parser.add_argument('--count',type=int,default=64);parser.add_argument('--seed',type=int,default=20261004)
    parser.add_argument('--only',choices=['all','fk','cursor'],default='all')
    args=parser.parse_args()
    benchmark(args.output,args.count,args.seed,args.only)
