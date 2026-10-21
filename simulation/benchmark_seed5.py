"""Held-out 5D endpoint/refinement/planning benchmark; no robot transport."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
import numpy as np
from .kinematics import ARM, rotation
from . import pose5, seed5, planning, service
from .prm import planning_space
from .seed5_data import sample_pairs, load_dataset


def held_out_tasks(count,seed,space):
    data,stats=sample_pairs(count,seed,space)
    tasks=[{'id':f'fk-{i}','group':'held_out_fk','current':q.tolist(),'xyz':p.tolist(),
            'axis':u.tolist(),'witness':g.tolist()} for i,(q,p,u,g) in enumerate(zip(
            data['q_current_rad'],data['target_xyz_m'],data['target_axis_base'],data['q_goal_rad']))]
    # Keep identical XYZ and change the directed tool axis, with FK witnesses
    # checked within a stricter 0.1 mm / 0.1 degree numerical tolerance.
    families=0
    for i,base in enumerate(data['q_goal_rad']):
        t=ARM.transform(base);desired=rotation([1,0,0],.12)@t[:3,2]
        solved=pose5.solve(base,t[:3,3],target_axis=desired,method='bounded_dls',
                           steps=400,tolerance=.1,orientation_tolerance_deg=.1)
        witness=np.deg2rad(solved['frames'][-1]['q'])
        if not solved['converged'] or not space.valid(witness):continue
        axis=ARM.transform(witness)[:3,2]
        if pose5.axis_error(t[:3,2],axis)[2]<2:continue
        for direction,goal in ((t[:3,2],base),(axis,witness)):
            for start in (data['q_current_rad'][i],data['q_current_rad'][(i+1)%count]):
                tasks.append({'id':f'family-{families}-{len(tasks)}','group':f'same_xyz_{families}',
                              'current':start.tolist(),'xyz':t[:3,3].tolist(),
                              'axis':direction.tolist(),'witness':goal.tolist()})
        families+=1
        if families==2:break
    if families<2:raise RuntimeError('Could not generate two valid same-XYZ direction families')
    # Common UI starts exercise singular/near-singular configurations absent
    # from generic continuous uniform sampling. These may require refinement
    # beyond one NN proposal's radius, although all robot limits still apply.
    for name,start in (('zero',np.zeros(6)),('rest',np.deg2rad([0,-40,-130,80,0,50]))):
        if not space.valid(start):raise RuntimeError(f'Common {name} start is invalid for this collision profile')
        for i in range(min(4,count)):
            tasks.append({'id':f'common-{name}-{i}','group':f'common_{name}',
                'current':start.tolist(),'xyz':data['target_xyz_m'][i].tolist(),
                'axis':data['target_axis_base'][i].tolist(),'witness':data['q_goal_rad'][i].tolist()})
    return tasks,stats


def summarize(rows):
    success=[r for r in rows if r['ik_success']]
    return {'tasks':len(rows),'valid_ik_success':len(success),'planned_success':sum(r['planned'] for r in rows),
            'error_averaging':'IK-successful tasks only; per-task failures and errors are retained in rows',
            'direct_plans':sum(r['planning_outcome']=='direct_connection' for r in rows),
            'prm_plans':sum(r['planning_outcome']=='roadmap_path' for r in rows),
            'mean_position_error_mm':float(np.mean([r['position_error_mm'] for r in success])) if success else None,
            'mean_direction_error_deg':float(np.mean([r['direction_error_deg'] for r in success])) if success else None,
            'mean_inference_ms':float(np.mean([r['inference_ms'] for r in rows])),
            'mean_refinement_ms':float(np.mean([r['refinement_ms'] for r in rows])),
            'mean_ik_ms':float(np.mean([r['ik_ms'] for r in rows])),
            'mean_planning_ms':float(np.mean([r['planning_ms'] for r in rows])),
            'goal_sources':{s:sum(s in r['goal_sources'] for r in rows) for s in sorted({s for r in rows for s in r['goal_sources']})}}


def benchmark(output,count=24,seed=20261002,checkpoint=seed5.DEFAULT_CHECKPOINT,validation=False,max_candidates=4):
    loading=perf_counter()
    space=planning_space('self',1.)
    road=service.load_roadmap(Path('simulation/artifacts/roadmap.json'),space)
    roadmap_load_ms=(perf_counter()-loading)*1000
    if validation:
        data,_=load_dataset(Path('simulation/artifacts/seed5_v1/dataset.npz'))
        # Fixed validation subset used for candidate selection, never test tasks.
        indices=np.random.default_rng(1102).choice(len(data['validation_q_current_rad']),count,replace=False)
        tasks=[{'id':f'validation-{i}','group':'selection_validation','current':data['validation_q_current_rad'][i].tolist(),
                'xyz':data['validation_target_xyz_m'][i].tolist(),'axis':data['validation_target_axis_base'][i].tolist(),
                'witness':data['validation_q_goal_rad'][i].tolist()} for i in indices]
        stats={'subset_seed':1102}
    else:tasks,stats=held_out_tasks(count,seed,space)
    config=planning.IKConfig(task='pose5',nn='seed5',max_candidates=max_candidates)
    variants={'numerical_current':replace(config,nn='none',random_seeds=0,roadmap_seeds=0,max_candidates=1),
              'numerical_multiseed':replace(config,nn='none'),
              'seed5_raw':replace(config,refine=False,numerical_fallback=False),
              'seed5_refined':config}
    if validation:variants={'seed5_refined':replace(config,numerical_fallback=False,max_candidates=1)}
    rows={name:[] for name in variants};original_propose=seed5.propose;began=perf_counter()
    with patch.object(seed5,'propose',side_effect=lambda q,p,u:original_propose(q,p,u,checkpoint)):
        for index,task in enumerate(tasks):
            q=np.array(task['current']);xyz=np.array(task['xyz']);axis=np.array(task['axis'])
            for name,cfg in variants.items():
                result=planning.query(q,xyz,road,cfg,target_axis=axis,shortcut=True)
                ik=result.get('ik',{});goals=ik.get('goals',[]);best=ik.get('best_error') or {}
                rows[name].append({'task_id':task['id'],'group':task['group'],'ik_success':bool(goals),
                    'planned':result['converged'],'ik_outcome':result['ik_outcome'],'planning_outcome':result['planning_outcome'],
                    'status':result['status'],'position_error_mm':result.get('final_error_mm',best.get('error_mm')),
                    'direction_error_deg':result.get('final_orientation_error_deg',best.get('orientation_error_deg')),
                    'inference_ms':ik.get('inference_ms',0.),'refinement_ms':sum(a.get('refinement_ms',0.) for a in ik.get('attempts',[])),
                    'ik_ms':result['ik_ms'],'planning_ms':result['planning_ms'],'goal_sources':sorted({g['source'] for g in goals}),
                    'raw_accepted':ik.get('raw_accepted_count',0),'goals':len(goals)})
            print(f"Benchmark {index+1}/{len(tasks)}: "+', '.join(f'{name}={rows[name][-1]["status"]}' for name in variants),flush=True)
    report={'kind':'validation_selection' if validation else 'held_out_benchmark','seed':seed,'generation':stats,
            'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            'tolerance_mm':1.,'orientation_tolerance_deg':1.,'steps_per_seed':240,'max_candidates':max_candidates,
            'seconds':perf_counter()-began,'roadmap_load_ms':roadmap_load_ms,'validity':space.metadata(),'roadmap_reused':'simulation/artifacts/roadmap.json',
            'tasks':tasks,'summary':{name:summarize(values) for name,values in rows.items()},'rows':rows,
            'limitations':'Finite held-out FK sample. Base pairs are proposal-representable; common and cross-family starts may exceed one proposal radius, and numerical refinement remains joint-limit bounded. No physical execution. Intermediate direction is unconstrained. CPU NumPy IK; training uses GPU.'}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report['summary']),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('simulation/artifacts/seed5_v1/benchmark.json'))
    parser.add_argument('--checkpoint',type=Path,default=seed5.DEFAULT_CHECKPOINT)
    parser.add_argument('--count',type=int,default=24);parser.add_argument('--seed',type=int,default=20261002)
    parser.add_argument('--validation',action='store_true');parser.add_argument('--max-candidates',type=int,default=4)
    args=parser.parse_args();benchmark(args.output,args.count,args.seed,args.checkpoint,args.validation,args.max_candidates)


if __name__=='__main__':main()
