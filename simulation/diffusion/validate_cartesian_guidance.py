"""Validation-only paired one-decision neural/guided controller prototype.

Uses the frozen twelve validation episodes, each fixed to its final Cartesian
target, from both its initial and near-terminal pose. This measures proposed
checked reference accuracy, not closed-loop success or trained-model accuracy.
The same denoised samples feed both branches; no test examples are accessed.
"""
import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import json
import numpy as np
from ..continuous import prepare
from .cartesian_conditioning import supports
from .cartesian_guidance import GuidanceConfig,guide_chunk
from .common import write_json
from .env import ArmEnv
from .policy import DiffusionPolicy,checked_chunk
from .service import condition,cartesian_score,errors,static_capture_prefix
from .validate_cartesian import tasks


def forbidden_ik(*args,**kwargs):raise AssertionError('Guidance must not call IK')


def run(checkpoint,source,augmentation,cache,output,seed=20261014,device='cuda'):
    chosen,cache_metadata=tasks(source,augmentation,cache)
    policy=DiffusionPolicy(checkpoint,device,seed)
    if not supports(policy):raise ValueError('Expected Cartesian checkpoint')
    config=GuidanceConfig();cases=[]
    for task in chosen:
        for phase,index in (('initial',0),('near_terminal',max(0,len(task['q'])-4))):
            cases.append({'task':task,'phase':phase,'q':task['q'][index].copy(),
                'target':task['targets'][-1],'axis':task['axes'][-1]})
    observations=np.asarray([condition(policy,c['q'],None,c['target'],c['axis'],c['task']['box']) for c in cases])
    chunks=policy.sample(observations,4,32)
    rows=[]
    with ArmEnv() as env:
        env.inverse_kinematics=forbidden_ik
        for case,proposals in zip(cases,chunks):
            task=case['task'];q=case['q'];target=case['target'];axis=case['axis']
            env.set_obstacle(task['box']);env.reset(q)
            if not env.path_valid(task['q']):raise ValueError('Preserved complete validation expert route failed collision replay')
            pure_candidates=[];guided_candidates=[];rejects=Counter();guidance_rows=[];guidance_ms=[]
            for raw in proposals:
                chunk,reason=checked_chunk(env,q,raw)
                if chunk is None:rejects[reason]+=1;continue
                began=perf_counter()
                guided,metadata=guide_chunk(env,q,chunk,target,axis,config=config)
                guidance_ms.append((perf_counter()-began)*1000);guidance_rows.append(metadata)
                for route,bucket in ((chunk,pure_candidates),(guided,guided_candidates)):
                    prefix,captured=static_capture_prefix(route,target,axis)
                    score=cartesian_score(prefix[-1],target,axis)+.25*cartesian_score(prefix[min(1,len(prefix)-1)],target,axis)
                    bucket.append((0 if captured else 1,score,prefix))

            def select(candidates):
                for _,_,route in sorted(candidates,key=lambda item:item[:2]):
                    try:_,_,motion=prepare(env,np.vstack((q,route)),np.zeros(6),np.zeros(6))
                    except ValueError:continue
                    error=errors(route[-1],target,axis)
                    return {**error,'strict_pose':error['error_mm']<=1 and error['orientation_error_deg']<=1,
                        'collision_checked':True,'c2_validated':True,
                        'max_velocity_rad_s':motion['max_velocity_rad_s'],
                        'max_acceleration_rad_s2':motion['max_acceleration_rad_s2']}
                return None
            row={'id':task['id'],'kind':task['kind'],'phase':case['phase'],
                'initial_error':errors(q,target,axis),'raw_rejections':dict(rejects),
                'pure':select(pure_candidates),'guided':select(guided_candidates),
                'guidance':guidance_rows,'mean_guidance_ms':float(np.mean(guidance_ms)) if guidance_ms else None}
            rows.append(row)
            print(json.dumps({k:v for k,v in row.items() if k not in ('guidance',)}),flush=True)
    summaries={}
    for phase in ('initial','near_terminal'):
        group=[row for row in rows if row['phase']==phase]
        paired=[row for row in group if row['pure'] is not None and row['guided'] is not None]
        summaries[phase]={'cases':len(group),'paired_playable_candidates':len(paired),
            'position_improved_cases':sum(row['guided']['error_mm']<row['pure']['error_mm']-1e-6 for row in paired),
            'pure_strict_proposed_pose':sum(row['pure'] is not None and row['pure']['strict_pose'] for row in group),
            'guided_strict_proposed_pose':sum(row['guided'] is not None and row['guided']['strict_pose'] for row in group),
            'pure_median_error_mm':float(np.median([row['pure']['error_mm'] for row in paired])) if paired else None,
            'guided_median_error_mm':float(np.median([row['guided']['error_mm'] for row in paired])) if paired else None,
            'pure_median_axis_deg':float(np.median([row['pure']['orientation_error_deg'] for row in paired])) if paired else None,
            'guided_median_axis_deg':float(np.median([row['guided']['orientation_error_deg'] for row in paired])) if paired else None,
            'mean_guidance_ms_per_safe_proposal':float(np.mean([v for row in group for v in [row['mean_guidance_ms']] if v is not None]))}
    report={'kind':'optional_gradient_controller_validation_prototype','split':'validation only',
        'checkpoint_sha256':policy.sha256,'dataset_sha256':cache_metadata['identity']['dataset_sha256'],
        'seed':seed,'configuration':asdict(config),'production_integrated':False,
        'task_ids':[task['id'] for task in chosen],
        'scope':'Same twelve frozen validation episodes fixed to final Cartesian target; initial and near-terminal static rest states; '
            'one decision, shared four neural proposals, 32 denoising steps. Checked proposed endpoint accuracy, not execution or closed-loop/model success.',
        'no_goal_joint_inputs':True,'ik_disabled':True,'summaries':summaries,'rows':rows}
    write_json(output,report);print(json.dumps(summaries,indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=Path('simulation/artifacts/diffusion_cartesian_v3/model/diffusion_policy_6dof.pth'))
    parser.add_argument('--source',type=Path,default=Path('simulation/artifacts/diffusion_100k_v1/demonstrations.h5'))
    parser.add_argument('--augmentation',type=Path,default=Path('simulation/artifacts/diffusion_motion_v2/augmentation.h5'))
    parser.add_argument('--cache',type=Path,default=Path('simulation/artifacts/diffusion_cartesian_v3/cache'))
    parser.add_argument('--output',type=Path,default=Path('output/diffusion/cartesian_v3_guidance_validation.json'))
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    run(**vars(parser.parse_args()))
