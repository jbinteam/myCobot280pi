"""Compare collision-aware Seed5 candidates with exact, held-out checks."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
import numpy as np
from . import planning, seed5
from .kinematics import ARM
from .prm import planning_space
from .seed5_data import sample_pairs


def compare(checkpoints, output, count=128, seed=20261004):
    space=planning_space('self',1.)
    data,generation=sample_pairs(count,seed,space)
    config=planning.IKConfig(task='pose5',nn='seed5',numerical_fallback=False,max_candidates=1)
    report={'kind':'held_out_collision_training_comparison','simulation_only':True,
            'generation':generation,'seed':seed,'tasks':count,'validity':space.metadata(),
            'configuration':{'tolerance_mm':1.,'orientation_tolerance_deg':1.,'steps':240,
                             'max_candidates':1,'numerical_fallback':False,'path_sample_rad':.04},
            'models':{},'limitations':'Finite held-out FK sample. Direct connections are checked; PRM detours are not evaluated. No physical execution.'}
    for checkpoint in checkpoints:
        checkpoint=Path(checkpoint);model=seed5.Seed5Model(checkpoint)
        rows=[];began=perf_counter()
        for index,(q,xyz,axis) in enumerate(zip(data['q_current_rad'],data['target_xyz_m'],data['target_axis_base'])):
            candidates=model.predict(q,xyz,axis)
            fk=np.stack([ARM.transform(c) for c in candidates])
            position=np.linalg.norm(fk[:,:3,3]-xyz,axis=1)*1000
            angle=np.rad2deg(np.arctan2(np.linalg.norm(np.cross(fk[:,:3,2],axis),axis=1),np.clip(fk[:,:3,2]@axis,-1,1)))
            valid=np.array([space.valid(c) for c in candidates])
            scores=(position/1000/ARM.scale)**2+.72*(1-np.cos(np.deg2rad(angle)))
            best=int(np.argmin(scores))
            with patch.object(seed5,'propose',return_value=(candidates,{'status':'ready'},0.)):
                result=planning.generate_goals(q,xyz,space=space,config=config,target_axis=axis)
            goals=result['goals']
            direct=any(space.edge_valid(q,g['q_rad'],.04) for g in goals)
            rows.append({'task':index,'raw_invalid_proposals':int((~valid).sum()),
                         'raw_valid_success':bool(np.any(valid & (position<=1.) & (angle<=1.))),
                         'any_valid_proposal':bool(valid.any()),'pose_best_invalid':bool(not valid[best]),
                         'pose_best_error_mm':float(position[best]),'pose_best_direction_deg':float(angle[best]),
                         'refined_valid_success':bool(goals),'refined_direct_success':direct,
                         'refinement_iterations':sum(a.get('iterations',0) for a in result['attempts'])})
        summary={'raw_proposals':count*4,'raw_invalid_proposals':sum(r['raw_invalid_proposals'] for r in rows),
                 'raw_invalid_fraction':sum(r['raw_invalid_proposals'] for r in rows)/(count*4),
                 'raw_valid_success':sum(r['raw_valid_success'] for r in rows),
                 'tasks_with_valid_proposal':sum(r['any_valid_proposal'] for r in rows),
                 'pose_best_invalid_tasks':sum(r['pose_best_invalid'] for r in rows),
                 'mean_pose_best_error_mm':float(np.mean([r['pose_best_error_mm'] for r in rows])),
                 'mean_pose_best_direction_deg':float(np.mean([r['pose_best_direction_deg'] for r in rows])),
                 'refined_valid_success':sum(r['refined_valid_success'] for r in rows),
                 'refined_direct_success':sum(r['refined_direct_success'] for r in rows),
                 'mean_refinement_iterations':float(np.mean([r['refinement_iterations'] for r in rows])),
                 'seconds':perf_counter()-began}
        report['models'][str(checkpoint)]={'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'training':{k:model.data['training'].get(k) for k in ('kind','completed_steps','seed','initialization','collision_training')},
            'summary':summary,'rows':rows}
        print(json.dumps({'checkpoint':str(checkpoint),'summary':summary}),flush=True)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoints',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,default=Path('output/collision_training/comparison.json'))
    parser.add_argument('--count',type=int,default=128);parser.add_argument('--seed',type=int,default=20261004)
    args=parser.parse_args();compare(args.checkpoints,args.output,args.count,args.seed)


if __name__=='__main__':main()
