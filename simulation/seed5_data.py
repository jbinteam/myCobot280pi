"""Reproducible FK-labeled, validity-checked pairs for the versioned 5D Seed NN."""
import argparse
import json
from pathlib import Path
import numpy as np
from .kinematics import ARM
from .prm import planning_space
from .seed5 import configuration
from .workspace import transforms


def sample_pairs(count,seed,space):
    rng=np.random.default_rng(seed);pairs=[];nearby_flags=[];attempted=0
    while sum(len(x[0]) for x in pairs)<count:
        if attempted>max(1000,count*200):raise RuntimeError('Insufficient valid representable FK pairs within sampling budget')
        size=min(256,max(16,count-sum(len(x[0]) for x in pairs)))
        current=ARM.center+ARM.radius*rng.uniform(-.9,.9,(size,6))
        goal=ARM.center+ARM.radius*rng.uniform(-.9,.9,(size,6))
        nearby=rng.random(size)<.75
        goal[nearby]=ARM.clip(current[nearby]+rng.normal(0,.35,(nearby.sum(),6)))
        representable=np.all(np.abs(goal-current)<ARM.radius*(1-1e-6),axis=1)
        valid=np.array([fits and space.valid(a) and space.valid(b) for a,b,fits in zip(current,goal,representable)])
        if valid.any():pairs.append((current[valid],goal[valid]));nearby_flags.extend(nearby[valid].tolist())
        attempted+=size
    current=np.concatenate([x[0] for x in pairs])[:count];goal=np.concatenate([x[1] for x in pairs])[:count]
    fk=transforms(goal)
    return {'q_current_rad':current,'q_goal_rad':goal,'target_xyz_m':fk[:,:3,3],
            'target_axis_base':fk[:,:3,2]}, {'attempted':attempted,'accepted':count,'nearby_accepted':sum(nearby_flags[:count])}


def generate(path,count=24000,validation_count=2048,seed=42,validation_seed=43,collision='self',clearance_mm=1.):
    if count<1 or validation_count<1 or seed==validation_seed:raise ValueError('Positive split counts and different train/validation seeds required')
    path=Path(path)
    if path.resolve()==(Path(__file__).parent/'artifacts/dataset.npz').resolve():raise ValueError('Do not overwrite the existing XYZ dataset')
    space=planning_space(collision,clearance_mm)
    train,train_stats=sample_pairs(count,seed,space)
    validation,validation_stats=sample_pairs(validation_count,validation_seed,space)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**train,**{'validation_'+key:value for key,value in validation.items()})
    metadata={'configuration':configuration(),'generation':{'train_seed':seed,'validation_seed':validation_seed,
              'strategy':'75% nearby normal delta sigma=0.35 rad; 25% independent uniform goals; reject invalid/unrepresentable pairs',
              'proposal_bound':'abs(q_goal-q_current) < joint_radius*(1-1e-6)',
              'train':train_stats,'validation':validation_stats},'validity':space.metadata(),
              'units':{'q_current_rad':'radians','q_goal_rad':'radians','target_xyz_m':'meters',
                       'target_axis_base':'directed unit vector in g_base'},
              'target_offset':'Resolved flange XYZ; no extra tool/TCP transform in FK'}
    path.with_suffix('.json').write_text(json.dumps(metadata,indent=2,allow_nan=False))
    return metadata


def load_dataset(path):
    path=Path(path);metadata=json.loads(path.with_suffix('.json').read_text())
    if metadata.get('configuration')!=configuration():raise ValueError('Dataset task/FK/normalization metadata mismatch')
    with np.load(path,allow_pickle=False) as archive:data={key:archive[key].copy() for key in archive.files}
    for prefix in ('','validation_'):
        q=data[prefix+'q_current_rad'];goal=data[prefix+'q_goal_rad'];xyz=data[prefix+'target_xyz_m'];axis=data[prefix+'target_axis_base']
        if q.ndim!=2 or q.shape[1]!=6 or len(q)<1 or goal.shape!=q.shape or xyz.shape!=(len(q),3) or axis.shape!=(len(q),3):raise ValueError('Invalid Seed5 dataset shapes')
        if not all(np.isfinite(a).all() for a in (q,goal,xyz,axis)):raise ValueError('Nonfinite Seed5 dataset values')
        if np.any(q<ARM.lower) or np.any(q>ARM.upper) or np.any(goal<ARM.lower) or np.any(goal>ARM.upper):raise ValueError('Seed5 dataset joint limits violated')
        if np.any(np.abs(goal-q)>=ARM.radius):raise ValueError('Seed5 dataset goal exceeds proposal delta bounds')
        fk=transforms(goal)
        if not np.allclose(xyz,fk[:,:3,3],atol=1e-10,rtol=0) or not np.allclose(axis,fk[:,:3,2],atol=1e-10,rtol=0):raise ValueError('Seed5 dataset FK labels inconsistent')
    return data,metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('simulation/artifacts/seed5_v1/dataset.npz'))
    parser.add_argument('--count',type=int,default=24000);parser.add_argument('--validation-count',type=int,default=2048)
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--validation-seed',type=int,default=43)
    parser.add_argument('--collision',choices=['self','limits'],default='self');parser.add_argument('--clearance-mm',type=float,default=1.)
    args=parser.parse_args()
    metadata=generate(args.output,args.count,args.validation_count,args.seed,args.validation_seed,args.collision,args.clearance_mm)
    print(json.dumps({'output':str(args.output),'generation':metadata['generation'],'validation':metadata['validity']['validation']}))


if __name__=='__main__':main()
