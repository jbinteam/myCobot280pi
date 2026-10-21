"""Offline scene-valid FK supervision with target-group-disjoint splits."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.optimize import least_squares
from ..kinematics import ARM
from ..workspace import transforms
from .env import ArmEnv
from .ik_seed import configuration
from .endpoint_ik import residual_and_jacobian

FIELDS = ('q_current_rad', 'q_goal_rad', 'target_xyz_m', 'target_axis_base', 'target_id')
SPLITS = ('train', 'validation', 'test')


def generate(path, train_targets=16000, validation_targets=512, test_targets=256,
             currents_per_target=2, seed=202610071, current_pool=2048):
    counts=(train_targets,validation_targets,test_targets)
    if any(isinstance(n,bool) or not isinstance(n,int) or n<1 for n in (*counts,currents_per_target,current_pool)):
        raise ValueError('Positive integer target counts, current pool and pairs required')
    path=Path(path); rng=np.random.default_rng(seed);arrays={};stats={};offset=0
    with ArmEnv() as env:
        env.set_obstacle(None)
        scene=env.metadata()
        def sample(count):
            accepted=[];attempts=0
            while len(accepted)<count:
                if attempts>count*100+1000:raise RuntimeError('Scene-valid sampling budget exhausted')
                q=rng.uniform(ARM.lower,ARM.upper);attempts+=1
                if env.valid(q):accepted.append(q)
            return np.asarray(accepted),attempts
        for name,count in zip(SPLITS,counts):
            witness,attempts=sample(count)
            pool,pool_attempts=sample(min(current_pool,max(32,count)))
            fk=transforms(witness)
            repeat=np.repeat(np.arange(count),currents_per_target)
            # Independent starting poses are selected only after target splitting.
            selection=rng.integers(len(pool),size=len(repeat))
            arrays[name+'_q_current_rad']=pool[selection]
            arrays[name+'_q_goal_rad']=witness[repeat]
            arrays[name+'_target_xyz_m']=fk[repeat,:3,3]
            arrays[name+'_target_axis_base']=fk[repeat,:3,2]
            arrays[name+'_target_id']=offset+repeat
            offset+=count
            stats[name]={'target_groups':count,'pairs':len(repeat),'witness_sampling_attempts':attempts,
                         'current_pool':len(pool),'current_sampling_attempts':pool_attempts}
            print(f'{name}: {count} valid targets, {len(repeat)} pairs',flush=True)
    metadata={'configuration':configuration(),'schema_version':1,
              'generation':{'seed':seed,'currents_per_target':currents_per_target,'splits':stats,
                  'strategy':'Uniform operating-box joint witnesses; rejection by exact runtime self/table validity; independent current pools',
                  'split_unit':'FK target group before current pairing','target_leakage':'No target_id or exact XYZ+directed-axis label crosses splits',
                  'supervision':'Known reachable FK witness, not an IK solver success claim'},
              'scene':scene,'obstacle':None,'validity':'ArmEnv.valid including operating limits, conservative self collision, table and 1 mm clearance',
              'units':{'joints':'radians','position':'meters','axis':'directed unit local Z in base'},
              'safety':'All joint samples and starts are virtual; no robot IO or trajectories'}
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**arrays)
    metadata['dataset_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix('.json').write_text(json.dumps(metadata,indent=2,allow_nan=False))
    return metadata


def load_dataset(path, check_scene=False):
    path=Path(path);metadata=json.loads(path.with_suffix('.json').read_text())
    if metadata.get('configuration')!=configuration():raise ValueError('Dataset IK schema/FK/limits mismatch')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=metadata.get('dataset_sha256'):raise ValueError('Dataset content hash mismatch')
    with np.load(path,allow_pickle=False) as archive:arrays={k:archive[k].copy() for k in archive.files}
    targets=set();ids=set()
    for split in SPLITS:
        q,witness,xyz,axis,group=(arrays[split+'_'+key] for key in FIELDS)
        if q.ndim!=2 or q.shape[1]!=6 or not len(q) or witness.shape!=q.shape or xyz.shape!=(len(q),3) or axis.shape!=xyz.shape or group.shape!=(len(q),):
            raise ValueError('Dataset shape mismatch')
        if not all(np.isfinite(x).all() for x in (q,witness,xyz,axis)) or group.dtype.kind not in 'iu':raise ValueError('Nonfinite data or invalid target IDs')
        if any(np.any(x<ARM.lower) or np.any(x>ARM.upper) for x in (q,witness)):raise ValueError('Dataset joint limits violated')
        fk=transforms(witness)
        if not np.allclose(xyz,fk[:,:3,3],atol=1e-10,rtol=0) or not np.allclose(axis,fk[:,:3,2],atol=1e-10,rtol=0):raise ValueError('Dataset FK labels inconsistent')
        split_ids=set(group.tolist());split_targets={row.tobytes() for row in np.c_[xyz,axis]}
        if ids&split_ids or targets&split_targets:raise ValueError('Target leakage across data splits')
        _,first,inverse=np.unique(group,return_index=True,return_inverse=True)
        if not np.array_equal(xyz,xyz[first][inverse]) or not np.array_equal(axis,axis[first][inverse]):raise ValueError('Target group has differing labels')
        ids.update(split_ids);targets.update(split_targets)
    if check_scene:
        with ArmEnv() as env:
            env.set_obstacle(None)
            if metadata['scene']!=env.metadata():raise ValueError('Dataset collision scene metadata changed')
            for split in SPLITS:
                for key in FIELDS[:2]:
                    if not all(env.valid(q) for q in np.unique(arrays[split+'_'+key],axis=0)):raise ValueError('Dataset fails runtime scene validity')
    return arrays,metadata


def label_starts(dataset,output,count=64,attempts=8,seed=202610073):
    """Audit broad numerical starts on training groups only, never test targets.

    The FK witness itself is a certified useful initialization. Additional
    random starts record successful solver basins and accepted final angles;
    this audit does not alter target labels or select model weights.
    """
    if count<1 or attempts<1:raise ValueError('Positive label budgets required')
    arrays,metadata=load_dataset(dataset);rng=np.random.default_rng(seed)
    _,unique=np.unique(arrays['train_target_id'],return_index=True)
    rows=rng.permutation(unique)[:count];labels=[];rejected=0;tick=perf_counter()
    with ArmEnv() as env:
        env.set_obstacle(None)
        for row in rows:
            xyz=arrays['train_target_xyz_m'][row];axis=arrays['train_target_axis_base'][row]
            witness=arrays['train_q_goal_rad'][row]
            starts=[witness,arrays['train_q_current_rad'][row],*rng.uniform(ARM.lower,ARM.upper,(attempts-1,6))]
            for index,start in enumerate(starts):
                result=least_squares(lambda q:residual_and_jacobian(q,xyz,axis)[0],start,
                    jac=lambda q:residual_and_jacobian(q,xyz,axis)[1],bounds=(ARM.lower,ARM.upper),
                    max_nfev=160,ftol=1e-9,xtol=1e-9,gtol=1e-9)
                fk=ARM.transform(result.x);position=np.linalg.norm(fk[:3,3]-xyz)*1000
                angle=np.rad2deg(np.arctan2(np.linalg.norm(np.cross(fk[:3,2],axis)),np.clip(fk[:3,2]@axis,-1,1)))
                if position<=1 and angle<=1 and env.valid(result.x):
                    labels.append((arrays['train_target_id'][row],index,start.copy(),result.x.copy(),result.nfev))
                else:rejected+=1
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,target_id=np.array([x[0] for x in labels]),start_kind=np.array([x[1] for x in labels]),
        q_initial_rad=np.array([x[2] for x in labels]),q_refined_rad=np.array([x[3] for x in labels]),nfev=np.array([x[4] for x in labels]))
    report={'schema_version':1,'configuration':configuration(),'dataset_sha256':metadata['dataset_sha256'],
        'seed':seed,'target_split':'train only','targets':len(rows),'starts_per_target':attempts+1,
        'start_kind':'0=known FK witness,1=paired current,2+=uniform joint guesses',
        'accepted_labels':len(labels),'rejected':rejected,'seconds':perf_counter()-tick,
        'acceptance':'≤1 mm, directed tool-Z≤1 degree, bounds and runtime scene collision-valid',
        'used_for_training':False,'scope':'Optional successful-initialization audit; strict solver refinement always required at runtime'}
    output.with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False));return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('simulation/artifacts/ik_seed_v1/dataset.npz'))
    parser.add_argument('--train-targets',type=int,default=16000);parser.add_argument('--validation-targets',type=int,default=512)
    parser.add_argument('--test-targets',type=int,default=256);parser.add_argument('--currents-per-target',type=int,default=2)
    parser.add_argument('--seed',type=int,default=202610071)
    args=parser.parse_args();print(json.dumps(generate(args.output,args.train_targets,args.validation_targets,args.test_targets,args.currents_per_target,args.seed)['generation']))


if __name__=='__main__':main()
