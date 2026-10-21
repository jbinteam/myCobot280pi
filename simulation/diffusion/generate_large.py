"""Resumable process-sharded expert generation and checked atomic HDF5 merge."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import fcntl
import json
import multiprocessing
from pathlib import Path
from time import perf_counter
import h5py
import numpy as np
from ..kinematics import ARM
from .common import file_sha256, robot_schema, write_json
from .env import ArmEnv
from .generate_dataset import generate
from .planner import RRTConfig


def worker(job):
    path,count,seed,fraction,configuration=job
    # Native libraries must not start additional CPU pools per process.
    return generate(path,count,seed,fraction,RRTConfig(**configuration))


def valid_shard(path, specification, identity):
    if not path.exists():return False
    record=json.loads(path.with_suffix('.json').read_text())
    if (record['completed_episodes']!=specification['episodes'] or record['seed']!=specification['seed']
            or record['planner']!=identity['planner'] or record['environment']['robot']!=identity['robot']
            or record['blocked_fraction_requested']!=identity['blocked_fraction']
            or record['sha256']!=file_sha256(path)):
        raise ValueError(f'Existing shard differs from this generation plan: {path}')
    return True


def merge(output, shards, identity, began, audit_episodes=1000):
    """Stream records; global stratified splits; independent scene sample audit."""
    temporary=output.with_suffix('.partial.h5');counts=Counter();failures=Counter();strata={}
    provenance=[];environment=None;number=0
    with h5py.File(temporary,'w') as destination:
        root=destination.create_group('episodes')
        for path in shards:
            record=json.loads(path.with_suffix('.json').read_text())
            if record['sha256']!=file_sha256(path):raise ValueError('Shard changed before merge')
            provenance.append({'file':path.name,'seed':record['seed'],'episodes':record['completed_episodes'],'sha256':record['sha256']})
            failures.update(record['failures'])
            with h5py.File(path,'r') as source:
                metadata=json.loads(source.attrs['metadata_json'])
                if environment is None:environment=metadata['environment']
                if metadata['environment']!=environment:raise ValueError('Shard collision environments differ')
                if metadata['completed_episodes']!=len(source['episodes']):raise ValueError('Incomplete shard')
                for name,group in source['episodes'].items():
                    q=group['q_rad'][:];times=group['time_s'][:];goal=group['q_goal_rad'][:]
                    box=group['obstacle_m'][:];pose=ARM.transform(goal)
                    if (q.ndim!=2 or q.shape[1]!=6 or len(q)<2 or times.shape!=(len(q),)
                            or not np.isfinite(q).all() or not np.isfinite(times).all()
                            or np.any(q<ARM.lower) or np.any(q>ARM.upper) or times[0]!=0
                            or not np.allclose(np.diff(times),.1,atol=1e-9,rtol=0)
                            or not np.array_equal(q[-1],goal) or box.shape!=(6,) or not np.isfinite(box).all()
                            or not np.allclose(group['goal_xyz_m'][:],pose[:3,3],atol=1e-9,rtol=0)
                            or not np.allclose(group['goal_axis_base'][:],pose[:3,2],atol=1e-9,rtol=0)):
                        raise ValueError(f'Invalid shard trajectory {path.name}/{name}')
                    velocity=np.diff(q,axis=0)/np.diff(times)[:,None]
                    # Include the initial and final hold in the acceleration audit.
                    acceleration=np.diff(np.vstack((np.zeros(6),velocity,np.zeros(6))),axis=0)/.1
                    if np.linalg.norm(velocity,axis=1).max()>1+1e-8 or np.linalg.norm(acceleration,axis=1).max()>3+1e-8:
                        raise ValueError('Expert timing violates speed/acceleration limits')
                    episode=json.loads(group.attrs['record_json']);episode.update(episode=number,source_shard=path.name,source_episode=name)
                    destination.copy(group,root,name=f'{number:06d}')
                    copied=root[f'{number:06d}'];copied.attrs['record_json']=json.dumps(episode)
                    strata.setdefault((episode['direct_blocked'],episode['nearby']),[]).append(number)
                    counts['direct_blocked']+=int(episode['direct_blocked']);counts['nearby']+=int(episode['nearby']);counts['frames']+=len(q)
                    number+=1
            print(json.dumps({'phase':'merge','completed_episodes':number,'requested':identity['episodes']}),flush=True)
        if number!=identity['episodes']:raise ValueError('Merged episode count differs from request')
        rng=np.random.default_rng(identity['seed']);splits=Counter()
        for indices in strata.values():
            rng.shuffle(indices);train_end=int(len(indices)*.8);validation_end=train_end+max(1,int(len(indices)*.1))
            for i,index in enumerate(indices):
                split='train' if i<train_end else 'validation' if i<validation_end else 'test'
                root[f'{index:06d}'].attrs['split']=split;splits[split]+=1
        selected=np.random.default_rng(identity['seed']+1).choice(number,min(audit_episodes,number),replace=False)
        audit_states=0
        with ArmEnv() as env:
            if env.metadata()!=environment:raise ValueError('Final audit scene differs from generation')
            for i,index in enumerate(selected):
                group=root[f'{index:06d}'];env.set_obstacle(group['obstacle_m'][:]);q=group['q_rad'][:]
                if not env.path_valid(q):raise ValueError(f'Independent scene audit failed at episode {index}')
                audit_states+=len(q)
                if (i+1)%100==0:print(json.dumps({'phase':'independent_collision_audit','episodes':i+1,'states':audit_states}),flush=True)
        metadata={'kind':'rrt_star_expert_demonstrations','seed':identity['seed'],'requested_episodes':number,
            'completed_episodes':number,'environment':environment,'planner':identity['planner'],
            'dt':.1,'speed_rad_s':1.,'acceleration_rad_s2':3.,'blocked_fraction_requested':identity['blocked_fraction'],
            'split':'global episode-level stratified train/validation/test 80/10/10; never window-level',
            'counts':dict(counts),'split_counts':dict(splits),'failures':dict(failures),'shards':provenance,
            'generation_identity':identity,'seconds':perf_counter()-began,
            'validation':{'every_episode':'generated scene-valid; merged finite shapes/FK/limits/speed/acceleration checked',
                          'independent_collision_episodes':len(selected),'independent_collision_states':audit_states}}
        destination.attrs['metadata_json']=json.dumps(metadata);destination.flush()
    temporary.replace(output);metadata['sha256']=file_sha256(output)
    write_json(output.with_suffix('.json'),metadata);return metadata


def generate_large(output, episodes=100000, seed=20261005, workers=8, shard_size=1000, blocked_fraction=.2,
                   config=RRTConfig(), resume=False, audit_episodes=1000):
    if isinstance(episodes,bool) or not isinstance(episodes,int) or episodes<10:raise ValueError('At least 10 episodes required')
    if not isinstance(workers,int) or isinstance(workers,bool) or not 1<=workers<=32:raise ValueError('Workers must be 1..32')
    if not isinstance(shard_size,int) or shard_size<10 or episodes%shard_size:raise ValueError('Shard size ≥10 must divide the requested count')
    if not 0<=blocked_fraction<=1 or audit_episodes<1:raise ValueError('Invalid blocked fraction or audit count')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    directory=output.parent/(output.stem+'_shards');directory.mkdir(exist_ok=True)
    identity={'version':1,'episodes':episodes,'seed':seed,'shard_size':shard_size,
              'blocked_fraction':blocked_fraction,'planner':asdict(config),'robot':robot_schema(),
              'source_sha256':{name:file_sha256(Path(__file__).parent/name)
                               for name in ('generate_dataset.py','planner.py','env.py','common.py')}}
    seeds=np.random.SeedSequence(seed).generate_state(episodes//shard_size,dtype=np.uint32)
    specifications=[{'file':f'shard-{i:04d}.h5','episodes':shard_size,'seed':int(value)} for i,value in enumerate(seeds)]
    plan_path=directory/'generation_plan.json';progress=directory/'progress.json';began=perf_counter()
    with (directory/'generation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if plan_path.exists():
            if not resume:raise ValueError('Generation directory exists; use --resume for this exact plan')
            if json.loads(plan_path.read_text())!={'identity':identity,'shards':specifications}:raise ValueError('Resume configuration differs from saved plan')
        else:
            if output.exists():raise ValueError('Existing dataset is preserved; choose a new output')
            write_json(plan_path,{'identity':identity,'shards':specifications})
        if output.exists():
            metadata=json.loads(output.with_suffix('.json').read_text())
            if metadata['generation_identity']!=identity or metadata['sha256']!=file_sha256(output):raise ValueError('Completed dataset differs from saved plan')
            return metadata
        complete=[];jobs=[]
        for spec in specifications:
            path=directory/spec['file']
            if valid_shard(path,spec,identity):complete.append(path)
            else:jobs.append((path,spec['episodes'],spec['seed'],blocked_fraction,asdict(config)))
        def status(phase):
            value={'phase':phase,'completed_episodes':len(complete)*shard_size,'requested_episodes':episodes,
                   'completed_shards':len(complete),'total_shards':len(specifications),'workers':workers,'seconds':perf_counter()-began}
            temporary=progress.with_suffix('.tmp');write_json(temporary,value);temporary.replace(progress)
            print(json.dumps(value),flush=True)
        status('generation')
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures={pool.submit(worker,job):job[0] for job in jobs}
            for future in as_completed(futures):
                future.result();complete.append(futures[future]);status('generation')
        status('merging')
        metadata=merge(output,[directory/s['file'] for s in specifications],identity,began,audit_episodes)
        status('complete');return metadata


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--episodes',type=int,default=100000)
    parser.add_argument('--seed',type=int,default=20261005);parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--shard-size',type=int,default=1000);parser.add_argument('--blocked-fraction',type=float,default=.2)
    parser.add_argument('--iterations',type=int,default=800);parser.add_argument('--post-solution',type=int,default=40)
    parser.add_argument('--audit-episodes',type=int,default=1000);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    result=generate_large(args.output,args.episodes,args.seed,args.workers,args.shard_size,args.blocked_fraction,
                          RRTConfig(iterations=args.iterations,post_solution=args.post_solution),args.resume,args.audit_episodes)
    print(json.dumps({'phase':'complete','episodes':result['completed_episodes'],'sha256':result['sha256']}),flush=True)
