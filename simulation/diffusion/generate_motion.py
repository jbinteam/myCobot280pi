"""Checked scene-absence, moving-goal and nonzero-motion retarget augmentation.

Derived episodes retain their parent's split. Moving goals look ahead along an
expert route, so future action labels do not anticipate an unseen target jump.
Retargets start on an old route with nonzero motion and join a new checked goal.
The original datasets and their source-pinned generators are never modified.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
from time import perf_counter
import h5py
import numpy as np
from ..kinematics import ARM
from ..continuous import prepare, sample_motion
from .common import file_sha256, write_json, robot_schema
from .env import ArmEnv
from .planner import smooth, trajectory


def derivatives(q,times):
    velocity=np.gradient(q,times,axis=0)
    velocity[0]=0;velocity[-1]=0
    acceleration=np.gradient(velocity,times,axis=0)
    acceleration[0]=0;acceleration[-1]=0
    return velocity,acceleration


def sample_curve(metadata,times):
    states=[];velocities=[];accelerations=[]
    for time in times:
        q,v,a=sample_motion(metadata,time);states.append(q);velocities.append(v);accelerations.append(a)
    return np.asarray(states),np.asarray(velocities),np.asarray(accelerations)


def worker(job):
    source_path,output,names,kinds,seed=job
    rng=np.random.default_rng(seed);counts=Counter();began=perf_counter()
    if Path(output).exists():
        with h5py.File(output,'r') as previous:
            if 'counts_json' in previous.attrs and len(previous['episodes'])==len(names):
                parents=[g.attrs['parent_episode'] for g in previous['episodes'].values()]
                if parents!=names:raise ValueError('Existing augmentation shard has different parents')
                print(f'Reusing completed checked augmentation shard {output}',flush=True)
                return str(output)
    with h5py.File(source_path,'r') as source,h5py.File(output,'w') as result,ArmEnv() as env:
        root=result.create_group('episodes')
        for number,(name,kind) in enumerate(zip(names,kinds)):
            parent=source['episodes'][name];original=parent['q_rad'][:];original_times=parent['time_s'][:]
            no_block='absent' in kind;box=None if no_block else parent['obstacle_m'][:]
            env.set_obstacle(box);q=original.copy();times=original_times.copy();record={}
            if kind=='static_absent':
                # Removing a block permits shorter direct routes; every shortcut
                # and the newly timed path are checked in the actual scene.
                q,times=trajectory(smooth(env,original))
                if not env.path_valid(q):
                    counts['rejected_retimed_shortcut']+=1
                    q=original.copy();times=original_times.copy()
            velocity,acceleration=derivatives(q,times)
            if kind.startswith('retarget'):
                old_v,old_a=derivatives(original,original_times)
                indices=np.flatnonzero((np.linalg.norm(old_v,axis=1)>.15)&(np.linalg.norm(old_a,axis=1)<2.5)&(np.arange(len(original))<len(original)-3))
                accepted=False
                for attempt in range(16):
                    if not len(indices):break
                    index=int(rng.choice(indices));current=original[index]
                    goal=ARM.clip(original[-1]+rng.normal(0,.18,6))
                    if not env.valid(goal):continue
                    route=np.vstack((original[index:],goal))
                    if not env.path_valid(route):continue
                    try:_,fine_times,metadata=prepare(env,route,old_v[index],old_a[index])
                    except ValueError:continue
                    times=np.arange(0,fine_times[-1]+1e-9,.1)
                    # Use a uniform 0.1 s learning grid and keep an exact resting
                    # endpoint; duration need not be a multiple of the UI's dt.
                    if times[-1]<fine_times[-1]-1e-9:times=np.r_[times,times[-1]+.1]
                    q,velocity,acceleration=sample_curve(metadata,np.minimum(times,fine_times[-1]))
                    q[-1]=goal;velocity[-1]=0;acceleration[-1]=0
                    record={'retarget_source_frame':index,'previous_goal_rad':original[-1].tolist(),
                            'initial_speed_rad_s':float(np.linalg.norm(velocity[0])),
                            'curve':metadata}
                    accepted=True;break
                if not accepted:
                    # A failed nonzero join must not be labelled as a retarget.
                    counts['rejected_retarget']+=1;kind='moving_absent' if no_block else 'moving_block'
                    q=original.copy();times=original_times.copy();velocity,acceleration=derivatives(q,times)
            if kind.startswith('moving'):
                # Cropping retains the true incoming motion instead of forcing
                # every episode to start at rest. The target advances ahead of
                # the arm and then holds at the original endpoint.
                index=int(rng.integers(0,max(1,len(q)//4)))
                q=q[index:];times=times[index:]-times[index];velocity=velocity[index:];acceleration=acceleration[index:]
                lookahead=int(rng.integers(16,33));goal_index=np.minimum(np.arange(len(q))+lookahead,len(q)-1)
                goals=q[goal_index];record.update(lookahead_steps=lookahead,crop_frame=index)
            else:goals=np.repeat(q[-1][None],len(q),axis=0)
            xyz=np.asarray([ARM.fk(goal) for goal in goals]);target_velocity=np.gradient(xyz,times,axis=0)
            lengths=np.linalg.norm(target_velocity,axis=1);target_velocity*=np.minimum(1.,.5/np.maximum(lengths,1e-12))[:,None]
            if (not env.path_valid(q) or not all(env.valid(goal) for goal in goals)
                    or np.max(np.linalg.norm(velocity,axis=1))>1+1e-6
                    or np.max(np.linalg.norm(acceleration,axis=1))>3+1e-6):
                raise RuntimeError(f'Augmented episode failed scene/motion checks: {name}/{kind}; '
                                   f'path={env.path_valid(q)}, speed={np.linalg.norm(velocity,axis=1).max()}, acceleration={np.linalg.norm(acceleration,axis=1).max()}')
            group=root.create_group(f'{number:06d}')
            for key,value in [('q_rad',q),('time_s',times),('goal_rad',goals),('velocity_rad_s',velocity),
                              ('acceleration_rad_s2',acceleration),('target_velocity_m_s',target_velocity)]:
                group.create_dataset(key,data=value,compression='gzip',compression_opts=1)
            group.create_dataset('obstacle_m',data=np.zeros(6) if no_block else box)
            group.attrs['obstacle_present']=not no_block;group.attrs['split']=parent.attrs['split']
            group.attrs['parent_episode']=name
            group.attrs['record_json']=json.dumps({'kind':kind,'frames':len(q),**record})
            counts[kind]+=1;counts['frames']+=len(q)
            if number%100==0:print(json.dumps({'shard':Path(output).name,'completed':number+1,'counts':dict(counts),'seconds':perf_counter()-began}),flush=True)
        result.attrs['environment_json']=json.dumps(env.metadata())
        result.attrs['counts_json']=json.dumps(dict(counts))
    return str(output)


def generate(source,output,episodes=20000,workers=8,seed=20261006):
    source=Path(source);output=Path(output)
    if output.exists():raise ValueError('Augmentation output already exists')
    began=perf_counter();source_sha=file_sha256(source)
    with h5py.File(source,'r') as data:
        metadata=json.loads(data.attrs['metadata_json'])
        if metadata['environment']['robot']!=robot_schema():raise ValueError('Source robot mismatch')
        names=sorted(data['episodes'])
    if episodes<10 or episodes>len(names):raise ValueError('Choose 10..source-count augmentations')
    rng=np.random.default_rng(seed);selected=rng.choice(names,episodes,replace=False).tolist()
    proportions=['static_absent']*4+['moving_absent']*3+['moving_block']*2+['retarget_absent','retarget_block']
    kinds=[proportions[i%len(proportions)] for i in range(episodes)]
    folder=output.parent/(output.stem+'_shards');folder.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for index in range(workers):
        part=slice(index*episodes//workers,(index+1)*episodes//workers)
        jobs.append((str(source),str(folder/f'{index:03d}.h5'),selected[part],kinds[part],seed+index+1))
    context=multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(workers,mp_context=context) as pool:
        for result in as_completed([pool.submit(worker,job) for job in jobs]):result.result()
    counts=Counter();splits=Counter();number=0;temporary=output.with_suffix('.partial.h5')
    with h5py.File(temporary,'w') as result:
        root=result.create_group('episodes')
        for job in jobs:
            with h5py.File(job[1],'r') as shard:
                counts.update(json.loads(shard.attrs['counts_json']))
                for group in shard['episodes'].values():
                    shard.copy(group,root,name=f'{number:06d}');number+=1;splits[group.attrs['split']]+=1
        metadata={'kind':'motion_conditioned_augmentation_v2','source':str(source),'source_sha256':source_sha,
                  'completed_episodes':number,'counts':dict(counts),'split_counts':dict(splits),'seed':seed,
                  'environment':metadata['environment'],'dt':.1,'speed_rad_s':1.,'acceleration_rad_s2':3.,
                  'split':'inherits parent episode split; no parent-child leakage',
                  'limitations':'Moving goals follow expert routes; not unrestricted cursor trajectories. No physical dynamics.',
                  'seconds':perf_counter()-began}
        result.attrs['metadata_json']=json.dumps(metadata)
    temporary.replace(output);metadata['sha256']=file_sha256(output);write_json(output.with_suffix('.json'),metadata)
    print(json.dumps(metadata),flush=True);return metadata


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('simulation/artifacts/diffusion_100k_v1/demonstrations.h5'))
    parser.add_argument('--output',type=Path,default=Path('simulation/artifacts/diffusion_motion_v2/augmentation.h5'))
    parser.add_argument('--episodes',type=int,default=20000);parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();generate(args.source,args.output,args.episodes,args.workers)
