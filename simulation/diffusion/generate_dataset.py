"""Generate seeded, collision-checked myCobot RRT* demonstrations into HDF5."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
import h5py
import numpy as np
from ..kinematics import ARM
from .common import ARTIFACTS, file_sha256, write_json
from .env import ArmEnv
from .planner import RRTConfig, plan, smooth, trajectory


def sample_task(env,rng,blocked=False,attempts=1000):
    """Sample valid endpoints and a table-mounted box; optionally block the edge."""
    env.set_obstacle(None)
    for _ in range(attempts):
        start=rng.uniform(ARM.lower*.9,ARM.upper*.9)
        if not env.valid(start):continue
        nearby=rng.random()<.65
        goal=ARM.clip(start+rng.normal(0,.4,6)) if nearby else rng.uniform(ARM.lower*.9,ARM.upper*.9)
        if not env.valid(goal) or np.linalg.norm(goal-start)<.2:continue
        if blocked:
            middle=start+rng.uniform(.25,.75)*(goal-start);point=ARM.fk(middle)
            if np.linalg.norm(point[:2])<.095 or point[2]<.015:continue
            half_xy=rng.uniform(.012,.028,2);top=max(point[2]+.025,.06)
            height=top-env.table_top
            box=np.r_[point[:2]+rng.normal(0,.006,2),env.table_top+height/2,half_xy,height/2]
        else:
            angle=rng.uniform(-np.pi,np.pi);radius=rng.uniform(.13,.27);height=rng.uniform(.10,.32)
            box=np.r_[radius*np.cos(angle),radius*np.sin(angle),env.table_top+height/2,
                      rng.uniform(.015,.035,2),height/2]
        env.set_obstacle(box)
        if not env.valid(start) or not env.valid(goal):
            env.set_obstacle(None);continue
        direct=env.edge_valid(start,goal)
        if blocked and direct:
            env.set_obstacle(None);continue
        return start,goal,box,{'nearby':nearby,'direct_blocked':not direct}
    raise RuntimeError('Could not sample a task with valid endpoints and requested blockage')


def generate(output,episodes=1000,seed=42,blocked_fraction=.2,config=RRTConfig(),dt=.1,speed=1.,acceleration=3.):
    if isinstance(episodes,bool) or episodes<10:raise ValueError('At least ten episodes are required for three independent splits')
    if not 0<=blocked_fraction<=1:raise ValueError('Invalid blocked fraction')
    output=Path(output)
    if output.exists():raise ValueError('Choose a new dataset output; existing demonstrations are preserved')
    output.parent.mkdir(parents=True,exist_ok=True);temporary=output.with_suffix('.partial.h5')
    rng=np.random.default_rng(seed);began=perf_counter();failures=Counter();counts=Counter();rows=[]
    with ArmEnv() as env,h5py.File(temporary,'w') as data:
        metadata={'kind':'rrt_star_expert_demonstrations','seed':seed,'requested_episodes':episodes,
                  'environment':env.metadata(),'planner':asdict(config),'dt':dt,'speed_rad_s':speed,
                  'acceleration_rad_s2':acceleration,'blocked_fraction_requested':blocked_fraction,
                  'split':'episode-level stratified train/validation/test 80/10/10; never window-level'}
        data.attrs['metadata_json']=json.dumps(metadata)
        root=data.create_group('episodes')
        for attempt in range(episodes*10):
            if len(rows)==episodes:break
            want_blocked=counts['direct_blocked']<int(np.ceil((len(rows)+1)*blocked_fraction))
            try:start,goal,box,task=sample_task(env,rng,want_blocked)
            except RuntimeError:failures['sampling']+=1;continue
            path,diagnostic=plan(env,start,goal,int(rng.integers(2**31)),config)
            if path is None:failures[diagnostic['status']]+=1;continue
            path=smooth(env,path);q,times=trajectory(path,dt,speed,acceleration)
            # Retiming shifts the sampling grid. Recheck the actual frames and
            # connecting edges; coarse planner samples can miss grazing contact.
            invalid=[index for index,state in enumerate(q) if not env.valid(state)]
            timed_valid=not invalid and all(env.edge_valid(a,b) for a,b in zip(q[:-1],q[1:]))
            if not timed_valid:
                failures['timed_trajectory_rejected']+=1
                fixture=output.with_name(output.stem+'.rejected_timing.json')
                if not fixture.exists():
                    write_json(fixture,{'start':start.tolist(),'goal':goal.tolist(),'obstacle':box.tolist(),
                        'path':path.tolist(),'trajectory':q.tolist(),'invalid_frames':invalid,
                        'reason':'Retimed trajectory failed final frame/edge collision checks'})
                continue
            number=len(rows);group=root.create_group(f'{number:06d}')
            group.create_dataset('q_rad',data=q,compression='gzip',compression_opts=1)
            group.create_dataset('time_s',data=times)
            group.create_dataset('q_goal_rad',data=goal);group.create_dataset('obstacle_m',data=box)
            target=ARM.transform(goal)
            group.create_dataset('goal_xyz_m',data=target[:3,3]);group.create_dataset('goal_axis_base',data=target[:3,2])
            record={'episode':number,**task,'planner':diagnostic,'frames':len(q),'waypoints':len(path)}
            group.attrs['record_json']=json.dumps(record);rows.append(record)
            counts['direct_blocked']+=int(task['direct_blocked']);counts['nearby']+=int(task['nearby']);counts['frames']+=len(q)
            if number%25==0 or number==episodes-1:
                print(json.dumps({'output':str(output),'episodes':number+1,'requested':episodes,'blocked':counts['direct_blocked'],
                                  'failures':dict(failures),'seconds':perf_counter()-began}),flush=True)
        if len(rows)!=episodes:raise RuntimeError(f'Generated {len(rows)}/{episodes} successful demonstrations; partial file retained')
        strata={}
        for row in rows:strata.setdefault((row['direct_blocked'],row['nearby']),[]).append(row['episode'])
        split_counts=Counter()
        for indices in strata.values():
            rng.shuffle(indices);train_end=int(len(indices)*.8);validation_end=train_end+max(1,int(len(indices)*.1))
            for i,index in enumerate(indices):
                split='train' if i<train_end else 'validation' if i<validation_end else 'test'
                root[f'{index:06d}'].attrs['split']=split;split_counts[split]+=1
        metadata.update(completed_episodes=len(rows),counts=dict(counts),failures=dict(failures),split_counts=dict(split_counts),seconds=perf_counter()-began)
        data.attrs['metadata_json']=json.dumps(metadata);data.flush()
    temporary.replace(output);metadata['sha256']=file_sha256(output)
    write_json(output.with_suffix('.json'),metadata)
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ARTIFACTS/'demonstrations.h5')
    parser.add_argument('--episodes',type=int,default=1000);parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--blocked-fraction',type=float,default=.2);parser.add_argument('--iterations',type=int,default=800)
    parser.add_argument('--post-solution',type=int,default=40)
    args=parser.parse_args();print(json.dumps(generate(args.output,args.episodes,args.seed,args.blocked_fraction,
        RRTConfig(iterations=args.iterations,post_solution=args.post_solution)),allow_nan=False))


if __name__=='__main__':main()
