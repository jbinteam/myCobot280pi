"""Same held-out tasks for v1/v2: fixed goals, absence, moving goals, retargets.

Pure neural suites deliberately omit local IK to expose policy changes. The
motion suite plays checked C2 curves with nonzero boundary motion. CPU latency
is measured separately; batched GPU rollout timing is not browser latency.
"""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
from time import perf_counter
import h5py
import numpy as np
import torch
from ..kinematics import ARM
from ..continuous import prepare,evaluate
from .common import observation,file_sha256,write_json
from .env import ArmEnv
from .generate_motion import sample_curve
from .motion_conditioning import encode,bounded_vector
from .policy import DiffusionPolicy,checked_chunk
from .eval import errors


def tasks(source,augmentation,seed=20261008):
    rng=np.random.default_rng(seed);fixed=[];moving=[]
    with h5py.File(source,'r') as data:
        strata={True:[],False:[]}
        for name,group in data['episodes'].items():
            if group.attrs['split']!='test':continue
            strata[json.loads(group.attrs['record_json'])['direct_blocked']].append(name)
        for blocked in (True,False):
            for name in rng.choice(strata[blocked],16,replace=False):
                group=data['episodes'][name]
                fixed.append({'id':'block_'+name,'kind':'block','blocked':blocked,'q':group['q_rad'][:],
                              'goal':group['q_goal_rad'][:],'box':group['obstacle_m'][:]})
    with h5py.File(augmentation,'r') as data:
        by_kind={}
        for name,group in data['episodes'].items():
            if group.attrs['split']=='test':by_kind.setdefault(json.loads(group.attrs['record_json'])['kind'],[]).append(name)
        for kind,count in [('static_absent',32),('moving_absent',8),('moving_block',8),('retarget_absent',4),('retarget_block',4)]:
            for name in rng.choice(by_kind[kind],count,replace=False):
                group=data['episodes'][name];row={'id':kind+'_'+name,'kind':kind,'q':group['q_rad'][:],
                    'box':group['obstacle_m'][:] if group.attrs['obstacle_present'] else None,
                    'parent_episode':group.attrs['parent_episode']}
                if kind=='static_absent':row['goal']=group['goal_rad'][-1];fixed.append(row)
                else:
                    row.update(goals=group['goal_rad'][:],times=group['time_s'][:],
                               velocities=group['velocity_rad_s'][:],accelerations=group['acceleration_rad_s2'][:],
                               target_velocities=group['target_velocity_m_s'][:]);moving.append(row)
    return fixed,moving


def fixed_suite(policy,tasks,max_decisions=200):
    envs=[];rows=[];velocities=[np.zeros(6) for _ in tasks];accelerations=[np.zeros(6) for _ in tasks];began=perf_counter()
    try:
        for task in tasks:
            env=ArmEnv();env.set_obstacle(task['box']);env.reset(task['q'][0]);envs.append(env)
            if not env.path_valid(task['q']):raise ValueError('Held-out expert route failed independent collision replay')
            rows.append({'id':task['id'],'kind':task['kind'],'status':'running','success':False,'decisions':0,
                         'rejections':Counter(),'proposals':0,'stalls':0,'executed_steps':0,'execution_collisions':0})
        active=list(range(len(tasks)))
        for decision in range(max_decisions):
            remaining=[]
            for i in active:
                q=envs[i].current();goal=tasks[i]['goal'];position,angle=errors(q,goal)
                if position<=1 and angle<=1:rows[i].update(success=True,status='success');continue
                if np.linalg.norm(q-goal)<=.1 and envs[i].execute_step(q,goal,.1,1.):
                    rows[i].update(success=True,status='success');continue
                remaining.append(i)
            active=remaining
            if not active:break
            obs=np.stack([encode(policy,observation(envs[i].current(),tasks[i]['goal'],tasks[i]['box']),velocities[i],accelerations[i]) for i in active])
            chunks=policy.sample(obs,4,32)
            for j,i in enumerate(active):
                row=rows[i];q=envs[i].current();goal=tasks[i]['goal'];candidates=[];row['decisions']+=1;row['proposals']+=4
                for raw in chunks[j]:
                    chunk,reason=checked_chunk(envs[i],q,raw)
                    if chunk is None:row['rejections'][reason]+=1;continue
                    score=np.linalg.norm((chunk[1]-goal)/ARM.radius)+.25*np.linalg.norm((chunk[-1]-goal)/ARM.radius)
                    candidates.append((score,chunk))
                if not candidates:
                    velocities[i]=np.zeros(6);accelerations[i]=np.zeros(6)
                    row['stalls']+=1
                    if row['stalls']>=20:row['status']='no_safe_chunk'
                    continue
                row['stalls']=0
                for target in min(candidates,key=lambda item:item[0])[1][:2]:
                    if not envs[i].execute_step(q,target,.1,1.):raise RuntimeError('Checked fixed-goal execution failed')
                    if not envs[i].valid(target):raise RuntimeError('Executed collision')
                    next_velocity=(target-q)/.1
                    accelerations[i]=bounded_vector((next_velocity-velocities[i])/.1,6,3.,'virtual reference acceleration',clip=True)
                    velocities[i]=next_velocity;row['executed_steps']+=1;q=target
            active=[i for i in active if rows[i]['status']=='running']
            if decision%20==0:print(f'{policy.path.parent.name} fixed {decision}: {sum(r["success"] for r in rows)}/{len(rows)}',flush=True)
        for row,env,task in zip(rows,envs,tasks):
            position,angle=errors(env.current(),task['goal']);row.update(final_error_mm=position,final_angle_deg=angle)
            if row['status']=='running':row['status']='decision_limit'
            row['rejections']=dict(row['rejections'])
        summary={'episodes':len(rows),'successes':sum(r['success'] for r in rows),'execution_collisions':0,
                 'proposals':sum(r['proposals'] for r in rows),'raw_collision_rejections':sum(r['rejections'].get('raw_collision',0) for r in rows),
                 'blocked_successes':sum(r['success'] for r,t in zip(rows,tasks) if t.get('blocked')),
                 'by_kind':{kind:{'episodes':sum(r['kind']==kind for r in rows),'successes':sum(r['success'] and r['kind']==kind for r in rows)} for kind in ('block','static_absent')},
                 'seconds':perf_counter()-began}
        return {'summary':summary,'rows':rows}
    finally:
        for env in envs:env.close()


def motion_suite(policy,tasks,max_decisions=120,dt=.4):
    envs=[];rows=[];qs=[];velocities=[];accelerations=[];curves=[];curve_times=[];began=perf_counter()
    try:
        for task in tasks:
            env=ArmEnv();env.set_obstacle(task['box']);envs.append(env);q=task['q'][0].copy()
            if not env.path_valid(task['q']):raise ValueError('Motion expert collision replay failed')
            qs.append(q);velocities.append(task['velocities'][0].copy());accelerations.append(task['accelerations'][0].copy())
            # Initial continuation is only a checked brake along incoming
            # motion, never the held-out expert route or an endpoint planner.
            brake=ARM.clip(q+velocities[-1]*.25)
            try:_,_,prior=prepare(env,np.vstack((q,brake)),velocities[-1],accelerations[-1])
            except ValueError:prior=None
            curves.append(prior);curve_times.append(0.)
            rows.append({'id':task['id'],'kind':task['kind'],'status':'running','success':False,'decisions':0,'stalls':0,
                         'tracking_error_mm':[],'rejections':Counter(),'proposals':0,'join_rejections':0,
                         'max_velocity_rad_s':0.,'max_acceleration_rad_s2':0.,'max_jerk_rad_s3':0.,
                         'max_boundary_velocity_error':0.,'max_boundary_acceleration_error':0.,'execution_collisions':0})
            if prior is None:
                rows[-1]['status']='initial_continuation_rejected'
                rows[-1]['tracking_error_mm']=[errors(q,task['goals'][0])[0]]
        active=[i for i,row in enumerate(rows) if row['status']=='running']
        def advance(i,metadata,start_time):
            row=rows[i];duration=min(dt,max(0.,metadata['duration_s']-start_time))
            if duration<1e-10:
                velocities[i]=np.zeros(6);accelerations[i]=np.zeros(6);return
            time=np.linspace(start_time,start_time+duration,max(2,int(np.ceil(duration/.01))+1))
            states,v,a=sample_curve(metadata,time)
            if not envs[i].path_valid(states):raise RuntimeError('Continuous execution collision')
            row['max_boundary_velocity_error']=max(row['max_boundary_velocity_error'],float(np.linalg.norm(v[0]-velocities[i])))
            row['max_boundary_acceleration_error']=max(row['max_boundary_acceleration_error'],float(np.linalg.norm(a[0]-accelerations[i])))
            row['max_velocity_rad_s']=max(row['max_velocity_rad_s'],float(np.linalg.norm(v,axis=1).max()))
            row['max_acceleration_rad_s2']=max(row['max_acceleration_rad_s2'],float(np.linalg.norm(a,axis=1).max()))
            row['max_jerk_rad_s3']=max(row['max_jerk_rad_s3'],float(np.linalg.norm(np.diff(a,axis=0)/np.diff(time)[:,None],axis=1).max()))
            qs[i]=states[-1];velocities[i]=v[-1];accelerations[i]=a[-1];curve_times[i]=start_time+duration
        for decision in range(max_decisions):
            current_goals={};target_velocities={}
            for i in active:
                task=tasks[i];index=min(int(round(decision*dt/.1)),len(task['goals'])-1)
                goal=task['goals'][index];position,angle=errors(qs[i],goal)
                rows[i]['tracking_error_mm'].append(position)
                if np.array_equal(goal,task['goals'][-1]) and position<=1 and angle<=1 and np.linalg.norm(velocities[i])<.05:
                    rows[i].update(success=True,status='success');continue
                current_goals[i]=goal;target_velocities[i]=task['target_velocities'][index]
            active=[i for i in active if rows[i]['status']=='running']
            if not active:break
            obs=np.stack([encode(policy,observation(qs[i],current_goals[i],tasks[i]['box']),velocities[i],accelerations[i],target_velocities[i]) for i in active])
            chunks=policy.sample(obs,4,32)
            for j,i in enumerate(active):
                row=rows[i];q=qs[i];goal=current_goals[i];candidates=[];row['decisions']+=1;row['proposals']+=4
                # A checked final capture has the same contract for both models.
                if np.linalg.norm(goal-q)<=.1 and envs[i].edge_valid(q,goal):candidates.append((-1.,np.repeat(goal[None],16,axis=0)))
                for raw in chunks[j]:
                    chunk,reason=checked_chunk(envs[i],q,raw)
                    if chunk is None:row['rejections'][reason]+=1;continue
                    score=np.linalg.norm((chunk[-1]-goal)/ARM.radius);candidates.append((score,chunk))
                if not candidates:
                    row['stalls']+=1;advance(i,curves[i],curve_times[i]);continue
                metadata=None
                for _,selected in sorted(candidates,key=lambda item:item[0]):
                    try:_,_,metadata=prepare(envs[i],np.vstack((q,selected)),velocities[i],accelerations[i]);break
                    except ValueError:row['join_rejections']+=1
                if metadata is None:
                    row['stalls']+=1;advance(i,curves[i],curve_times[i]);continue
                curves[i]=metadata;advance(i,metadata,0.)
                if np.linalg.norm(qs[i]-q)<.004:row['stalls']+=1
            if decision%20==0:print(f'{policy.path.parent.name} motion {decision}: {sum(r["success"] for r in rows)}/{len(rows)}',flush=True)
        for row,q,task in zip(rows,qs,tasks):
            row['final_error_mm'],row['final_angle_deg']=errors(q,task['goals'][-1]);row['rejections']=dict(row['rejections'])
            row['tracking_rmse_mm']=float(np.sqrt(np.mean(np.square(row['tracking_error_mm']))))
            if row['status']=='running':row['status']='decision_limit'
        summary={'episodes':len(rows),'successes':sum(r['success'] for r in rows),'execution_collisions':0,
                 'mean_tracking_rmse_mm':float(np.mean([r['tracking_rmse_mm'] for r in rows])),
                 'stalls':sum(r['stalls'] for r in rows),'join_rejections':sum(r['join_rejections'] for r in rows),
                 'max_velocity_rad_s':max(r['max_velocity_rad_s'] for r in rows),'max_acceleration_rad_s2':max(r['max_acceleration_rad_s2'] for r in rows),
                 'max_jerk_rad_s3':max(r['max_jerk_rad_s3'] for r in rows),
                 'max_boundary_velocity_error':max(r['max_boundary_velocity_error'] for r in rows),
                 'max_boundary_acceleration_error':max(r['max_boundary_acceleration_error'] for r in rows),
                 'proposals':sum(r['proposals'] for r in rows),'raw_collision_rejections':sum(r['rejections'].get('raw_collision',0) for r in rows),
                 'seconds':perf_counter()-began}
        return {'summary':summary,'rows':rows}
    finally:
        for env in envs:env.close()


def cpu_latency(checkpoint,task):
    policy=DiffusionPolicy(checkpoint,'cpu',42);obs=encode(policy,observation(task['q'][0],task['goal'],task['box']))[None]
    policy.sample(obs,4,32);timings=[]
    for _ in range(20):
        began=perf_counter();policy.sample(obs,4,32);timings.append((perf_counter()-began)*1000)
    return {'samples':20,'median_ms':float(np.median(timings)),'p95_ms':float(np.percentile(timings,95)),
            'scope':'Single observation, four candidates, 32 denoising steps, CPU, four threads; inference only'}


def compare(source,augmentation,baseline,candidate,output,baseline_only=False,candidate_only=False):
    fixed,moving=tasks(source,augmentation);results={}
    models=[('candidate',candidate)] if candidate_only else [('baseline',baseline),('candidate',candidate)]
    for name,checkpoint in models:
        cache_path=Path(output).with_name(name+'_comparison.json')
        if cache_path.exists() and not baseline_only and not candidate_only:
            saved=json.loads(cache_path.read_text())
            if (saved.get('checkpoint_sha256')==file_sha256(checkpoint)
                    and saved.get('source_sha256')==file_sha256(source) and saved.get('augmentation_sha256')==file_sha256(augmentation)
                    and [r['id'] for r in saved['fixed']['rows']]==[r['id'] for r in fixed]
                    and [r['id'] for r in saved['motion']['rows']]==[r['id'] for r in moving]):
                results[name]=saved;results[name]['cpu_inference_latency']=cpu_latency(checkpoint,fixed[0]);continue
            raise ValueError('Cached comparison provenance/tasks differ')
        policy=DiffusionPolicy(checkpoint,'cuda',20261008)
        results[name]={'checkpoint_sha256':policy.sha256,'observation_dim':policy.observation_dim,
                       'source_sha256':file_sha256(source),'augmentation_sha256':file_sha256(augmentation),
                       'fixed':fixed_suite(policy,fixed),'motion':motion_suite(policy,moving)}
        del policy;gc.collect();torch.cuda.empty_cache()
        results[name]['cpu_inference_latency']=cpu_latency(checkpoint,fixed[0])
        write_json(cache_path,results[name])
        if baseline_only or candidate_only:return results[name]
    old=results['baseline'];new=results['candidate'];checks={
        'fixed_success_not_lower':new['fixed']['summary']['successes']>=old['fixed']['summary']['successes'],
        'no_block_success_improves':new['fixed']['summary']['by_kind']['static_absent']['successes']>old['fixed']['summary']['by_kind']['static_absent']['successes'],
        'motion_success_not_lower':new['motion']['summary']['successes']>=old['motion']['summary']['successes'],
        'tracking_error_not_higher':new['motion']['summary']['mean_tracking_rmse_mm']<=old['motion']['summary']['mean_tracking_rmse_mm'],
        'executed_collision_free':new['fixed']['summary']['execution_collisions']==0 and new['motion']['summary']['execution_collisions']==0,
        'continuous_motion_limits':new['motion']['summary']['max_velocity_rad_s']<=1+1e-6 and new['motion']['summary']['max_acceleration_rad_s2']<=3+1e-6,
        'continuous_boundaries_match':new['motion']['summary']['max_boundary_velocity_error']<1e-6 and new['motion']['summary']['max_boundary_acceleration_error']<1e-6}
    report={'kind':'paired_held_out_policy_comparison','source_sha256':file_sha256(source),'augmentation_sha256':file_sha256(augmentation),
            'seed':20261008,'configuration':{'samples':4,'inference_steps':32,'fixed_max_decisions':200,'motion_max_decisions':120,'motion_replan_period_s':.4},
            'results':results,'deployment_checks':checks,'recommended_for_deployment':all(checks.values()),
            'limitations':'One training seed and 88 paired scenarios. Fixed suite conditions on virtual reference velocity and clipped acceleration, bounds speed only before final UI smoothing. Motion suite evaluates checked C2 execution, with zero compute delay; its stalls are safe rejected/no-progress decisions, not measured browser pauses. Moving targets follow expert routes. No hardware tests.'}
    write_json(output,report);print(json.dumps({'checks':checks,'summaries':{name:{kind:row[kind]['summary'] for kind in ('fixed','motion')} for name,row in results.items()}}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('simulation/artifacts/diffusion_100k_v1/demonstrations.h5'))
    parser.add_argument('--augmentation',type=Path,default=Path('simulation/artifacts/diffusion_motion_v2/augmentation.h5'))
    parser.add_argument('--baseline',type=Path,default=Path('simulation/artifacts/diffusion_v1/model/diffusion_policy_6dof.pth'))
    parser.add_argument('--candidate',type=Path,default=Path('simulation/artifacts/diffusion_motion_v2/model/diffusion_policy_6dof.pth'))
    parser.add_argument('--output',type=Path,default=Path('output/diffusion/motion_v2_comparison.json'))
    parser.add_argument('--baseline-only',action='store_true')
    parser.add_argument('--candidate-only',action='store_true')
    args=parser.parse_args();compare(args.source,args.augmentation,args.baseline,args.candidate,args.output,args.baseline_only,args.candidate_only)
