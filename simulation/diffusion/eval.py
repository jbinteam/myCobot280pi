"""Held-out receding-horizon PyBullet evaluation; no hidden RRT/IK fallback."""
import argparse
from collections import Counter
from pathlib import Path
from time import perf_counter, sleep
import numpy as np
from ..kinematics import ARM
from ..pose5 import axis_error
from .common import ARTIFACTS, observation, write_json, file_sha256
from .dataset import read_episodes
from .env import ArmEnv
from .policy import DiffusionPolicy, checked_chunk


def errors(q,goal):
    actual=ARM.transform(q);target=ARM.transform(goal)
    return float(np.linalg.norm(actual[:3,3]-target[:3,3])*1000),axis_error(actual[:3,2],target[:3,2])[2]


def evaluate(dataset,checkpoint,output,episodes=50,seed=20261004,device='auto',samples=4,
             inference_steps=32,max_decisions=200,gui=False):
    """Batch denoising across isolated scenes; execute at most two checked steps.

    Endpoint capture is explicitly logged: if the exact goal is within one
    speed-bounded step, check and execute that final connection. RRT* supplies
    an expert baseline only; it never repairs a policy rollout.
    """
    if episodes<1 or max_decisions<1:raise ValueError('Positive episodes and decision limit required')
    if gui and episodes!=1:raise ValueError('GUI evaluation displays one episode; use --episodes 1')
    metadata,data=read_episodes(dataset,'test')
    if episodes>len(data):raise ValueError('Not enough independent test episodes')
    selected=np.random.default_rng(seed).choice(len(data),episodes,replace=False)
    data=[data[i] for i in selected];policy=DiffusionPolicy(checkpoint,device,seed)
    if policy.training['dataset_sha256']!=file_sha256(dataset):raise ValueError('Evaluation dataset differs from training provenance')
    dt=metadata['dt'];speed=metadata['speed_rad_s'];began=perf_counter();envs=[];rows=[]
    try:
        for episode in data:
            env=ArmEnv(gui=gui);envs.append(env);env.set_obstacle(episode['obstacle']);env.reset(episode['q'][0])
            if env.metadata()!=metadata['environment']:raise ValueError('Scene/checker metadata changed since demonstration generation')
            expert_valid=env.path_valid(episode['q'])
            expert_speed=bool(np.all(np.linalg.norm(np.diff(episode['q'],axis=0)/np.diff(episode['times'])[:,None],axis=1)<=speed+1e-8))
            if not expert_valid or not expert_speed:raise ValueError('Expert baseline failed current scene checks')
            rows.append({'episode':episode['id'],'direct_blocked':episode['record']['direct_blocked'],
                'nearby':episode['record']['nearby'],'goal_rad':episode['goal'].tolist(),'obstacle_m':episode['obstacle'].tolist(),
                'expert_success':True,'status':'running','success':False,'decisions':0,'executed_steps':0,
                'endpoint_capture_steps':0,'proposed_chunks':0,'rejections':{},'rejected_decisions':0,
                'execution_collision':False,'max_velocity_rad_s':0.,'max_acceleration_rad_s2':0.,
                'trajectory_rad':[episode['q'][0].tolist()]})
        active=list(range(episodes));velocities=[np.zeros(6) for _ in data];stalls=[0]*episodes
        for decision in range(max_decisions):
            remaining=[]
            for i in active:
                env=envs[i];row=rows[i];q=env.current();goal=data[i]['goal']
                position,angle=errors(q,goal)
                if position<=1. and angle<=1.:
                    row.update(status='success',success=True);continue
                # An exact, fully checked final reference step, not a planner fallback.
                if np.linalg.norm(goal-q)<=speed*dt and env.execute_step(q,goal,dt,speed):
                    velocity=(goal-q)/dt
                    row['max_velocity_rad_s']=max(row['max_velocity_rad_s'],float(np.linalg.norm(velocity)))
                    row['max_acceleration_rad_s2']=max(row['max_acceleration_rad_s2'],float(np.linalg.norm(velocity-velocities[i])/dt))
                    row['endpoint_capture_steps']+=1;row['executed_steps']+=1;row['trajectory_rad'].append(goal.tolist())
                    row.update(status='success',success=True);continue
                remaining.append(i)
            active=remaining
            if not active:break
            observations=np.stack([observation(envs[i].current(),data[i]['goal'],data[i]['obstacle']) for i in active])
            chunks=policy.sample(observations,samples,inference_steps)
            for n,i in enumerate(active):
                env=envs[i];row=rows[i];current=env.current();goal=data[i]['goal'];candidates=[]
                rejects=Counter(row['rejections']);row['decisions']+=1;row['proposed_chunks']+=samples
                for raw in chunks[n]:
                    chunk,status=checked_chunk(env,current,raw,dt,speed)
                    if chunk is None:rejects[status]+=1;continue
                    score=np.linalg.norm((chunk[1]-goal)/ARM.radius)+.25*np.linalg.norm((chunk[-1]-goal)/ARM.radius)
                    candidates.append((score,chunk))
                row['rejections']=dict(rejects)
                if not candidates:
                    row['rejected_decisions']+=1;stalls[i]+=1
                    if stalls[i]>=20:row['status']='no_safe_chunk'
                    continue
                stalls[i]=0;chunk=min(candidates,key=lambda item:item[0])[1]
                for target in chunk[:2]:
                    if not env.execute_step(current,target,dt,speed):
                        row['status']='execution_gate_rejected';break
                    actual=env.current()
                    if not np.allclose(actual,target,rtol=0,atol=1e-12):raise RuntimeError('Bullet reference execution differs from checked command')
                    velocity=(actual-current)/dt
                    row['max_velocity_rad_s']=max(row['max_velocity_rad_s'],float(np.linalg.norm(velocity)))
                    row['max_acceleration_rad_s2']=max(row['max_acceleration_rad_s2'],float(np.linalg.norm(velocity-velocities[i])/dt))
                    velocities[i]=velocity;current=actual
                    row['trajectory_rad'].append(current.tolist());row['executed_steps']+=1
                    if not env.valid(current):row['execution_collision']=True;row['status']='execution_collision';break
                    if gui:sleep(dt)
            active=[i for i in active if rows[i]['status']=='running']
            if decision%10==0:
                print(f'RHC {decision+1}/{max_decisions}: {sum(r["success"] for r in rows)}/{episodes} reached, {len(active)} active',flush=True)
        for row,env,episode in zip(rows,envs,data):
            row['final_position_error_mm'],row['final_direction_error_deg']=errors(env.current(),episode['goal'])
            if row['status']=='running':
                success=row['final_position_error_mm']<=1. and row['final_direction_error_deg']<=1.
                row.update(success=success,status='success' if success else 'decision_limit')
        proposals=sum(r['proposed_chunks'] for r in rows)
        summary={'episodes':episodes,'successes':sum(r['success'] for r in rows),
                 'success_rate':sum(r['success'] for r in rows)/episodes,
                 'execution_collisions':sum(r['execution_collision'] for r in rows),
                 'execution_collision_rate':sum(r['execution_collision'] for r in rows)/episodes,
                 'episodes_with_rejected_raw_collision':sum(bool(r['rejections'].get('raw_collision')) for r in rows),
                 'proposed_chunks':proposals,'raw_collision_rejections':sum(r['rejections'].get('raw_collision',0) for r in rows),
                 'raw_collision_rejection_fraction':sum(r['rejections'].get('raw_collision',0) for r in rows)/max(1,proposals),
                 'blocked_tasks':sum(r['direct_blocked'] for r in rows),
                 'blocked_successes':sum(r['success'] and r['direct_blocked'] for r in rows),
                 'expert_successes':sum(r['expert_success'] for r in rows),
                 'endpoint_capture_steps':sum(r['endpoint_capture_steps'] for r in rows),
                 'mean_executed_steps':float(np.mean([r['executed_steps'] for r in rows])),
                 'max_velocity_rad_s':max(r['max_velocity_rad_s'] for r in rows),
                 'max_acceleration_rad_s2':max(r['max_acceleration_rad_s2'] for r in rows),'seconds':perf_counter()-began}
        report={'kind':'held_out_diffusion_rhc','checkpoint':str(checkpoint),'checkpoint_sha256':policy.sha256,
                'dataset':str(dataset),'dataset_sha256':file_sha256(dataset),'seed':seed,
                'configuration':{'inference_steps':inference_steps,'samples':samples,'max_decisions':max_decisions,
                                 'horizon':16,'execute_steps':2,'dt':dt,'speed_rad_s':speed,'tolerance_mm':1.,'direction_tolerance_deg':1.,
                                 'planner_fallback':False,'endpoint_capture':'exact goal within one speed-bounded checked step'},
                'summary':summary,'rows':rows,
                'limitations':'Kinematic PyBullet reference execution. Safety rejection contributes to zero executed collisions; raw unsafe proposals are counted separately. Policy acceleration is measured, not bounded. One training seed and one table/box family; no physical execution.'}
        write_json(output,report);print(summary,flush=True);return report
    finally:
        for env in envs:env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=ARTIFACTS/'demonstrations.h5')
    parser.add_argument('--checkpoint',type=Path,default=ARTIFACTS/'model/diffusion_policy_6dof.pth')
    parser.add_argument('--output',type=Path,default=Path('output/diffusion/evaluation.json'))
    parser.add_argument('--episodes',type=int,default=50);parser.add_argument('--seed',type=int,default=20261004)
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    parser.add_argument('--samples',type=int,default=4);parser.add_argument('--inference-steps',type=int,default=32)
    parser.add_argument('--max-decisions',type=int,default=200);parser.add_argument('--gui',action='store_true')
    args=parser.parse_args();evaluate(args.dataset,args.checkpoint,args.output,args.episodes,args.seed,args.device,
        args.samples,args.inference_steps,args.max_decisions,args.gui)


if __name__=='__main__':main()
