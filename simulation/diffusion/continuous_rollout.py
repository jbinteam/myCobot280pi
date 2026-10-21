"""Full-horizon receding previews using the same checked C2 tracking controller."""
from collections import Counter
from time import perf_counter
import numpy as np
from ..kinematics import ARM
from ..continuous import prepare,sample_motion
from .policy import checked_chunk,projected_chunk


def rollout(env,policy,start,goal,target,axis,box,config,deadline):
    from .service import errors,condition,cartesian_score,static_capture_prefix
    from .cartesian_conditioning import supports
    cartesian=supports(policy)
    q=start.copy();v=np.zeros(6);a=np.zeros(6);points=[q.copy()];rejects=Counter()
    decisions=proposals=capture=stalls=0;status='decision_limit';curve=None;elapsed=0.
    for _ in range(config['max_decisions']):
        error=errors(q,target,axis)
        reached=error['error_mm']<=1 and error['orientation_error_deg']<=1
        stopped=(np.linalg.norm(v)<1e-10 and np.linalg.norm(a)<1e-10) if cartesian else np.linalg.norm(v)<.05
        if reached and stopped:
            status='success';break
        if perf_counter()>=deadline:status='time_limit';break
        candidates=[]
        if not cartesian and goal is not None and np.linalg.norm(goal-q)<=.1 and env.edge_valid(q,goal):
            candidates.append(((0,-1.),np.repeat(goal[None],16,axis=0),True,None))
        # A reached Cartesian pose needs a checked C2 stop, not a new goal.
        if cartesian and reached:
            candidates.append(((-1,0.),q[None].copy(),False,None))
            chunks=[]
        else:
            chunks=policy.sample(condition(policy,q,goal,target,axis,box,v,a)[None],config['samples'],config['inference_steps'])[0]
        proposals+=len(chunks);decisions+=1
        for raw in chunks:
            if not cartesian:
                # Rank the identical speed-projected geometry before expensive
                # validation. Only candidates considered for selection need
                # full raw/projected checks; inference/RNG ordering is unchanged.
                chunk=projected_chunk(q,raw)
                if chunk is None:rejects['malformed']+=1;continue
                score=np.linalg.norm((chunk[-1]-goal)/ARM.radius)
                candidates.append(((1,score),chunk,False,raw))
                continue
            chunk,reason=checked_chunk(env,q,raw)
            if chunk is None:rejects[reason]+=1;continue
            captured=False
            if cartesian:chunk,captured=static_capture_prefix(chunk,target,axis)
            score=(cartesian_score(chunk[-1],target,axis)+.25*cartesian_score(chunk[min(1,len(chunk)-1)],target,axis)) if cartesian else np.linalg.norm((chunk[-1]-goal)/ARM.radius)
            candidates.append(((0 if captured else 1,score),chunk,False,None))
        accepted=None
        for _,chunk,endpoint,raw in sorted(candidates,key=lambda item:item[0]):
            if raw is not None:
                chunk,reason=checked_chunk(env,q,raw)
                if chunk is None:rejects[reason]+=1;continue
            try:_,_,new_curve=prepare(env,np.vstack((q,chunk)),v,a)
            except ValueError:rejects['continuous_join']+=1;continue
            accepted=new_curve
            if endpoint:capture+=1
            break
        if accepted is not None:curve=accepted;elapsed=0.;stalls=0
        else:stalls+=1
        # Keep an existing checked braking tail when a replacement is rejected.
        if curve is not None and curve['duration_s']>elapsed+1e-10:
            end=min(elapsed+.4,curve['duration_s'])
            times=np.r_[np.arange(elapsed+.02,end-1e-9,.02),end]
            for time in times:
                next_q,next_v,next_a=sample_motion(curve,time)
                dt=time-elapsed
                if not env.execute_step(q,next_q,dt,1.):
                    return points,'execution_gate_rejected',decisions,proposals,dict(rejects),capture
                q=next_q;v=next_v;a=next_a;points.append(q.copy());elapsed=time
        else:v=np.zeros(6);a=np.zeros(6)
        if stalls>=20:status='no_safe_chunk';break
    error=errors(q,target,axis)
    stopped=(np.linalg.norm(v)<1e-10 and np.linalg.norm(a)<1e-10) if cartesian else np.linalg.norm(v)<.05
    if error['error_mm']<=1 and error['orientation_error_deg']<=1 and stopped:status='success'
    return points,status,decisions,proposals,dict(rejects),capture
