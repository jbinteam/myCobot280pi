"""Five-dimensional XYZ + tool-Z-direction task; local tool-Z spin is free.

Orientation inputs use R_ref = Ry(pitch) Rx(roll). The accepted orientations
are R_ref Rz(spin), for any spin. Thus the constraint is the directed unit
vector R[:,2], rather than Euler roll/pitch extracted from the current flange.
"""
import json
from time import perf_counter
import numpy as np
from .kinematics import ARM, rotation
from .solvers import capped, validate, ARTIFACTS

METHODS=[
 ('pinv','5D pseudoinverse','XYZ + flange Z direction; rotation around flange Z is free'),
 ('dls','5D damped least squares','Five-row task Jacobian; preserves free spin where possible'),
 ('null_classical','5D null-space posture','Five-row DLS + exact null-space centering policy'),
 ('bounded_dls','5D bounded DLS','Joint-limit-aware DLS with task-error backtracking'),
]
ORIENTATION_WEIGHT=.6


def bounded_velocity(j, error, q, damping, dt, speed):
    """Active-set box DLS: minimize ||J v - 3 e||² + damping² ||v||².

    Joint position bounds are constraints in the solve, rather than clipping
    an unconstrained command afterwards. The full vector speed cap is applied
    by the caller; scaling towards zero preserves feasibility.
    """
    h=j.T@j+damping*damping*np.eye(6)
    b=3*j.T@error
    lower=np.maximum((ARM.lower-q)/dt,-speed)
    upper=np.minimum((ARM.upper-q)/dt,speed)
    v=np.zeros(6);fixed=np.zeros(6,dtype=int)
    for _ in range(64):
        free=np.flatnonzero(fixed==0);active=np.flatnonzero(fixed!=0)
        if len(free):
            v[free]=np.linalg.solve(h[np.ix_(free,free)],b[free]-h[np.ix_(free,active)]@v[active])
        violation=np.maximum(lower-v,v-upper);violation[fixed!=0]=-1
        if violation.max()>1e-10:
            k=int(np.argmax(violation));fixed[k]=-1 if v[k]<lower[k] else 1
            v[k]=lower[k] if fixed[k]<0 else upper[k]
            continue
        gradient=h@v-b
        wrong=np.where(fixed==-1,-gradient,np.where(fixed==1,gradient,-1))
        if wrong.max()>1e-10:
            fixed[int(np.argmax(wrong))]=0
            continue
        return v
    # Bounded fallback; the caller's nonlinear error check still decides
    # whether this candidate can be used.
    return np.clip(v,lower,upper)


def axis_from_xy(orientation_xy_deg):
    if isinstance(orientation_xy_deg,(list,tuple)) and any(isinstance(v,bool) for v in orientation_xy_deg):
        raise ValueError('Orientation angles must be numbers, not booleans')
    xy=np.asarray(orientation_xy_deg,float)
    if xy.shape!=(2,) or not np.isfinite(xy).all() or np.any(np.abs(xy)>180):
        raise ValueError('Orientation requires two finite X/Y angles in degrees, each within -180 to 180')
    roll,pitch=np.deg2rad(xy)
    return rotation([0,1,0],pitch)@rotation([1,0,0],roll)@np.array([0.,0.,1.])


def unit_axis(axis):
    """Directed tool-Z in g_base. Normalize finite nonzero vectors robustly."""
    if isinstance(axis,(list,tuple)) and any(isinstance(v,(bool,np.bool_)) for v in axis):
        raise ValueError('Tool direction components must be numbers, not booleans')
    raw=np.asarray(axis)
    if raw.shape!=(3,) or raw.dtype.kind not in 'iuf' or not np.isfinite(raw).all():
        raise ValueError('Tool direction requires three finite numeric base-frame components')
    axis=np.asarray(axis,float)
    scale=float(np.max(np.abs(axis)))
    if scale==0:raise ValueError('Tool direction must be a nonzero vector')
    axis=axis/scale
    return axis/np.linalg.norm(axis)


def resolve_axis(orientation_xy_deg=None,target_axis=None):
    """Explicit base vector takes precedence; legacy reference tilts still work."""
    return unit_axis(target_axis) if target_axis is not None else axis_from_xy(orientation_xy_deg)


def xy_from_axis(axis):
    axis=unit_axis(axis)
    roll=np.arcsin(np.clip(-axis[1],-1,1))
    pitch=np.arctan2(axis[0],axis[2]) if np.hypot(axis[0],axis[2])>1e-12 else 0.
    return np.rad2deg([roll,pitch]).tolist()


def _cross3(a,b):
    # Both operands have already passed directed unit-vector validation.
    # Avoid general-dimensional NumPy axis dispatch for these tiny vectors.
    return np.array([a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]])


def axis_error(current,target):
    """Shortest pointing rotation, including a deterministic antipodal branch."""
    current=unit_axis(current);target=unit_axis(target)
    cross=_cross3(current,target);norm=np.linalg.norm(cross)
    angle=np.arctan2(norm,np.clip(current@target,-1.,1.))
    reference=np.eye(3)[np.argmin(np.abs(current))]
    b1=_cross3(current,reference);b1/=np.linalg.norm(b1)
    basis=np.stack([b1,_cross3(current,b1)])
    vector=(cross/norm*angle if norm>1e-10 else (b1*angle if current@target<0 else np.zeros(3)))
    return vector,basis,float(np.rad2deg(angle))


def task(q,target,target_axis):
    t,j6=ARM.spatial_jacobian(q)
    pointing=t[:3,2]
    error,basis,angle=axis_error(pointing,target_axis)
    j=np.vstack([j6[:3]/ARM.scale,ORIENTATION_WEIGHT*basis@j6[3:]])
    e=np.r_[(target-t[:3,3])/ARM.scale,ORIENTATION_WEIGHT*basis@error]
    return t,j,e,angle


def solve(q_start,target,orientation_xy_deg=None,method='dls',steps=240,speed=1.2,tolerance=1.,damping=.04,orientation_tolerance_deg=1.,target_axis=None,goal_constraint=None,cancel_check=None):
    if method not in [m[0] for m in METHODS]:
        raise ValueError('5D pose control supports classical solvers; XYZ neural networks require position-only mode')
    q,target=validate(q_start,target,'dls' if method=='bounded_dls' else method,steps,speed,tolerance,damping)
    if isinstance(orientation_tolerance_deg,bool) or not isinstance(orientation_tolerance_deg,(int,float)) or not np.isfinite(orientation_tolerance_deg) or not .1<=orientation_tolerance_deg<=10:
        raise ValueError('Orientation tolerance must be between 0.1 and 10 degrees')
    target_axis=resolve_axis(orientation_xy_deg,target_axis)
    if goal_constraint is not None and not callable(goal_constraint):
        raise ValueError('Goal constraint must be callable')
    if cancel_check is not None and not callable(cancel_check):
        raise ValueError('Cancellation check must be callable')
    dt=.05;frames=[];hits=0;leaks=[];times=[];backtracks=0;start=perf_counter()
    for iteration in range(int(steps)+1):
        if cancel_check is not None:cancel_check()
        t,j,e,angle=task(q,target,target_axis)
        error=float(np.linalg.norm(t[:3,3]-target)*1000)
        frames.append({'q':np.rad2deg(q).tolist(),'position':(t[:3,3]*1000).tolist(),
                       'tool_axis':t[:3,2].tolist(),'error_mm':error,'orientation_error_deg':angle,'time':iteration*dt})
        task_reached=error<=tolerance and angle<=orientation_tolerance_deg
        if (task_reached and (goal_constraint is None or goal_constraint(q))) or iteration==steps:break
        tick=perf_counter()
        if method=='bounded_dls':velocity=bounded_velocity(j,e,q,damping,dt,speed)
        elif method=='pinv':velocity=3*np.linalg.pinv(j,rcond=1e-5)@e
        else:velocity=3*j.T@np.linalg.solve(j@j.T+damping*damping*np.eye(5),e)
        if method=='null_classical':
            null=(np.eye(6)-np.linalg.pinv(j,rcond=1e-5)@j)@(-.6*(q-ARM.center)/ARM.radius)
            leaks.append(float(np.linalg.norm(j@null)))
            velocity+=.3*null
        proposed=q+dt*capped(velocity,speed);next_q=ARM.clip(proposed)
        if method=='bounded_dls':
            merit=float(e@e)
            for attempt in range(11):
                if cancel_check is not None:cancel_check()
                candidate_error=task(next_q,target,target_axis)[2]
                if float(candidate_error@candidate_error)<=merit+1e-14:break
                if attempt==10:
                    next_q=q.copy()
                    break
                backtracks+=1
                next_q=ARM.clip(q+(next_q-q)*.5)
        hits+=int(np.any(np.abs(next_q-proposed)>1e-9));times.append((perf_counter()-tick)*1000)
        if np.linalg.norm(next_q-q)<1e-9:break
        q=next_q
    last=frames[-1]
    reached=bool(last['error_mm']<=tolerance and last['orientation_error_deg']<=orientation_tolerance_deg
                 and (goal_constraint is None or goal_constraint(q)))
    travel=np.linalg.norm(np.diff(np.deg2rad([f['q'] for f in frames]),axis=0),axis=1).sum()
    return {'task':'pose5','method':method,'frames':frames,'target_mm':(target*1000).tolist(),
            'orientation_xy_deg':xy_from_axis(target_axis),'target_axis':target_axis.tolist(),
            'converged':reached,'final_error_mm':last['error_mm'],
            'final_orientation_error_deg':last['orientation_error_deg'],'iterations':len(frames)-1,
            'joint_travel_deg':float(np.rad2deg(travel)),'elapsed_ms':(perf_counter()-start)*1000,
            'mean_step_ms':float(np.mean(times)) if times else 0.,'limit_hits':hits,'backtracks':backtracks,
            'null_residual_max':max(leaks,default=0.),'tolerance_mm':tolerance,
            'orientation_tolerance_deg':orientation_tolerance_deg,'free_rotation':'local flange Z',
            'stop_reason':'target_reached' if reached else ('iteration_limit' if len(frames)-1>=steps else 'stalled')}


def cases(seed=9001,count=24):
    # Identical position cases/seeds to the XYZ benchmark, with an additional
    # reachable direction from the same FK goal. No training data is reused.
    rng=np.random.default_rng(seed);out=[]
    for i in range(count):
        q0=ARM.center+ARM.radius*rng.uniform(-.8,.8,6)
        if i%4==0:q0=np.zeros(6)
        if i%4<2:goal=ARM.clip(q0+rng.normal(0,.25,6));kind='nearby'
        else:goal=ARM.center+ARM.radius*rng.uniform(-.8,.8,6);kind='global'
        t=ARM.transform(goal)
        out.append({'name':f'{kind}-{i+1:02d}','start_deg':np.rad2deg(q0).tolist(),
                    'target_mm':(t[:3,3]*1000).tolist(),'orientation_xy_deg':xy_from_axis(t[:3,2]),'kind':kind})
    return out


def benchmark():
    fixtures=cases();rows=[]
    for method,name,_ in METHODS:
        results=[solve(np.deg2rad(c['start_deg']),np.array(c['target_mm'])/1000,c['orientation_xy_deg'],method) for c in fixtures]
        row={'method':method,'name':name,'success':sum(r['converged'] for r in results),'total':len(results),
             'mean_error_mm':float(np.mean([r['final_error_mm'] for r in results])),
             'mean_orientation_error_deg':float(np.mean([r['final_orientation_error_deg'] for r in results])),
             'mean_travel_deg':float(np.mean([r['joint_travel_deg'] for r in results])),
             'mean_elapsed_ms':float(np.mean([r['elapsed_ms'] for r in results])),
             'cases':[{k:r[k] for k in ('converged','final_error_mm','final_orientation_error_deg','iterations','joint_travel_deg')} for r in results]}
        rows.append(row);print(f'{name}: {row["success"]}/24; XYZ {row["mean_error_mm"]:.2f} mm; direction {row["mean_orientation_error_deg"]:.2f} deg',flush=True)
    result={'task':'pose5','seed':9001,'cases':fixtures,'rows':rows,'steps':240,'dt':.05,'speed_rad_s':1.2,
            'tolerance_mm':1.,'orientation_tolerance_deg':1.,'orientation_weight':ORIENTATION_WEIGHT,
            'urdf_sha256':ARM.fingerprint,'limits_rad':{'min':ARM.lower.tolist(),'max':ARM.upper.tolist()},'joint_limit_profile':ARM.limit_profile,
            'limitations':'24 reachable poses within the recorded operating limits; no collision checking or hardware validation'}
    (ARTIFACTS/'benchmark_pose5.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':benchmark()
