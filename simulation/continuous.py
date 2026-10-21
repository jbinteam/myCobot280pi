"""Checked C2 simulation trajectories that can start with nonzero motion."""
from math import comb
import numpy as np
from .kinematics import ARM
from .tracking import simplify_path
from .motion_limits import curve_envelope, scale_needed, profile
from . import trajectory_math

DT = .02
DURATION_LIMIT = 12.
METHOD = 'collision_checked_bezier_c2'


def initial_motion(body, speed):
    enabled = body.get('continuous_motion', False)
    if not isinstance(enabled, bool):raise ValueError('continuous_motion must be a boolean')
    values = []
    for name, limit in [('start_velocity_rad_s', speed), ('start_acceleration_rad_s2', 3.)]:
        value = body.get(name, [0.]*6)
        if (not isinstance(value, list) or len(value)!=6 or any(
                isinstance(v,bool) or not isinstance(v,(int,float)) for v in value)):
            raise ValueError(f'{name} must contain six finite numbers')
        value = np.asarray(value,float)
        if not np.isfinite(value).all() or np.linalg.norm(value)>limit+1e-8:
            raise ValueError(f'{name} exceeds the vector motion limit')
        if not enabled and np.linalg.norm(value)>1e-12:
            raise ValueError('Nonzero starting motion requires continuous_motion')
        values.append(value)
    return enabled, *values


def evaluate(controls, u):
    controls = np.asarray(controls); u = np.asarray(u)
    degree = len(controls)-1
    weights = np.stack([comb(degree,i)*u**i*(1-u)**(degree-i) for i in range(degree+1)],axis=-1)
    return weights@controls


def bound(controls, depth=4):
    """Convex-hull norm bound, tightened by exact de Casteljau subdivision."""
    leaves = trajectory_math.subdivision_leaves(controls, depth)
    if leaves is not None:
        return float(np.linalg.norm(leaves,axis=1).max(initial=0))
    return _bound_recursive(controls,depth)


def _bound_recursive(controls, depth=4):
    if not depth:return float(np.linalg.norm(controls,axis=1).max(initial=0))
    rows = [controls]
    while len(rows[-1])>1:rows.append((rows[-1][:-1]+rows[-1][1:])/2)
    left = np.array([row[0] for row in rows])
    right = np.array([row[-1] for row in rows[::-1]])
    return max(_bound_recursive(left,depth-1),_bound_recursive(right,depth-1))


def controls_for(a,b,v,acc,duration):
    return np.array([a,a+v*duration/5,a+2*v*duration/5+acc*duration**2/20,b,b,b])


def _path_valid(space,path):
    """Use the scene's endpoint-sharing checks on the identical 0.04 rad grid."""
    if hasattr(space,'path_valid'):return space.path_valid(path,.04)
    return all(space.valid(q) for q in path) and all(space.edge_valid(a,b,.04) for a,b in zip(path,path[1:]))


def prepare(space, path, velocity, acceleration, speed=1.):
    """Check the original route and every curved replacement, including joins.

    Bezier position hulls prove joint limits, derivative hulls prove vector
    speed/acceleration bounds. Curves and connecting edges are collision sampled
    at 100 Hz. Retained route corners and the final endpoint remain at rest.
    """
    path = np.asarray(path,float)
    velocity=np.asarray(velocity,float);acceleration=np.asarray(acceleration,float)
    if (velocity.shape!=(6,) or acceleration.shape!=(6,) or not np.isfinite(velocity).all()
            or not np.isfinite(acceleration).all() or not np.isfinite(speed) or not 0<speed<=1
            or np.linalg.norm(velocity)>speed+1e-8 or np.linalg.norm(acceleration)>3.+1e-8):
        raise ValueError('Invalid continuous boundary motion or speed limit')
    if path.ndim!=2 or path.shape[1:]!=(6,) or not len(path) or not np.isfinite(path).all():
        raise ValueError('Expected a finite continuous joint path')
    if not _path_valid(space,path):
        raise ValueError('Continuous input route fails collision checks')
    path = path[np.r_[True,np.linalg.norm(np.diff(path,axis=0),axis=1)>1e-12]]
    path = simplify_path(path,space)
    at_rest = bool(np.linalg.norm(velocity)<1e-12 and np.linalg.norm(acceleration)<1e-12)
    if len(path)==1 and not at_rest:path = np.vstack((path,path))
    states = [path[0]]; times = [0.]; segments = []; vmax = amax = 0.
    specification_maxima = curve_envelope(np.zeros((6,6)), 1.)
    v,acc = velocity,acceleration
    for a,b in zip(path,path[1:]):
        distance = np.linalg.norm(b-a)
        duration = max(.2,1.875*distance/speed,np.sqrt(5.78*distance/3))*1.05
        accepted = None
        for _ in range(25):
            pieces = int(np.ceil(duration/DT)); duration = pieces*DT
            if times[-1]+duration>DURATION_LIMIT+1e-9:break
            controls = controls_for(a,b,v,acc,duration)
            vc = 5*np.diff(controls,axis=0)/duration
            ac = 20*np.diff(controls,n=2,axis=0)/duration**2
            vb,ab = bound(vc),bound(ac)
            specification_bounds = curve_envelope(controls, duration)
            if (np.all(controls>=ARM.lower) and np.all(controls<=ARM.upper)
                    and vb<=speed+1e-8 and ab<=3.+1e-8
                    and scale_needed(specification_bounds)<=1.+1e-8):
                checked = evaluate(controls,np.linspace(0,1,pieces*2+1))
                if _path_valid(space,checked):
                    accepted = controls;break
            duration *= 1.16
        if accepted is None:raise ValueError('No collision-free continuous join within motion/timing limits')
        specification_maxima = {key:max(specification_maxima[key], specification_bounds[key])
                                for key in specification_maxima}
        start_frame = len(states)-1; start_time = times[-1]
        samples = evaluate(accepted,np.arange(1,pieces+1)/pieces);samples[-1]=b
        states.extend(samples);times.extend(start_time+np.arange(1,pieces+1)*DT)
        segments.append({'start_frame':start_frame,'end_frame':len(states)-1,
            'start_time_s':start_time,'duration_s':duration,
            'start_deg':np.rad2deg(a).tolist(),'end_deg':np.rad2deg(b).tolist(),
            'controls_deg':np.rad2deg(accepted).tolist()})
        vmax=max(vmax,vb);amax=max(amax,ab);v=np.zeros(6);acc=np.zeros(6)
    states=np.array(states);times=np.array(times)
    return states,times,{'method':METHOD,'applied':True,'collision_rechecked':True,
        'collision_sample_dt_s':.01,'dt':DT,'frames':len(states),'duration_s':float(times[-1]),
        'speed_limit_rad_s':speed,'acceleration_limit_rad_s2':3.,
        'max_velocity_rad_s':vmax,'max_acceleration_rad_s2':amax,
        'start_velocity_rad_s':velocity.tolist(),'start_acceleration_rad_s2':acceleration.tolist(),
        'starts_and_ends_at_rest':at_rest,'ends_at_rest':True,'stops_at_corners':True,
        'segments':segments,'path_vertices':len(path),
        'specification_checked':True,'motion_specification':profile(),
        'motion_bounds':specification_maxima}


def frames_for(states,times,target,axis):
    from .pose5 import axis_error
    frames=[]
    for q,time in zip(states,times):
        transform=ARM.transform(q)
        frames.append({'q':np.rad2deg(q).tolist(),'time':float(time),
            'position':(transform[:3,3]*1000).tolist(),'tool_axis':transform[:3,2].tolist(),
            'error_mm':float(np.linalg.norm(transform[:3,3]-target)*1000),
            'orientation_error_deg':float(axis_error(transform[:3,2],axis)[2])})
    return frames


def sample_motion(metadata,time):
    """Analytical q/v/a at a checked curve time, including resting endpoints."""
    if not metadata['segments']:raise ValueError('A resting one-point curve has no motion segments')
    segment=next((s for s in metadata['segments'] if time<=s['start_time_s']+s['duration_s']+1e-9),metadata['segments'][-1])
    controls=np.deg2rad(segment['controls_deg']);duration=segment['duration_s']
    u=np.clip((time-segment['start_time_s'])/duration,0,1)
    return (evaluate(controls,u),evaluate(5*np.diff(controls,axis=0)/duration,u),
            evaluate(20*np.diff(controls,n=2,axis=0)/duration**2,u))
