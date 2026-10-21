"""Collision-preserving quintic timing for final simulation references."""
import numpy as np
from .planner import smooth
from ..motion_limits import quintic_path

PLAYBACK_DT=.02
ACCELERATION=3.
TRACKING_DURATION_LIMIT=1.3


def playback_segments(path,states,times):
    """Export the exact checked segments and their already-rounded timing."""
    segments=[];start=0
    for endpoint in path[1:]:
        if np.array_equal(endpoint,states[start]):continue
        matches=np.flatnonzero(np.all(states[start+1:]==endpoint,axis=1))
        if not len(matches):raise ValueError('Timed path is missing a checked corner')
        end=start+1+int(matches[0])
        segments.append({'start_frame':start,'end_frame':end,
            'start_time_s':float(times[start]),'duration_s':float(times[end]-times[start]),
            'start_deg':np.rad2deg(states[start]).tolist(),'end_deg':np.rad2deg(states[end]).tolist()})
        start=end
    return segments


def motion_bounds(states,times):
    """Finite-difference vector bounds including initial/final zero-velocity holds."""
    if len(states)<2:return 0.,0.
    intervals=np.diff(times)
    velocity=np.diff(states,axis=0)/intervals[:,None]
    padded=np.vstack((np.zeros(6),velocity,np.zeros(6)))
    # Velocity samples live at interval midpoints; holds are one interval away.
    acceleration_intervals=np.r_[intervals[0],(intervals[:-1]+intervals[1:])/2,intervals[-1]]
    acceleration=np.diff(padded,axis=0)/acceleration_intervals[:,None]
    return float(np.linalg.norm(velocity,axis=1).max(initial=0)),float(np.linalg.norm(acceleration,axis=1).max(initial=0))


def prepare(env,points,speed=1.,acceleration=ACCELERATION,dt=PLAYBACK_DT):
    """Recheck raw path, checked shortcuts, timing, every output state and edge.

    Monotone quintics stay on straight segments and stop at their corners.
    If a shortcut's new timing samples reveal grazing contact, retry the original
    safe corners. Failure never returns an unsmoothed executable fallback.
    """
    points=np.asarray(points,float)
    if (points.ndim!=2 or points.shape[1]!=6 or not len(points) or not np.isfinite(points).all()
            or not all(np.isfinite(v) and v>0 for v in (speed,acceleration,dt))):
        raise ValueError('Expected finite joint path and positive timing limits')
    if not env.path_valid(points):raise ValueError('Input path fails collision validation')
    original=points[np.r_[True,np.linalg.norm(np.diff(points,axis=0),axis=1)>1e-12]]
    if len(original)==1:
        states=original.copy();times=np.array([0.]);path=original;fallback=False
    else:
        path=smooth(env,original);fallback=False
        states,times,specification=quintic_path(path,dt=dt,speed=speed,acceleration=acceleration)
        if not env.path_valid(states):
            path=original;fallback=True
            states,times,specification=quintic_path(path,dt=dt,speed=speed,acceleration=acceleration)
        if not env.path_valid(states):raise ValueError('Smoothed final trajectory fails collision validation')
    maximum_speed,maximum_acceleration=motion_bounds(states,times)
    if maximum_speed>speed+1e-8 or maximum_acceleration>acceleration+1e-8:
        raise ValueError('Smoothed trajectory violates speed or acceleration limits')
    if not np.array_equal(states[0],points[0]) or not np.array_equal(states[-1],points[-1]):
        raise ValueError('Smoothing changed the captured start or final endpoint')
    metadata={'method':'collision_checked_shortcuts_quintic','applied':True,
        'input_vertices':len(points),'path_vertices':len(path),'frames':len(states),'duration_s':float(times[-1]),
        'dt':dt,'speed_limit_rad_s':speed,'acceleration_limit_rad_s2':acceleration,
        'starts_and_ends_at_rest':True,'stops_at_corners':True,'collision_rechecked':True,
        'preserved_corner_fallback':fallback,'max_velocity_rad_s':maximum_speed,
        'max_acceleration_rad_s2':maximum_acceleration,
        'segments':playback_segments(path,states,times)}
    if len(original)==1:
        _,_,specification=quintic_path(original,dt=dt,speed=speed,acceleration=acceleration)
    metadata.update({key:specification[key] for key in ('specification_checked','motion_specification','motion_bounds')})
    return states,times,metadata


def prepare_tracking(env, points, speed=1., duration_limit=TRACKING_DURATION_LIMIT):
    """Keep the bounded tracking horizon by shortening geometry before timing.

    Never truncate a moving curve in time: the returned safe prefix is retimed
    to finish at rest, and the remaining goal still requires replanning.
    """
    states, times, metadata = prepare(env, points, speed=speed)
    if times[-1] <= duration_limit+1e-9:
        return states, times, metadata
    path = [states[0]]
    for segment in metadata['segments']:
        endpoint = np.deg2rad(segment['end_deg'])
        if segment['start_time_s']+segment['duration_s'] <= duration_limit+1e-9:
            path.append(endpoint)
            continue
        start = path[-1]; lo, hi = 0., 1.
        for _ in range(20):
            fraction = (lo+hi)/2
            _, candidate_times, _ = quintic_path([*path, start+fraction*(endpoint-start)], speed=speed)
            if candidate_times[-1] <= duration_limit+1e-9:
                lo = fraction
            else:
                hi = fraction
        if lo > 1e-8:
            path.append(start+lo*(endpoint-start))
        break
    result = prepare(env, np.asarray(path), speed=speed)
    if result[1][-1] > duration_limit+1e-9:
        raise ValueError('Specification-checked tracking prefix exceeds its duration budget')
    result[2].update(tracking_prefix_limited=True, requested_duration_s=float(times[-1]))
    return result
