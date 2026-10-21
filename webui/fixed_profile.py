"""Signed, jointly normalized fixed-target velocities scaled by SDK allowance."""
import numpy as np
from simulation.motion_limits import JOINT_SPEED, retime_result
from webui.motion_settings import MAX_SDK_SPEED
from simulation.reference_shape import settings as shape_settings, evaluate as shape_evaluate


def selected_speed(body, default):
    speed = body.get('speed', default)
    if isinstance(speed, bool) or not isinstance(speed, int) or not 1 <= speed <= MAX_SDK_SPEED:
        raise ValueError(f'Arm speed must be an integer from 1 to {MAX_SDK_SPEED}%')
    return speed


def retime_fixed(result, speed, *, path=None, velocity_shape=None):
    velocity_shape = shape_settings(velocity_shape)
    allowance = JOINT_SPEED*speed/100.
    if not result.get('frames'):
        return result
    if path is None:
        segments = result.get('smoothing', {}).get('segments')
        if segments:
            # Use the checked route's vertices, never treat dense samples as
            # extra stops or independent, differently scaled joint motions.
            path = np.deg2rad([segments[0]['start_deg'], *[s['end_deg'] for s in segments]])
    result = retime_result(result, allowance, path=path, per_joint_speed=True, reference_shape=velocity_shape)
    result['smoothing'].update(speed_percent=speed,
        joint_velocity_allowance_deg_s=float(np.rad2deg(allowance)),
        velocity_normalization='signed joint directions / common segment peak; '
            'quintic rate / 1.875; common time stretch for acceleration and Cartesian limits')
    if velocity_shape['kind'] != 'quintic':
        result['smoothing']['velocity_normalization'] = 'signed joint directions / common segment peak; unit-area velocity ramps / common peak; checked common time stretch'
        result['velocity_shape'] = velocity_shape
    result.update(planned_sdk_speed=speed,
                  joint_velocity_allowance_deg_s=float(np.rad2deg(allowance)))
    return result


def waypoint_intervals(result, waypoints):
    """Invert the shared normalized quintic for each retained SDK waypoint."""
    segments=result['smoothing']['segments']; index=0; previous=0.; intervals=[]
    for point in waypoints:
        point=np.asarray(point,float)
        if index>=len(segments):raise ValueError('Fixed-target waypoint is outside its timed profile')
        segment=segments[index]
        a=np.asarray(segment['start_deg']); b=np.asarray(segment['end_deg']); delta=b-a
        fraction=float(np.dot(point-a,delta)/np.dot(delta,delta))
        if not -1e-8<=fraction<=1.+1e-8 or not np.allclose(point,a+fraction*delta,atol=1e-7,rtol=0):
            raise ValueError('Fixed-target waypoint differs from its normalized profile route')
        lo,hi=0.,1.
        for _ in range(40):
            u=(lo+hi)/2
            shape = result['smoothing'].get('reference_shape')
            progress = 10*u**3-15*u**4+6*u**5 if shape is None else shape_evaluate(u, shape)[0]
            if progress < fraction:lo=u
            else:hi=u
        phase=segment['start_time_s']+segment['duration_s']*(lo+hi)/2
        if fraction>=1.-1e-10:
            phase=segment['start_time_s']+segment['duration_s'];index+=1
        if phase<=previous:raise ValueError('Fixed-target profile times must increase')
        intervals.append(phase-previous);previous=phase
    if index!=len(segments):raise ValueError('Fixed-target profile does not reach its reviewed endpoint')
    return intervals
