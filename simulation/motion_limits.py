"""Manufacturer motion envelope for the bare myCobot 280 Pi flange.

The URDF has placeholder velocity=0, so it is not a dynamics specification.
Joint limits are per joint. XYZ component limits are conservatively enforced
as translation norms. The rotational policy bounds angular velocity, not Euler
angle derivatives (which are singular at some orientations).
"""
import numpy as np
from .kinematics import ARM
from . import trajectory_math
from .reference_shape import settings as shape_settings, peaks as shape_peaks, evaluate as shape_evaluate

SOURCE = ('https://docs.elephantrobotics.com/docs/mycobot_280_pi_en/'
          '3-FunctionsAndApplications/6.developmentGuide/'
          'CommunicationProtocolPackage/18-communication.html')
JOINT_SPEED = float(np.deg2rad(150.))
JOINT_ACCELERATION = float(np.deg2rad(200.))
TCP_SPEED = .1
TCP_ACCELERATION = .4
ANGULAR_SPEED = float(np.deg2rad(40.))
ANGULAR_ACCELERATION = float(np.deg2rad(66.))
PREVIEW_SPEED = 2.6  # Inward-rounded vector cap, also bounds every joint.


def profile():
    return {'robot': 'myCobot 280 Pi', 'source': SOURCE,
            'joint_velocity_deg_s': [150.]*6, 'joint_acceleration_deg_s2': [200.]*6,
            'tcp_velocity_mm_s': 100., 'tcp_acceleration_mm_s2': 400.,
            'angular_velocity_deg_s': 40., 'angular_acceleration_deg_s2': 66.,
            'scope': 'bare joint6_flange; conservative XYZ norm and angular-vector policy; '
                     'not an Euler-rate, payload torque or measured servo guarantee'}


# p(q) is a sum of rotated URDF translations. A translation after joint k
# has speed <= length*sum(abs(v[:k])) and acceleration <=
# length*(sum(abs(a[:k])) + sum(abs(v[:k]))**2). Constant origin rotations
# preserve these norms. This proves bounds between samples without treating
# a finite FK difference as an instantaneous TCP derivative.
_translations = []
_active = 0
for _origin, _axis in ARM.chain:
    if _active:
        _translations.append((float(np.linalg.norm(_origin[:3, 3])), _active))
    if _axis is not None:
        _active += 1


def envelope(velocity_abs, acceleration_abs):
    v = np.asarray(velocity_abs, float); a = np.asarray(acceleration_abs, float)
    if (v.shape != (6,) or a.shape != (6,) or not np.isfinite(v).all()
            or not np.isfinite(a).all() or np.any(v < 0) or np.any(a < 0)):
        raise ValueError('Expected six finite nonnegative derivative bounds')
    linear_v = linear_a = 0.
    for length, count in _translations:
        total_v = float(v[:count].sum())
        linear_v += length*total_v
        linear_a += length*(float(a[:count].sum())+total_v**2)
    # Each world joint axis has derivative from preceding joint rotations.
    rotational_a = float(a.sum())+sum(float(v[j]*v[:j].sum()) for j in range(6))
    return {'joint_velocity_rad_s': float(v.max()),
            'joint_acceleration_rad_s2': float(a.max()),
            'tcp_velocity_m_s': linear_v, 'tcp_acceleration_m_s2': linear_a,
            'angular_velocity_rad_s': float(v.sum()),
            'angular_acceleration_rad_s2': rotational_a}


def scale_needed(bounds):
    """Uniform time stretch needed for the conservative derivative envelope."""
    return max(1., bounds['joint_velocity_rad_s']/JOINT_SPEED,
               np.sqrt(bounds['joint_acceleration_rad_s2']/JOINT_ACCELERATION),
               bounds['tcp_velocity_m_s']/TCP_SPEED,
               np.sqrt(bounds['tcp_acceleration_m_s2']/TCP_ACCELERATION),
               bounds['angular_velocity_rad_s']/ANGULAR_SPEED,
               np.sqrt(bounds['angular_acceleration_rad_s2']/ANGULAR_ACCELERATION))


def derivative_hull(controls, depth=4):
    """Per-component absolute bounds from subdivided Bernstein hulls."""
    controls = np.asarray(controls, float)
    leaves = trajectory_math.subdivision_leaves(controls, depth)
    if leaves is not None:
        return np.max(np.abs(leaves),axis=0)
    return _derivative_hull_recursive(controls,depth)


def _derivative_hull_recursive(controls, depth=4):
    controls = np.asarray(controls, float)
    if not depth:
        return np.max(np.abs(controls), axis=0)
    rows = [controls]
    while len(rows[-1]) > 1:
        rows.append((rows[-1][:-1]+rows[-1][1:])/2)
    return np.maximum(_derivative_hull_recursive(np.array([r[0] for r in rows]), depth-1),
                      _derivative_hull_recursive(np.array([r[-1] for r in rows[::-1]]), depth-1))


def curve_envelope(controls, duration):
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError('Curve duration must be positive')
    v = 5*np.diff(np.asarray(controls, float), axis=0)/duration
    a = 4*np.diff(v, axis=0)/duration
    return envelope(derivative_hull(v), derivative_hull(a))


def quintic_path(path, speed=1., acceleration=3., dt=.02, per_joint_speed=False, reference_shape=None):
    """Retain every route vertex, with rest-to-rest C2 motion at all joins."""
    reference_shape = shape_settings(reference_shape)
    peak_rate, peak_acceleration = shape_peaks(reference_shape)
    shaped = reference_shape["kind"] != "quintic"
    path = np.asarray(path, float)
    if (path.ndim != 2 or path.shape[1:] != (6,) or not len(path)
            or not np.isfinite(path).all() or np.any(path < ARM.lower-1e-9)
            or np.any(path > ARM.upper+1e-9)
            or not all(np.isfinite(x) and x > 0 for x in (speed, acceleration, dt))):
        raise ValueError('Expected an operating-box path and positive timing limits')
    speed = min(speed, JOINT_SPEED if per_joint_speed else PREVIEW_SPEED)
    acceleration = min(acceleration, JOINT_ACCELERATION)
    states = [path[0]]; times = [0.]; segments = []; maxima = envelope(np.zeros(6), np.zeros(6))
    for start, end in zip(path, path[1:]):
        delta = end-start; distance = float(np.linalg.norm(delta))
        if distance < 1e-12:
            continue
        peak_distance = float(np.max(np.abs(delta)))
        velocity_distance = peak_distance if per_joint_speed else distance
        duration = max(peak_rate*velocity_distance/speed, np.sqrt(peak_acceleration*distance/acceleration))
        bounds = envelope(peak_rate*np.abs(delta)/duration,
                          peak_acceleration*np.abs(delta)/duration**2)
        duration *= scale_needed(bounds)*1.001
        pieces = max(1, int(np.ceil(duration/dt))); duration = pieces*dt
        bounds = envelope(peak_rate*np.abs(delta)/duration,
                          peak_acceleration*np.abs(delta)/duration**2)
        maxima = {k: max(maxima[k], bounds[k]) for k in maxima}
        index = len(states)-1; offset = times[-1]
        u = np.arange(1, pieces+1)/pieces
        values = start+shape_evaluate(u, reference_shape)[0][:, None]*delta; values[-1] = end
        states.extend(values); times.extend(offset+np.arange(1, pieces+1)*dt)
        segments.append({'start_frame': index, 'end_frame': len(states)-1,
            'start_time_s': offset, 'duration_s': duration,
            'start_deg': np.rad2deg(start).tolist(), 'end_deg': np.rad2deg(end).tolist(),
            'normalized_joint_direction': (delta/peak_distance).tolist(),
            'velocity_scale_rad_s': float(peak_rate*peak_distance/duration),
            'velocity_allowance_scale': float(peak_rate*peak_distance/(duration*speed))})
    metadata = {'method': 'specification_checked_quintic', 'applied': True,
        'starts_and_ends_at_rest': True, 'ends_at_rest': True, 'stops_at_corners': True,
        'speed_limit_rad_s': speed, 'acceleration_limit_rad_s2': acceleration,
        'speed_limit_kind': 'per_joint' if per_joint_speed else 'vector_norm',
        'dt': dt, 'segments': segments, 'duration_s': float(times[-1]),
        'motion_specification': profile(), 'specification_checked': True,
        'motion_bounds': maxima}
    if shaped:
        metadata.update(method='specification_checked_velocity_shape', reference_shape=reference_shape)
    return np.asarray(states), np.asarray(times), metadata


def retime_result(result, speed=1.2, *, path=None, per_joint_speed=False, reference_shape=None):
    """Time solver references without changing its numerical solve or geometry."""
    frames = result.get('frames', [])
    if not frames:
        return result
    if path is None:
        path = np.deg2rad([f['q'] for f in frames])
    states, times, metadata = quintic_path(path, speed, per_joint_speed=per_joint_speed, reference_shape=reference_shape)
    target = np.asarray(result['target_mm'])/1000
    from .pose5 import axis_error
    output = []
    for q, time in zip(states, times):
        t = ARM.transform(q)
        frame = {'q': np.rad2deg(q).tolist(), 'time': float(time),
                 'position': (t[:3, 3]*1000).tolist(), 'tool_axis': t[:3, 2].tolist(),
                 'error_mm': float(np.linalg.norm(t[:3, 3]-target)*1000)}
        if result.get('target_axis') is not None:
            frame['orientation_error_deg'] = float(axis_error(t[:3, 2], np.asarray(result['target_axis']))[2])
        output.append(frame)
    return {**result, 'frames': output, 'smoothing': metadata,
            'motion_specification': profile(), 'solver_frame_count': len(frames)}
