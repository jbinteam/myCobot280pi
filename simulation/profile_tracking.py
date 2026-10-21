"""Shared fixed-target progress law with checked C2 tracking boundaries.

A resting leg follows reference_shape.evaluate exactly. A moving retarget adds
quintic Hermite boundary terms, preserving initial q/v/a and ending at rest.
Piecewise Bernstein hulls bound the complete profile between collision samples.
"""
from math import comb
import numpy as np
from numpy.polynomial import Polynomial as Poly
from .kinematics import ARM
from .reference_shape import settings, peaks, evaluate as shape_evaluate
from .continuous import DT, DURATION_LIMIT, bound, evaluate, _path_valid, frames_for
from .tracking import simplify_path
from .motion_limits import curve_envelope, scale_needed, profile

METHOD = 'collision_checked_velocity_profile_c2'


def motion_for(segment, phase, shape):
    duration = segment['duration_s']
    u = float(np.clip((phase-segment['start_time_s'])/duration, 0., 1.))
    a = np.deg2rad(segment['start_deg']); delta = np.deg2rad(segment['end_deg'])-a
    v = np.asarray(segment['start_velocity_rad_s']); acc = np.asarray(segment['start_acceleration_rad_s2'])
    p, rate, acceleration = shape_evaluate(u, shape)
    h = u-6*u**3+8*u**4-3*u**5
    dh = 1-18*u**2+32*u**3-15*u**4
    ddh = -36*u+96*u**2-60*u**3
    k = .5*(u**2-3*u**3+3*u**4-u**5)
    dk = u-4.5*u**2+6*u**3-2.5*u**4
    ddk = 1-9*u+18*u**2-10*u**3
    return (a+delta*p+v*duration*h+acc*duration**2*k,
            delta*rate/duration+v*dh+acc*duration*dk,
            delta*acceleration/duration**2+v*ddh/duration+acc*ddk)


def sample_motion(curve, phase):
    if not curve['segments']:
        return np.deg2rad(curve['hold_deg']), np.zeros(6), np.zeros(6)
    segment = next((s for s in curve['segments'] if phase <= s['start_time_s']+s['duration_s']+1e-9), curve['segments'][-1])
    return motion_for(segment, phase, curve['reference_shape'])


def pieces_for(a, b, velocity, acceleration, duration, shape):
    """Exact degree-five Bernstein representation of each time-law piece."""
    x = Poly([0., 1.]); delta = b-a
    if shape['kind'] == 'quintic':
        pieces = [(0., 1., 10*x**3-15*x**4+6*x**5)]
    else:
        r = shape['rise_percent']/100; f = shape['fall_percent']/100
        peak = 1/(1-(r+f)/2)
        pieces = [(0., r, peak*r*(x**3-.5*x**4))]
        if 1-f-r > 1e-12:
            pieces.append((r, 1-f, peak*(r/2+(1-f-r)*x)))
        pieces.append((1-f, 1., peak*(1-f-r/2+f*(x-x**3+.5*x**4))))
    result = []
    for lo, hi, progress in pieces:
        u = Poly([lo, hi-lo])
        h = u-6*u**3+8*u**4-3*u**5
        k = .5*(u**2-3*u**3+3*u**4-u**5)
        coeff = np.zeros((6, 6)); coeff[0] = a
        for polynomial, scale in [(progress, delta), (h, velocity*duration), (k, acceleration*duration**2)]:
            coeff[:len(polynomial.coef)] += polynomial.coef[:, None]*scale
        controls = np.array([sum(coeff[i]*comb(j, i)/comb(5, i) for i in range(j+1)) for j in range(6)])
        result.append((controls, (hi-lo)*duration))
    return result


def prepare(space, path, velocity, acceleration, speed=1., *, velocity_shape=None, minimum_duration_s=.2):
    shape = settings(velocity_shape)
    path = np.asarray(path, float); velocity = np.asarray(velocity, float); acceleration = np.asarray(acceleration, float)
    if (path.ndim != 2 or path.shape[1:] != (6,) or not len(path) or not np.isfinite(path).all()
            or velocity.shape != (6,) or acceleration.shape != (6,) or not np.isfinite(velocity).all()
            or not np.isfinite(acceleration).all() or not np.isfinite(speed) or not 0 < speed <= 1
            or np.linalg.norm(velocity) > speed+1e-8 or np.linalg.norm(acceleration) > 3+1e-8):
        raise ValueError('Invalid shared-profile tracking path or initial motion')
    if not _path_valid(space, path): raise ValueError('Profile input route fails collision checks')
    path = path[np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1)>1e-12]]
    path = simplify_path(path, space)
    at_rest = bool(np.linalg.norm(velocity)<1e-12 and np.linalg.norm(acceleration)<1e-12)
    if len(path) == 1 and not at_rest: path = np.vstack((path, path))
    states = [path[0]]; times = [0.]; segments = []; vmax = amax = 0.
    maxima = curve_envelope(np.zeros((6, 6)), 1.)
    v, acc = velocity, acceleration
    peak_rate, peak_acc = peaks(shape)
    for a, b in zip(path, path[1:]):
        distance = np.linalg.norm(b-a)
        duration = max(minimum_duration_s, peak_rate*distance/speed, np.sqrt(peak_acc*distance/3))*1.05
        accepted = None
        for _ in range(25):
            count = int(np.ceil(duration/DT)); duration = count*DT
            if times[-1]+duration > DURATION_LIMIT+1e-9: break
            parts = pieces_for(a, b, v, acc, duration, shape)
            bounds = [curve_envelope(c, d) for c, d in parts]
            vb = max(bound(5*np.diff(c, axis=0)/d) for c, d in parts)
            ab = max(bound(20*np.diff(c, n=2, axis=0)/d**2) for c, d in parts)
            if (all(np.all(c>=ARM.lower) and np.all(c<=ARM.upper) for c, _ in parts)
                    and vb<=speed+1e-8 and ab<=3+1e-8 and all(scale_needed(x)<=1+1e-8 for x in bounds)
                    and all(_path_valid(space, evaluate(c, np.linspace(0, 1, max(2, int(np.ceil(d/.01))+1)))) for c, d in parts)):
                accepted = True; break
            duration *= 1.16
        if not accepted: raise ValueError('No checked shared-profile join within motion/timing limits')
        segment = {'start_frame': len(states)-1, 'start_time_s': times[-1], 'duration_s': duration,
            'start_deg': np.rad2deg(a).tolist(), 'end_deg': np.rad2deg(b).tolist(),
            'start_velocity_rad_s': v.tolist(), 'start_acceleration_rad_s2': acc.tolist()}
        phase = times[-1]+np.arange(1, count+1)*DT
        samples = [motion_for(segment, t, shape)[0] for t in phase]; samples[-1] = b
        states.extend(samples); times.extend(phase)
        segment['end_frame'] = len(states)-1; segments.append(segment)
        maxima = {key: max(maxima[key], *(x[key] for x in bounds)) for key in maxima}
        vmax = max(vmax, vb); amax = max(amax, ab); v = np.zeros(6); acc = np.zeros(6)
    states = np.asarray(states); times = np.asarray(times)
    if not _path_valid(space, states): raise ValueError('Profile preview fails final scene checks')
    return states, times, {'method': METHOD, 'applied': True, 'collision_rechecked': True,
        'reference_shape': shape, 'boundary_blended': not at_rest, 'hold_deg': np.rad2deg(states[0]).tolist(),
        'dt': DT, 'frames': len(states), 'duration_s': float(times[-1]), 'collision_sample_dt_s': .01,
        'speed_limit_rad_s': speed, 'acceleration_limit_rad_s2': 3., 'max_velocity_rad_s': vmax,
        'max_acceleration_rad_s2': amax, 'start_velocity_rad_s': velocity.tolist(),
        'start_acceleration_rad_s2': acceleration.tolist(), 'starts_and_ends_at_rest': at_rest,
        'ends_at_rest': True, 'stops_at_corners': True, 'segments': segments,
        'path_vertices': len(path), 'specification_checked': True,
        'motion_specification': profile(), 'motion_bounds': maxima}


def prepare_tracking(space, path, speed=1., *, velocity_shape=None, duration_limit=1.3):
    zero = np.zeros(6); path = np.asarray(path, float)
    if not _path_valid(space, path): raise ValueError('Profile prefix input fails collision checks')
    path = simplify_path(path, space); prefix = [path[0]]
    for end in path[1:]:
        try:
            complete = prepare(space, [*prefix, end], zero, zero, speed, velocity_shape=velocity_shape)
        except ValueError:
            complete = None
        if complete is not None and complete[1][-1] <= duration_limit+1e-9:
            prefix.append(end); continue
        lo, hi = 0., 1.
        for _ in range(16):
            fraction = (lo+hi)/2
            try:
                candidate = prepare(space, [*prefix, prefix[-1]+fraction*(end-prefix[-1])], zero, zero, speed, velocity_shape=velocity_shape)
            except ValueError:
                hi = fraction; continue
            if candidate[1][-1] <= duration_limit+1e-9: lo = fraction
            else: hi = fraction
        if lo > 1e-8: prefix.append(prefix[-1]+lo*(end-prefix[-1]))
        break
    result = prepare(space, prefix, zero, zero, speed, velocity_shape=velocity_shape)
    result[2]['tracking_prefix_limited'] = len(prefix)<len(path) or not np.allclose(prefix[-1],path[-1],atol=1e-12,rtol=0)
    return result


def validate(curve, frames, expected, speed, space):
    """Reconstruct the exact profile independently of reported derivative bounds."""
    shape = settings(curve.get('reference_shape'))
    if (curve.get('method') != METHOD or not isinstance(curve.get('segments'), list)
            or curve.get('ends_at_rest') is not True or curve.get('collision_rechecked') is not True
            or curve.get('dt') != DT or curve.get('frames') != len(frames) or not frames or len(frames)>601
            or not np.isfinite(curve.get('duration_s', np.nan)) or not 0 <= curve['duration_s'] <= DURATION_LIMIT):
        raise ValueError('Missing checked shared-profile mouse curve')
    if not all(np.allclose(a, b, atol=1e-8, rtol=0) for a, b in zip(sample_motion(curve, 0), expected)):
        raise ValueError('Shared profile changed the owned q/v/a join')
    previous = expected; elapsed = 0.; checked = [expected[0]]
    for segment in curve['segments']:
        a = np.deg2rad(np.asarray(segment['start_deg'], float)); b = np.deg2rad(np.asarray(segment['end_deg'], float))
        v = np.asarray(segment['start_velocity_rad_s'], float); acc = np.asarray(segment['start_acceleration_rad_s2'], float)
        duration = segment['duration_s']
        if (any(x.shape!=(6,) or not np.isfinite(x).all() for x in (a, b, v, acc))
                or not np.isfinite(duration) or duration<=0 or abs(segment['start_time_s']-elapsed)>1e-8
                or not all(np.allclose(x, y, atol=1e-8, rtol=0) for x, y in zip((a, v, acc), previous))):
            raise ValueError('Invalid shared-profile segment or join')
        for controls, dt in pieces_for(a, b, v, acc, duration, shape):
            if (np.any(controls<ARM.lower) or np.any(controls>ARM.upper)
                    or bound(5*np.diff(controls, axis=0)/dt)>speed+1e-8
                    or bound(20*np.diff(controls, n=2, axis=0)/dt**2)>3+1e-8
                    or scale_needed(curve_envelope(controls, dt))>1+1e-8):
                raise ValueError('Shared profile exceeds joint or motion limits')
            checked.extend(evaluate(controls, np.linspace(0, 1, max(2, int(np.ceil(dt/.01))+1))))
        elapsed += duration; previous = (b, np.zeros(6), np.zeros(6))
    if abs(elapsed-curve['duration_s'])>1e-8 or abs(frames[-1]['time']-elapsed)>1e-8:
        raise ValueError('Shared-profile duration differs from preview')
    for i, frame in enumerate(frames):
        if (abs(frame['time']-i*DT)>1e-8 or not np.allclose(np.deg2rad(frame['q']), sample_motion(curve, frame['time'])[0], atol=1e-8, rtol=0)):
            raise ValueError('Shared profile differs from a preview frame')
    checked = np.asarray(checked)
    scene_valid = space.path_valid(checked, .01) if hasattr(space, 'path_valid') else (all(space.valid(q) for q in checked) and all(space.edge_valid(a, b, .01) for a, b in zip(checked, checked[1:])))
    if not np.allclose(previous[0], np.deg2rad(frames[-1]['q']), atol=1e-8, rtol=0) or not scene_valid:
        raise ValueError('Shared-profile endpoint or scene validation failed')
