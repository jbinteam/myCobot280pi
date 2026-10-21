"""Monotone progress laws shared by fixed-path timing and encoder execution."""
import numpy as np


def settings(value=None):
    if value is None:
        return {'kind': 'quintic'}
    if not isinstance(value, dict):
        raise ValueError('Reference velocity shape must be an object')
    if value.get('kind') == 'quintic' and set(value) == {'kind'}:
        return {'kind': 'quintic'}
    if value.get('kind') != 'smooth_trapezoid' or set(value) != {'kind', 'rise_percent', 'fall_percent'}:
        raise ValueError('Choose the original bell or adjustable reference velocity ramps')
    values = [value['rise_percent'], value['fall_percent']]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v)
           or not 5 <= v <= 80 for v in values) or sum(values) > 100:
        raise ValueError('Acceleration and deceleration time must each be 5–80%, with a total at most 100%')
    return {'kind': 'smooth_trapezoid', 'rise_percent': float(values[0]), 'fall_percent': float(values[1])}


def peaks(shape=None):
    shape = settings(shape)
    if shape['kind'] == 'quintic':
        return 1.875, 10*np.sqrt(3)/3
    rise, fall = shape['rise_percent']/100, shape['fall_percent']/100
    rate = 1/(1-(rise+fall)/2)
    return rate, 1.5*rate/min(rise, fall)


def evaluate(u, shape=None):
    """Progress, its normalized velocity, and acceleration; area is exactly one."""
    u = np.clip(np.asarray(u, float), 0., 1.)
    if shape is None or shape['kind'] == 'quintic':
        return 10*u**3-15*u**4+6*u**5, 30*u**2*(1-u)**2, 60*u*(1-u)*(1-2*u)
    rise, fall = shape['rise_percent']/100, shape['fall_percent']/100
    peak = 1/(1-(rise+fall)/2)
    x = np.clip(u/rise, 0., 1.)
    w = np.clip((u-(1-fall))/fall, 0., 1.)
    progress = np.where(u < rise, peak*rise*(x**3-.5*x**4),
        np.where(u <= 1-fall, peak*(u-rise/2),
            peak*(1-fall-rise/2+fall*(w-w**3+.5*w**4))))
    rate = np.where(u < rise, peak*(3*x**2-2*x**3),
        np.where(u <= 1-fall, peak, peak*(1-3*w**2+2*w**3)))
    acceleration = np.where(u < rise, peak*(6*x-6*x**2)/rise,
        np.where(u <= 1-fall, 0., -peak*(6*w-6*w**2)/fall))
    return np.where(u == 1., 1., progress), rate, acceleration
