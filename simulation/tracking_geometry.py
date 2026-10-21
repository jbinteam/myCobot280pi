"""Frozen cone/disk cursor goals in g_base millimeters.

With the camera origin at the EEF, let u = (camera - object) / |camera - object|.
The acceptable plane has center object + distance*u and normal u; its bounded
disk radius is distance*tan(half_angle). Points must stay within the axial
allowance around that plane, within its radius, and point tool-Z at the object.
The stored request axis is -u (captured EEF toward cursor). Reversing the normal
does not change the plane. The disk is
fixed for that cursor target and distance stage; a new approach stage captures
a fresh measured camera-to-object line. A selected point has its own cursor-facing
tool axis. Client-supplied centers/radii are never authoritative.
"""
import numpy as np
from .kinematics import ARM
from . import pose5


def approach_settings(value, region):
    """Validate an opt-in staged approach; distances are axial millimeters."""
    if value is None:
        return None
    if (not isinstance(value, dict) or not {'final_distance_mm', 'step_mm'} <= set(value)
            or set(value)-{'final_distance_mm', 'step_mm', 'distance_tolerance_mm'}):
        raise ValueError('Approach requires final_distance_mm and step_mm')
    if region is None:
        raise ValueError('Approach requires a cone tracking region')
    final, step = value['final_distance_mm'], value['step_mm']
    for name, number, low, high in [('final_distance_mm', final, 10., region['distance_mm']),
                                    ('step_mm', step, 1., 50.)]:
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not np.isfinite(number) or not low <= number <= high:
            raise ValueError(f'Approach {name} must be {low}–{high}')
    tolerance = value.get('distance_tolerance_mm', 10.)
    if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
            or not np.isfinite(tolerance) or not 0 < tolerance <= min(step/2, final-1)):
        raise ValueError('Approach distance tolerance must be positive and at most half the stage step')
    return {'final_distance_mm': float(final), 'step_mm': float(step),
            'distance_tolerance_mm': float(tolerance)}


def region_for(cursor, axis, settings):
    if settings is None:
        return None
    if not isinstance(settings, dict) or settings.get('mode', 'cone_disk') != 'cone_disk':
        raise ValueError('tracking_region must describe cone_disk mode')
    if set(settings)-{'mode', 'distance_mm', 'half_angle_deg'}:
        raise ValueError('tracking_region contains unknown or derived fields')
    values = {}
    for name, default, low, high in [('distance_mm', 100., 10., 500.),
                                    ('half_angle_deg', 20., 1., 45.)]:
        value = settings.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not low <= value <= high:
            raise ValueError(f'tracking_region {name} must be {low}–{high}')
        values[name] = float(value)
    cursor = np.asarray(cursor, float)
    if cursor.shape != (3,) or not np.isfinite(cursor).all():
        raise ValueError('Cursor requires three finite millimeter coordinates')
    axis = pose5.unit_axis(axis)
    return {'mode': 'cone_disk', **values, 'cursor_mm': cursor.tolist(),
            'axis': axis.tolist(), 'center_mm': (cursor-values['distance_mm']*axis).tolist(),
            'radius_mm': float(values['distance_mm']*np.tan(np.deg2rad(values['half_angle_deg'])))}


def region_metrics(q, region, position_tol, angle_tol):
    transform = ARM.transform(q)
    return position_metrics(transform[:3, 3]*1000, transform[:3, 2], region, position_tol, angle_tol)


def position_metrics(position, tool_axis, region, position_tol, angle_tol):
    position = np.asarray(position, float)
    axis = np.asarray(region['axis'], float)
    relative = position-np.asarray(region['center_mm'], float)
    axial = float(relative@axis)
    radial = float(np.linalg.norm(relative-axial*axis))
    delta = np.asarray(region['cursor_mm'], float)-position
    distance = float(np.linalg.norm(delta))
    angle = pose5.axis_error(tool_axis, delta)[2] if distance > 1e-6 else 180.
    accepted = bool(abs(axial) <= position_tol and radial <= region['radius_mm']+1e-9 and angle <= angle_tol)
    return {'accepted': accepted, 'tracking_accepted': accepted, 'distance_mm': distance,
            'gap_error_mm': abs(axial), 'axial_error_mm': abs(axial),
            'radial_distance_mm': radial, 'radius_mm': region['radius_mm'],
            'direction_error_deg': angle}


def candidate_poses(region, start):
    """Try the nearest disk point, then bounded interior/rim alternatives.

    Each virtual endpoint is numerically solved from the current joints so the
    free tool-Z spin and current IK branch are retained where possible. The
    finite sample is a bounded search, not a proof of disk unreachability.
    """
    axis = np.asarray(region['axis'], float)
    center = np.asarray(region['center_mm'], float)
    cursor = np.asarray(region['cursor_mm'], float)
    radius = region['radius_mm']
    current = ARM.fk(start)*1000
    offset = current-center
    radial = offset-axis*(offset@axis)
    nearest = center+radial*min(1., radius/max(np.linalg.norm(radial), 1e-12))
    basis1 = np.cross(axis, np.eye(3)[np.argmin(np.abs(axis))]); basis1 /= np.linalg.norm(basis1)
    basis2 = np.cross(axis, basis1)
    points = [nearest, center]
    for fraction in (.5, .98):
        points.extend(center+fraction*radius*(np.cos(angle)*basis1+np.sin(angle)*basis2)
                      for angle in np.arange(8)*np.pi/4)
    # Preserve nearest-first, then rank remaining alternatives by Cartesian
    # travel. Current-state IK further prefers a nearby joint branch.
    ordered = [points[0]]+sorted(points[1:], key=lambda point: np.linalg.norm(point-current))
    unique = []
    for point in ordered:
        if any(np.linalg.norm(point-old) < 1e-7 for old in unique):
            continue
        unique.append(point)
        yield point/1000, pose5.unit_axis(cursor-point)


def gaze_score(states, region):
    """Rank checked tracking proposals by where the tool actually looks.

    Pointing at the frozen disk axis is insufficient away from its center:
    each EEF position has its own line of sight to the cursor. Disk position
    and joint travel break ties after terminal and mean pointing error.
    This ranking never replaces the caller's collision or capture checks.
    """
    states = np.asarray(states, float)
    metrics = [region_metrics(q, region, 1., 1.) for q in states]
    final = metrics[-1]
    return (final['direction_error_deg'],
            float(np.mean([item['direction_error_deg'] for item in metrics])),
            final['axial_error_mm']+max(0., final['radial_distance_mm']-region['radius_mm']),
            float(np.linalg.norm(np.diff(states, axis=0), axis=1).sum()))


def annotate_gaze(frames, cursor, region=None):
    """Expose cursor-relative gaze quality without changing playback timing."""
    cursor = np.asarray(cursor, float)
    angles = []
    for frame in frames:
        delta = cursor-np.asarray(frame['position'], float)
        angle = pose5.axis_error(frame['tool_axis'], delta)[2] if np.linalg.norm(delta) > 1e-6 else 180.
        frame['cursor_direction_error_deg'] = angle
        if region is not None:
            metrics = position_metrics(frame['position'], frame['tool_axis'], region, 1., 1.)
            frame['cursor_axial_error_mm'] = metrics['axial_error_mm']
            frame['cursor_radial_distance_mm'] = metrics['radial_distance_mm']
        angles.append(angle)
    if not angles:
        return None
    return {'initial_direction_error_deg': angles[0],
            'final_direction_error_deg': angles[-1],
            'max_direction_error_deg': max(angles),
            'mean_direction_error_deg': float(np.mean(angles)),
            'free_rotation': 'local flange Z'}
