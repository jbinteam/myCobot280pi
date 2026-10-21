"""Shared degree-valued waypoint conversion for simulation and joint control."""
import numpy as np


def joint_waypoints(start_deg, points_deg, max_step_deg=3.5, compress=False):
    """Split known joint segments; no waypoint IK.

    compress reproduces the existing hardware solver-frame compression. Keep
    it False for validated PRM paths: skipping a point would require a fresh
    segment validity check. The result excludes the initial joint state.
    """
    start = np.asarray(start_deg, float); points = np.asarray(points_deg, float).reshape(-1, 6)
    if start.shape != (6,) or not np.isfinite(start).all() or not np.isfinite(points).all():
        raise ValueError('Expected finite six-joint degree waypoints')
    if not np.isfinite(max_step_deg) or max_step_deg <= 0: raise ValueError('Waypoint step must be positive')
    waypoints = []; previous = start
    for i, point in enumerate(points):
        if compress and i != len(points)-1 and np.max(np.abs(point-previous)) < 2: continue
        delta = point-previous
        pieces = max(1, int(np.ceil(np.max(np.abs(delta))/max_step_deg)))
        segment = previous+np.arange(1, pieces+1)[:, None]/pieces*delta
        segment[-1] = point
        waypoints.extend(segment.tolist())
        previous = point
    return waypoints
