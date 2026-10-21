"""Local, fully checked 5D paths for tracking from the current joints."""
from dataclasses import asdict
from time import perf_counter
import numpy as np
from .kinematics import ARM
from . import pose5
from .planning import checked_goal, trajectory
from .trajectory import joint_waypoints


def simplify_path(path, space, max_step_rad=.04):
    """Skip solver ticks only through newly validated joint segments.

    Try the endpoint first, then halve the lookahead after a blocked shortcut.
    This keeps checking bounded without treating every solver tick as a turn.
    The caller must have already checked every original segment.
    """
    kept = [path[0]]; index = 0
    while index < len(path)-1:
        candidate = len(path)-1
        while candidate > index+1 and not space.edge_valid(path[index], path[candidate], max_step_rad):
            candidate = index+max(1, (candidate-index)//2)
        kept.append(path[candidate]); index = candidate
    return np.asarray(kept)


def local_pose_plan(start, target, target_axis, space, config, goal_constraint=None, *, solved=None):
    """Return an executable path only after complete local motion validation.

    A failed attempt returns diagnostics, never a playable partial prefix.
    No neural model or roadmap is loaded here. The service owns fallback.
    """
    began = perf_counter()
    attempt = {'attempted': True, 'method': config.pose5_method, 'status': 'not_started'}
    def rejected(status, reason):
        attempt.update(status=status, reason=reason, elapsed_ms=(perf_counter()-began)*1000)
        return {'converged': False, 'local_ik': attempt}

    if not space.valid(start):
        return rejected('invalid_start', space.diagnostic(start)['reason'])
    solved = solved if solved is not None else pose5.solve(start, target, method=config.pose5_method, target_axis=target_axis,
                         steps=config.steps, speed=config.speed_rad_s, damping=config.damping,
                         tolerance=config.tolerance_mm, orientation_tolerance_deg=config.orientation_tolerance_deg,
                         goal_constraint=goal_constraint)
    ik_ms = (perf_counter()-began)*1000
    attempt.update(solver_ms=solved['elapsed_ms'], iterations=solved['iterations'],
                   stop_reason=solved['stop_reason'], final_error_mm=solved['final_error_mm'],
                   final_orientation_error_deg=solved['final_orientation_error_deg'])
    if not solved['converged']:
        return rejected('not_converged', solved['stop_reason'])
    path = np.deg2rad([f['q'] for f in solved['frames']])
    attempt['raw_path_vertices'] = len(path)
    if (path.ndim != 2 or path.shape[1:] != (6,) or not len(path) or not np.isfinite(path).all()
            or not np.allclose(path[0], start, atol=1e-10, rtol=0)):
        return rejected('invalid_path', 'Local path must begin at the captured current joints')
    # Validate the complete solver path before simplifying it. A valid endpoint
    # does not authorize motion through an invalid intermediate configuration.
    if not all(space.valid(q) for q in path):
        return rejected('path_invalid', 'Local path violates joint limits or self-collision clearance')
    if not all(space.edge_valid(a, b, .04) for a, b in zip(path, path[1:])):
        return rejected('path_invalid', 'Local path segment violates joint limits or self-collision clearance')
    goal = checked_goal(path[-1], target, space, config, target_axis)
    if goal is None or (goal_constraint is not None and not goal_constraint(path[-1])):
        return rejected('goal_rejected', 'Local endpoint misses the requested pose or cursor tolerances')
    path = simplify_path(path, space)
    try:
        frames = trajectory(path, target, space, .04, config.speed_rad_s, target_axis=target_axis)
    except ValueError as exc:
        return rejected('path_invalid', str(exc))
    planning_ms = (perf_counter()-began)*1000-ik_ms
    source = 'current_state' if solved['iterations'] == 0 else 'current_local_ik'
    goal['source'] = source
    attempt.update(status='success', route_vertices=len(path), elapsed_ms=(perf_counter()-began)*1000)
    length = float(sum(np.linalg.norm(space.delta(a, b)) for a, b in zip(path, path[1:])))
    return {'simulation_only': True, 'method': 'local_ik', 'task': 'pose5',
            'converged': True, 'status': 'success', 'reason': 'local_ik_path',
            'validation': space.validation, 'self_collision_assumptions': space.metadata()['assumptions'],
            'local_ik': attempt, 'ik_ms': ik_ms, 'planning_ms': planning_ms, 'search_ms': 0.,
            'ik_outcome': 'accepted_goals',
            'planning_outcome': 'direct_connection' if source == 'current_state' else 'local_ik_path',
            'ik': {'goals': [goal], 'attempts': [{'source': source, 'accepted': True,
                    'valid_configuration': True, 'refinement_method': config.pose5_method,
                    'iterations': solved['iterations'], 'refinement_ms': solved['elapsed_ms'],
                    'stop_reason': solved['stop_reason']}], 'nn_checkpoint': None,
                   'seed_queue': [], 'config': asdict(config), 'ik_ms': ik_ms},
            'frames': frames, 'waypoints_rad': path.tolist(), 'waypoints_deg': np.rad2deg(path).tolist(),
            'fk_transforms_m': [ARM.transform(q).tolist() for q in path],
            'controller_waypoints_deg': joint_waypoints(np.rad2deg(start), [f['q'] for f in frames[1:]]),
            'start_deg': np.rad2deg(start).tolist(), 'target_mm': (target*1000).tolist(),
            'target_axis': target_axis.tolist(), 'orientation_xy_deg': pose5.xy_from_axis(target_axis),
            'final_error_mm': frames[-1]['error_mm'], 'final_orientation_error_deg': frames[-1]['orientation_error_deg'],
            'achieved_axis': frames[-1]['tool_axis'], 'tolerance_mm': config.tolerance_mm,
            'orientation_tolerance_deg': config.orientation_tolerance_deg, 'stop_reason': 'target_reached', 'dt': .05,
            'path_cost': float(sum(space.distance(a, b) for a, b in zip(path, path[1:]))),
            'path_length_rad': length, 'joint_travel_deg': float(np.rad2deg(length)),
            'limitations': 'Discretized URDF self-collision validation; external obstacles, tools and cables are not modeled.'}
