"""Learned-assisted numerical IK and checked direct/RRT* cursor tracking.

Virtual IK starts never become motion. Only scene-checked, timed routes can be
played; continuous replacements preserve the server-owned q/v/a boundary.
"""
from dataclasses import asdict
from time import perf_counter
import numpy as np
from ..kinematics import ARM
from .. import pose5
from ..continuous import initial_motion, prepare as continuous_prepare, frames_for
from . import service, classical_service
from .planner import RRTConfig, plan as rrt_plan
from .tracking import cursor_metrics
from ..tracking_geometry import region_for, candidate_poses, annotate_gaze
from ..physical_reach import ENVELOPE, target_precheck

_env = None  # Access only while holding the shared planner lock.


def verified_status():
    status = classical_service.info()
    if status['status'] != 'ready':
        raise RuntimeError(status.get('reason', 'Learned IK + RRT* unavailable'))
    return status


def plan(body):
    global _env
    began = perf_counter()
    cursor = service.vector(body.get('cursor_mm'), 3, 'cursor_mm')
    start_deg = service.vector(body.get('start_deg'), 6, 'start_deg')
    start = np.deg2rad(start_deg)
    axis = pose5.unit_axis(service.vector(body['target_axis'], 3, 'target_axis')
        if 'target_axis' in body else cursor-ARM.fk(start)*1000)
    region = region_for(cursor, axis, body.get("tracking_region"))
    orientation_mode = body.get('orientation_mode','point_at_object')
    if orientation_mode not in ('point_at_object','camera_fov','strict_then_fov'):
        raise ValueError('Unknown tracking orientation mode')
    if orientation_mode in ('camera_fov','strict_then_fov') and region is None:
        raise ValueError('Camera-FOV orientation requires a cone disk')
    target = (cursor-(region["distance_mm"] if region else 50)*axis)/1000
    start, target, axis, box, _, config = service.request({
        'start_deg': start_deg.tolist(), 'target_mm': (target*1000).tolist(),
        'target_axis': axis.tolist(), 'obstacle_mm': body.get('obstacle_mm'),
        **{k: body[k] for k in ('seed',) if k in body}})
    if 'obstacle_mm' not in body:
        raise ValueError('Include obstacle_mm as six numbers or null for no block')
    settings = {}
    for key, default, low, high in [('speed_rad_s', 1., .001, 1.),
            ('tolerance_mm', 1., .1, 10.), ('orientation_tolerance_deg', 1., .1, 10.)]:
        value = body.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{key} must be {low}–{high}')
        settings[key] = value
    speed = settings['speed_rad_s']
    continuous, velocity, acceleration = initial_motion(body, speed)
    from ..reference_shape import settings as shape_settings
    shape = shape_settings(body["velocity_shape"]) if "velocity_shape" in body else None
    tolerance = settings['tolerance_mm']; angle_tolerance = settings['orientation_tolerance_deg']
    acceptance = lambda q: cursor_metrics(q, cursor, tolerance, angle_tolerance, region)
    if not service._lock.acquire(blocking=False):
        raise service.PlannerBusy('Classical tracker busy; retry the latest cursor')
    try:
        status = verified_status()
        from .env import ArmEnv
        from .endpoint_ik import IKSearchExhausted
        from .smoothing import prepare_tracking
        if _env is None or _env.closed:
            _env = ArmEnv()
        env = _env
        if env.metadata() != status['environment']:
            env.close(); _env = None
            raise RuntimeError('Classical tracking scene differs from evaluated geometry')
        if box is not None and abs(box[2]-box[5]-env.table_top) > 1e-6:
            raise ValueError('Block must rest on the table (top −35 mm)')
        env.set_obstacle(box); env.reset(start)
        result = {'method': 'learned_rrt', 'planner': 'learned_rrt', 'task': 'pose5',
            'mouse_aim': True, 'simulation_only': True, 'tracking_strategy': 'learned_ik_rrt',
            'tracking_chunk': True, 'tracking_source': 'learned_ik_rrt',
            'playable': False, 'converged': False, 'frames': [], 'replan_required': True,
            'validation': 'joint_limits_self_table_block', 'coordinate_frame': 'g_base',
            'tool_frame': ARM.tcp, 'gap_mm': region['distance_mm'] if region else 50., 'cursor_mm': cursor.tolist(),
            'start_deg': start_deg.tolist(), 'target_mm': (target*1000).tolist(),
            'target_axis': axis.tolist(), 'obstacle_mm': None if box is None else (box*1000).tolist(),
            'scene_mode': 'no_block' if box is None else 'block',
            'collision_checks': ['joint_limits', 'self', 'table'] + ([] if box is None else ['block']),
            'checkpoint_sha256': status['checkpoint_sha256'], 'environment': status['environment'],
            'continuous_motion': continuous, 'speed_rad_s': speed, 'acceleration_rad_s2': 3.,
            'cursor_tolerance_mm': tolerance,
            'cursor_direction_tolerance_deg': angle_tolerance if orientation_mode=='point_at_object' else None,
            'orientation_mode': 'point_at_object' if orientation_mode=='strict_then_fov' else orientation_mode,
            'configuration': {'seed': config['seed'], 'learned_ik': True, 'ik_attempts': 32,
                              'robust_ik': True, 'region_candidate_budget': 8 if region else 1,
                              'region_ik_attempts_per_candidate': 4 if region else 32, 'rrt': asdict(RRTConfig())}}
        if region is not None:
            result['tracking_region'] = region
            result['tracking_objective'] = 'camera_visible_joint_travel' if orientation_mode=='camera_fov' else 'gaze_first'
        def finish(code, reason):
            result.update(status=code, reason=reason, server_compute_ms=(perf_counter()-began)*1000)
            return result
        if not env.valid(start):
            return finish('invalid_start', 'Starting joints fail self/table/block checks; pose held')
        result['physical_reach_precheck'] = target_precheck(target, region, tolerance)
        result['region_candidates_filtered'] = 0
        result['region_candidates_tested'] = 0
        if not result['physical_reach_precheck']['possible']:
            return finish('outside_workspace', 'EEF goal region is outside the URDF physical reach bound; pose held')
        if region is not None:
            from .region_tracking import plan_region
            route_planner = rrt_plan
            if callable(body.get('_rrt_observer')):
                from functools import partial
                route_planner = partial(rrt_plan, observer=body['_rrt_observer'])
            from .orientation_policy import plan_with_orientation
            camera = body.get('_visibility_camera')
            if orientation_mode=='strict_then_fov' and camera is None:
                from ..camera_model import configured_camera
                camera = configured_camera()
            poses = list(candidate_poses(region,start))[:8]
            def attempt(mode):
                cancel_check = body.get('_cancel_check')
                if cancel_check is not None:
                    cancel_check()
                return plan_region(env, start, cursor, region, config['seed'],
                    poses, speed, velocity, acceleration, continuous, shape,
                    tolerance, angle_tolerance, route_planner,
                    candidate_validator=body.get('_candidate_validator'),camera=camera,
                    orientation_mode=mode,cancel_check=cancel_check)
            continuation = plan_with_orientation(attempt,orientation_mode)
            result.update(continuation)
            effective_mode=result['orientation_mode']
            result['cursor_direction_tolerance_deg']=angle_tolerance if effective_mode=='point_at_object' else None
            result['tracking_objective']='camera_visible_joint_travel' if effective_mode=='camera_fov' else 'gaze_first'
            result['configuration'].update(ik_attempts=64, robust_ik=True,
                endpoint_selection='candidate_limited_visible', alternate_seed_pass=True)
            if verified_status()['checkpoint_sha256'] != status['checkpoint_sha256']:
                raise RuntimeError('Learned IK checkpoint changed during tracking planning')
            return finish(result['status'], result['reason'])
        ik_began = perf_counter()
        candidates = list(candidate_poses(region, start))[:8] if region else [(target, axis)]
        goal = None
        path = None
        route_attempts = []
        route_elapsed_ms = 0.
        if region is not None and acceptance(start)['accepted']:
            goal = start.copy()
            target = ARM.fk(start)
            axis = pose5.unit_axis(cursor-target*1000)
            route_began = perf_counter()
            path, diagnostics = rrt_plan(env, start, goal, seed=config['seed'], config=RRTConfig())
            route_elapsed_ms += (perf_counter()-route_began)*1000
        for candidate_target, candidate_axis in ([] if goal is not None else candidates):
            if not ENVELOPE.point(candidate_target*1000, min(1., tolerance))['possible']:
                result['region_candidates_filtered'] += 1
                continue
            result['region_candidates_tested'] += 1
            try:
                candidate = env.inverse_kinematics(candidate_target, candidate_axis,
                    current=start, seed=config['seed'], attempts=4 if region else 32,
                    robust=(region is None or result['region_candidates_tested'] == 1), learned=True)
            except IKSearchExhausted:
                continue
            result['ik_diagnostics'] = getattr(env, 'ik_diagnostics', {})
            error = service.errors(candidate, candidate_target, candidate_axis)
            if (not acceptance(candidate)['accepted'] or error['error_mm'] > min(1., tolerance)
                    or error['orientation_error_deg'] > min(1., angle_tolerance)):
                refined = pose5.solve(candidate, candidate_target, target_axis=candidate_axis,
                    # Numerical IK iterations have a separate minimum step
                    # speed; the checked trajectory retains the hardware cap.
                    method='bounded_dls', steps=240, speed=max(.1, speed), damping=.04,
                    tolerance=min(.25, tolerance/2), orientation_tolerance_deg=min(.25, angle_tolerance/2),
                    goal_constraint=lambda q: env.valid(q) and acceptance(q)['accepted'])
                candidate = np.deg2rad(refined['frames'][-1]['q'])
                result['cursor_refinement'] = True
            error = service.errors(candidate, candidate_target, candidate_axis)
            if (env.valid(candidate) and acceptance(candidate)['accepted']
                    and error['error_mm'] <= min(1., tolerance)
                    and error['orientation_error_deg'] <= min(1., angle_tolerance)):
                route_began = perf_counter()
                candidate_path, diagnostics = rrt_plan(env, start, candidate, seed=config['seed'], config=RRTConfig())
                route_elapsed_ms += (perf_counter()-route_began)*1000
                route_attempts.append({'target_mm': (candidate_target*1000).tolist(),
                    'status': diagnostics['status'], 'joint_travel_rad': float(np.linalg.norm(candidate-start))})
                if candidate_path is None:
                    continue
                goal, target, axis, path = candidate, candidate_target, candidate_axis, candidate_path
                break
        result['region_route_attempts'] = route_attempts
        result['ik_elapsed_ms'] = (perf_counter()-ik_began)*1000-route_elapsed_ms
        if goal is None:
            result['ik_diagnostics'] = (getattr(env, 'ik_diagnostics', {}) if result['region_candidates_tested'] else
                {'status': 'not_attempted', 'reason': 'Bounded candidate set is outside the physical reach bound'})
            return finish('no_path' if route_attempts else 'ik_failed', 'No checked cursor route within the bounded disk/IK search')
        result.update(target_mm=(target*1000).tolist(), target_axis=axis.tolist())
        result.update(goal_deg=np.rad2deg(goal).tolist(), goal_cursor_acceptance=acceptance(goal),
                      ik_outcome='accepted_goals')
        result['rrt_diagnostics'] = diagnostics
        if path is None:
            return finish(diagnostics['status'], 'No checked direct/RRT* route within the search budget')
        path_began = perf_counter()
        try:
            if continuous:
                if shape is None:
                    states, times, smoothing = continuous_prepare(env, path, velocity, acceleration, speed)
                else:
                    from ..profile_tracking import prepare as profile_prepare
                    try:
                        states, times, smoothing = profile_prepare(env, path, velocity, acceleration, speed, velocity_shape=shape)
                    except ValueError:
                        # A slow hardware startup can exceed the bounded curve
                        # duration even when IK and the route are valid. From
                        # rest, execute a checked prefix and replan from feedback.
                        # Moving joins must retain their owned q/v/a boundary.
                        if np.linalg.norm(velocity) >= 1e-12 or np.linalg.norm(acceleration) >= 1e-12:
                            raise
                        from ..continuous import DURATION_LIMIT
                        from ..profile_tracking import prepare_tracking as profile_prepare_tracking
                        states, times, smoothing = profile_prepare_tracking(env, path, speed,
                            velocity_shape=shape, duration_limit=DURATION_LIMIT)
                        if len(path) > 1 and np.linalg.norm(states[-1]-start) <= 1e-12:
                            raise ValueError('No checked shared-profile startup progress')
            else:
                if shape is None:
                    states, times, smoothing = prepare_tracking(env, path, speed=speed)
                else:
                    from ..profile_tracking import prepare_tracking as profile_prepare_tracking
                    states, times, smoothing = profile_prepare_tracking(env, path, speed=speed, velocity_shape=shape)
        except ValueError as exc:
            return finish('smoothing_validation_failed', str(exc))
        if not np.array_equal(states[0], start) or not env.path_valid(states):
            return finish('final_validation_failed', 'Timed route fails start or scene validation')
        if verified_status()['checkpoint_sha256'] != status['checkpoint_sha256']:
            raise RuntimeError('Learned IK checkpoint changed during tracking planning')
        accepted = acceptance(states[-1])
        result.update(playable=True, converged=accepted['accepted'], replan_required=not accepted['accepted'],
            frames=frames_for(states, times, target, axis), smoothing=smoothing, dt=.02,
            cursor_acceptance=accepted, planning_outcome=diagnostics['status'],
            max_velocity_rad_s=smoothing['max_velocity_rad_s'],
            max_acceleration_rad_s2=smoothing['max_acceleration_rad_s2'],
            path_elapsed_ms=(perf_counter()-path_began)*1000+route_elapsed_ms,
            **{f'final_{key}': value for key, value in service.errors(states[-1], target, axis).items()})
        result['gaze_tracking'] = annotate_gaze(result['frames'], cursor, region)
        return finish('target_reached' if accepted['accepted'] else 'chunk_ready', 'Checked learned IK + classical route')
    finally:
        service._lock.release()
