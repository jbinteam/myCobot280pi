"""Fixed-target learned IK and checked direct/RRT* planning; no motor IO."""
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import numpy as np
from . import service
from .common import robot_schema
from .ik_seed import checkpoint_status
from .planner import RRTConfig, plan as rrt_plan
from ..physical_reach import target_precheck

IK_SHA256 = '168ab43a2d4d4dcc0e61acf8c3c89e6dff6e9723843b1fd023a4519108ea04f5'
SCENE = Path(__file__).resolve().parents[1] / 'artifacts/diffusion_motion_v2/webui.json'


def info():
    status = checkpoint_status()
    result = {'status': 'unavailable', 'model_name': 'learned_ik_rrt',
              'planner': 'learned_rrt', 'simulation_only': True}
    try:
        if status['status'] != 'ready' or status.get('checkpoint_sha256') != IK_SHA256:
            raise ValueError('The evaluated learned IK checkpoint is missing or changed')
        environment = json.loads(SCENE.read_text())['environment']
        if environment['robot'] != robot_schema():
            raise ValueError('Classical planning scene differs from evaluated robot geometry')
        result.update(status='ready', checkpoint_sha256=IK_SHA256, ik_seed=status,
                      environment=environment, defaults={'seed': 42, 'ik_attempts': 32,
                      'robust_ik': True, 'rrt': asdict(RRTConfig())})
    except (OSError, ValueError, KeyError) as exc:
        result['reason'] = str(exc)
    return result


def plan(body):
    if not service._lock.acquire(blocking=False):
        raise service.PlannerBusy('Fixed-target planner busy; wait for the current preview')
    began = perf_counter()
    try:
        status = info()
        if status['status'] != 'ready': raise RuntimeError(status['reason'])
        start, target, axis, box, goal, config = service.request(body)
        from .env import ArmEnv
        from .endpoint_ik import IKSearchExhausted
        from .smoothing import prepare
        from ..trajectory_math import frame_transforms
        with ArmEnv() as env:
            if env.metadata() != status['environment']:
                raise RuntimeError('Classical scene/checker differs from evaluated scene')
            if box is not None and abs(box[2]-box[5]-env.table_top) > 1e-6:
                raise ValueError('Block must rest on the table (top −35 mm)')
            env.set_obstacle(box); env.reset(start)
            result = {'method': 'learned_rrt', 'planner': 'learned_rrt', 'task': 'pose5',
                'simulation_only': True, 'validation': 'joint_limits_self_table_block',
                'converged': False, 'frames': [], 'start_deg': np.rad2deg(start).tolist(),
                'target_mm': (target*1000).tolist(), 'target_axis': axis.tolist(),
                'obstacle_mm': None if box is None else (box*1000).tolist(),
                'scene_mode': 'no_block' if box is None else 'block',
                'collision_checks': ['joint_limits', 'self', 'table'] + ([] if box is None else ['block']),
                'checkpoint_sha256': IK_SHA256, 'environment': status['environment'],
                'configuration': {'seed': config['seed'], 'ik_attempts': 32, 'robust_ik': True,
                                  'learned_ik': True, 'rrt': asdict(RRTConfig())},
                'dt': .02, 'speed_rad_s': 1., 'acceleration_rad_s2': 3.}
            def fail(code, reason):
                result.update(status=code, reason=reason, elapsed_ms=(perf_counter()-began)*1000)
                return result
            if not env.valid(start): return fail('invalid_start', 'Starting pose fails joint/self/table/block checks')
            result['physical_reach_precheck'] = target_precheck(target, allowance_mm=1.)
            if not result['physical_reach_precheck']['possible']:
                return fail('outside_workspace', 'EEF target is outside the URDF physical reach bound')
            ik_began = perf_counter()
            if goal is None:
                try:
                    goal = env.inverse_kinematics(target, axis, current=start,
                        seed=config['seed'], attempts=32, robust=True, learned=True)
                except IKSearchExhausted:
                    result['ik_diagnostics'] = env.ik_diagnostics
                    return fail('ik_failed', 'No collision-free IK endpoint found within the search budget')
                result['ik_diagnostics'] = getattr(env, 'ik_diagnostics', {})
            result['ik_elapsed_ms'] = (perf_counter()-ik_began)*1000
            endpoint = service.errors(goal, target, axis)
            if not env.valid(goal) or endpoint['error_mm'] > 1. or endpoint['orientation_error_deg'] > 1.:
                return fail('final_validation_failed', 'IK endpoint fails strict pose or collision checks')
            result['goal_deg'] = np.rad2deg(goal).tolist()
            path_began = perf_counter()
            points, diagnostics = rrt_plan(env, start, goal, seed=config['seed'], config=RRTConfig())
            result['rrt_diagnostics'] = diagnostics
            if points is None: return fail(diagnostics['status'], 'No checked path found within the RRT* budget')
            if not env.path_valid(points): return fail('final_validation_failed', 'Complete path rejected')
            try: points, times, smoothing = prepare(env, points)
            except ValueError as exc: return fail('smoothing_validation_failed', str(exc))
            final = service.errors(points[-1], target, axis)
            if (not np.array_equal(points[0], start) or final['error_mm'] > 1.
                    or final['orientation_error_deg'] > 1. or not env.path_valid(points)):
                return fail('final_validation_failed', 'Timed path fails endpoint/start/collision checks')
            if info().get('checkpoint_sha256') != IK_SHA256:
                raise RuntimeError('Learned IK checkpoint changed during planning')
            frames = [{'q': np.rad2deg(q).tolist(), 'time': float(t),
                'position': (transform[:3, 3]*1000).tolist(), 'tool_axis': transform[:3, 2].tolist(),
                **service.errors(q, target, axis, transform=transform)}
                for q, t, transform in zip(points, times, frame_transforms(points))]
            result.update(status='success', reason='Complete checked classical path', converged=True,
                frames=frames, smoothing=smoothing, iterations=len(frames)-1,
                stop_reason='checked_smoothed_classical_path',
                elapsed_ms=(perf_counter()-began)*1000, path_elapsed_ms=(perf_counter()-path_began)*1000,
                final_error_mm=final['error_mm'], final_orientation_error_deg=final['orientation_error_deg'],
                joint_travel_deg=float(np.rad2deg(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())),
                max_velocity_rad_s=smoothing['max_velocity_rad_s'],
                max_acceleration_rad_s2=smoothing['max_acceleration_rad_s2'])
            return result
    finally:
        service._lock.release()
