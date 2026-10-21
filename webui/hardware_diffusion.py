"""Pinned v2 fixed-target plans for telemetry-paced hardware trials.

Planning sends no motor commands. The hardware executor follows checked straight
joint segments, stopping at corners and the endpoint. Normalized preview timing
paces waypoint sends; physical servo interpolation remains firmware controlled.
"""
import numpy as np
from time import perf_counter
from simulation.kinematics import ARM
from simulation.pose5 import xy_from_axis
from simulation.trajectory import joint_waypoints
from simulation.diffusion import service
from simulation.diffusion.motion_conditioning import CONDITIONING

V2_SHA256 = '6582cbf4e1d2575007cfead8f99ed5f15a7b7021083d0eb200c61b224401bddc'
from webui.motion_settings import MAX_SDK_SPEED, FIXED_V2_JOINT_COMMAND
from webui.fixed_profile import selected_speed, retime_fixed, waypoint_intervals
from simulation.reference_shape import settings as shape_settings
MAX_STEP_DEG = 2.
# Match the continuous v2 hardware executor's stopped-joint allowance for
# servo deadband. Final Cartesian acceptance is checked separately by HardwareIK.
ARRIVAL_DEG = 1.5
PATH_ARRIVAL_DEG = 3.
MAX_LIVE_COMMAND_DEG = 4.
COLLISION_STEP_RAD = .01


def verified_status():
    status = service.info()
    if (status.get('status') != 'ready' or status.get('model_name') != 'diffusion_motion_v2'
            or status.get('checkpoint_sha256') != V2_SHA256
            or status.get('conditioning') != CONDITIONING):
        raise RuntimeError('Hardware diffusion requires the pinned diffusion_motion_v2 checkpoint and schema')
    return status


def prepare(start_deg, body):
    began = perf_counter()
    allowed = {'target_mm', 'eef_offset_mm', 'target_axis', 'seed', 'samples',
               'inference_steps', 'max_decisions', 'speed', 'joint_command', 'velocity_shape', 'planner', 'goal_deg'}
    if set(body)-allowed:
        raise ValueError('Hardware diffusion accepts target/settings only; starting joints are read live. First trials use no block')
    planner = body.get('planner', 'diffusion_v2')
    if planner not in ('diffusion_v2', 'learned_rrt'): raise ValueError('Choose learned_rrt or diffusion_v2 fixed-target planning')
    if 'goal_deg' in body and planner != 'learned_rrt':
        raise ValueError('Explicit joint goals require the checked classical RRT* planner')
    from simulation.diffusion import classical_service
    engine = classical_service if planner == 'learned_rrt' else service
    status = classical_service.info() if planner == 'learned_rrt' else verified_status()
    if status.get('status') != 'ready': raise RuntimeError(status.get('reason', 'Fixed-target planner unavailable'))
    speed = selected_speed(body, 5)
    velocity_shape = shape_settings(body.get('velocity_shape'))
    if velocity_shape['kind'] != 'quintic' and body.get('joint_command', FIXED_V2_JOINT_COMMAND) != 'encoder_drag_profile':
        raise ValueError('Adjustable reference velocity shapes require v2 velocity-profile mode')
    if body.get('joint_command', FIXED_V2_JOINT_COMMAND) not in ('send_angles', 'encoder_drag', 'encoder_drag_stream', 'encoder_drag_staged', 'encoder_drag_profile'):
        raise ValueError('Choose joint angle targets, encoder waypoints or encoder streaming')
    start = np.deg2rad(service.vector(start_deg, 6, 'live starting joints'))
    rotation = ARM.transform(start)[:3, :3]
    base = service.vector(body.get('target_mm'), 3, 'target_mm')
    offset = service.vector(body.get('eef_offset_mm', [0, 0, 0]), 3, 'eef_offset_mm')
    target = base+rotation@offset
    request = {k: v for k, v in body.items() if k not in ('eef_offset_mm','speed','joint_command','velocity_shape','planner')}
    generation_began = perf_counter()
    result = engine.plan({**request, 'start_deg': start_deg,
                           'target_mm': target.tolist(), 'obstacle_mm': None})
    generation_done = perf_counter()
    if (result.get('checkpoint_sha256') != status['checkpoint_sha256']
            or result.get('environment') != status['environment']
            or (planner == 'diffusion_v2' and result.get('motion_conditioning', {}).get('schema') != CONDITIONING['schema'])):
        raise RuntimeError('Diffusion deployment changed while planning; plan again')
    waypoints, stop_indices = [], set()
    if result.get('converged'):
        if result.get('validation') != 'joint_limits_self_table_block' or result.get('obstacle_mm') is not None:
            raise ValueError('Hardware diffusion needs the checked no-block scene')
        frames = result.get('frames', [])
        if not frames:
            raise ValueError('Converged diffusion plan has no checked frames')
        states = np.asarray([service.vector(f['q'], 6, 'planned joints') for f in frames])
        if not np.allclose(states[0], start_deg, atol=1e-8, rtol=0):
            raise ValueError('Diffusion path differs from the captured live start')
        if 'goal_deg' in body and not np.allclose(states[-1], service.vector(body['goal_deg'], 6, 'goal_deg'), atol=1e-7, rtol=0):
            raise ValueError('Checked joint path changed the requested endpoint')
        # Quintics are monotone on these straight legs. Preserve every checked
        # corner, then independently check the executor's straight segments.
        segments = result.get('smoothing', {}).get('segments')
        if not isinstance(segments, list):
            raise ValueError('Diffusion plan is missing its checked corner path')
        corners = [start_deg]
        for segment in segments:
            if not np.allclose(segment['start_deg'], corners[-1], atol=1e-8, rtol=0):
                raise ValueError('Diffusion corner path is discontinuous')
            corners.append(service.vector(segment['end_deg'], 6, 'checked corner').tolist())
        if not np.allclose(corners[-1], states[-1], atol=1e-8, rtol=0):
            raise ValueError('Diffusion corner path changed the endpoint')
        for previous, corner in zip(corners, corners[1:]):
            waypoints.extend(joint_waypoints(previous, [corner], MAX_STEP_DEG, compress=False))
            stop_indices.add(len(waypoints)-1)
        if len(waypoints) > 200 and body.get('joint_command', FIXED_V2_JOINT_COMMAND) not in ('encoder_drag_staged', 'encoder_drag_profile'):
            raise ValueError('Hardware trial exceeds 200 waypoints; choose a closer target')
        from simulation.diffusion.env import ArmEnv
        with ArmEnv() as env:
            env.set_obstacle(None)
            if env.metadata() != status['environment']:
                raise RuntimeError('Hardware collision scene differs from the pinned v2 scene')
            if (not env.path_valid(np.deg2rad(states), COLLISION_STEP_RAD)
                    or not env.path_valid(np.deg2rad([start_deg, *waypoints]), COLLISION_STEP_RAD)):
                raise ValueError('Hardware waypoint path fails self/table/limit collision validation')
        error = service.errors(np.deg2rad(states[-1]), target/1000, np.asarray(result['target_axis']))
        if error['error_mm'] > 1. or error['orientation_error_deg'] > 1.:
            raise ValueError('Hardware diffusion endpoint misses the strict Cartesian target')
    result = {**result, 'simulation_only': False, 'hardware_diffusion': True,
              'hardware_classical': planner == 'learned_rrt', 'planner': planner,
              'orientation_xy_deg': xy_from_axis(np.asarray(result['target_axis'])),
              'limitations': 'Relaxed interior tracking; stopped corners and endpoint. Preview speed/acceleration do not measure servo dynamics. '
                             'URDF self/table checks; no external obstacles or added tool geometry.'}
    validation_done = perf_counter()
    if result.get('converged'):
        result=retime_fixed(result,speed,path=np.deg2rad(corners),velocity_shape=velocity_shape)
        # Retiming follows the same monotone straight legs validated above.
        result['smoothing']['collision_rechecked']=True
    intervals = waypoint_intervals(result,waypoints) if result.get('converged') else []
    done = perf_counter()
    result['v2_prepare_timing_ms'] = {
        'deployment_and_request': (generation_began-began)*1000,
        'trajectory_generation': (generation_done-generation_began)*1000,
        'hardware_route_validation': (validation_done-generation_done)*1000,
        'velocity_retiming': (done-validation_done)*1000,
        'total': (done-began)*1000}
    return {'result': result, 'waypoints': waypoints, 'stop_indices': stop_indices, 'target_mm': target.tolist(),
            'eef_rotation': rotation.tolist(), 'environment': status['environment'],
            'waypoint_intervals_s':intervals}
