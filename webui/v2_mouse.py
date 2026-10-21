"""Server-owned continuous C2 mouse curves with fresh hardware feedback.

Diffusion v2 and learned IK + classical RRT* use feedback-paced encoder tracking.

The planner and curve evaluator are shared with simulation. No client joint
arrays are accepted. Replacements anchor to fresh live joints, retain measured
motion where available, and require a bounded, collision-checked connection.
Trajectory lag is diagnostic; each command retains collision and joint limits.
"""
import copy
import secrets
import threading
import time
import numpy as np
from simulation.kinematics import ARM
from simulation.camera_model import configured_camera
from simulation.continuous import evaluate, bound, METHOD
from simulation import profile_tracking
from simulation.reference_shape import settings as shape_settings
from simulation.diffusion import tracking, classical_tracking
from simulation.diffusion.classical_service import IK_SHA256
from simulation.diffusion.service import vector
from webui.hardware_diffusion import verified_status, V2_SHA256
from webui.mouse_motion import StaleMousePlan
from webui import encoder_profile, encoder_drag
from webui.robot_transport import reply_timeout
from webui.motion_settings import MAX_SDK_SPEED
from simulation.motion_limits import curve_envelope, scale_needed, profile
from webui.tracking_scene import scene_for, step_for, set_candidate_object, region_acceptance
from simulation.diffusion.orientation_policy import effective_orientation_mode

ARRIVAL_DEG = encoder_profile.ARRIVAL_DEG
PLAN_TTL = 60.
JOIN_TOL = 1e-8


def motion(curve, phase):
    if curve.get('method') == profile_tracking.METHOD:
        return profile_tracking.sample_motion(curve, phase)
    if not curve['segments']:
        return np.deg2rad(curve['hold_deg']), np.zeros(6), np.zeros(6)
    segment = next((s for s in curve['segments']
                    if phase <= s['start_time_s']+s['duration_s']+1e-9), curve['segments'][-1])
    controls = np.deg2rad(segment['controls_deg'])
    duration = segment['duration_s']
    u = np.clip((phase-segment['start_time_s'])/duration, 0., 1.)
    velocity = 5*np.diff(controls, axis=0)/duration
    acceleration = 4*np.diff(velocity, axis=0)/duration
    # Integrate the quartic velocity exactly from its checked segment anchor.
    # Stateless reconstruction preserves the checked boundary motion without
    # accumulating telemetry-cycle integration drift.
    nodes = u*np.array([.5-np.sqrt(3/5)/2, .5, .5+np.sqrt(3/5)/2])
    q = controls[0]+duration*u*(np.array([5/18, 4/9, 5/18])@evaluate(velocity, nodes))
    return q, evaluate(velocity, u), evaluate(acceleration, u)


def same_motion(a, b):
    return all(np.allclose(x, y, atol=JOIN_TOL, rtol=0) for x, y in zip(a, b))


def validate_curve(result, expected, speed, environment, engine="diffusion_v2", visibility_object_m=None,
                   visibility_camera=None):
    scene_settings = {'visibility_object_m': visibility_object_m,
                      'visibility_camera': visibility_camera if visibility_camera is not None
                      else configured_camera().configuration() if visibility_object_m is not None else None}
    checkpoint = IK_SHA256 if engine == "learned_rrt" else V2_SHA256
    if engine == "learned_rrt" and (result.get("method") != "learned_rrt" or result.get("tracking_strategy") != "learned_ik_rrt"):
        raise ValueError("Classical mouse requires a checked learned IK + RRT* result")
    """Independently recheck exact curve bounds, q/v/a joins and collision scene."""
    if (result.get('checkpoint_sha256') != checkpoint
            or result.get('validation') != 'joint_limits_self_table_block'
            or result.get('continuous_motion') is not True or result.get('obstacle_mm') is not None
            or result.get('playable') is not True or not result.get('frames')):
        raise ValueError('v2 mouse execution requires a pinned, playable continuous no-block path')
    curve = copy.deepcopy(result.get('smoothing', {}))
    if curve.get('method') == profile_tracking.METHOD:
        with scene_for(scene_settings) as env:
            env.set_obstacle(None)
            if env.metadata() != environment:
                raise ValueError('Shared-profile scene differs from the pinned geometry')
            profile_tracking.validate(curve, result['frames'], expected, speed, env)
            if not env.path_valid(np.deg2rad([f['q'] for f in result['frames']]), step_for(env)):
                raise ValueError('Shared-profile curve fails scene/object visibility checks')
        return curve
    if curve.get('method') != METHOD or not isinstance(curve.get('segments'), list):
        raise ValueError('Missing checked C2 curve')
    start = vector(result['frames'][0]['q'], 6, 'curve start')
    curve['hold_deg'] = start.tolist()
    if not same_motion(motion(curve, 0.), expected):
        raise ValueError('v2 planner changed the server-owned join motion')
    previous = expected
    elapsed = 0.; checked = [expected[0]]
    for segment in curve['segments']:
        controls = np.deg2rad(np.asarray(segment.get('controls_deg'), float))
        duration = segment.get('duration_s')
        if (controls.shape != (6, 6) or not np.isfinite(controls).all()
                or not isinstance(duration, (int, float)) or not np.isfinite(duration) or duration <= 0
                or abs(segment['start_time_s']-elapsed) > JOIN_TOL):
            raise ValueError('Invalid C2 controls or timing')
        v = 5*np.diff(controls, axis=0)/duration
        a = 20*np.diff(controls, n=2, axis=0)/duration**2
        if (np.any(controls < ARM.lower) or np.any(controls > ARM.upper)
                or bound(v) > speed+1e-8 or bound(a) > 3.+1e-8
                or scale_needed(curve_envelope(controls, duration)) > 1.+1e-8
                or not same_motion((controls[0], v[0], a[0]), previous)):
            raise ValueError('C2 join, joint or derivative bounds failed')
        from simulation.continuous import evaluate
        checked.extend(evaluate(controls, np.linspace(0, 1, int(np.ceil(duration/.01))+1)))
        previous = (controls[-1], v[-1], a[-1]); elapsed += duration
    if elapsed > 12.+JOIN_TOL or abs(curve.get('duration_s', -1)-elapsed) > JOIN_TOL:
        raise ValueError('Invalid C2 duration')
    if not np.allclose(previous[1:], np.zeros((2, 6)), atol=1e-8, rtol=0):
        raise ValueError('v2 path has no resting braking tail')
    if not np.allclose(previous[0], np.deg2rad(result['frames'][-1]['q']), atol=JOIN_TOL, rtol=0):
        raise ValueError('v2 curve endpoint differs from the preview')
    frames = result['frames']
    if (len(frames) > 601 or abs(frames[-1]['time']-elapsed) > JOIN_TOL
            or curve.get('frames') != len(frames) or curve.get('dt') != .02):
        raise ValueError('v2 curve and preview timing differ')
    for index, frame in enumerate(frames):
        if (abs(frame['time']-index*.02) > JOIN_TOL
                or not np.allclose(np.deg2rad(vector(frame['q'], 6, 'preview joints')),
                                   motion(curve, frame['time'])[0], atol=JOIN_TOL, rtol=0)):
            raise ValueError('v2 curve differs from a preview frame')
    with scene_for(scene_settings) as env:
        env.set_obstacle(None)
        if env.metadata() != environment or not env.path_valid(np.asarray(checked), step_for(env)):
            raise ValueError('v2 hardware C2 curve fails pinned scene/object visibility checks')
    return curve


class V2Mouse:
    def __init__(self, ik, speed, status, reference_speed=1., engine="diffusion_v2", velocity_shape=None, tracking_region=None, approach=None, orientation_mode='point_at_object'):
        self.velocity_shape = shape_settings(velocity_shape)
        if engine not in ("diffusion_v2", "learned_rrt"):
            raise ValueError("Unknown continuous tracking engine")
        self.engine = engine
        self.checkpoint_sha256 = IK_SHA256 if engine == "learned_rrt" else V2_SHA256
        self.ik = ik
        self.speed = speed
        # A vector limit also bounds each joint; firmware speed is a separate cap.
        self.speed_rad_s = min(reference_speed, np.deg2rad(20.*speed/30.))
        self.environment = status['environment']
        self.curve = None
        self.phase = 0.
        self.active_id = None
        self.last_compute = .2
        self.last_target = None
        self.encoder_mapping = None
        self.clock_origin = None
        self.feedback_hold_until = 0.
        self.live_q = None
        self.live_at = None
        self.live_velocity = np.zeros(6)
        self.region_settings = None
        if tracking_region is not None:
            from simulation.tracking_geometry import region_for
            canonical = region_for(np.zeros(3), np.array([0., 0., 1.]), tracking_region)
            self.region_settings = {k: canonical[k] for k in ('mode', 'distance_mm', 'half_angle_deg')}
        self.visibility_camera = None
        if self.engine == 'learned_rrt' and self.region_settings is not None:
            self.visibility_camera = getattr(ik.controller, 'visibility_camera', None) or configured_camera()
        from simulation.tracking_geometry import approach_settings
        self.approach = approach_settings(approach, self.region_settings)
        if orientation_mode not in ('point_at_object', 'strict_then_fov'):
            raise ValueError('Hardware tracking requires strict pointing or strict-first camera fallback')
        if orientation_mode == 'strict_then_fov' and (self.engine != 'learned_rrt' or self.approach is None):
            raise ValueError('Camera-FOV fallback requires a learned-RRT approach session')
        self.orientation_mode = orientation_mode
        self.active_orientation_mode = 'point_at_object'
        self.proposed_region = None
        self.active_region = None
        self.stage_region = None

    def reference_phase(self):
        return self.phase

    def join_phase_now(self):
        return self.phase

    def feedback_hold(self, now):
        return self.ik.mouse_pending_plan is not None or now < self.feedback_hold_until

    def observe_live(self, degrees, now):
        q = np.deg2rad(degrees)
        if self.live_q is not None and .05 <= now-self.live_at <= .6:
            velocity = (q-self.live_q)/(now-self.live_at)
            length = np.linalg.norm(velocity)
            self.live_velocity = velocity*min(1., self.speed_rad_s/max(length, 1e-12))
        else:
            self.live_velocity = np.zeros(6)
        self.live_q, self.live_at = q.copy(), now

    def deployment_status(self):
        status = classical_tracking.verified_status() if self.engine == "learned_rrt" else verified_status()
        if status["checkpoint_sha256"] != self.checkpoint_sha256 or status["environment"] != self.environment:
            raise RuntimeError("Tracking deployment changed; start a new session")
        return status

    def prepare(self, body, observer=None):
        try:
            return self._prepare(body, observer)
        except Exception:
            with self.ik.controller.lock:
                self.feedback_hold_until = 0.
            raise

    def _prepare(self, body, observer=None):
        allowed = {'session_id', 'cursor_mm', 'target_axis', 'pose5_method', 'steps',
                   'speed_rad_s', 'tolerance_mm', 'orientation_tolerance_deg', 'damping',
                   'seed', 'samples', 'inference_steps', 'tracking_engine', 'obstacle_mm',
                   'guided_route_repair', 'route_repair_samples', 'velocity_shape', 'tracking_region'}
        if set(body)-allowed:
            raise ValueError('v2 mouse accepts cursor/settings only; reference joints and derivatives are owned by the server')
        if body.get('obstacle_mm') is not None:
            raise ValueError('First v2 mouse hardware trials use no block')
        for name, limit in [('tolerance_mm', 5.), ('orientation_tolerance_deg', 3.)]:
            value = body.get(name, 1.)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not .1 <= value <= limit:
                raise ValueError(f'Hardware {name} must be 0.1–{limit}')
        requested_shape = shape_settings(body.get('velocity_shape')) if 'velocity_shape' in body else self.velocity_shape
        if requested_shape != self.velocity_shape:
            raise ValueError('Velocity profile changed; stop and restart mouse tracking')
        cursor = vector(body.get('cursor_mm'), 3, 'cursor_mm')
        from simulation.tracking_geometry import region_for
        requested = self.region_settings
        if 'tracking_region' in body:
            requested = region_for(np.zeros(3), np.array([0., 0., 1.]), body['tracking_region'])
            requested = {k: requested[k] for k in ('mode', 'distance_mm', 'half_angle_deg')} if requested is not None else None
        ik = self.ik; c = ik.controller
        began = time.monotonic()
        with c.lock:
            ik._mouse_session(body.get('session_id'))
            if ik.v2_mouse is not self or not c.robot:
                raise StaleMousePlan('v2 mouse session changed')
            if requested != self.region_settings:
                current = self.region_settings
                if (self.approach is None or requested is None or current is None
                        or requested['mode'] != current['mode']
                        or requested['half_angle_deg'] != current['half_angle_deg']):
                    raise ValueError('Tracking region changed; stop and restart mouse tracking')
                expected_distance = max(self.approach['final_distance_mm'],
                                        current['distance_mm']-self.approach['step_mm'])
                if (current['distance_mm'] <= self.approach['final_distance_mm']
                        or abs(requested['distance_mm']-expected_distance) > 1e-9):
                    raise ValueError('Approach must use the next bounded distance stage')
                completed_region = ik.status.get('tracking_region') or {}
                if (ik.status.get('state') != 'completed' or c.motion or c.moving
                        or ik.status.get('final_goal_accepted') is not True
                        or ik.status.get('replan_required')
                        or completed_region.get('distance_mm') != current['distance_mm']
                        or np.linalg.norm(cursor-np.asarray(completed_region.get('cursor_mm', [np.inf]*3))) > 1e-9):
                    raise StaleMousePlan('Approach waits for the current target region to be reached')
            running = ik.owns_running_mouse(ik.mouse_session)
            # The executor already measures joints. Avoid duplicate serial
            # queries during replanning when that measured sample is fresh.
            if (not running or c.feedback_stamp_unix_s is None
                    or not 0 <= time.time()-c.feedback_stamp_unix_s <= .25):
                with reply_timeout(c.robot, encoder_profile.STREAM_REPLY_TIMEOUT_S,
                        budget_s=encoder_profile.STREAM_QUERY_BUDGET_S):
                    c._read()
            if requested != self.region_settings:
                completed_contract = {'tracking_region':completed_region,
                    'orientation_mode':self.active_orientation_mode,
                    'visibility_object_m':(cursor/1000).tolist(),
                    'visibility_camera':self.visibility_camera.configuration() if self.visibility_camera else None}
                if c.motion or c.moving or not region_acceptance(np.deg2rad(c.angles), completed_contract,
                        self.approach['distance_tolerance_mm'], 3.)['accepted']:
                    raise StaleMousePlan('Approach waits for measured arrival in the current region')
                if self.active_orientation_mode == 'camera_fov':
                    with scene_for(completed_contract) as view:
                        view.set_obstacle(None)
                        if not view.valid(np.deg2rad(c.angles)):
                            raise StaleMousePlan('Approach waits for measured camera visibility')
                # Authorize this stage once; failed planning retries it without
                # shrinking again. The active curve/region remains unchanged.
                self.region_settings = copy.deepcopy(requested)
            if (c.motion or c.moving) and not running:
                if ik.status.get('state') == 'paused' and not c.motion:
                    raise StaleMousePlan('Waiting for the paused arm to settle; retry fresh feedback')
                raise RuntimeError('Stop the arm before v2 mouse planning')
            live, generation = c.angles[:], c.generation
            ik.plan = None
            join = None
            if running and self.curve is not None:
                if ik.mouse_pending_plan is not None:
                    raise StaleMousePlan('A checked cursor replacement is still awaiting acknowledgement')
                # Stop advancing the old reference while planning from actual
                # feedback. Keep measured velocity to avoid repeated cursor
                # changes restarting the speed ramp at zero.
                velocity = self.live_velocity.copy() if self.live_at is not None and began-self.live_at <= .6 else np.zeros(6)
                expected = (np.deg2rad(live), velocity, np.zeros(6))
                join = {'curve_id': self.active_id, 'mode': 'live_feedback', 'phase': self.phase}
                self.feedback_hold_until = began+5.
            else:
                expected = (np.deg2rad(live), np.zeros(6), np.zeros(6))
            axis = vector(body['target_axis'], 3, 'target_axis') if 'target_axis' in body else cursor-ARM.fk(expected[0])*1000
            from simulation.pose5 import unit_axis
            axis = unit_axis(axis)
            region = None
            if self.region_settings is not None:
                # Capture the measured EEF-to-object line again at each new
                # distance stage. Keep that plane fixed for retries within
                # the stage; failed planning never changes the active region.
                candidates = [self.stage_region, self.proposed_region,
                              ik.mouse_pending_plan['plan'].get('tracking_region') if ik.mouse_pending_plan else None,
                              self.active_region]
                prior = next((r for r in candidates if r is not None
                              and r['mode'] == self.region_settings['mode']
                              and abs(r['distance_mm']-self.region_settings['distance_mm']) <= 1e-9
                              and abs(r['half_angle_deg']-self.region_settings['half_angle_deg']) <= 1e-9
                              and np.linalg.norm(cursor-np.asarray(r['cursor_mm'])) <= .5), None)
                axis = np.asarray(prior['axis']) if prior is not None else unit_axis(cursor-ARM.fk(np.deg2rad(live))*1000)
                region = region_for(cursor, axis, self.region_settings)
                self.stage_region = copy.deepcopy(region)
            target = np.asarray(region['center_mm'])/1000 if region else (cursor-50*axis)/1000
            target_velocity = np.zeros(3)
            if self.last_target is not None:
                dt = began-self.last_target[0]
                if .02 <= dt <= 2:
                    target_velocity = (target-self.last_target[1])/dt
                    length = np.linalg.norm(target_velocity)
                    if length > .5: target_velocity *= .5/length
            self.last_target = (began, target)
        read_done = time.monotonic()
        status = self.deployment_status()
        if status['environment'] != self.environment:
            raise RuntimeError('v2 scene changed; start a new session')
        request = {k: body[k] for k in ('pose5_method', 'steps', 'tolerance_mm',
                   'orientation_tolerance_deg', 'damping', 'seed', 'samples', 'inference_steps') if k in body}
        if self.velocity_shape is not None:
            request['velocity_shape'] = self.velocity_shape
        if region is not None:
            request['tracking_region'] = copy.deepcopy(self.region_settings)
        requested_speed = body.get('speed_rad_s', self.speed_rad_s)
        if isinstance(requested_speed, bool) or not isinstance(requested_speed, (int, float)) or not np.isfinite(requested_speed) or requested_speed <= 0:
            raise ValueError('Invalid v2 planning speed')
        if abs(min(requested_speed, 1., np.deg2rad(20.*self.speed/30.))-self.speed_rad_s) > JOIN_TOL:
            raise ValueError('Reference speed changed; stop and restart v2 mouse tracking')
        # Speed remains fixed during a session so a live reanchor cannot violate
        # newly lowered derivative limits. User changes stop and restart tracking.
        planner = classical_tracking if self.engine == "learned_rrt" else tracking
        if self.engine == 'learned_rrt' and region is not None:
            # This callable is installed in-process from server-owned state;
            # the public request allowlist never accepts it. Reject rounded
            # encoder paths while selecting endpoints, so another candidate
            # can be tried before the independent final hardware gates.
            mapping = copy.deepcopy(self.encoder_mapping)
            request['_visibility_camera'] = self.visibility_camera
            request['orientation_mode'] = self.orientation_mode
            def check_current_session():
                # GIL-protected identity/generation reads avoid taking the arm
                # lock at each numerical iteration or visibility sample.
                if (ik.v2_mouse is not self or c.generation != generation
                        or ik.mouse_session != body['session_id']):
                    raise StaleMousePlan('Tracking session changed; planning cancelled')
            request['_cancel_check'] = check_current_session
            def check_encoder_candidate(states, times, smoothing, deadline, mode):
                check_current_session()
                candidate = {'curve':copy.deepcopy(smoothing), 'environment':self.environment,
                    'tracking_region':region, 'visibility_object_m':(cursor/1000).tolist(),
                    'visibility_camera':self.visibility_camera.configuration(),
                    'orientation_mode':mode,
                    'result':{'frames':[{'q':np.rad2deg(q).tolist(),'time':float(t)} for q,t in zip(states,times)]}}
                candidate['result']['converged'] = region_acceptance(states[-1],candidate,
                    body.get('tolerance_mm',1.),body.get('orientation_tolerance_deg',1.))['accepted']
                encoder_profile.prepare_mouse(candidate,mapping,self.speed,deadline=deadline)
                check_current_session()
            request['_candidate_validator'] = check_encoder_candidate
            if callable(observer):
                request['_rrt_observer'] = observer
        planner_began = time.monotonic()
        result = planner.plan({**request, 'start_deg': np.rad2deg(expected[0]).tolist(),
            'cursor_mm': cursor.tolist(), 'target_axis': axis.tolist(), 'obstacle_mm': None,
            'speed_rad_s': self.speed_rad_s, 'continuous_motion': True,
            'start_velocity_rad_s': expected[1].tolist(), 'start_acceleration_rad_s2': expected[2].tolist(),
            'target_velocity_m_s': target_velocity.tolist()})
        planner_done = time.monotonic()
        if callable(request.get('_cancel_check')):
            request['_cancel_check']()
        playable = result.get('playable') is True
        mode = effective_orientation_mode(result, self.orientation_mode) if playable else result.get('orientation_mode', 'point_at_object')
        endpoint_contract = {'tracking_region':region, 'orientation_mode':mode,
            'visibility_object_m':(cursor/1000).tolist(),
            'visibility_camera':self.visibility_camera.configuration() if self.visibility_camera else None}
        if region is not None:
            # The planner owns selected poses, but cannot alter the frozen
            # region or claim convergence outside its checked acceptance area.
            returned_region = result.get('tracking_region')
            if (not isinstance(returned_region, dict) or set(returned_region) != set(region)
                    or returned_region.get('mode') != region['mode']
                    or any(not np.allclose(returned_region[k], region[k], atol=1e-8, rtol=0)
                           for k in region if k != 'mode')):
                raise ValueError('Planner changed the server-owned tracking region')
            if result.get('converged') and not region_acceptance(np.deg2rad(result['frames'][-1]['q']), endpoint_contract,
                    body.get('tolerance_mm', 1.), body.get('orientation_tolerance_deg', 1.))['accepted']:
                raise ValueError('Planner endpoint is outside the tracking region')
        if playable and self.velocity_shape is not None and (result.get('smoothing', {}).get('method') != profile_tracking.METHOD or result['smoothing'].get('reference_shape') != self.velocity_shape):
            raise ValueError('Planner did not preserve the selected mouse velocity profile')
        visibility_object = (cursor/1000).tolist() if self.engine == 'learned_rrt' and region is not None else None
        visibility_camera = self.visibility_camera.configuration() if self.visibility_camera is not None else None
        curve = validate_curve(result, expected, self.speed_rad_s, self.environment, self.engine,
                               visibility_object, visibility_camera) if playable else None
        if playable and any(s['duration_s'] < 4*encoder_profile.PERIOD_S for s in curve['segments']):
            # Rebuild and recheck timing; scaling a moving curve would alter its
            # initial velocity/acceleration and break the checked live boundary.
            from simulation.diffusion.tracking import SceneSpace
            from simulation.continuous import frames_for
            with scene_for({'visibility_object_m':visibility_object, 'visibility_camera':visibility_camera}) as env:
                env.set_obstacle(None)
                path = np.deg2rad([curve['segments'][0]['start_deg'], *[s['end_deg'] for s in curve['segments']]])
                states, times, smoothing = profile_tracking.prepare(SceneSpace(env), path, expected[1], expected[2],
                    self.speed_rad_s, velocity_shape=self.velocity_shape, minimum_duration_s=4*encoder_profile.PERIOD_S)
            result = {**result, 'smoothing':smoothing,
                'frames':frames_for(states, times, np.asarray(result['target_mm'])/1000, np.asarray(result['target_axis'])),
                'max_acceleration_rad_s2':smoothing['max_acceleration_rad_s2']}
            from simulation.tracking_geometry import annotate_gaze
            result['gaze_tracking'] = annotate_gaze(result['frames'], cursor, region)
            curve = validate_curve(result, expected, self.speed_rad_s, self.environment, self.engine,
                                   visibility_object, visibility_camera)
        result = {**result, 'simulation_only': False, 'hardware_v2_mouse': True,
                  'hardware_diffusion': self.engine == 'diffusion_v2', 'hardware_classical_mouse': self.engine == 'learned_rrt', 'environment': self.environment,
                  'execution_mode': 'encoder_feedback_tracking', 'hardware_sdk_speed': self.speed,
                  'hardware_reference_speed_rad_s': self.speed_rad_s}
        result['visibility_camera'] = visibility_camera
        plan = {'id': secrets.token_urlsafe(24), 'created': time.monotonic(),
                'generation': generation, 'start_deg': np.rad2deg(expected[0]).tolist(),
                'live_start_deg': live, 'target_mm': result['target_mm'], 'target_axis': result['target_axis'],
                'result': result, 'waypoints': [f['q'] for f in result.get('frames', [])[1:]],
                'mouse_aim_session_id': body['session_id'], 'cursor_mm': cursor.tolist(),
                'tracking_region': copy.deepcopy(region),
                'orientation_mode': mode,
                'visibility_object_m': visibility_object,
                'visibility_camera': visibility_camera,
                'hardware_v2_mouse': True, 'hardware_diffusion': self.engine == 'diffusion_v2', 'hardware_classical_mouse': self.engine == 'learned_rrt', 'checkpoint_sha256': self.checkpoint_sha256,
                'environment': self.environment, 'curve': curve, 'expected_motion': expected, 'join': join,
                'execution_mode': 'encoder_feedback_tracking', 'arrival_deg': ARRIVAL_DEG,
                'max_sdk_speed': MAX_SDK_SPEED, 'table_top_mm': self.environment['table_top_m']*1000}
        if playable:
            encoder_profile.prepare_mouse(plan, self.encoder_mapping, self.speed)
        with c.lock:
            ik._mouse_session(body['session_id'])
            if ik.v2_mouse is not self or c.generation != generation:
                raise StaleMousePlan('Robot session changed while planning')
            if join:
                self.check_join(plan)
            else:
                c._read()
                if c.motion or c.moving or max(abs(a-b) for a, b in zip(c.angles, live)) > 1:
                    raise StaleMousePlan('Live pose changed while planning')
            self.last_compute = time.monotonic()-began
            result['hardware_prepare_timing_ms'] = {
                'live_feedback_read': (read_done-began)*1000.,
                'planner': (planner_done-planner_began)*1000.,
                'validation_and_encoder_preparation': (time.monotonic()-planner_done)*1000.,
                'total': self.last_compute*1000.}
            if join:
                self.feedback_hold_until = time.monotonic()+2. if playable else 0.
            ik.plan = plan if playable else None
            if playable and region is not None:
                self.proposed_region = copy.deepcopy(region)
            if not running:
                ik.status = {'state': 'planned', 'samples': [], 'mouse_engine': self.engine,
                             'message': f'Checked {self.engine} C2 mouse path' if playable else f'No checked {self.engine} continuation'}
            ik.status['last_orientation_selection'] = copy.deepcopy(result.get('orientation_selection'))
            ik.status['last_planning_orientation_mode'] = mode
            return {'session_id': body['session_id'], 'plan': ik.snapshot()['plan'], 'result': result,
                    'start_deg': plan['start_deg'], 'simulation_only': False, 'hardware_mouse_aim': True,
                    'hardware_v2_mouse': True, 'execution_error_mm': 5., 'execution_orientation_deg': 3.}

    def check_join(self, plan):
        join = plan['join']
        if (not self.ik.owns_running_mouse(plan['mouse_aim_session_id']) or self.curve is None
                or self.active_id != join['curve_id'] or join.get('mode') != 'live_feedback'
                or np.max(np.abs(np.deg2rad(self.ik.controller.angles)-plan['expected_motion'][0])) > np.deg2rad(encoder_profile.MOUSE_LEAD_DEG)):
            self.feedback_hold_until = 0.
            raise StaleMousePlan('Live cursor anchor moved beyond its bounded connection; retain checked path and replan')

    def execute(self, plan, speed, armed, frame_verified, scene_verified):
        ik = self.ik; c = ik.controller
        if not plan['result'].get('playable'):
            raise ValueError('No playable v2 mouse curve')
        if armed is not True or frame_verified is not True or scene_verified is not True:
            raise ValueError('Verify movement, flange frame and table scene before v2 mouse execution')
        if speed != self.speed:
            raise ValueError('SDK speed changed; stop and restart v2 mouse tracking')
        if time.monotonic()-plan['created'] > PLAN_TTL:
            raise ValueError('v2 mouse plan expired')
        status = self.deployment_status()
        if status['environment'] != self.environment:
            raise ValueError('v2 deployment changed; start again')
        if c.generation != plan['generation']:
            raise StaleMousePlan('Robot session changed; replan')
        if plan['join']:
            self.check_join(plan)
            if ik.mouse_pending_plan is not None:
                raise StaleMousePlan('Previous v2 continuation is still pending')
            with reply_timeout(c.robot, encoder_profile.STREAM_REPLY_TIMEOUT_S,
                    budget_s=encoder_profile.STREAM_QUERY_BUDGET_S):
                c._read(); c._health()
            self.check_join(plan)
            ik.plan = None
            ik.mouse_pending_plan = {'plan': plan, 'speed': speed, 'created': time.monotonic(), 'confirmed': False}
            ik.status['accepted_plan_id'] = plan['id']
            return c.snapshot()
        c._ready(armed, plan['mouse_aim_session_id'])
        if max(abs(a-b) for a, b in zip(c.angles, plan['live_start_deg'])) > 1:
            raise StaleMousePlan('Live start changed; replan')
        encoder_drag.verify(c.robot, c.angles, self.encoder_mapping)
        c._cancel_timer(); ik.plan = None
        c.error = None; c.motion = 'ik'; c.target = None
        ik.heartbeat_at = time.monotonic(); ik.heartbeat_confirmed = False
        self.curve, self.phase, self.active_id = plan['curve'], 0., plan['id']
        self.active_region = copy.deepcopy(plan.get('tracking_region'))
        self.active_orientation_mode = plan.get('orientation_mode', 'point_at_object')
        self.clock_origin = None
        ik.status = {'state': 'running', 'execution_id': plan['id'], 'active_plan_id': plan['id'],
            'accepted_plan_id': plan['id'], 'mouse_aim_session_id': ik.mouse_session,
            'method': plan['result']['method'], 'mouse_engine': self.engine,
            'hardware_diffusion': self.engine == 'diffusion_v2', 'hardware_classical_mouse': self.engine == 'learned_rrt', 'hardware_v2_mouse': True, 'checkpoint_sha256': self.checkpoint_sha256,
            'target_mm': plan['target_mm'], 'target_axis': plan['target_axis'], 'task': 'pose5',
            'tracking_region': copy.deepcopy(plan.get('tracking_region')),
            'orientation_mode': self.active_orientation_mode,
            'orientation_selection': copy.deepcopy(plan['result'].get('orientation_selection')),
            'sdk_speed': speed, 'arrival_deg': ARRIVAL_DEG, 'execution_mode': 'encoder_feedback_tracking',
            'joint_command':'encoder_drag_profile',
            'completed_waypoints': 0, 'total_waypoints': len(self.curve['segments']),
            'replan_required': plan['result']['replan_required'], 'samples': [],
            'velocity_limit_deg_s': float(np.rad2deg(self.speed_rad_s)),
            'acceleration_limit_deg_s2': float(np.rad2deg(3.)), 'motion_specification': profile(),
            'message': 'Waiting for owning browser acknowledgement'}
        threading.Thread(target=self.run, args=(plan, c.generation), daemon=True).start()
        return c.snapshot()

    def pending_join_limit(self):
        pending = self.ik.mouse_pending_plan
        # A held old path must not start a fresh lookahead during retargeting.
        return max(self.phase, self.ik.status.get('target_phase_s', self.phase)) if pending or time.monotonic() < self.feedback_hold_until else None

    def advance_pending(self, plan, started, now, scene=None):
        """Apply acknowledged live anchors under the existing execution owner."""
        ik = self.ik; c = ik.controller
        pending = ik.mouse_pending_plan
        if not pending:
            return plan, started
        candidate = pending['plan']; join = candidate['join']
        if join['curve_id'] != self.active_id or now-pending['created'] > 2.:
            ik.mouse_pending_plan = None
            self.feedback_hold_until = 0.
            ik.status.update(accepted_plan_id=self.active_id,
                             retarget_rejected='Expired cursor acknowledgement; retaining checked path', feedback_replan_required=True)
            return plan, started
        if not pending['confirmed']:
            return plan, started
        boundary = motion(candidate['curve'], 0.)
        if not same_motion(boundary, candidate['expected_motion']):
            raise RuntimeError('Encoder continuation changed q/v/a at its join')
        if np.max(np.abs(np.rad2deg(boundary[0])-c.angles)) > encoder_profile.MOUSE_LEAD_DEG:
            ik.mouse_pending_plan = None
            self.feedback_hold_until = 0.
            ik.status.update(accepted_plan_id=self.active_id,
                             retarget_rejected='Live anchor drifted during acknowledgement; retaining checked path', feedback_replan_required=True)
            return plan, started
        initial = encoder_profile.packet(candidate['curve'], 0., candidate['encoder_mapping'], self.speed,
            live_deg=c.angles, maximum_lead_deg=encoder_profile.MOUSE_LEAD_DEG)
        previous_object = set_candidate_object(scene, candidate) if scene is not None else None
        connection_ok = False
        try:
            connection_ok = (initial['command_lead_deg'] <= encoder_profile.MOUSE_LEAD_DEG+1e-8
                and (scene is None or scene.path_valid(np.deg2rad([c.angles, initial['target_deg']]), step_for(scene))))
        finally:
            if not connection_ok and previous_object is not None:
                scene.set_object(previous_object)
        if not connection_ok:
            ik.mouse_pending_plan = None
            self.feedback_hold_until = 0.
            ik.status.update(accepted_plan_id=self.active_id,
                retarget_rejected='Cursor connection failed live collision/visibility/lead checks; retaining checked path',
                feedback_replan_required=True)
            return plan, started
        ik.status.update(join_position_error_rad=float(np.max(np.abs(boundary[0]-candidate['expected_motion'][0]))),
            join_velocity_error_rad_s=float(np.max(np.abs(boundary[1]-candidate['expected_motion'][1]))),
            join_acceleration_error_rad_s2=float(np.max(np.abs(boundary[2]-candidate['expected_motion'][2]))))
        self.curve, self.active_id = candidate['curve'], candidate['id']
        self.active_region = copy.deepcopy(candidate.get('tracking_region'))
        self.active_orientation_mode = candidate.get('orientation_mode', 'point_at_object')
        self.clock_origin = now
        self.phase = 0.
        self.feedback_hold_until = 0.
        ik.mouse_pending_plan = None
        ik.status.update(active_plan_id=self.active_id, accepted_plan_id=self.active_id,
            target_mm=candidate['target_mm'], target_axis=candidate['target_axis'], retarget_rejected=None,
            feedback_replan_required=False,
            tracking_region=copy.deepcopy(candidate.get('tracking_region')),
            orientation_mode=self.active_orientation_mode,
            orientation_selection=copy.deepcopy(candidate['result'].get('orientation_selection')),
            replan_required=candidate['result']['replan_required'], total_waypoints=len(self.curve['segments']))
        return candidate, self.clock_origin

    def run(self, plan, generation):
        encoder_profile.run(self.ik, plan, self.speed, generation, mouse=self)
