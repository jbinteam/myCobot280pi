"""Live-start IK planning and telemetry-paced waypoint execution.

All SDK access is serialized by the existing controller's RLock. No simulated
clock is used to stream robot commands. Only server-owned, converged plans can
be executed; client-supplied joint trajectories are never accepted.
"""
import copy
import re
import secrets
import threading
import time
import numpy as np
from simulation.kinematics import ARM
from simulation import service
from simulation.pose5 import axis_error, xy_from_axis
from simulation.trajectory import joint_waypoints
from webui.mouse_motion import command_path, JointServo, rebase_path, StaleMousePlan
from webui.command_profile import CommandProfile
from webui.motion_settings import MAX_SDK_SPEED, FIXED_V2_JOINT_COMMAND
from webui.fixed_profile import selected_speed, retime_fixed, waypoint_intervals
from simulation.motion_limits import profile
from pymycobot.common import ProtocolCode

PLAN_TTL = 60
HEARTBEAT_TIMEOUT = 5
SEGMENT_TIMEOUT = 10
EXECUTION_TIMEOUT = 180
ARRIVAL_DEG = .5
FINAL_ERROR_MM = 5
FINAL_ORIENTATION_DEG = 3
MAX_STEP_DEG = 3.5
MOUSE_STEP_DEG = 3.
MOUSE_ARRIVAL_DEG = 1.
MOUSE_POLL_SECONDS = .05
MOUSE_MAX_CYCLE_SECONDS = 2.


class HardwareIK:
    def __init__(self, controller):
        self.controller = controller
        self.plan = None
        self.status = {'state': 'idle', 'samples': []}
        self.heartbeat_at = 0
        self.heartbeat_confirmed = False
        self.mouse_session = None
        self.mouse_client_id = None
        self.mouse_heartbeat_at = 0
        self.mouse_original_mode = None
        self.mouse_mode_changed = False
        self.refresh_mode = None
        self.mouse_servo = None
        self.mouse_pending_plan = None
        self.v2_mouse = None

    def snapshot(self):
        # Caller owns controller.lock; return detached JSON-safe data.
        status = dict(self.status)
        status['samples'] = list(self.status.get('samples', []))
        status['mouse_aim_session_id'] = self.mouse_session
        status['mouse_aim_client_id'] = self.mouse_client_id
        status['mouse_session_owner_protocol'] = 1
        status['mouse_pause_protocol'] = 1
        status['refresh_mode'] = self.refresh_mode
        status['original_refresh_mode'] = self.mouse_original_mode
        status['tracking_protocol'] = 2
        status['v2_tracking_protocol'] = 1
        status['mouse_velocity_profile_protocol'] = 1
        status['mouse_encoder_profile_protocol'] = 1
        status['mouse_feedback_tracking_protocol'] = 1
        status['mouse_tracking_region_protocol'] = 1
        status['mouse_approach_protocol'] = 1
        status['mouse_orientation_policy_protocol'] = 1
        status['requested_orientation_mode'] = self.v2_mouse.orientation_mode if self.v2_mouse else None
        status['approach_settings'] = copy.deepcopy(self.v2_mouse.approach) if self.v2_mouse else None
        status['mouse_executor'] = 'encoder_feedback_tracking' if self.v2_mouse else None
        status['velocity_shape'] = self.v2_mouse.velocity_shape if self.v2_mouse else None
        status['tracking_region_settings'] = copy.deepcopy(self.v2_mouse.region_settings) if self.v2_mouse else None
        camera = self.v2_mouse.visibility_camera if self.v2_mouse else None
        status['visibility_camera'] = camera.configuration() if camera is not None else None
        if 'tracking_region' in status:
            status['tracking_region'] = copy.deepcopy(status['tracking_region'])
        plan = self.plan
        status['plan'] = ({'id':plan['id'], 'method':plan['result']['method'],
                           'target_mm':plan['target_mm'], 'start_deg':plan['start_deg'],
                           'waypoints':len(plan['waypoints']),
                           'arrival_deg':plan.get('arrival_deg', ARRIVAL_DEG),
                           'path_arrival_deg':plan.get('path_arrival_deg', plan.get('arrival_deg', ARRIVAL_DEG)),
                           'max_live_command_deg':plan.get('max_live_command_deg', plan.get('max_step_deg', MAX_STEP_DEG)+plan.get('arrival_deg', ARRIVAL_DEG)),
                           'execution_mode':plan.get('execution_mode', 'settled_waypoints'),
                           'execution_timeout_s':plan.get('execution_timeout_s', EXECUTION_TIMEOUT),
                           'joint_command':plan.get('joint_command', 'send_angles'),
                           'planning_timing_ms':plan['result'].get('planning_timing_ms'),
                           'encoder_drag':plan['result'].get('encoder_drag'),
                           'hardware_diffusion':plan.get('hardware_diffusion', False),
                           'hardware_joint_diagnostic':plan.get('hardware_joint_diagnostic', False),
                           'diagnostic':plan['result'].get('diagnostic'),
                           'hardware_v2_mouse':plan.get('hardware_v2_mouse', False),
                           'hardware_classical_mouse':plan.get('hardware_classical_mouse', False),
                           'planner':plan['result'].get('planner'),
                           'playable':plan['result'].get('playable', plan['result']['converged']),
                           'checkpoint_sha256':plan.get('checkpoint_sha256'),
                           'table_top_mm':plan.get('table_top_mm'),
                           'max_step_deg':plan.get('max_step_deg', MAX_STEP_DEG),
                           'max_sdk_speed':plan.get('max_sdk_speed', MAX_SDK_SPEED),
                           'planned_sdk_speed':plan.get('planned_sdk_speed'),
                           'velocity_shape':plan['result'].get('velocity_shape', {'kind':'quintic'}),
                           'joint_velocity_allowance_deg_s':plan['result'].get('joint_velocity_allowance_deg_s'),
                           'preview_frames':len(plan['result']['frames']),
                           'expires_in':max(0, plan.get('review_ttl_s', PLAN_TTL)-(time.monotonic()-plan['created'])),
                           'converged':plan['result']['converged'],
                           'task':plan['result'].get('task','position'),
                           'orientation_xy_deg':plan['result'].get('orientation_xy_deg'),
                           'target_axis':plan['target_axis'],
                           'tracking_region':copy.deepcopy(plan.get('tracking_region')),
                           'final_orientation_error_deg':plan['result'].get('final_orientation_error_deg'),
                           'final_error_mm':plan['result']['final_error_mm']} if plan else None)
        if self.controller.angles is not None:
            t=ARM.transform(np.deg2rad(self.controller.angles))
            status['estimated_flange_mm'] = (t[:3,3]*1000).tolist()
            status['flange_z_axis'] = t[:3,2].tolist()
            status['orientation_xy_deg'] = xy_from_axis(t[:3,2])
        return status

    def cancel(self, reason='stopped'):
        self.plan = None
        self.mouse_session = None
        self.mouse_client_id = None
        self.mouse_pending_plan = None
        self.mouse_servo = None
        self.v2_mouse = None
        if self.status['state'] == 'running':
            self.status['state'] = reason
            self.status['message'] = 'Execution stopped; a new live-start plan is required'

    def read_refresh_mode(self):
        getter = getattr(self.controller.robot, 'get_fresh_mode', None)
        if not callable(getter):
            raise RuntimeError('Robot SDK cannot verify command refresh mode; hardware tracking is disabled')
        value = getter()
        if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
            raise RuntimeError('Robot did not verify command refresh mode; hardware tracking is disabled')
        self.refresh_mode = value
        return value

    def enable_refresh_mode(self):
        if self.read_refresh_mode() != 1:
            # The old socket SDK retries setters for missing ACKs. Send once,
            # just as for motor/Stop packets, then verify with the getter.
            self.controller.robot._mesg(ProtocolCode.SET_FRESH_MODE, 1, _async=True)
            self.mouse_mode_changed = True
            if self.read_refresh_mode() != 1:
                raise RuntimeError('Robot refused refresh mode 1; hardware tracking is disabled')

    def release_mouse_mode(self):
        original = self.mouse_original_mode
        if original is not None and self.controller.robot and (self.mouse_mode_changed or self.refresh_mode != original):
            try:
                self.controller.robot._mesg(ProtocolCode.SET_FRESH_MODE, original, _async=True)
                if self.read_refresh_mode() != original:
                    raise RuntimeError('Previous command mode was not verified')
            except Exception as exc:
                self.controller.error = f'Robot stopped, but its previous command mode could not be restored: {exc}'
                self.controller._close()
                raise RuntimeError(self.controller.error) from exc
        self.mouse_original_mode = None
        self.mouse_mode_changed = False

    def start_mouse_aim(self, body):
        if set(body)-{'armed', 'frame_verified', 'tracking_engine', 'scene_verified', 'speed', 'speed_rad_s', 'velocity_shape', 'tracking_region', 'approach', 'client_id', 'orientation_mode'}:
            raise ValueError('Mouse aim start accepts confirmations and engine/speed settings only')
        client_id = body.get('client_id', 'webui')
        if not isinstance(client_id, str) or re.fullmatch(r'[A-Za-z0-9_.:/-]{1,64}', client_id) is None:
            raise ValueError('Tracking client_id requires 1–64 letters, digits or _ . : / -')
        engine = body.get('tracking_engine', 'classical')
        if engine not in ('classical', 'diffusion_v2', 'learned_rrt'):
            raise ValueError('Unknown hardware mouse engine')
        if engine == 'classical' and 'velocity_shape' in body:
            raise ValueError('Shared mouse velocity profiles require learned IK + RRT* or diffusion v2')
        if engine == 'classical' and 'tracking_region' in body:
            raise ValueError('Cone tracking requires learned IK + RRT* or diffusion v2')
        if engine == 'classical' and body.get('approach') is not None:
            raise ValueError('Approach requires checked cone tracking')
        if engine in ('diffusion_v2', 'learned_rrt'):
            from webui.hardware_diffusion import verified_status
            from webui.v2_mouse import V2Mouse
            if body.get('scene_verified') is not True:
                raise ValueError('Verify the table scene before v2 mouse tracking')
            speed = body.get('speed', 5)
            if isinstance(speed, bool) or not isinstance(speed, int) or not 1 <= speed <= MAX_SDK_SPEED:
                raise ValueError(f'v2 SDK speed must be 1–{MAX_SDK_SPEED}')
            reference_speed = body.get('speed_rad_s', 1.)
            if (isinstance(reference_speed, bool) or not isinstance(reference_speed, (int, float))
                    or not np.isfinite(reference_speed) or not 0 < reference_speed <= 1.):
                raise ValueError('v2 reference speed must be above zero and at most 1 rad/s')
            if engine == "learned_rrt":
                from simulation.diffusion.classical_tracking import verified_status as classical_status
                deployment = classical_status()
            else:
                deployment = verified_status()
        if body.get('frame_verified') is not True:
            raise ValueError('Verify the URDF joint zero/axes and flange frame before mouse aim')
        c = self.controller
        with c.lock:
            self.expire_mouse_aim()
            if self.mouse_session:
                raise RuntimeError(f'A tracking session is already active for {self.mouse_client_id}; stop that owner before starting another')
            c._ready(body.get('armed'))
            mapping = self.capture_encoder_mapping(c.angles) if engine in ('diffusion_v2', 'learned_rrt') else None
            self.mouse_original_mode = self.read_refresh_mode()
            self.v2_mouse = V2Mouse(self, speed, deployment, reference_speed, engine, body.get("velocity_shape"), body.get('tracking_region'), body.get('approach'), body.get('orientation_mode', 'point_at_object')) if engine in ('diffusion_v2', 'learned_rrt') else None
            if self.v2_mouse:
                self.v2_mouse.encoder_mapping = mapping
            self.plan = None
            self.mouse_session = secrets.token_urlsafe(24)
            self.mouse_client_id = client_id
            self.mouse_heartbeat_at = time.monotonic()
            self.status = {'state': 'idle', 'message': 'Mouse aim ready; waiting for a cursor target', 'samples': [], 'mouse_engine': engine}
            return {**c.snapshot(), 'session_id': self.mouse_session}

    def _mouse_session(self, session_id):
        self.expire_mouse_aim()
        if not self.mouse_session or session_id != self.mouse_session:
            raise ValueError('Mouse aim session is no longer active; start again')

    def stop_mouse_aim(self, session_id):
        c = self.controller
        with c.lock:
            # A delayed stop from an old session cannot interrupt a new owner.
            if not session_id or session_id != self.mouse_session:
                return c.snapshot()
            if c.motion == 'ik' and self.status.get('mouse_aim_session_id') == session_id:
                c._stop()
            else:
                self.mouse_session = None
                self.mouse_client_id = None
                self.v2_mouse = None
                if self.plan and self.plan.get('mouse_aim_session_id') == session_id:
                    self.plan = None
                self.status.update(state='stopped', message='Mouse aim stopped')
                self.release_mouse_mode()
            return c.snapshot()

    def pause_mouse_aim(self, session_id):
        """Hold hardware and invalidate curves without releasing the owner."""
        c = self.controller
        with c.lock:
            # Delayed requests from an old owner cannot pause a new session.
            if not session_id or session_id != self.mouse_session:
                raise ValueError('Mouse aim session is no longer active')
            if self.status.get('state') == 'paused':
                return c.snapshot()
            c._cancel_timer()
            self.plan = self.mouse_pending_plan = None
            self.mouse_servo = None
            self.heartbeat_confirmed = False
            if self.v2_mouse:
                mouse = self.v2_mouse
                mouse.curve = mouse.active_id = mouse.clock_origin = None
                mouse.phase = mouse.feedback_hold_until = 0.
                mouse.live_q = mouse.live_at = None
                mouse.live_velocity = np.zeros(6)
                mouse.proposed_region = mouse.active_region = None
                mouse.stage_region = None
            if c.robot:
                c._send_motion(lambda: c.robot._mesg(ProtocolCode.STOP, _async=True))
            c.motion = c.target = None
            self.status = {'state': 'paused', 'samples': [],
                'mouse_aim_session_id': session_id,
                'mouse_engine': self.v2_mouse.engine if self.v2_mouse else 'classical',
                'message': 'Holding position; waiting for a fresh detected target'}
            return c.snapshot()

    def expire_mouse_aim(self):
        if self.mouse_session and time.monotonic()-self.mouse_heartbeat_at > HEARTBEAT_TIMEOUT:
            self.stop_mouse_aim(self.mouse_session)

    def mouse_aim_heartbeat(self, session_id, execution_id=None, plan_id=None):
        with self.controller.lock:
            self._mouse_session(session_id)
            if execution_id is not None:
                if self.status.get('mouse_aim_session_id') != session_id:
                    raise ValueError('Mouse aim does not own this execution')
                self.heartbeat(execution_id)
            if plan_id is not None:
                pending = self.mouse_pending_plan
                if pending and pending['plan']['id'] == plan_id and time.monotonic()-pending['created'] <= 2:
                    if self.v2_mouse is not None:
                        self.v2_mouse.check_join(pending['plan'])
                    if not pending['confirmed']:
                        pending['confirmed_at'] = time.monotonic()
                    pending['confirmed'] = True
                elif self.status.get('active_plan_id') != plan_id:
                    if pending and pending['plan']['id'] == plan_id and self.v2_mouse is not None:
                        self.mouse_pending_plan = None
                        self.v2_mouse.feedback_hold_until = 0.
                        self.status.update(accepted_plan_id=self.status['active_plan_id'],
                            retarget_rejected='Expired cursor acknowledgement; retaining checked path', feedback_replan_required=True)
                        raise StaleMousePlan('Cursor acknowledgement expired; replan from fresh live feedback')
                    if self.v2_mouse is not None and self.status.get('retarget_rejected'):
                        raise StaleMousePlan('Encoder continuation is no longer awaiting acknowledgement; retain braking tail and replan')
                    raise ValueError('Replacement mouse plan is no longer awaiting acknowledgement')
            self.mouse_heartbeat_at = time.monotonic()
            return {'active': True, 'plan_acknowledged': plan_id}

    def owns_running_mouse(self, session_id):
        return (self.controller.motion == 'ik' and self.status.get('state') == 'running'
                and self.status.get('mouse_aim_session_id') == session_id)

    def prepare_mouse_aim(self, body, observer=None):
        if self.v2_mouse is not None:
            return self.v2_mouse.prepare(body) if observer is None else self.v2_mouse.prepare(body, observer=observer)
        allowed = {'session_id', 'cursor_mm', 'pose5_method', 'steps', 'speed_rad_s',
                   'tolerance_mm', 'orientation_tolerance_deg', 'damping',
                   'guided_route_repair', 'route_repair_samples'}
        if set(body)-allowed:
            raise ValueError('Mouse aim accepts a cursor and settings only; starting joints are read live')
        for key, limit in (('tolerance_mm', FINAL_ERROR_MM), ('orientation_tolerance_deg', FINAL_ORIENTATION_DEG)):
            value = body.get(key, 1.)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not .1 <= value <= limit:
                raise ValueError(f'Hardware mouse aim {key} must be 0.1–{limit}')
        c = self.controller
        session_id = body.get('session_id')
        with c.lock:
            self._mouse_session(session_id)
            if not c.robot:
                raise RuntimeError('Connect to the robot before planning')
            c._read()
            moving_start = self.owns_running_mouse(session_id)
            if (c.motion or c.moving) and not moving_start:
                raise RuntimeError('Wait for the current path to finish before planning')
            start, generation = c.angles[:], c.generation
            self.plan = None
        from simulation.prm import planning_space
        space = planning_space('self', 1.)
        result = service.mouse_aim_plan({**{k: v for k, v in body.items() if k != 'session_id'}, 'start_deg': start}, space=space)
        waypoints, stop_indices = [], set()
        if result.get('converged') and (result.get('validation') != 'joint_limits_and_self_collision'
                                      or result.get('cursor_acceptance', {}).get('accepted') is not True):
            raise ValueError('Mouse aim did not return a collision-checked, accepted cursor goal')
        if result.get('converged'):
            waypoints, stop_indices = command_path(start, result.get('waypoints_deg'), space, MOUSE_STEP_DEG)
            endpoint = waypoints[-1] if waypoints else start
            if not np.allclose(endpoint, result['frames'][-1]['q'], atol=1e-8, rtol=0):
                raise ValueError('Hardware mouse command endpoint differs from the validated goal')
        plan = {'id': secrets.token_urlsafe(24), 'created': time.monotonic(),
                'generation': generation, 'start_deg': start, 'target_mm': result.get('target_mm'),
                'result': result, 'waypoints': waypoints, 'target_axis': result.get('target_axis'),
                'mouse_aim_session_id': session_id, 'cursor_mm': body['cursor_mm'], 'space': space,
                'stop_indices': stop_indices, 'arrival_deg': MOUSE_ARRIVAL_DEG,
                'vertices_deg': result.get('waypoints_deg'), 'execution_mode': 'velocity_feedback'}
        with c.lock:
            self._mouse_session(session_id)
            if not c.robot or c.generation != generation or (c.motion and not self.owns_running_mouse(session_id)):
                raise RuntimeError('Robot session changed while planning; start again')
            c._read()
            if not moving_start and (c.moving or max(abs(a-b) for a, b in zip(c.angles, start)) > 1):
                raise RuntimeError('Robot pose changed while planning; plan again')
            self.plan = plan if result.get('converged') else None
            if not self.owns_running_mouse(session_id):
                self.status = {'state': 'planned', 'message': 'Mouse aim route prepared' if self.plan else 'No accepted mouse aim route', 'samples': []}
            return {'session_id': session_id, 'plan': self.snapshot()['plan'], 'result': result,
                    'start_deg': start, 'simulation_only': False, 'hardware_mouse_aim': True,
                    'execution_error_mm': FINAL_ERROR_MM, 'execution_orientation_deg': FINAL_ORIENTATION_DEG}

    def prepare(self, body):
        if set(body)-{'task','target_mm','eef_offset_mm','orientation_xy_deg','orientation_tolerance_deg','method','steps','speed_rad_s','tolerance_mm','damping','speed'}:
            raise ValueError('Hardware planning accepts target and solver settings only; the start is read from live joints')
        speed = selected_speed(body, 30)
        if body.get('task')=='pose5' and body.get('orientation_tolerance_deg',1.)>FINAL_ORIENTATION_DEG:
            raise ValueError('Hardware orientation planning tolerance must not exceed 3 degrees')
        c = self.controller
        with c.lock:
            if self.mouse_session:
                raise RuntimeError('Stop mouse aim before preparing a different hardware plan')
            if not c.robot: raise RuntimeError('Connect to the robot before planning')
            c._read()
            if c.motion or c.moving: raise RuntimeError('Stop the arm before planning')
            start = c.angles[:]
            generation = c.generation
            self.plan = None
        method = body.get('method', 'dls')
        solved = service.run({**{k:v for k,v in body.items() if k!='speed'}, 'start_deg':start, 'methods':[method]})
        result = solved['results'][0]
        points = [f['q'] for f in result['frames'][1:]]
        # Compress tiny solver updates but retain its joint-space path. Split any
        # retained segment to preserve the 3.5-degree per-joint command bound.
        waypoints=joint_waypoints(start, points, MAX_STEP_DEG, compress=True)
        result=retime_fixed(result,speed,path=np.deg2rad([start,*waypoints]))
        plan={'id':secrets.token_urlsafe(24), 'created':time.monotonic(),
              'generation':generation, 'start_deg':start, 'target_mm':result['target_mm'],
              'result':result, 'waypoints':waypoints, 'target_axis':result.get('target_axis'),
              'planned_sdk_speed':speed,'waypoint_intervals_s':waypoint_intervals(result,waypoints)}
        with c.lock:
            if not c.robot or c.generation!=generation or c.motion:
                raise RuntimeError('Robot session changed while planning; plan again')
            c._read()
            if c.moving or max(abs(a-b) for a,b in zip(c.angles,start))>1:
                raise RuntimeError('Robot pose changed while planning; plan again')
            self.plan=plan
            self.status={'state':'planned', 'message':'Preview the path before executing', 'samples':[]}
            return {'plan':self.snapshot()['plan'], 'result':result, 'start_deg':start,
                    'target_mm':result['target_mm'],'eef_rotation':solved['eef_rotation'], 'simulation_only':False,
                    'frame':'URDF base → joint6_flange', 'arrival_deg':ARRIVAL_DEG,
                    'execution_error_mm':FINAL_ERROR_MM, 'execution_orientation_deg':FINAL_ORIENTATION_DEG}

    def prepare_diffusion(self, body):
        return self._prepare_checked(body)

    def prepare_joint_diagnostic(self, body):
        return self._prepare_checked(body, diagnostic=True)

    def prepare_jog_qualification(self, body):
        return self._prepare_checked(body, diagnostic='jog_qualification')

    def encoder_preflight(self, body):
        if body:
            raise ValueError('Encoder preflight accepts no settings')
        c = self.controller
        with c.lock:
            if not c.robot or c.motion or c.moving or self.mouse_session:
                raise RuntimeError('Encoder preflight requires a connected idle arm with mouse tracking stopped')
            c._read()
            return {'encoder_drag': self.capture_encoder_mapping(c.angles), 'motion_sent': False}

    def capture_encoder_mapping(self,start):
        """Caller owns controller lock; retain only verified paired directions."""
        from webui.encoder_drag import capture, EncoderDirectionChanged
        c=self.controller;key=(c.host,c.port,c.firmware)
        previous=c.encoder_calibration.lookup(*key)
        try: mapping=capture(c.robot,start,verified_mapping=previous)
        except EncoderDirectionChanged:
            c.encoder_calibration.forget(*key)
            raise
        c.encoder_calibration.remember(*key,mapping)
        return mapping

    def small_target(self, body):
        from webui.small_targets import target, PRESETS
        if set(body) != {'preset'} or not isinstance(body['preset'], str) or body['preset'] not in PRESETS:
            raise ValueError('Small target accepts a listed preset only')
        c = self.controller
        with c.lock:
            if not c.robot or c.motion or self.mouse_session:
                raise RuntimeError('Small target requires a connected idle arm with mouse tracking stopped')
            c._read()
            if c.moving:
                raise RuntimeError('Stop the arm before choosing a small target')
            return target(c.angles, body['preset'])

    def _prepare_checked(self, body, diagnostic=False):
        began = time.perf_counter()
        timings = {}
        from webui import hardware_diffusion
        c = self.controller
        joint_command = body.get('joint_command', FIXED_V2_JOINT_COMMAND) if not diagnostic else 'send_angles'
        if joint_command not in ('send_angles', 'encoder_drag', 'encoder_drag_stream', 'encoder_drag_staged', 'encoder_drag_profile'):
            raise ValueError('Choose joint angle targets, encoder waypoints or encoder streaming')
        encoder_mapping = None
        waiting = time.perf_counter()
        with c.lock:
            locked = time.perf_counter()
            timings['initial_lock_wait'] = (locked-waiting)*1000
            if self.mouse_session:
                raise RuntimeError('Stop mouse aim before preparing a fixed hardware plan')
            if not c.robot:
                raise RuntimeError('Connect to the robot before planning')
            read_began = time.perf_counter()
            c._read()
            timings['live_joint_read'] = (time.perf_counter()-read_began)*1000
            if c.motion or c.moving:
                raise RuntimeError('Stop the arm before planning')
            start, generation = c.angles[:], c.generation
            self.plan = None
            if joint_command in ('encoder_drag', 'encoder_drag_stream', 'encoder_drag_staged', 'encoder_drag_profile'):
                from webui.encoder_drag import MAX_TRIAL_SPEED
                if joint_command != 'encoder_drag_profile' and selected_speed(body, 5) > MAX_TRIAL_SPEED:
                    raise ValueError('Encoder trial requires arm speed at most 5%')
                capture_began = time.perf_counter()
                encoder_mapping = self.capture_encoder_mapping(start)
                timings['encoder_mapping_capture'] = (time.perf_counter()-capture_began)*1000
        prepare_began = time.perf_counter()
        if diagnostic=='jog_qualification':
            from webui import hardware_jog
            prepared=hardware_jog.prepare(start,body)
        elif diagnostic:
            from webui import hardware_diagnostic
            prepared=hardware_diagnostic.prepare(start,body)
        else:
            prepared = hardware_diffusion.prepare(start, body)
        timings['trajectory_prepare'] = (time.perf_counter()-prepare_began)*1000
        profile_began = time.perf_counter()
        if encoder_mapping is not None:
            if joint_command == 'encoder_drag_profile':
                from webui.encoder_profile import prepare
                prepare(prepared, encoder_mapping, selected_speed(body, 5))
            elif joint_command in ('encoder_drag_stream', 'encoder_drag_staged'):
                from webui.encoder_stream import prepare
                prepare(prepared, encoder_mapping, selected_speed(body, 5),
                        staged=joint_command == 'encoder_drag_staged')
            else:
                from webui.encoder_drag import prepare
                prepare(prepared, encoder_mapping, selected_speed(body, 5))
        timings['encoder_profile_prepare'] = (time.perf_counter()-profile_began)*1000
        result = prepared['result']
        plan = {'id': secrets.token_urlsafe(24), 'created': time.monotonic(),
                'generation': generation, 'start_deg': start, 'target_mm': prepared['target_mm'],
                'result': result, 'waypoints': prepared['waypoints'], 'target_axis': result['target_axis'],
                'hardware_diffusion': not diagnostic, 'hardware_joint_diagnostic':bool(diagnostic),
                'checkpoint_sha256': result['checkpoint_sha256'],
                'environment': prepared['environment'], 'max_step_deg':
                    result['diagnostic']['maximum_step_deg'] if diagnostic else hardware_diffusion.MAX_STEP_DEG,
                'arrival_deg': .5 if diagnostic else hardware_diffusion.ARRIVAL_DEG,
                'path_arrival_deg': hardware_diffusion.PATH_ARRIVAL_DEG,
                'max_live_command_deg': hardware_diffusion.MAX_LIVE_COMMAND_DEG,
                'stop_indices': prepared['stop_indices'],
                'execution_mode':prepared.get('execution_mode','path_waypoints'),
                'joint_command':joint_command,
                'encoder_mapping':encoder_mapping,
                'encoder_packets':prepared.get('encoder_packets'),
                'max_sdk_speed': 5 if encoder_mapping is not None and joint_command != 'encoder_drag_profile' else hardware_diffusion.MAX_SDK_SPEED,
                'planned_sdk_speed':result.get('planned_sdk_speed'),
                'waypoint_intervals_s':prepared.get('waypoint_intervals_s',[]),
                'table_top_mm': prepared['environment']['table_top_m']*1000}
        if prepared.get('execution_mode') == 'encoder_stream':
            from webui.encoder_stream import ARRIVAL_DEG, MAX_LEAD_DEG, SETTLE_S
            plan.update(arrival_deg=ARRIVAL_DEG, path_arrival_deg=ARRIVAL_DEG,
                        max_live_command_deg=MAX_LEAD_DEG, max_step_deg=.35)
            if result['encoder_drag'].get('staged'):
                plan['review_ttl_s'] = result['smoothing']['duration_s'] + PLAN_TTL
                plan['execution_timeout_s'] = max(EXECUTION_TIMEOUT,
                    sum(plan['waypoint_intervals_s']) + len(plan['waypoints'])*(SEGMENT_TIMEOUT+2.)
                    + result['encoder_drag']['stage_count']*(SETTLE_S+2.))
        if prepared.get('execution_mode') == 'encoder_profile':
            from webui.encoder_profile import ARRIVAL_DEG, TRACKING_DEG, LEAD_DEG
            duration = result['smoothing']['duration_s']
            plan.update(arrival_deg=ARRIVAL_DEG, path_arrival_deg=TRACKING_DEG,
                        max_live_command_deg=LEAD_DEG, review_ttl_s=duration+PLAN_TTL,
                        execution_timeout_s=duration+30.)
        waiting = time.perf_counter()
        with c.lock:
            locked = time.perf_counter()
            timings['final_lock_wait'] = (locked-waiting)*1000
            if not c.robot or c.generation != generation or c.motion or self.mouse_session:
                raise RuntimeError('Robot session changed while planning; plan again')
            recheck_began = time.perf_counter()
            c._read()
            timings['live_start_recheck'] = (time.perf_counter()-recheck_began)*1000
            if c.moving or max(abs(a-b) for a, b in zip(c.angles, start)) > 1:
                raise RuntimeError('Robot pose changed while planning; plan again')
            self.plan = plan if result.get('converged') else None
            self.status = {'state': 'planned', 'message': ('Joint command diagnostic prepared; no motion sent' if diagnostic
                else 'Review the complete v2 hardware path' if self.plan else 'No converged v2 path'), 'samples': []}
            timings['server_total'] = (time.perf_counter()-began)*1000
            timings['v2_prepare'] = result.get('v2_prepare_timing_ms')
            result['planning_timing_ms'] = timings
            return {'plan': self.snapshot()['plan'], 'result': result, 'start_deg': start,
                    'target_mm': prepared['target_mm'], 'eef_rotation': prepared['eef_rotation'],
                    'simulation_only': False, 'hardware_diffusion': not diagnostic,
                    'hardware_joint_diagnostic':bool(diagnostic),
                    'frame': 'URDF base → joint6_flange', 'execution_error_mm': FINAL_ERROR_MM,
                    'execution_orientation_deg': FINAL_ORIENTATION_DEG}

    def execute(self, plan_id, speed, armed, frame_verified, scene_verified=False, session_id=None):
        if isinstance(speed,bool) or not isinstance(speed,int) or not 1<=speed<=MAX_SDK_SPEED:
            raise ValueError(f'IK SDK speed must be an integer from 1 to {MAX_SDK_SPEED}')
        c=self.controller
        with c.lock:
            plan=self.plan
            if not plan or plan_id!=plan['id']:raise ValueError('No matching live-start plan; plan again')
            if session_id is not None:
                self._mouse_session(session_id)
                if plan.get('mouse_aim_session_id') != session_id:
                    raise ValueError('Tracking session does not own this plan')
            if plan.get('execution_mode')=='jog_qualification':
                from webui.hardware_jog import verify_execution_ready
                verify_execution_ready()
            if plan.get('hardware_v2_mouse'):
                self._mouse_session(plan['mouse_aim_session_id'])
                if self.v2_mouse is None:
                    raise ValueError('v2 mouse session is no longer active')
                return self.v2_mouse.execute(plan, speed, armed, frame_verified, scene_verified)
            if plan.get('mouse_aim_session_id'):
                self._mouse_session(plan['mouse_aim_session_id'])
            elif speed != plan.get('planned_sdk_speed'):
                raise ValueError('Arm speed changed after planning; plan again to review the scaled velocity profile')
            if time.monotonic()-plan['created']>plan.get('review_ttl_s', PLAN_TTL):raise ValueError('Plan expired; plan again')
            if not plan['result']['converged']:raise ValueError('IK failed to reach tolerance; execution refused')
            if frame_verified is not True:raise ValueError('Verify the URDF joint zero/axes and flange frame before execution')
            if plan.get('hardware_diffusion') or plan.get('hardware_joint_diagnostic'):
                from webui import hardware_diffusion
                if scene_verified is not True:
                    raise ValueError('Verify the table at −35 mm in g_base and clear unmodelled obstacles/tools before v2 execution')
                if speed > plan['max_sdk_speed']:
                    raise ValueError(f'Hardware execution requires SDK speed ≤{plan["max_sdk_speed"]}')
                if plan['result'].get('planner') == 'learned_rrt':
                    from simulation.diffusion.classical_service import info
                    status = info()
                else:
                    status = hardware_diffusion.verified_status()
                if (status.get('status') != 'ready' or status.get('environment') != plan['environment']
                        or status.get('checkpoint_sha256') != plan['checkpoint_sha256']):
                    raise ValueError('Fixed-target deployment changed after review; plan again')
            if plan.get('mouse_aim_session_id') and self.owns_running_mouse(plan['mouse_aim_session_id']):
                if armed is not True: raise ValueError('Enable movement after clearing the robot path')
                if c.generation != plan['generation']: raise StaleMousePlan('Robot session changed; plan again')
                if self.mouse_servo is None or self.mouse_pending_plan is not None:
                    raise StaleMousePlan('Mouse servo is applying its previous plan; retry the latest cursor')
                c._read(); c._health()
                rebase_path(self.mouse_servo.reference, c.angles, plan['vertices_deg'], plan['space'])
                self.plan = None
                self.mouse_pending_plan = {'plan':plan, 'speed':speed, 'created':time.monotonic(), 'confirmed':False}
                self.status['accepted_plan_id'] = plan['id']
                return c.snapshot()
            c._ready(armed, plan.get('mouse_aim_session_id'))
            if c.generation!=plan['generation'] or max(abs(a-b) for a,b in zip(c.angles,plan['start_deg']))>1:
                self.plan=None
                raise ValueError('Live pose or robot session changed; plan again')
            if plan.get('encoder_mapping') is not None:
                from webui.encoder_drag import verify
                verify(c.robot, c.angles, plan['encoder_mapping'])
            if plan.get('execution_mode') in ('path_waypoints','encoder_stream','encoder_profile','jog_qualification'):
                # Replacements on a straight leg must update the active command
                # rather than accumulate in a firmware queue. Stop/disconnect
                # already restore this server-owned mode lease.
                self.mouse_original_mode = self.read_refresh_mode()
                try:
                    self.enable_refresh_mode()
                except Exception:
                    self.release_mouse_mode()
                    raise
            c._cancel_timer()
            self.plan=None  # A plan is consumed once, never replayed.
            c.error=None; c.motion='ik';c.target=None
            self.heartbeat_at=time.monotonic()
            self.heartbeat_confirmed=False
            self.status={'state':'running', 'execution_id':plan['id'], 'method':plan['result']['method'],
                         'target_mm':plan['target_mm'], 'task':plan['result'].get('task','position'),
                         'target_axis':plan['target_axis'], 'completed_waypoints':0,
                         'total_waypoints':len(plan['waypoints']), 'sdk_speed':speed,
                         'joint_velocity_allowance_deg_s':plan['result'].get('joint_velocity_allowance_deg_s'),
                         'planned_duration_s':plan['result'].get('smoothing',{}).get('duration_s'),
                         'planning_timing_ms':plan['result'].get('planning_timing_ms'),
                         'execution_timeout_s':plan.get('execution_timeout_s', EXECUTION_TIMEOUT),
                         'arrival_deg':plan.get('arrival_deg', ARRIVAL_DEG),
                         'path_arrival_deg':plan.get('path_arrival_deg', plan.get('arrival_deg', ARRIVAL_DEG)),
                         'max_live_command_deg':plan.get('max_live_command_deg', plan.get('max_step_deg', MAX_STEP_DEG)+plan.get('arrival_deg', ARRIVAL_DEG)),
                         'execution_mode':plan.get('execution_mode', 'settled_waypoints'),
                         'joint_command':plan.get('joint_command', 'send_angles'),
                         'encoder_drag':plan['result'].get('encoder_drag'),
                         'completed_stages':0,
                         'total_stages':plan['result'].get('encoder_drag',{}).get('stage_count',0),
                         'hardware_diffusion':plan.get('hardware_diffusion', False),
                         'hardware_joint_diagnostic':plan.get('hardware_joint_diagnostic',False),
                         'diagnostic':plan['result'].get('diagnostic'),
                         'checkpoint_sha256':plan.get('checkpoint_sha256'),
                         'preview_frames':len(plan['result']['frames']),
                         'message':'Waiting for owning browser heartbeat', 'samples':[]}
            if plan.get('mouse_aim_session_id'):
                self.status['mouse_aim_session_id'] = plan['mouse_aim_session_id']
                self.status['active_plan_id'] = self.status['accepted_plan_id'] = plan['id']
            generation=c.generation
            thread=threading.Thread(target=self._run,args=(plan,speed,generation),daemon=True)
            thread.start()
            return c.snapshot()

    def heartbeat(self, execution_id):
        with self.controller.lock:
            if self.status['state']!='running' or execution_id!=self.status.get('execution_id'):
                raise ValueError('Execution is no longer active')
            self.heartbeat_at=time.monotonic()
            self.heartbeat_confirmed=True
            self.status['message']=('Tracking the v2 analytic joint velocity profile'
                                    if self.status.get('execution_mode') == 'encoder_profile'
                                    else 'Keeping the tool aimed at the cursor with collision-checked live feedback'
                                    if self.status.get('execution_mode') == 'encoder_feedback_tracking'
                                    else 'Streaming encoder targets with live tracking feedback'
                                    if self.status.get('execution_mode') == 'encoder_stream'
                                    else 'Following checked path; stopping at corners and endpoint'
                                    if self.status.get('execution_mode') == 'path_waypoints'
                                    else 'Executing telemetry-paced joint waypoints')
            return {'active':True}

    def _run(self, plan, speed, generation):
        if plan.get('execution_mode') == 'encoder_profile':
            from webui.encoder_profile import run
            return run(self, plan, speed, generation)
        if plan.get('execution_mode')=='jog_qualification':
            from webui.hardware_jog import run
            return run(self,plan,generation)
        if plan.get('mouse_aim_session_id'):
            return self._run_mouse(plan, speed, generation)
        c=self.controller;started=time.monotonic();index=0;sent_at=None
        allowance=plan['result'].get('joint_velocity_allowance_deg_s')
        command_profile = CommandProfile(c.angles[:], started,
            None if allowance is None else np.deg2rad(allowance))
        profile_clock_started=False
        mouse = bool(plan.get('mouse_aim_session_id'))
        arrival = plan.get('arrival_deg', ARRIVAL_DEG)
        max_step = plan.get('max_step_deg', MAX_STEP_DEG)
        stream = plan.get('execution_mode') == 'encoder_stream'
        trial = plan['result'].get('encoder_drag', {})
        staged = stream and trial.get('staged', False)
        stage_start = np.asarray(c.angles, float)
        execution_timeout = plan.get('execution_timeout_s', EXECUTION_TIMEOUT)
        path_tracking = plan.get('execution_mode') in ('path_waypoints', 'encoder_stream')
        settled_at = None
        settled_pose = None
        previous_encoders = (np.asarray(plan['encoder_mapping']['start_encoders'])
                             if stream else None)
        command_bound = plan.get('max_live_command_deg', max_step+arrival)
        scene = None
        try:
            if plan.get('hardware_diffusion') or plan.get('hardware_joint_diagnostic'):
                from simulation.diffusion.env import ArmEnv
                scene = ArmEnv()
                scene.set_obstacle(None)
                if scene.metadata() != plan['environment']:
                    raise RuntimeError('Execution collision scene differs from the reviewed v2 scene')
            while not c.closed.wait(MOUSE_POLL_SECONDS if mouse else .2):
                with c.lock:
                    if c.generation!=generation or c.motion!='ik' or not c.robot:return
                    now=time.monotonic()
                    if now-self.heartbeat_at>HEARTBEAT_TIMEOUT:raise RuntimeError('Browser heartbeat lost')
                    if now-started>execution_timeout:raise RuntimeError(f'IK execution exceeded its {execution_timeout:g} second route budget')
                    if not self.heartbeat_confirmed:continue
                    c._read()
                    telemetry_at = time.monotonic()
                    if not profile_clock_started:
                        command_profile=CommandProfile(c.angles[:],time.monotonic(),
                            None if allowance is None else np.deg2rad(allowance))
                        profile_clock_started=True
                    if path_tracking and self.read_refresh_mode() != 1:
                        raise RuntimeError('Hardware path tracking lost refresh mode 1')
                    if stream:
                        c._health()
                    if stream and time.monotonic()-now > 2.:
                        raise RuntimeError('Encoder stream telemetry/health cycle exceeded two seconds; Stop sent')
                    if scene is not None and not scene.valid(np.deg2rad(c.angles)):
                        raise RuntimeError('Live pose fails the reviewed v2 self/table/limit collision check')
                    if (stream and trial.get('maximum_excursion_deg') is not None
                            and max(abs(a-b) for a,b in zip(c.angles,plan['start_deg'])) > trial['maximum_excursion_deg']+.25):
                        raise RuntimeError('Encoder stream measured excursion exceeds the bounded trial')
                    if staged and np.max(np.abs(np.asarray(c.angles)-stage_start)) > 3.25:
                        raise RuntimeError('Encoder stage measured excursion exceeds the local trial bound')
                    position=ARM.fk(np.deg2rad(c.angles))*1000
                    error=float(np.linalg.norm(position-plan['target_mm']))
                    orientation_error=axis_error(ARM.transform(np.deg2rad(c.angles))[:3,2],np.array(plan['target_axis']))[2] if plan['target_axis'] is not None else None
                    sample={'orientation_error_deg':orientation_error,'time':(telemetry_at if stream else now)-started,'q':c.angles[:],'position_mm':position.tolist(),'error_mm':error}
                    if c.target is not None:
                        sample['command_deg'] = c.target[:]
                        sample['max_joint_error_deg'] = max(abs(a-b) for a,b in zip(c.angles,c.target))
                        sample['moving'] = c.moving
                    self.status['samples'].append(sample)
                    self.status['samples']=self.status['samples'][-1000:]
                    self.status['estimated_error_mm']=error
                    self.status['estimated_orientation_error_deg']=orientation_error
                    if sent_at is None and c.moving and command_profile.interval == 0:
                        raise RuntimeError('Unplanned motion detected before the next waypoint')
                    if sent_at is not None:
                        target=plan['waypoints'][index]
                        joint_error=max(abs(a-b) for a,b in zip(c.angles,target))
                        stop_required = (index in plan['stop_indices'] if path_tracking or mouse else True)
                        point_arrival = arrival if stop_required else plan.get('path_arrival_deg', arrival)
                        # Only interior points on the same checked straight leg
                        # may advance before stopped status. Turns and the goal
                        # still wait for the arm to stop; never queue blindly.
                        reached = joint_error <= point_arrival and (not stop_required or not c.moving)
                        if stream:
                            self.status['tracking_error_deg'] = joint_error
                            sample['tracking_error_deg'] = joint_error
                            if stop_required:
                                from webui.encoder_stream import SETTLE_S, SETTLE_DRIFT_DEG
                                if not reached:
                                    settled_at = settled_pose = None
                                elif settled_at is None or np.max(np.abs(np.asarray(c.angles)-settled_pose)) > SETTLE_DRIFT_DEG:
                                    settled_at, settled_pose = telemetry_at, np.asarray(c.angles)
                                reached = reached and telemetry_at-settled_at >= SETTLE_S
                        if reached and index+1 < len(plan['waypoints']):
                            next_error=max(abs(a-b) for a,b in zip(c.angles,plan['waypoints'][index+1]))
                            reached = next_error <= command_bound+1e-6
                        if not reached:
                            if stream:
                                self.status['tracking_paused'] = True
                                self.status['message'] = ('Observing stopped corner/endpoint' if stop_required
                                                          else 'Waiting for live joints before the next encoder reference')
                            if now-sent_at>SEGMENT_TIMEOUT:
                                self.status['command_deg'] = target[:]
                                self.status['max_joint_error_deg'] = joint_error
                                failure = 'settle' if stop_required else 'enter the path tracking allowance'
                                raise RuntimeError(f'Waypoint {index+1} failed to {failure} within {SEGMENT_TIMEOUT:g} seconds '
                                                   f'(max joint error {joint_error:.2f}°, tolerance {point_arrival:g}°, moving={c.moving})')
                            continue
                        if (plan.get('hardware_joint_diagnostic') and index==len(plan['waypoints'])-1
                                and now-sent_at<2.):
                            # Capture stopped-angle behaviour after the last packet,
                            # rather than ending a diagnostic on its first arrival.
                            continue
                        if staged and stop_required:
                            from webui.encoder_drag import verify
                            verify(c.robot, c.angles, plan['encoder_mapping'])
                            stage_pose = ARM.transform(np.deg2rad(plan['waypoints'][index]))
                            stage_error = float(np.linalg.norm(position-stage_pose[:3,3]*1000))
                            stage_orientation = axis_error(ARM.transform(np.deg2rad(c.angles))[:3,2], stage_pose[:3,2])[2]
                            if stage_error > FINAL_ERROR_MM or stage_orientation > FINAL_ORIENTATION_DEG:
                                raise RuntimeError('Stopped encoder stage misses its Cartesian/direction tolerance; plan again')
                            stage_start = np.asarray(c.angles, float)
                            self.status['completed_stages'] += 1
                            self.status['stage_error_mm'] = stage_error
                            sample.update(completed_stages=self.status['completed_stages'], stage_error_mm=stage_error)
                        index+=1;sent_at=None
                        settled_at=None
                        settled_pose=None
                        self.status['completed_waypoints']=index
                    if index==len(plan['waypoints']):
                        if error>FINAL_ERROR_MM:raise RuntimeError(f'Final URDF FK error {error:.1f} mm exceeds 5 mm')
                        if orientation_error is not None and orientation_error>FINAL_ORIENTATION_DEG:
                            raise RuntimeError(f'Final pointing error {orientation_error:.1f} deg exceeds 3 degrees')
                        if plan.get('cursor_mm') is not None:
                            delta = np.asarray(plan['cursor_mm'])-position
                            distance = float(np.linalg.norm(delta))
                            angle = axis_error(ARM.transform(np.deg2rad(c.angles))[:3, 2], delta)[2] if distance > 1e-6 else 180.
                            self.status.update(cursor_distance_mm=distance, cursor_direction_error_deg=angle)
                            if abs(distance-50.) > FINAL_ERROR_MM or angle > FINAL_ORIENTATION_DEG:
                                raise RuntimeError('Final live cursor gap or direction is outside the execution tolerance')
                        if path_tracking:
                            self.release_mouse_mode()
                        c.motion=c.target=None
                        self.status.update(state='completed', tracking_paused=False,
                            message=('Encoder stream completed with stopped endpoint; error is estimated from URDF FK'
                                     if stream else 'Joint waypoints completed; error is estimated from URDF FK'))
                        return
                    if not stream:
                        c._health()
                    if plan.get('encoder_mapping') is not None and time.monotonic()-now > 2.:
                        raise RuntimeError('Encoder trial telemetry/health cycle exceeded two seconds; Stop sent')
                    if time.monotonic()-self.heartbeat_at>HEARTBEAT_TIMEOUT:
                        raise RuntimeError('Browser heartbeat lost during robot health checks')
                    target=np.round(plan['waypoints'][index],2).tolist()
                    encoder_packet = None
                    if plan.get('encoder_mapping') is not None:
                        encoder_packet = plan['encoder_packets'][index]
                        target = plan['result']['encoder_drag']['quantized_waypoints_deg'][index]
                    intervals=plan.get('waypoint_intervals_s')
                    if intervals and time.monotonic()-command_profile.timestamp < intervals[index]:
                        self.status['message']='Following the speed-scaled fixed-target profile'
                        continue
                    if scene is not None and not scene.edge_valid(np.deg2rad(c.angles), np.deg2rad(target), .01):
                        raise RuntimeError('Live-to-command segment fails the reviewed v2 self/table/limit collision check')
                    if plan.get('space') is not None and not plan['space'].edge_valid(
                            np.deg2rad(c.angles), np.deg2rad(target), .04):
                        raise RuntimeError('Live-to-command segment violates joint limits or self-collision clearance')
                    # Require a small command from *actual* telemetry, including tracking tolerance.
                    if max(abs(a-b) for a,b in zip(c.angles,target))>command_bound+1e-6:
                        raise RuntimeError('Live tracking drift exceeds the waypoint bound')
                    if time.monotonic()-self.heartbeat_at>HEARTBEAT_TIMEOUT:
                        raise RuntimeError('Browser heartbeat lost during command validation')
                    command_sample = command_profile.candidate(target, time.monotonic())
                    if not command_sample['accepted']:
                        self.status['message'] = 'Waiting for specification-limited command timing'
                        continue
                    if encoder_packet is not None:
                        from webui.encoder_drag import send_once
                        encoder_speeds = encoder_packet['speeds']
                        if stream:
                            from webui.encoder_stream import paced_speeds
                            encoder_speeds = paced_speeds(encoder_packet, previous_encoders, command_sample['dt'])
                        c._send_motion(lambda:send_once(c.robot, encoder_packet['encoders'], encoder_speeds))
                    else:
                        c._send_motion(lambda:c.robot.send_angles(target,speed,_async=True))
                    checked_command = command_profile.commit(command_sample)
                    checked_command.update(command_send_time_s=command_sample['time']-started,
                                           sent_command_deg=target[:])
                    if encoder_packet is not None:
                        checked_command.update(sent_encoder_targets=encoder_packet['encoders'],
                            sent_encoder_speeds=encoder_speeds,
                            encoder_speed_units='nominal step/s', joint_command=plan['joint_command'])
                        if stream:
                            previous_encoders = np.asarray(encoder_packet['encoders'])
                            checked_command.update(reference_phase_s=encoder_packet['reference_phase_s'],
                                reference_velocity_deg_s=encoder_packet['reference_velocity_deg_s'],
                                actual_send_interval_s=command_sample['dt'], tracking_paused=False)
                    self.status.update(checked_command, motion_specification=profile())
                    self.status['samples'][-1].update(checked_command)
                    c.target=target;sent_at=time.monotonic()
                    self.status['command_deg'] = target[:]
        except Exception as exc:
            with c.lock:
                if c.generation!=generation or c.motion!='ik':return
                try:c._stop()
                except Exception:
                    c._close()
                c.error=f'IK execution stopped: {exc}. Plan again before retrying.'
                self.status.update(state='failed',message=c.error)
        finally:
            if scene is not None:
                scene.close()

    def _run_mouse(self, plan, speed, generation):
        c = self.controller
        started = previous_tick = progress_at = time.monotonic()
        execution_started = started
        command_profile = CommandProfile(c.angles[:], started)
        progress_q = np.asarray(plan['start_deg'])
        servo = None
        last_mode_check = 0
        targets, stops, index = [], set(), 0
        try:
            while not c.closed.wait(MOUSE_POLL_SECONDS):
                cycle_started = time.monotonic()
                with c.lock:
                    if c.generation != generation or c.motion != 'ik' or not c.robot:
                        return
                    now = time.monotonic()
                    if now-self.heartbeat_at > HEARTBEAT_TIMEOUT: raise RuntimeError('Browser heartbeat lost')
                    if now-started > EXECUTION_TIMEOUT: raise RuntimeError('Mouse route exceeded 180 seconds')
                    if not self.heartbeat_confirmed:
                        previous_tick = now
                        continue
                    telemetry_started = time.monotonic()
                    c._read()
                    telemetry_ms = 1000*(time.monotonic()-telemetry_started)
                    now = time.monotonic()
                    live = np.asarray(c.angles)
                    if servo is None:
                        if c.moving: raise RuntimeError('Unplanned motion detected before tracking started')
                        self.enable_refresh_mode()
                        last_mode_check = time.monotonic()
                        servo = self.mouse_servo = JointServo(live, speed)
                        targets, stops, _ = rebase_path(live, live, plan['vertices_deg'], plan['space'])
                        self.status['total_waypoints'] = len(targets)
                        previous_tick = time.monotonic()
                    elif now-last_mode_check >= .5:
                        if self.read_refresh_mode() != 1: raise RuntimeError('Command refresh mode changed during tracking')
                        last_mode_check = time.monotonic()
                    pending = self.mouse_pending_plan
                    if pending and now-pending['created'] > 2 and not pending['confirmed']:
                        self.mouse_pending_plan = None
                        self.status['accepted_plan_id'] = self.status['active_plan_id']
                        pending = None
                    if pending and pending['confirmed']:
                        candidate = pending['plan']
                        try:
                            targets_new, stops_new, distance = rebase_path(servo.reference, live, candidate['vertices_deg'], candidate['space'])
                        except StaleMousePlan as exc:
                            self.status['retarget_rejected'] = str(exc)
                        else:
                            plan, speed, targets, stops, index = candidate, pending['speed'], targets_new, stops_new, 0
                            servo.set_speed(speed)
                            started = progress_at = now
                            progress_q = live.copy()
                            self.status.update(active_plan_id=plan['id'], target_mm=plan['target_mm'], target_axis=plan['target_axis'],
                                               total_waypoints=len(targets), completed_waypoints=0, sdk_speed=speed,
                                               route_join_deg=distance, retarget_rejected=None)
                        self.mouse_pending_plan = None
                    position = ARM.fk(np.deg2rad(live))*1000
                    angle = axis_error(ARM.transform(np.deg2rad(live))[:3,2], np.asarray(plan['target_axis']))[2]
                    error = float(np.linalg.norm(position-plan['target_mm']))
                    sample = {'time':now-execution_started,'q':live.tolist(),'position_mm':position.tolist(),'error_mm':error,
                              'telemetry_read_ms':telemetry_ms,
                              'orientation_error_deg':angle,'active_plan_id':plan['id'],'moving':c.moving}
                    sample.update(reference_deg=servo.reference.tolist(), reference_velocity_deg_s=servo.velocity.tolist())
                    if c.target is not None:
                        sample.update(command_deg=c.target[:], max_joint_error_deg=float(np.max(np.abs(live-c.target))))
                    self.status['samples'] = (self.status['samples']+[sample])[-1000:]
                    self.status.update(estimated_error_mm=error, estimated_orientation_error_deg=angle,
                                       velocity_limit_deg_s=servo.max_velocity, acceleration_limit_deg_s2=servo.max_acceleration)
                    if np.max(np.abs(live-progress_q)) >= .1:
                        progress_at, progress_q = now, live.copy()
                    target = targets[index]
                    arrived = np.max(np.abs(live-target)) <= MOUSE_ARRIVAL_DEG
                    reference_settled = (np.max(np.abs(servo.reference-target)) <= MOUSE_ARRIVAL_DEG
                                         and np.max(np.abs(servo.velocity)) <= .1 and arrived and not c.moving)
                    joined = index not in stops and np.max(np.abs(servo.reference-target)) <= .25
                    while (servo.at_target(target) or reference_settled or joined) and (index not in stops or (arrived and not c.moving)):
                        index += 1
                        self.status['completed_waypoints'] = index
                        if index == len(targets):
                            if self.mouse_pending_plan:
                                index -= 1
                                break  # Release the lock so the pending replacement can be acknowledged.
                            if error > FINAL_ERROR_MM: raise RuntimeError(f'Final URDF FK error {error:.1f} mm exceeds 5 mm')
                            if angle > FINAL_ORIENTATION_DEG: raise RuntimeError(f'Final pointing error {angle:.1f} deg exceeds 3 degrees')
                            delta = np.asarray(plan['cursor_mm'])-position
                            distance = float(np.linalg.norm(delta))
                            cursor_angle = axis_error(ARM.transform(np.deg2rad(live))[:3,2],delta)[2] if distance > 1e-6 else 180.
                            self.status.update(cursor_distance_mm=distance,cursor_direction_error_deg=cursor_angle)
                            if abs(distance-50.) > FINAL_ERROR_MM or cursor_angle > FINAL_ORIENTATION_DEG:
                                raise RuntimeError('Final live cursor gap or direction is outside the execution tolerance')
                            c.motion = c.target = None
                            self.mouse_servo = None
                            self.status.update(state='completed',message='Velocity tracking reached the checked cursor goal')
                            return
                        target = targets[index]
                        arrived = np.max(np.abs(live-target)) <= MOUSE_ARRIVAL_DEG
                        reference_settled = (np.max(np.abs(servo.reference-target)) <= MOUSE_ARRIVAL_DEG
                                             and np.max(np.abs(servo.velocity)) <= .1 and arrived and not c.moving)
                        joined = index not in stops and np.max(np.abs(servo.reference-target)) <= .25
                        progress_at = now
                    if now-progress_at > SEGMENT_TIMEOUT:
                        self.status['route_target_deg'] = list(target)
                        self.status['route_joint_error_deg'] = float(np.max(np.abs(live-target)))
                        self.status['max_joint_error_deg'] = float(np.max(np.abs(live-c.target))) if c.target is not None else None
                        raise RuntimeError(f'Mouse tracking stalled for {SEGMENT_TIMEOUT:g} seconds '
                                           f'(route joint error {self.status["route_joint_error_deg"]:.2f}°, '
                                           f'command error {self.status["max_joint_error_deg"]})')
                    elapsed = time.monotonic()-previous_tick
                    self.status.update(cycle_ms=1000*elapsed, telemetry_read_ms=telemetry_ms)
                    if elapsed > MOUSE_MAX_CYCLE_SECONDS:
                        raise RuntimeError(f'Mouse servo telemetry cycle exceeded {MOUSE_MAX_CYCLE_SECONDS:g} seconds '
                                           f'({1000*elapsed:.0f} ms; telemetry {telemetry_ms:.0f} ms)')
                    dt = min(.15,max(.001,elapsed))
                    previous_tick = time.monotonic()
                    old_reference = servo.reference.copy()
                    old_velocity = servo.velocity.copy()
                    reference = servo.step(live,target,dt)
                    # Check both the reference interval and actual-to-command
                    # interval, including any curved retarget transition.
                    command = np.round(reference,2)
                    if not (plan['space'].edge_valid(np.deg2rad(old_reference),np.deg2rad(command),.04)
                            and plan['space'].edge_valid(np.deg2rad(live),np.deg2rad(command),.04)):
                        raise RuntimeError('Live-to-command segment violates joint limits or self-collision clearance')
                    self.status.update(reference_deg=reference.tolist(), reference_velocity_deg_s=servo.velocity.tolist())
                    if c.target is not None and command.tolist() == c.target:
                        continue
                    if c.target is None and np.array_equal(command,np.round(live,2)):
                        continue  # Do not make the Pi wait for an unchanged motor target.
                    c._health()
                    if time.monotonic()-cycle_started > MOUSE_MAX_CYCLE_SECONDS:
                        raise RuntimeError('Mouse servo telemetry/health cycle exceeded 2 seconds')
                    if time.monotonic()-self.heartbeat_at > HEARTBEAT_TIMEOUT: raise RuntimeError('Browser heartbeat lost during health checks')
                    command_sample = command_profile.candidate(command.tolist(), time.monotonic())
                    if not command_sample['accepted']:
                        servo.reference, servo.velocity = old_reference, old_velocity
                        self.status.update(reference_deg=old_reference.tolist(), reference_velocity_deg_s=old_velocity.tolist(),
                                           message='Waiting for specification-limited command timing')
                        continue
                    c._send_motion(lambda:c.robot.send_angles(command.tolist(),speed,_async=True))
                    checked_command = command_profile.commit(command_sample)
                    self.status.update(checked_command, motion_specification=profile())
                    self.status['samples'][-1].update(checked_command)
                    c.target = command.tolist()
                    self.status['command_deg'] = c.target[:]
        except Exception as exc:
            with c.lock:
                if c.generation != generation or c.motion != 'ik': return
                try: c._stop()
                except Exception: c._close()
                c.error = f'Mouse tracking stopped: {exc}. Start again before retrying.'
                self.status.update(state='failed',message=c.error)
