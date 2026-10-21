"""ROS-independent validation and single-owner checked-plan HTTP adapter."""
import copy
import json
import math
import threading
import urllib.request
import urllib.error
from urllib.parse import urlsplit


class LocalAPI:
    def __init__(self, url='http://127.0.0.1:8000', timeout=3.):
        parts = urlsplit(url)
        if parts.scheme != 'http' or parts.hostname not in ('localhost', '127.0.0.1'):
            raise ValueError('Robot API must be local HTTP')
        self.url, self.timeout, self.token = url.rstrip('/'), timeout, None

    def call(self, route, body=None):
        headers = {'Content-Type': 'application/json'}
        if body is not None:
            if not self.token:
                self.token = self.call('/api/state')['token']
            headers['X-Robot-Token'] = self.token
        request = urllib.request.Request(self.url+route, headers=headers,
            data=None if body is None else json.dumps(body, allow_nan=False).encode())
        timeout = 90. if route.endswith('/plan') else self.timeout
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except TimeoutError as exc:
            raise TimeoutError(f'{route} timed out after {timeout:g} seconds') from exc
        except urllib.error.HTTPError as exc:
            try:
                exc.msg = json.load(exc).get('error', exc.reason)
            except (ValueError, OSError):
                pass
            raise

    def plan_with_observer(self, body, observer):
        # Same checked planning API, with a diagnostic stream instead of one
        # JSON response. No execution occurs on this endpoint.
        if not self.token:
            self.token = self.call('/api/state')['token']
        request = urllib.request.Request(self.url+'/api/ik/mouse-aim/plan-stream',
            headers={'Content-Type':'application/json', 'X-Robot-Token':self.token},
            data=json.dumps(body, allow_nan=False).encode())
        try:
            with urllib.request.urlopen(request, timeout=90.) as response:
                for line in response:
                    message = json.loads(line)
                    if message['type'] == 'rrt':
                        for event in message['events']:
                            observer(event)
                    elif message['type'] == 'result':
                        return message['result']
                    elif message['type'] == 'error':
                        error = RuntimeError(message['error'])
                        error.code = message.get('code', 503)
                        raise error
            raise RuntimeError('Planning stream ended without a checked result')
        except urllib.error.HTTPError as exc:
            try:
                exc.msg = json.load(exc).get('error', exc.reason)
            except (ValueError, OSError):
                pass
            raise


def validate_detection(point, stamp, now, frame, expected_frame, previous=0., max_age=.5):
    if frame != expected_frame:
        raise ValueError('Unexpected detection frame')
    if len(point) != 3 or not all(math.isfinite(v) for v in point):
        raise ValueError('Object centre must contain three finite metres')
    if point[2] <= 0 or point[2] > 5:
        raise ValueError('Invalid optical depth (expected metres)')
    if not math.isfinite(stamp) or stamp <= 0 or stamp <= previous:
        raise ValueError('Missing or out-of-order capture stamp')
    if now-stamp > max_age or stamp > now+.02:
        raise ValueError('Stale or future detection')


class FollowSession:
    """Exclusive session with independent lease and planning workers.

    ROS callbacks only mutate small state under lock. HTTP never holds that
    state lock. command_lock orders execute/ACK against scoped Stop.
    """
    route = '/api/ik/mouse-aim/'

    def __init__(self, api, clock, max_age=.5, dry_run=True, speed=70, reference_speed=1.,
                 target_deadband_mm=30., approach_final_distance_mm=None, approach_step_mm=20.,
                 approach_distance_tolerance_mm=10., cone_distance_mm=350., cone_half_angle_deg=30.,
                 timings=None, visibility_camera=None, planning_visualization=None,
                 approach_camera_fov_fallback=False):
        self.timings = timings
        self.planning_visualization = planning_visualization
        self.visibility_camera = copy.deepcopy(visibility_camera)
        if type(approach_camera_fov_fallback) is not bool:
            raise ValueError('approach_camera_fov_fallback must be a boolean')
        self.orientation_mode = ('strict_then_fov' if approach_camera_fov_fallback
                                 and approach_final_distance_mm is not None else 'point_at_object')
        self.orientation_selection = None
        if isinstance(target_deadband_mm, bool) or not math.isfinite(target_deadband_mm) or target_deadband_mm < 0:
            raise ValueError('Target deadband must be finite nonnegative millimetres')
        self.target_deadband_mm = target_deadband_mm
        from simulation.tracking_geometry import region_for
        cone = region_for([0.,0.,0.], [0.,0.,1.],
            {'distance_mm':cone_distance_mm,'half_angle_deg':cone_half_angle_deg})
        self.initial_distance_mm = self.region_distance_mm = cone['distance_mm']
        self.cone_half_angle_deg = cone['half_angle_deg']
        self.approach = None
        if approach_final_distance_mm is not None:
            from simulation.tracking_geometry import approach_settings
            self.approach = approach_settings({'final_distance_mm': approach_final_distance_mm,
                'step_mm': approach_step_mm, 'distance_tolerance_mm': approach_distance_tolerance_mm}, self.region_settings)
        self.accepted_distance_mm = None
        self.api, self.clock, self.max_age, self.dry_run = api, clock, max_age, dry_run
        self.speed, self.reference_speed = speed, reference_speed
        self.lock = threading.RLock()
        self.command_lock = threading.Lock()
        self.armed = False
        self.paused = self.pause_pending = False
        self.session = self.execution = self.plan = None
        self.latest = None
        self.generation = 0
        self.planning_seed = 42
        self.busy = self.control_busy = False
        self.last_error = ''
        self.accepted_target = None
        self.accepted_at = self.control_at = self.retry_at = -float('inf')
        self.replan_required = False
        self.stop_threads = []
        self.backend_status = {}

    @property
    def region_settings(self):
        return {'mode': 'cone_disk', 'distance_mm': self.region_distance_mm, 'half_angle_deg': self.cone_half_angle_deg}

    @property
    def approach_status(self):
        with self.lock:
            return {'enabled': self.approach is not None, 'initial_distance_mm': self.initial_distance_mm,
                    'distance_mm': self.region_distance_mm,
                    'final_distance_mm': self.approach['final_distance_mm'] if self.approach else self.initial_distance_mm,
                    'step_mm': self.approach['step_mm'] if self.approach else 0.,
                    'distance_tolerance_mm': self.approach['distance_tolerance_mm'] if self.approach else 0.,
                    'phase': ('disabled' if not self.approach else 'disarmed' if not self.armed
                              else 'waiting' if self.paused else 'holding' if
                              self.region_distance_mm == self.approach['final_distance_mm']
                              and self.backend_status.get('final_goal_accepted') is True
                              and (self.backend_status.get('tracking_region') or {}).get('distance_mm') == self.region_distance_mm
                              and self.latest is not None and not self.busy
                              and (self.backend_status.get('tracking_region') or {}).get('cursor_mm') == self.latest[0]
                              and self.backend_status.get('state') == 'completed' else 'approaching')}

    def _current(self, generation, session):
        return self.armed and generation == self.generation and session == self.session

    def arm(self):
        with self.lock:
            if self.armed:
                return
            self.region_distance_mm = self.initial_distance_mm
            self.accepted_distance_mm = None
            self.planning_seed = 42
            generation = self.generation
        session = None
        if not self.dry_run:
            body = {
                'armed': True, 'frame_verified': True, 'scene_verified': True,
                'client_id': 'ros2_object_follow', 'tracking_engine': 'learned_rrt',
                'speed': self.speed, 'speed_rad_s': self.reference_speed,
                'velocity_shape': {'kind': 'smooth_trapezoid', 'rise_percent': 5, 'fall_percent': 10},
                'tracking_region': self.region_settings}
            if self.approach is not None:
                body['approach'] = dict(self.approach)
            if self.orientation_mode == 'strict_then_fov':
                body['orientation_mode'] = self.orientation_mode
            result = self.api.call(self.route+'start', body)
            session = result['session_id']
            ik = result.get('ik', {})
            if (ik.get('mouse_aim_session_id') != session or ik.get('mouse_aim_client_id') != 'ros2_object_follow'
                or ik.get('mouse_executor') != 'encoder_feedback_tracking'
                or ik.get('mouse_feedback_tracking_protocol') != 1
                or ik.get('mouse_tracking_region_protocol') != 1
                or ik.get('mouse_session_owner_protocol') != 1
                or ik.get('mouse_pause_protocol') != 1
                or ik.get('tracking_region_settings') != self.region_settings
                or ik.get('velocity_shape') != {'kind':'smooth_trapezoid','rise_percent':5,'fall_percent':10}):
                self.api.call(self.route+'stop', {'session_id': session})
                raise ValueError('Server lacks the verified exclusive encoder tracking protocol')
            if self.approach is not None and (ik.get('mouse_approach_protocol') != 1
                    or ik.get('approach_settings') != self.approach):
                self.api.call(self.route+'stop', {'session_id': session})
                raise ValueError('Server lacks the verified approach protocol')
            if self.visibility_camera is not None and ik.get('visibility_camera') != self.visibility_camera:
                self.api.call(self.route+'stop', {'session_id': session})
                raise ValueError('ROS optical mount differs from the controller visibility camera')
            if self.orientation_mode == 'strict_then_fov' and (
                    ik.get('mouse_orientation_policy_protocol') != 1
                    or ik.get('requested_orientation_mode') != self.orientation_mode):
                self.api.call(self.route+'stop', {'session_id': session})
                raise ValueError('Server lacks the verified strict-first camera orientation protocol')
        with self.lock:
            if generation != self.generation:
                if session:
                    self._scoped_stop(session)
                raise ValueError('Arming cancelled by Stop')
            self.session = session
            self.armed = True
            self.paused = self.pause_pending = False
            self.last_error = ''
            self.backend_status = {}
            self.orientation_selection = None
            self.generation += 1
            self.accepted_target = None
            self.replan_required = False
            self.accepted_at = self.control_at = self.retry_at = -float('inf')
            if self.latest is None or self.clock()-self.latest[1] > self.max_age:
                self.pause('Waiting for object')

    def submit(self, point_mm, stamp):
        if len(point_mm) != 3 or not all(math.isfinite(v) for v in point_mm):
            raise ValueError('Invalid transformed target')
        with self.lock:
            if self.latest and stamp <= self.latest[1]:
                raise ValueError('Out-of-order transformed target')
            # Compare against the held target, so gradual motion eventually
            # crosses the threshold. Fresh accepted data still refreshes age.
            held = (self.latest[0] if self.latest and
                math.dist(point_mm, self.latest[0]) <= self.target_deadband_mm else list(point_mm))
            if self.latest is None or held != self.latest[0]:
                self.planning_seed = 42
            self.latest = (held, stamp)

    def _scoped_stop(self, session):
        def work():
            try:
                with self.command_lock:
                    self.api.call(self.route+'stop', {'session_id': session})
            except Exception as exc:
                self.last_error = 'Scoped stop request failed: '+str(exc)
        thread = threading.Thread(target=work, daemon=True)
        self.stop_threads.append(thread)
        thread.start()

    def stop(self, reason='Stopped'):
        with self.lock:
            self.armed = False
            self.paused = self.pause_pending = False
            self.generation += 1
            session = self.session
            self.session = self.execution = self.plan = None
            self.latest = None
            self.last_error = reason
        if session:
            self._scoped_stop(session)

    def pause(self, reason='Detector LOST'):
        """Stop motion but keep the session leased until a fresh target arrives."""
        with self.lock:
            self.latest = None
            if not self.armed:
                # LOST is normal while arming into the waiting state. Only
                # explicit Stop/faults invalidate an in-flight Arm request.
                if not self.last_error:
                    self.last_error = reason
                return
            self.last_error = reason
            if self.paused:
                return
            self.paused = True
            self.generation += 1
            generation, session = self.generation, self.session
            self.execution = self.plan = self.accepted_target = None
            self.replan_required = True
            self.accepted_at = self.retry_at = -float('inf')
            self.pause_pending = not self.dry_run
        if self.dry_run:
            return
        def work():
            try:
                with self.command_lock:
                    with self.lock:
                        if not self._current(generation, session):
                            return
                    state = self.api.call(self.route+'pause', {'session_id': session})
                    if state.get('ik', {}).get('state') != 'paused':
                        raise ValueError('Backend did not confirm holding position')
                    with self.lock:
                        if self._current(generation, session):
                            self.observe(state, generation, session)
                            self.pause_pending = False
            except Exception as exc:
                with self.lock:
                    if self._current(generation, session):
                        self.stop('Hold request failed: '+str(exc))
        thread = threading.Thread(target=work, daemon=True)
        self.stop_threads = [t for t in self.stop_threads if t.is_alive()]
        self.stop_threads.append(thread)
        thread.start()

    def observe(self, state, generation, session):
        with self.lock:
            if not self._current(generation, session):
                return
            ik = state.get('ik', {})
            self.backend_status = ik
            if not state.get('connected') or ik.get('mouse_aim_session_id') != session:
                self.stop('Backend session ended')
                return
            status = ik.get('state')
            if status in ('failed', 'stopped', 'disconnected'):
                self.stop(ik.get('message', 'Backend stopped'))
                return
            if status == 'completed' and ik.get('execution_id') == self.execution:
                self.execution = None
                self.plan = None
            # A completed partial prefix or a rejected join must continue from
            # new measured feedback even when the target has not moved.
            if self.paused:
                if self.approach is not None:
                    held = (ik.get('tracking_region_settings') or {}).get('distance_mm')
                    if held is not None and self.approach['final_distance_mm'] <= held <= self.initial_distance_mm:
                        self.region_distance_mm = held
                return
            self.replan_required = bool(ik.get('feedback_replan_required') or
                (status == 'completed' and (ik.get('replan_required') or not ik.get('tracking_accepted', True))))
            region = ik.get('tracking_region') or {}
            if (self.approach is not None and status == 'completed' and not self.busy
                    and ik.get('final_goal_accepted') is True and not self.replan_required
                    and self.latest is not None and self.clock()-self.latest[1] <= self.max_age
                    and region.get('distance_mm') == self.region_distance_mm
                    and region.get('cursor_mm') == self.latest[0]):
                self.region_distance_mm = max(self.approach['final_distance_mm'],
                    self.region_distance_mm-self.approach['step_mm'])

    def tick(self):
        """Nonblocking callback: stale guard plus independently paced workers."""
        with self.lock:
            if not self.armed:
                return
            now = self.clock()
            if not self.latest or now-self.latest[1] > self.max_age:
                if not self.paused:
                    self.pause('Accepted detection expired')
            elif self.paused and not self.pause_pending:
                self.paused = False
                self.last_error = ''
            if self.dry_run:
                return
            generation, session = self.generation, self.session
            if not self.control_busy and now-self.control_at >= .5:
                self.control_busy = True
                self.control_at = now
                threading.Thread(target=self._control, args=(generation, session), daemon=True).start()
            if self.paused:
                return
            moved = (self.accepted_target is None or math.dist(self.latest[0], self.accepted_target) > 1e-9
                     or self.accepted_distance_mm != self.region_distance_mm)
            if self.busy or now < self.retry_at or now-self.accepted_at < .15 or not (moved or self.replan_required):
                return
            self.busy = True
            target = self.latest
            threading.Thread(target=self._plan, args=(target, generation, session), daemon=True).start()

    def _control(self, generation, session):
        try:
            state = self.api.call('/api/state')
            self.observe(state, generation, session)
            with self.lock:
                if not self._current(generation, session):
                    return
                execution = self.execution
            body = {'session_id': session}
            if execution:
                body['execution_id'] = execution
            try:
                self.api.call(self.route+'heartbeat', body)
            except Exception:
                # A short checked path can complete between GET and heartbeat.
                state = self.api.call('/api/state')
                self.observe(state, generation, session)
                with self.lock:
                    completed = self._current(generation, session) and self.execution is None and state.get('ik', {}).get('state') == 'completed'
                if not completed:
                    raise
        except Exception as exc:
            with self.lock:
                if self._current(generation, session):
                    self.stop(str(exc))
        finally:
            with self.lock:
                self.control_busy = False

    def _plan(self, target, generation, session):
        try:
            if self.timings is not None:
                self.timings.add('detection.capture_to_plan_start', (self.clock()-target[1])*1000)
            with self.lock:
                region = self.region_settings
                seed = self.planning_seed
            body = {'session_id': session, 'cursor_mm': target[0], 'seed': seed}
            if self.approach is not None:
                body['tracking_region'] = region
            diagnostic = None
            if self.planning_visualization is not None and callable(getattr(self.api, 'plan_with_observer', None)):
                diagnostic = self.planning_visualization.begin()
            result = (self.api.plan_with_observer(body, diagnostic[1]) if diagnostic
                      else self.api.call(self.route+'plan', body))
            with self.command_lock:
                with self.lock:
                    if not self._current(generation, session):
                        return
                    if not self.latest or self.clock()-self.latest[1] > self.max_age:
                        self.pause('Accepted detection expired during planning')
                        return
                    # submit() already filters jitter against the held center.
                    # A changed held target invalidates the old proposal.
                    if math.dist(target[0], self.latest[0]) > 1e-9:
                        self.retry_at = self.clock()+.15
                        return
                plan = result.get('plan')
                if result.get('session_id') != session:
                    raise ValueError('Planning response changed session')
                with self.lock:
                    self.orientation_selection = copy.deepcopy(result.get('result', {}).get('orientation_selection'))
                if not plan or not result.get('result', {}).get('playable'):
                    with self.lock:
                        # A bounded stochastic search failure is not proof of
                        # infeasibility. Repeating the same seed from the held
                        # pose only repeats that exhausted search. Keep the
                        # frozen region and all per-request budgets unchanged.
                        if result.get('result', {}).get('status') in ('no_path', 'ik_failed'):
                            self.planning_seed = (seed+1) % (2**31)
                        self.last_error = 'No checked playable continuation'
                        self.retry_at = self.clock()+.15
                    return
                from simulation.diffusion.orientation_policy import effective_orientation_mode
                effective_orientation_mode(result['result'], self.orientation_mode)
                if plan.get('execution_mode') != 'encoder_feedback_tracking' or plan.get('joint_command') != 'encoder_drag_profile':
                    raise ValueError('Plan changed checked encoder executor')
                if diagnostic:
                    self.planning_visualization.accepted(diagnostic[0], result)
                plan_id = plan['id']
                response = self.api.call('/api/ik/execute', {'plan_id': plan_id,
                    'session_id': session, 'speed': self.speed, 'armed': True,
                    'frame_verified': True, 'scene_verified': True})
                if self.timings is not None:
                    self.timings.add('detection.capture_to_execute_return', (self.clock()-target[1])*1000)
                ik = response.get('ik', {})
                execution = ik.get('execution_id')
                if (not execution or ik.get('mouse_aim_session_id') != session or
                    (execution != plan_id and ik.get('accepted_plan_id') != plan_id)):
                    raise ValueError('Execute returned a different owner or candidate')
                with self.lock:
                    if not self._current(generation, session):
                        return
                    self.execution = execution
                ack = {'session_id': session, 'execution_id': execution}
                if execution != plan_id:
                    ack['plan_id'] = plan_id
                try:
                    self.api.call(self.route+'heartbeat', ack)
                except Exception:
                    state = self.api.call('/api/state')
                    final = state.get('ik', {})
                    if not (final.get('state') == 'completed' and final.get('execution_id') == execution
                        and final.get('mouse_aim_session_id') == session
                        and (execution == plan_id or final.get('active_plan_id') == plan_id)):
                        raise
                    self.observe(state, generation, session)
                with self.lock:
                    if self._current(generation, session):
                        self.plan = plan_id
                        self.accepted_target = target[0]
                        self.accepted_distance_mm = region['distance_mm']
                        self.accepted_at = self.clock()
                        self.planning_seed = 42
                        self.replan_required = False
                        self.last_error = ''
        except Exception as exc:
            with self.lock:
                if self._current(generation, session):
                    if getattr(exc, 'code', None) == 409:
                        self.last_error = str(exc)
                        self.retry_at = self.clock()+.15
                        self.replan_required = True
                    else:
                        self.stop(str(exc))
        finally:
            with self.lock:
                self.busy = False
