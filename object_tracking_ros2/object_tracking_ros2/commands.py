"""Exclusive checked joint and pose commands using the selected production planner."""
import math
import threading
import time


class CheckedCommands:
    def __init__(self, api, settings, tracking):
        self.api, self.settings, self.tracking = api, settings, tracking
        self.lock = threading.RLock()
        self.command_lock = threading.Lock()
        self.enabled = self.busy = self.stop_pending = self.control_busy = False
        self.generation = 0
        self.execution = None
        self.threads = []
        self.control_at = -math.inf
        self._status = {'state': 'disabled', 'reason': '', 'command_kind': None}

    @property
    def status(self):
        with self.lock:
            return {**self._status, 'enabled': self.enabled, 'planning': self.busy,
                'stop_pending': self.stop_pending, 'execution_id': self.execution,
                'planner': 'learned_rrt', 'joint_command': 'encoder_drag_profile',
                'orientation_mode': 'position_and_tool_z_axis_roll_free'}

    def _spawn(self, work):
        thread = threading.Thread(target=work, daemon=True)
        with self.lock:
            self.threads = [t for t in self.threads if t.is_alive()]
            self.threads.append(thread)
        thread.start()

    def enable(self):
        with self.lock:
            if self.tracking.armed or self.tracking.session:
                raise ValueError('Stop object following before enabling joint/pose commands')
            if self.busy or self.stop_pending or self.execution:
                raise ValueError('Wait for the active command or Stop to finish')
            if self.settings('dry_run'):
                raise ValueError('Joint/pose execution requires hardware mode')
            if not self.settings('frame_verified') or not self.settings('scene_verified'):
                raise ValueError('Verified robot frame and collision scene are required')
            generation = self.generation
        state = self.api.call('/api/state')
        if not state.get('connected') or state.get('motion') or state.get('moving') or state.get('ik', {}).get('mouse_aim_session_id'):
            raise ValueError('General commands require a connected idle arm with tracking stopped')
        with self.lock:
            if generation != self.generation or self.tracking.armed or self.tracking.session:
                raise ValueError('Enabling motion was cancelled')
            self.enabled = True
            self._status = {'state': 'ready', 'reason': '', 'command_kind': None}
        return 'Checked joint/pose commands enabled'

    def stop(self, reason='Stopped'):
        with self.lock:
            self.enabled = False
            self.generation += 1
            self.execution = None
            self._status.update(state='stopped', reason=reason)
            if self.stop_pending:
                return
            self.stop_pending = True
        def work():
            try:
                with self.command_lock:
                    state = self.api.call('/api/state')
                    if state.get('connected'):
                        self.api.call('/api/stop', {})
            except Exception as exc:
                with self.lock:
                    self._status['reason'] = reason+'; Stop delivery failed: '+str(exc)
            finally:
                with self.lock:
                    self.stop_pending = False
        self._spawn(work)

    def _stamp(self, stamp):
        now = time.time()
        if not math.isfinite(stamp) or stamp <= 0 or stamp > now+.02 or now-stamp > self.settings('max_detection_age_s'):
            raise ValueError('Command requires a fresh wall-clock header stamp')

    def submit_joints(self, names, positions, stamp):
        self._stamp(stamp)
        from simulation.kinematics import ARM
        import numpy as np
        if len(names) != 6 or len(set(names)) != 6 or set(names) != set(ARM.names) or len(positions) != 6:
            raise ValueError('Joint command must name each of the six URDF joints exactly once')
        if not all(math.isfinite(v) and not isinstance(v, bool) for v in positions):
            raise ValueError('Joint positions must be six finite radians')
        q = np.asarray([positions[names.index(name)] for name in ARM.names])
        if np.any(q < ARM.lower) or np.any(q > ARM.upper):
            raise ValueError('Joint command exceeds operating limits')
        transform = ARM.transform(q)
        self._submit('joints', {'goal_deg': np.rad2deg(q).tolist(),
            'target_mm': (transform[:3, 3]*1000).tolist(), 'target_axis': transform[:3, 2].tolist()})

    def submit_pose(self, frame, position_m, quaternion, stamp):
        self._stamp(stamp)
        if frame != self.settings('base_frame'):
            raise ValueError('Pose command must use the configured robot base frame')
        if len(position_m) != 3 or not all(math.isfinite(v) and abs(v) <= .6 for v in position_m):
            raise ValueError('Pose position must be finite metres within the planning workspace')
        if len(quaternion) != 4 or not all(math.isfinite(v) for v in quaternion) or abs(sum(v*v for v in quaternion)-1) > 1e-5:
            raise ValueError('Pose orientation must be a normalized [x,y,z,w] quaternion')
        x,y,z,w = quaternion
        # The evaluated IK task constrains XYZ and flange Z, with free tool roll.
        axis = [2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)]
        self._submit('pose', {'target_mm': [v*1000 for v in position_m], 'target_axis': axis})

    def _submit(self, kind, goal):
        with self.lock:
            if not self.enabled or self.tracking.armed or self.tracking.session:
                raise ValueError('Enable general motion with object following stopped first')
            if self.busy or self.execution or self.stop_pending:
                raise ValueError('A checked command is already active; wait or Stop it')
            self.busy = True
            generation = self.generation
            self._status.update(state='planning', reason='', command_kind=kind)
        self._spawn(lambda: self._run(goal, generation))

    def _current(self, generation):
        return self.enabled and generation == self.generation

    def _run(self, goal, generation):
        try:
            speed = self.settings('sdk_speed')
            result = self.api.call('/api/ik/diffusion/plan', {**goal,
                'planner': 'learned_rrt', 'joint_command': 'encoder_drag_profile', 'speed': speed,
                'velocity_shape': {'kind': 'smooth_trapezoid', 'rise_percent': 5, 'fall_percent': 10}})
            with self.command_lock:
                with self.lock:
                    if not self._current(generation):
                        return
                plan = result.get('plan')
                if not plan or not result.get('result', {}).get('converged'):
                    with self.lock:
                        self._status.update(state='rejected', reason=result.get('result', {}).get('reason', 'No checked path to the command'))
                    return
                if (plan.get('planner') != 'learned_rrt' or plan.get('joint_command') != 'encoder_drag_profile'
                        or plan.get('execution_mode') != 'encoder_profile'):
                    raise ValueError('Command changed the selected checked planner or executor')
                response = self.api.call('/api/ik/execute', {'plan_id': plan['id'], 'speed': speed,
                    'armed': True, 'frame_verified': True, 'scene_verified': True})
                if response.get('ik', {}).get('execution_id') != plan['id']:
                    raise ValueError('Command execution owner changed')
                with self.lock:
                    if not self._current(generation):
                        return
                    self.execution = plan['id']
                self.api.call('/api/ik/heartbeat', {'execution_id': plan['id']})
                with self.lock:
                    if self._current(generation):
                        self._status.update(state='running', reason='')
        except Exception as exc:
            with self.lock:
                if self._current(generation):
                    self.stop(str(exc))
        finally:
            with self.lock:
                self.busy = False

    def tick(self):
        with self.lock:
            if not self.enabled or self.execution is None or self.control_busy or time.monotonic()-self.control_at < .5:
                return
            generation, execution = self.generation, self.execution
            self.control_at = time.monotonic()
            self.control_busy = True
        def work():
            try:
                state = self.api.call('/api/state')
                ik = state.get('ik', {})
                with self.lock:
                    if not self._current(generation) or self.execution != execution:
                        return
                    if not state.get('connected') or ik.get('execution_id') != execution or ik.get('state') not in ('running', 'completed'):
                        raise ValueError(ik.get('message', 'Hardware command stopped'))
                    if ik['state'] == 'completed':
                        self.execution = None
                        self._status.update(state='completed', reason=ik.get('message', 'Command completed'))
                        return
                try:
                    self.api.call('/api/ik/heartbeat', {'execution_id': execution})
                except Exception:
                    # Completion can happen after the preceding state read.
                    final = self.api.call('/api/state').get('ik', {})
                    if final.get('state') != 'completed' or final.get('execution_id') != execution:
                        raise
            except Exception as exc:
                with self.lock:
                    if self._current(generation):
                        self.stop(str(exc))
            finally:
                with self.lock:
                    self.control_busy = False
        self._spawn(work)

    def shutdown(self):
        self.stop('Controller shutting down')
        for thread in list(self.threads):
            thread.join(2.)
