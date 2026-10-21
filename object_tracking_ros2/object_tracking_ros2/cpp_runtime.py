"""ROS-free checked runtime embedded by the rclcpp tracking node.

Python continues to own the reviewed planner, tracking state machine and arm
executor. This module does not import rclpy or the WebUI HTTP server.
"""
import math
import threading
import time
import json
from .core import FollowSession, LocalAPI
from .commands import CheckedCommands
from .timing import TimingStats, TimedBackend


class Runtime:
    def __init__(self, settings, clock, api=None):
        self.settings = dict(settings)
        self.clock = clock
        self.value = self.settings.__getitem__
        from simulation.camera_model import CameraModel
        self.visibility_camera = CameraModel.from_ros_settings(self.settings)
        if api is None:
            if self.value('backend') == 'direct':
                from .direct import DirectBackend
                api = DirectBackend(camera=self.visibility_camera)
            elif self.value('backend') == 'http':
                api = LocalAPI(self.value('api_url'))
            else:
                raise ValueError('backend must be direct or http')
        self.timings = TimingStats()
        from .planning_visualization import PlanningVisualization
        self.planning_visualization = PlanningVisualization()
        self.api = TimedBackend(api, self.timings)
        self.session = FollowSession(self.api, clock, self.value('max_detection_age_s'),
            self.value('dry_run'), self.value('sdk_speed'), self.value('reference_speed_rad_s'),
            self.value('target_deadband_mm'),
            self.value('approach_final_distance_mm') if self.value('approach_enabled') else None,
            self.value('approach_step_mm'), self.value('approach_distance_tolerance_mm'),
            self.value('cone_distance_mm'), self.value('cone_half_angle_deg'), timings=self.timings,
            visibility_camera=self.visibility_camera.configuration(),
            planning_visualization=self.planning_visualization,
            approach_camera_fov_fallback=self.settings.get('approach_camera_fov_fallback', True))
        self.commands = CheckedCommands(self.api, self.value, self.session)
        self.connection_lock = threading.Lock()
        self.connection_generation = self.connection_pending = 0
        self.shutting_down = False
        self.api_connected = self.feedback_available = False
        self.last_feedback_stamp = 0.
        self.feedback_error = 'Waiting for backend feedback'
        self.detector_tracking = False
        self.sequence = 0

    def set_detector_state(self, state):
        self.detector_tracking = state == 'TRACKING'
        if not self.detector_tracking:
            self.session.pause('Detector '+state)

    def detection_error(self, message, expired=False):
        self.session.last_error = message
        if self.session.armed:
            if expired:
                self.session.pause('Accepted detection expired')
            else:
                self.session.stop(message)

    def submit(self, point_mm, stamp):
        self.session.submit(point_mm, stamp)
        self.sequence += 1
        if self.session.armed and not self.session.paused:
            self.session.last_error = ''
        return self.session.latest[0]

    def connect(self, enabled):
        self.connection_pending += 1
        if not enabled:
            self.connection_generation += 1
            self.session.stop('Disconnect requested')
            self.commands.stop('Disconnect requested')
        generation = self.connection_generation
        try:
            with self.connection_lock:
                if enabled:
                    if generation != self.connection_generation or self.shutting_down:
                        raise RuntimeError('Connect cancelled')
                    self.api.call('/api/connect', {'host': self.value('robot_host'), 'port': self.value('robot_port')})
                    if generation != self.connection_generation or self.shutting_down:
                        self.api.call('/api/disconnect', {})
                        raise RuntimeError('Connect cancelled')
                else:
                    self.api.call('/api/disconnect', {})
                    self.api_connected = self.feedback_available = False
                    self.last_feedback_stamp = 0.
                    self.feedback_error = 'Robot disconnected'
            return 'Connected; motion remains disarmed' if enabled else 'Disconnected'
        finally:
            self.connection_pending -= 1

    def ready(self, use_sim_time):
        if self.connection_pending or self.shutting_down:
            raise ValueError('Robot connection is changing; motion is unavailable')
        if not self.value('dry_run'):
            if use_sim_time:
                raise ValueError('Hardware motion requires wall-clock ROS time')
            if not (self.value('frame_verified') and self.value('scene_verified')):
                raise ValueError('Hardware robot frame and scene verification required')
            if (not self.api_connected or not self.feedback_available
                    or self.clock()-self.last_feedback_stamp > self.value('max_detection_age_s')):
                raise ValueError('Fresh measured backend feedback is required before arming')

    def arm(self, use_sim_time=False):
        self.ready(use_sim_time)
        if self.commands.enabled or self.commands.busy or self.commands.stop_pending:
            raise ValueError('Stop general arm motion before arming tracking')
        if not (self.value('mount_configured') or self.value('mount_calibrated')):
            raise ValueError('Configured camera mount required')
        self.session.arm()
        return ('Armed (dry run)' if self.session.dry_run else 'Armed') + (
            '; waiting for object' if self.session.paused else '')

    def enable_motion(self, use_sim_time=False):
        self.ready(use_sim_time)
        return self.commands.enable()

    def stop(self):
        self.session.stop()
        self.commands.stop()
        return 'Disarmed; Stop requested'

    def command_joints(self, names, positions, stamp, use_sim_time=False):
        self.ready(use_sim_time)
        self.commands.submit_joints(names, positions, stamp)

    def command_pose(self, frame, position, quaternion, stamp, use_sim_time=False):
        self.ready(use_sim_time)
        self.commands.submit_pose(frame, position, quaternion, stamp)

    def feedback(self):
        state = self.api.call('/api/feedback')
        self.api_connected = True
        pose = state.get('feedback_pose')
        self.feedback_available = bool(pose)
        if not pose:
            self.feedback_error = 'Backend has no measured robot feedback (disconnected)'
            return []
        return state.get('feedback_poses', [pose])

    def feedback_health(self, stamp, error=''):
        self.feedback_available = not bool(error)
        self.feedback_error = error
        if not error:
            self.last_feedback_stamp = max(self.last_feedback_stamp, stamp)

    def feedback_failure(self, error):
        self.feedback_available = self.api_connected = False
        self.feedback_error = error

    def tick(self, show_rrt=None):
        enabled = self.settings.get('show_rrt', True) if show_rrt is None else show_rrt
        self.planning_visualization.set_enabled(enabled and self.session.armed and not self.session.paused)
        self.session.tick()
        self.commands.tick()

    def planning_display(self):
        return self.planning_visualization.message()

    def status(self):
        now = self.clock()
        with self.session.lock:
            s = self.session
            gaze_keys = ('estimated_orientation_error_deg', 'estimated_error_mm', 'tracking_accepted',
                'cursor_direction_error_deg', 'aim_error_deg', 'tracking_region_acceptance',
                'final_goal_accepted', 'cursor_distance_mm', 'reference_lag_s', 'self_collision_checked',
                'feedback_replan_required', 'replan_required', 'tracking_region',
                'orientation_mode', 'orientation_selection', 'last_planning_orientation_mode', 'last_orientation_selection')
            return json.dumps({'implementation': 'cpp', 'armed': s.armed, 'dry_run': s.dry_run,
                'backend': self.value('backend'), 'arm_motion': self.commands.status,
                'paused': s.paused, 'pause_pending': s.pause_pending, 'reason': s.last_error,
                'sequence': self.sequence, 'api_connected': self.api_connected,
                'feedback_error': self.feedback_error,
                'feedback_age_s': now-self.last_feedback_stamp if self.last_feedback_stamp else None,
                'detector_tracking': self.detector_tracking, 'mount_assumed': self.value('mount_assumed'),
                'target_deadband_mm': s.target_deadband_mm,
                'latest_age_s': now-s.latest[1] if s.latest else None,
                'session_id': s.session, 'execution_id': s.execution, 'plan_id': s.plan,
                'planning': s.busy, 'backend_state': s.backend_status.get('state'),
                'orientation_policy': s.orientation_mode, 'orientation_selection': s.orientation_selection,
                'visibility_camera': self.visibility_camera.configuration(),
                'gaze': {k: s.backend_status[k] for k in gaze_keys if k in s.backend_status},
                'tracking_region': s.region_settings, 'approach': s.approach_status}, allow_nan=False)

    def timing(self):
        return json.dumps({'schema_version': 1, 'implementation': 'cpp', 'stamp_unix_s': self.clock(),
            'armed': self.session.armed, 'dry_run': self.session.dry_run,
            'scope': 'Wall durations include waits; nested stages overlap. Capture ages require a shared clock.',
            **self.timings.snapshot()}, allow_nan=False)

    def shutdown(self):
        self.shutting_down = True
        self.connection_generation += 1
        self.session.stop('Node shutdown')
        self.commands.shutdown()
        if self.value('backend') == 'direct':
            self.api.shutdown()
        # CPython must outlive the shared planner/executor workers. join()
        # releases its GIL, allowing cancelled workers to finish.
        deadline = time.monotonic()+90.
        while True:
            workers = [t for t in threading.enumerate() if t is not threading.current_thread()]
            if not workers:
                return
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise RuntimeError('Runtime workers did not finish before interpreter shutdown')
            for worker in workers:
                worker.join(min(.1, remaining))
