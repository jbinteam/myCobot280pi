"""ROS arm control and accepted 3D detection following with a shared executor."""
from collections import deque
import json
import math
import signal
import threading
from time import perf_counter
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import SingleThreadedExecutor, ExternalShutdownException
from rclpy.task import Future
from geometry_msgs.msg import PoseStamped, PointStamped, TransformStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import TransformException, Buffer, TransformListener, TransformBroadcaster, StaticTransformBroadcaster
from .core import LocalAPI, FollowSession, validate_detection
from .commands import CheckedCommands
from .timing import TimingStats, TimedBackend, timed


def rotate(q, point):
    x, y, z, w = q
    vx, vy, vz = point
    # Quaternion vector rotation, assumes explicitly checked unit quaternion.
    tx, ty, tz = 2*(y*vz-z*vy), 2*(z*vx-x*vz), 2*(x*vy-y*vx)
    return (vx+w*tx+y*tz-z*ty, vy+w*ty+z*tx-x*tz, vz+w*tz+x*ty-y*tx)


def checked_quaternion(value):
    if len(value) != 4 or not all(math.isfinite(v) for v in value) or abs(sum(v*v for v in value)-1) > 1e-5:
        raise ValueError('Extrinsic quaternion must be finite and normalized [x,y,z,w]')
    return [float(v) for v in value]


class ObjectFollower(Node):
    def __init__(self, **kwargs):
        super().__init__('object_follow', **kwargs)
        defaults = {'backend': 'direct', 'api_url': 'http://127.0.0.1:8000', 'dry_run': True,
            'robot_host': '192.168.77.225', 'robot_port': 9000, 'auto_connect': False,
            'frame_verified': False, 'scene_verified': False, 'mount_configured': False,
            'mount_assumed': False, 'mount_calibrated': False, 'sdk_speed': 70, 'reference_speed_rad_s': 1.,
            'optical_frame': 'camera_color_optical_frame', 'base_frame': 'g_base',
            'eef_frame': 'joint6_flange', 'max_detection_age_s': .75,
            'target_deadband_mm': 30.,
            'cone_distance_mm': 350., 'cone_half_angle_deg': 30.,
            'approach_enabled': False, 'approach_final_distance_mm': 150., 'approach_step_mm': 20.,
            'approach_distance_tolerance_mm': 10., 'show_rrt': True,
            'approach_camera_fov_fallback': True,
            'horizontal_fov_deg': 82., 'vertical_fov_deg': 68.,
            'angular_margin_deg': 0., 'near_m': .005,
            'eef_to_optical_translation_m': [0., 0., 0.],
            'eef_to_optical_quaternion_xyzw': [0., 0., 0., 1.]}
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        self.value = lambda key: self.get_parameter(key).value
        from simulation.camera_model import CameraModel
        self.visibility_camera = CameraModel.from_ros_settings({key:self.value(key) for key in defaults})
        backend = self.value('backend')
        if backend == 'direct':
            from .direct import DirectBackend
            self.api = DirectBackend(camera=self.visibility_camera)
        elif backend == 'http':
            self.api = LocalAPI(self.value('api_url'))
        else:
            raise ValueError('backend must be direct or http')
        self.timings = TimingStats()
        from .planning_visualization import PlanningVisualization
        self.planning_visualization = PlanningVisualization()
        self.planning_publisher = self.create_publisher(String, '/object_follow/planning_visualization', 10)
        self.api = TimedBackend(self.api, self.timings)
        self.pose_received_at = {}
        self.last_tick_at = None
        self.connection_lock = threading.Lock()
        self.connection_generation = 0
        self.connection_pending = 0
        self.connection_threads = []
        self.shutting_down = False
        self.session = FollowSession(self.api, lambda: self.get_clock().now().nanoseconds/1e9,
            self.value('max_detection_age_s'), self.value('dry_run'),
            self.value('sdk_speed'), self.value('reference_speed_rad_s'),
            self.value('target_deadband_mm'),
            self.value('approach_final_distance_mm') if self.value('approach_enabled') else None,
            self.value('approach_step_mm'), self.value('approach_distance_tolerance_mm'),
            self.value('cone_distance_mm'), self.value('cone_half_angle_deg'), timings=self.timings,
            visibility_camera=self.visibility_camera.configuration(),
            planning_visualization=self.planning_visualization,
            approach_camera_fov_fallback=self.value('approach_camera_fov_fallback'))
        self.commands = CheckedCommands(self.api, self.value, self.session)
        self.tf = Buffer(cache_time=Duration(seconds=5))
        self.listener = TransformListener(self.tf, self)
        self.broadcaster = TransformBroadcaster(self)
        self.static = StaticTransformBroadcaster(self)
        self.last_stamp = self.last_feedback_stamp = 0.
        self.feedback_busy = False
        self.feedback_thread = None
        self.arm_thread = None
        self.enable_thread = None
        self.api_connected = False
        self.feedback_available = False
        self.feedback_error = 'Waiting for backend feedback'
        self.detector_tracking = False
        self.pose_queue = deque(maxlen=32)
        self.last_received_stamp = 0.
        qos = QoSProfile(depth=3, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(PoseStamped, '/object_pose/filtered', self.pose, qos)
        self.create_subscription(String, '/object_pose/tracking_state', self.state, 10)
        self.create_subscription(JointState, '/arm/command_joints', self.command_joints, 1)
        self.create_subscription(PoseStamped, '/arm/command_pose', self.command_pose, 1)
        # Hardware startup may wait on serial queries. Keep detection/timer
        # callbacks and Stop available while the Arm service is waiting.
        self.arm_group = MutuallyExclusiveCallbackGroup()
        self.stop_group = MutuallyExclusiveCallbackGroup()
        self.connect_group = MutuallyExclusiveCallbackGroup()
        self.disconnect_group = MutuallyExclusiveCallbackGroup()
        self.create_service(Trigger, '~/arm', self.arm_service, callback_group=self.arm_group)
        self.create_service(Trigger, '~/stop', self.stop, callback_group=self.stop_group)
        self.create_service(Trigger, '~/connect', self.connect_service, callback_group=self.connect_group)
        self.create_service(Trigger, '~/disconnect', self.disconnect_service, callback_group=self.disconnect_group)
        self.create_service(Trigger, '/arm/enable_motion', self.enable_motion_service, callback_group=self.arm_group)
        self.create_service(Trigger, '/arm/stop', self.stop, callback_group=self.stop_group)
        self.create_timer(.05, self.tick)
        self.sequence = 0
        self.target_publisher = self.create_publisher(PointStamped, '/object_follow/target', 10)
        self.status = self.create_publisher(String, '~/status', 10)
        self.timing_publisher = self.create_publisher(String, '~/timing', 10)
        self.create_timer(1., self.publish_timing)
        self.joint_publisher = self.create_publisher(JointState, '/joint_states', 10)
        self.eef_publisher = self.create_publisher(PoseStamped, '~/eef_pose', 10)
        if self.value('mount_configured') or self.value('mount_calibrated'):
            self.publish_mount()
        if self.value('auto_connect'):
            if self.value('dry_run'):
                self.get_logger().warning('Ignoring auto_connect in dry run; ~/connect remains explicit')
            else:
                self.start_connection_worker(True)

    def start_connection_worker(self, connect, response=None, done=None):
        self.connection_pending += 1
        if not connect:
            self.connection_generation += 1
            self.session.stop('Disconnect requested')
            self.commands.stop('Disconnect requested')
        generation = self.connection_generation
        def work():
            result = response if response is not None else Trigger.Response()
            try:
                with self.connection_lock:
                    if connect:
                        if generation != self.connection_generation or self.shutting_down:
                            raise RuntimeError('Connect cancelled')
                        self.api.call('/api/connect', {'host': self.value('robot_host'),
                                                     'port': self.value('robot_port')})
                        if generation != self.connection_generation or self.shutting_down:
                            self.api.call('/api/disconnect', {})
                            raise RuntimeError('Connect cancelled')
                    else:
                        self.api.call('/api/disconnect', {})
                        self.api_connected = self.feedback_available = False
                        self.last_feedback_stamp = 0.
                        self.feedback_error = 'Robot disconnected'
                        self.pose_queue.clear()
                result.success = True
                result.message = 'Connected; motion remains disarmed' if connect else 'Disconnected'
            except Exception as exc:
                result.success, result.message = False, str(exc)
                self.get_logger().error(result.message)
            finally:
                self.connection_pending -= 1
            if done is not None:
                done.set_result(result)
        thread = threading.Thread(target=work, daemon=True)
        self.connection_threads = [t for t in self.connection_threads if t.is_alive()]
        self.connection_threads.append(thread)
        thread.start()

    async def connect_service(self, request, response):
        done = Future(executor=self.executor)
        self.start_connection_worker(True, response, done)
        return await done

    async def disconnect_service(self, request, response):
        done = Future(executor=self.executor)
        self.start_connection_worker(False, response, done)
        return await done

    def check_motion_ready(self):
        if self.connection_pending or self.shutting_down:
            raise ValueError('Robot connection is changing; motion is unavailable')
        if self.get_parameter('use_sim_time').value and not self.session.dry_run:
            raise ValueError('Hardware motion requires wall-clock ROS time')
        if not self.session.dry_run:
            if not (self.value('frame_verified') and self.value('scene_verified')):
                raise ValueError('Hardware robot frame and scene verification required')
            if (not self.api_connected or not self.feedback_available
                    or self.get_clock().now().nanoseconds/1e9-self.last_feedback_stamp > self.value('max_detection_age_s')):
                raise ValueError('Fresh measured backend feedback is required before arming')

    def enable_motion(self, request, response):
        try:
            self.check_motion_ready()
            if self.arm_thread is not None and self.arm_thread.is_alive():
                raise ValueError('Tracking Arm is still pending')
            response.message = self.commands.enable()
            response.success = True
        except Exception as exc:
            response.success, response.message = False, str(exc)
        return response

    async def enable_motion_service(self, request, response):
        done = Future(executor=self.executor)
        def work():
            try:
                done.set_result(self.enable_motion(request, response))
            except Exception as exc:
                done.set_exception(exc)
        self.enable_thread = threading.Thread(target=work, daemon=True)
        self.enable_thread.start()
        return await done

    def command_joints(self, message):
        try:
            self.check_motion_ready()
            if message.velocity or message.effort:
                raise ValueError('Joint goals accept positions only')
            self.commands.submit_joints(list(message.name), list(message.position),
                message.header.stamp.sec+message.header.stamp.nanosec/1e9)
        except Exception as exc:
            self.get_logger().error('Joint goal rejected: '+str(exc))

    def command_pose(self, message):
        try:
            self.check_motion_ready()
            p, q = message.pose.position, message.pose.orientation
            self.commands.submit_pose(message.header.frame_id, [p.x,p.y,p.z],
                [q.x,q.y,q.z,q.w], message.header.stamp.sec+message.header.stamp.nanosec/1e9)
        except Exception as exc:
            self.get_logger().error('Pose goal rejected: '+str(exc))

    def publish_mount(self):
        translation = self.value('eef_to_optical_translation_m')
        q = checked_quaternion(self.value('eef_to_optical_quaternion_xyzw'))
        if len(translation) != 3 or not all(math.isfinite(v) for v in translation):
            raise ValueError('Explicit calibrated optical translation requires three finite metres')
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id, transform.child_frame_id = self.value('eef_frame'), self.value('optical_frame')
        transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = [float(v) for v in translation]
        transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w = q
        self.static.sendTransform(transform)

    def state(self, message):
        self.detector_tracking = message.data == 'TRACKING'
        if not self.detector_tracking:
            self.pose_queue.clear()
            self.session.pause('Detector '+message.data)

    @timed('ros.detection_callback')
    def pose(self, message):
        try:
            if not self.detector_tracking:
                return
            stamp = message.header.stamp.sec+message.header.stamp.nanosec/1e9
            if stamp <= self.last_stamp:
                return  # Already processed/delivered duplicate.
            point = [message.pose.position.x, message.pose.position.y, message.pose.position.z]
            now = self.get_clock().now().nanoseconds/1e9
            validate_detection(point, stamp, now,
                message.header.frame_id, self.value('optical_frame'), self.last_received_stamp,
                self.value('max_detection_age_s'))
            self.last_received_stamp = stamp
            self.timings.add('detection.capture_to_callback', (now-stamp)*1000)
            self.pose_received_at[stamp] = perf_counter()
            while len(self.pose_received_at) > self.pose_queue.maxlen:
                del self.pose_received_at[next(iter(self.pose_received_at))]
            self.pose_queue.append(message)
        except Exception as exc:
            self.session.last_error = str(exc)
            if self.session.armed:
                if str(exc) == 'Stale or future detection' and now-stamp > self.value('max_detection_age_s'):
                    self.session.pause('Accepted detection expired')
                else:
                    self.session.stop(str(exc))

    @timed('ros.process_pose')
    def process_pose(self):
        now = self.get_clock().now().nanoseconds/1e9
        max_age = self.value('max_detection_age_s')
        while self.pose_queue:
            message = self.pose_queue[0]
            stamp = message.header.stamp.sec+message.header.stamp.nanosec/1e9
            if stamp <= self.last_stamp or now-stamp > max_age:
                self.pose_queue.popleft()
            else:
                break
        # Preserve future samples until feedback brackets them. Select newest
        # eligible sample, avoiding starvation from a 30Hz camera / 2Hz robot.
        for message in reversed(self.pose_queue):
            stamp = message.header.stamp.sec+message.header.stamp.nanosec/1e9
            if stamp > self.last_feedback_stamp:
                continue  # No measured bracket yet; avoid futile TF retries.
            try:
                transform = self.tf.lookup_transform(self.value('base_frame'), message.header.frame_id,
                    Time.from_msg(message.header.stamp))
            except TransformException:
                continue
            try:
                stamp = message.header.stamp.sec+message.header.stamp.nanosec/1e9
                point = [message.pose.position.x, message.pose.position.y, message.pose.position.z]
                r = transform.transform.rotation
                rotated = rotate(checked_quaternion([r.x, r.y, r.z, r.w]), point)
                t = transform.transform.translation
                base_point = [rotated[i]+v for i, v in enumerate([t.x, t.y, t.z])]
                self.session.submit([1000*v for v in base_point], stamp)
                self.timings.add('detection.capture_to_target',
                    (self.get_clock().now().nanoseconds/1e9-stamp)*1000)
                received = self.pose_received_at.get(stamp)
                if received is not None:
                    self.timings.add('detection.callback_to_target', (perf_counter()-received)*1000)
                target = PointStamped()
                target.header.stamp = message.header.stamp
                target.header.frame_id = self.value('base_frame')
                target.point.x, target.point.y, target.point.z = [v/1000 for v in self.session.latest[0]]
                self.target_publisher.publish(target)
                self.sequence += 1
                self.last_stamp = stamp
                if self.session.armed and not self.session.paused:
                    self.session.last_error = ''
            except Exception as exc:
                self.session.last_error = str(exc)
                if self.session.armed:
                    self.session.stop(str(exc))
            break

    async def arm_service(self, request, response):
        # Yield the ROS executor while the exclusive hardware startup waits.
        # The Arm group remains held; detection and Stop groups stay runnable.
        done = Future(executor=self.executor)
        def work():
            try:
                done.set_result(self.arm(request, response))
            except Exception as exc:
                done.set_exception(exc)
        self.arm_thread = threading.Thread(target=work, daemon=True)
        self.arm_thread.start()
        return await done

    def arm(self, request, response):
        try:
            self.check_motion_ready()
            if self.commands.enabled or self.commands.busy or self.commands.stop_pending:
                raise ValueError('Stop general arm motion before arming tracking')
            if not (self.value('mount_configured') or self.value('mount_calibrated')):
                raise ValueError('Configured camera mount required')
            self.session.arm()
            response.success = True
            response.message = 'Armed (dry run)' if self.session.dry_run else 'Armed'
            if self.session.paused:
                response.message += '; waiting for object'
        except Exception as exc:
            response.success, response.message = False, str(exc)
        return response

    def stop(self, request, response):
        try:
            self.session.stop()
            self.commands.stop()
            response.success, response.message = True, 'Disarmed; Stop requested'
        except Exception as exc:
            response.success, response.message = False, str(exc)
        return response

    def publish_timing(self):
        message = String()
        message.data = json.dumps({'schema_version': 1,
            'stamp_unix_s': self.get_clock().now().nanoseconds/1e9,
            'armed': self.session.armed, 'dry_run': self.session.dry_run,
            'scope': 'Wall durations include waits; nested stages overlap. Capture ages require a shared clock.',
            **self.timings.snapshot()}, allow_nan=False)
        self.timing_publisher.publish(message)

    @timed('ros.tick')
    def tick(self):
        tick_at = perf_counter()
        if self.last_tick_at is not None:
            self.timings.add('ros.tick_interval', (tick_at-self.last_tick_at)*1000)
        self.last_tick_at = tick_at
        self.process_pose()
        self.planning_visualization.set_enabled(self.value('show_rrt') and self.session.armed and not self.session.paused)
        try:
            self.session.tick()
            self.commands.tick()
        except Exception as exc:
            self.get_logger().error(str(exc))
        if not self.feedback_busy:
            self.feedback_busy = True
            self.feedback_thread = threading.Thread(target=self.feedback, daemon=True)
            self.feedback_thread.start()
        msg = String()
        now = self.get_clock().now().nanoseconds/1e9
        msg.data = json.dumps({'armed': self.session.armed, 'dry_run': self.session.dry_run,
            'backend': self.value('backend'),
            'arm_motion': self.commands.status,
            'paused': self.session.paused, 'pause_pending': self.session.pause_pending,
            'reason': self.session.last_error, 'sequence': self.sequence,
            'api_connected': self.api_connected, 'feedback_error': self.feedback_error,
            'feedback_age_s': now-self.last_feedback_stamp if self.last_feedback_stamp else None,
            'detector_tracking': self.detector_tracking, 'mount_assumed': self.value('mount_assumed'),
            'target_deadband_mm': self.session.target_deadband_mm,
            'latest_age_s': now-self.session.latest[1] if self.session.latest else None,
            'session_id': self.session.session, 'execution_id': self.session.execution,
            'plan_id': self.session.plan, 'planning': self.session.busy,
            'backend_state': self.session.backend_status.get('state'),
            'orientation_policy': self.session.orientation_mode,
            'orientation_selection': self.session.orientation_selection,
            'visibility_camera': self.visibility_camera.configuration(),
            'gaze': {key: self.session.backend_status[key] for key in
                ('estimated_orientation_error_deg','estimated_error_mm','tracking_accepted',
                 'cursor_direction_error_deg','aim_error_deg','tracking_region_acceptance',
                 'final_goal_accepted','cursor_distance_mm','reference_lag_s','self_collision_checked',
                 'feedback_replan_required','replan_required','tracking_region',
                 'orientation_mode','orientation_selection','last_planning_orientation_mode','last_orientation_selection')
                 if key in self.session.backend_status},
            'tracking_region': self.session.region_settings,
            'approach': self.session.approach_status})
        self.status.publish(msg)
        display = self.planning_visualization.message()
        if display:
            self.planning_publisher.publish(String(data=display))

    @timed('ros.feedback_worker')
    def feedback(self):
        try:
            state = self.api.call('/api/feedback')
            self.api_connected = True
            pose = state.get('feedback_pose')
            self.feedback_available = bool(pose)
            if not pose:
                self.feedback_error = 'Backend has no measured robot feedback (disconnected)'
                return
            if pose.get('frame_id') != self.value('base_frame') or pose.get('child_frame_id') != self.value('eef_frame'):
                raise ValueError('Backend feedback frame mismatch')
            stamp = pose['stamp_unix_s']
            now = self.get_clock().now().nanoseconds/1e9
            if not math.isfinite(stamp) or stamp > now+.02 or now-stamp > self.value('max_detection_age_s'):
                raise ValueError('Stale backend feedback pose')
            self.feedback_error = ''
            transforms = []
            previous = self.last_feedback_stamp
            # Recover samples acquired between polls, preserving the two
            # measured endpoints needed for exact-stamp TF interpolation.
            for sample in state.get('feedback_poses', [pose]):
                captured = sample['stamp_unix_s']
                if not math.isfinite(captured) or captured > now+.02:
                    raise ValueError('Invalid backend feedback stamp')
                if captured <= previous:
                    continue
                if (sample.get('frame_id') != self.value('base_frame')
                        or sample.get('child_frame_id') != self.value('eef_frame')):
                    raise ValueError('Backend feedback frame mismatch')
                translation, q = sample['translation_m'], checked_quaternion(sample['quaternion_xyzw'])
                if len(translation) != 3 or not all(math.isfinite(v) for v in translation):
                    raise ValueError('Invalid backend feedback translation')
                transform = TransformStamped()
                transform.header.stamp = Time(nanoseconds=int(captured*1e9)).to_msg()
                transform.header.frame_id, transform.child_frame_id = self.value('base_frame'), self.value('eef_frame')
                transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = [float(v) for v in translation]
                transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w = q
                transforms.append(transform)
                eef = PoseStamped()
                eef.header = transform.header
                eef.pose.position.x, eef.pose.position.y, eef.pose.position.z = [float(v) for v in translation]
                eef.pose.orientation = transform.transform.rotation
                self.eef_publisher.publish(eef)
                angles = sample.get('angles_deg')
                names = sample.get('joint_names')
                if angles is not None and names is not None:
                    if (len(angles) != 6 or len(names) != 6 or len(set(names)) != 6
                            or not all(math.isfinite(v) for v in angles)):
                        raise ValueError('Invalid measured joint feedback')
                    joints = JointState()
                    joints.header = transform.header
                    joints.name = list(names)
                    joints.position = [math.radians(float(v)) for v in angles]
                    self.joint_publisher.publish(joints)
                previous = captured
                self.timings.add('feedback.capture_to_publish', (now-captured)*1000)
            if transforms:
                self.broadcaster.sendTransform(transforms)
                self.last_feedback_stamp = previous
        except Exception as exc:
            self.feedback_error = str(exc)
            self.feedback_available = False
            if not isinstance(exc, ValueError):
                self.api_connected = False
        finally:
            self.feedback_busy = False


def main(args=None):
    rclpy.init(args=args)
    node = ObjectFollower()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Terminal SIGINT and launch's forwarded SIGINT can both arrive.
        # Finish stopping/disconnecting once cleanup has begun.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        node.shutting_down = True
        node.connection_generation += 1
        node.session.stop('Node shutdown')
        node.commands.shutdown()
        for thread in node.session.stop_threads:
            thread.join(node.api.timeout+.5)
        executor.shutdown()
        if node.arm_thread is not None:
            node.arm_thread.join(node.api.timeout+.5)
        if node.enable_thread is not None:
            node.enable_thread.join(node.api.timeout+.5)
        if node.feedback_thread is not None:
            node.feedback_thread.join(node.api.timeout+.5)
        for thread in node.connection_threads:
            thread.join(node.api.timeout+.5)
        if node.value('backend') == 'direct':
            node.api.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
