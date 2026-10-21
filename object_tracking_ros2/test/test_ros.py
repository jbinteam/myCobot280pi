"""ROS-level exact-frame/extrinsic checks; require ROS Jazzy Python environment."""
import pytest
pytest.importorskip('rclpy')
from object_tracking_ros2.node import rotate, checked_quaternion


def make_node(**kwargs):
    from rclpy.parameter import Parameter
    from object_tracking_ros2.node import ObjectFollower
    kwargs.setdefault('parameter_overrides', [Parameter('backend', value='http')])
    return ObjectFollower(**kwargs)

@pytest.fixture(autouse=True)
def isolated_ros_domain(monkeypatch):
    # Integration tests publish real ROS services/topics but must never share
    # the live detector or hardware-control graph.
    monkeypatch.setenv('ROS_DOMAIN_ID','179')
    yield
    # Constructor failures must not leak the global context into later tests.
    import rclpy
    rclpy.try_shutdown()

def test_configured_optical_rotation_does_not_guess_camera_link_axes():
    # A supplied 90 degree Y mounting maps optical depth to EEF +X.
    import math
    q = checked_quaternion([0, math.sqrt(.5), 0, math.sqrt(.5)])
    assert rotate(q, [0,0,1]) == pytest.approx([1,0,0], abs=1e-12)
    assert rotate([0,0,0,1], [0,0,1]) == (0,0,1)

def test_invalid_extrinsic_rejected():
    with pytest.raises(ValueError): checked_quaternion([0,0,0,2])

def test_lost_keeps_armed_until_fresh_transformed_detection_and_stop_disarms():
    import rclpy
    from rclpy.time import Time
    from std_msgs.msg import String
    from std_srvs.srv import Trigger
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from object_tracking_ros2.node import ObjectFollower
    rclpy.init();node=make_node()
    try:
        now=node.get_clock().now().nanoseconds/1e9
        # Startup may arm without an object and must remain idle until a
        # TRACKING state plus a timestamped transformable pose both arrive.
        node.session.arm()
        assert node.session.armed and node.session.paused
        lost=String();lost.data='LOST';node.state(lost)
        assert node.session.armed and node.session.paused
        tracking=String();tracking.data='TRACKING';node.state(tracking)
        node.session.tick()
        assert node.session.paused  # State alone cannot revive the old target.
        tf=TransformStamped();tf.header.frame_id='g_base'
        tf.child_frame_id='camera_color_optical_frame';tf.transform.rotation.w=1.
        tf.header.stamp=Time(nanoseconds=int((now-.1)*1e9)).to_msg()
        node.tf.set_transform(tf,'test');node.last_feedback_stamp=now-.1
        pose=PoseStamped();pose.header.frame_id=tf.child_frame_id
        pose.header.stamp=tf.header.stamp;pose.pose.position.z=.35
        node.pose(pose);node.process_pose();node.session.tick()
        assert node.session.armed and not node.session.paused
        stale=PoseStamped();stale.header.frame_id=tf.child_frame_id
        stale.header.stamp=Time(nanoseconds=int((now-2)*1e9)).to_msg()
        stale.pose.position.z=.35
        node.last_stamp=node.last_received_stamp=0.
        node.pose(stale)
        assert node.session.armed and node.session.paused
        node.stop(Trigger.Request(),Trigger.Response())
        assert not node.session.armed
    finally:
        node.destroy_node();rclpy.shutdown()

def test_continuous_camera_30hz_feedback_2hz_exact_stamp_no_starvation():
    import rclpy
    from rclpy.time import Time
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from object_tracking_ros2.node import ObjectFollower
    from std_msgs.msg import String
    rclpy.init()
    node=make_node()
    try:
        state=String();state.data='TRACKING';node.state(state)
        now=node.get_clock().now().nanoseconds/1e9
        start=now-.6
        def feedback(stamp):
            tf=TransformStamped()
            tf.header.frame_id='g_base';tf.child_frame_id='camera_color_optical_frame'
            tf.header.stamp=Time(nanoseconds=int(stamp*1e9)).to_msg()
            tf.transform.rotation.w=1.
            # Known moving camera: exact interpolation x(t)=t-start metres.
            tf.transform.translation.x=stamp-start
            node.tf.set_transform(tf,'test-feedback')
            node.last_feedback_stamp=stamp
        feedback(start)
        for i in range(1,19):
            stamp=start+i/30
            if i==15:feedback(start+.5)
            pose=PoseStamped();pose.header.frame_id='camera_color_optical_frame'
            pose.header.stamp=Time(nanoseconds=int(stamp*1e9)).to_msg()
            pose.pose.position.z=.35
            node.pose(pose);node.process_pose()
        assert node.sequence>0
        assert node.last_stamp <= start+.5+1e-7
        assert node.session.latest[0][0] == pytest.approx((node.last_stamp-start)*1000,abs=.001)
        # Newest camera sample after the last robot reply is retained, never
        # transformed using the latest available robot pose.
        assert any(m.header.stamp.sec+m.header.stamp.nanosec/1e9 > start+.5
            for m in node.pose_queue)
        previous=node.sequence
        feedback(start+.6);node.process_pose()
        assert node.sequence>previous
        assert node.last_stamp==pytest.approx(start+.6,abs=1e-6)
    finally:
        node.destroy_node();rclpy.shutdown()

def test_dry_arm_uses_fresh_target_without_hardware_confirmation_or_http():
    import rclpy
    from rclpy.parameter import Parameter
    from std_srvs.srv import Trigger
    from object_tracking_ros2.node import ObjectFollower
    rclpy.init();node=make_node()
    try:
        node.set_parameters([Parameter('mount_configured',value=True)])
        calls=[]
        node.api.call=lambda *args:calls.append(args)
        now=node.get_clock().now().nanoseconds/1e9
        node.session.submit([0,0,350],now)
        response=node.arm(Trigger.Request(),Trigger.Response())
        assert response.success and node.session.armed
        assert calls==[]
        node.session.stop()
        node.session.dry_run=False
        node.session.submit([0,0,350],now+.0001)
        response=node.arm(Trigger.Request(),Trigger.Response())
        assert not response.success and not node.session.armed
        assert 'verification' in response.message
        assert calls==[]
    finally:
        node.destroy_node();rclpy.shutdown()

def test_null_feedback_is_reported_and_cannot_authorize_hardware_arm():
    import rclpy
    from rclpy.parameter import Parameter
    from std_srvs.srv import Trigger
    from object_tracking_ros2.node import ObjectFollower
    rclpy.init();node=make_node()
    try:
        node.api.call=lambda *args:{'connected':False,'feedback_pose':None}
        node.feedback()
        assert node.api_connected and not node.feedback_available
        assert 'disconnected' in node.feedback_error
        node.set_parameters([Parameter('mount_configured',value=True),
            Parameter('frame_verified',value=True),Parameter('scene_verified',value=True)])
        node.session.dry_run=False
        node.session.submit([0,0,350],node.get_clock().now().nanoseconds/1e9)
        response=node.arm(Trigger.Request(),Trigger.Response())
        assert not response.success and 'feedback' in response.message
    finally:
        node.destroy_node();rclpy.shutdown()

def test_feedback_uses_nonblocking_endpoint_preserves_history_and_rejects_stale_data():
    import rclpy
    from unittest.mock import patch
    from object_tracking_ros2.node import ObjectFollower
    rclpy.init();node=make_node()
    try:
        now=node.get_clock().now().nanoseconds/1e9
        def pose(stamp):
            return {'stamp_unix_s':stamp,'frame_id':'g_base','child_frame_id':'joint6_flange',
                'translation_m':[0,0,.4],'quaternion_xyzw':[0,0,0,1],
                'joint_names':['joint2_to_joint1','joint3_to_joint2','joint4_to_joint3',
                    'joint5_to_joint4','joint6_to_joint5','joint6output_to_joint6'],
                'angles_deg':[0,90,-90,0,10,20]}
        poses=[pose(now-.25),pose(now-.05)];routes=[]
        def call(route):
            routes.append(route)
            return {'connected':True,'feedback_pose':poses[-1],'feedback_poses':poses}
        node.api.call=call
        with patch.object(node.broadcaster,'sendTransform') as send, \
                patch.object(node.joint_publisher,'publish') as joints, \
                patch.object(node.eef_publisher,'publish') as eef:
            node.feedback()
            assert routes==['/api/feedback']
            transforms=send.call_args.args[0]
            assert len(transforms)==2
            assert joints.call_count==eef.call_count==2
            joint_state=joints.call_args.args[0]
            import math
            assert joint_state.position==pytest.approx([0,math.pi/2,-math.pi/2,0,math.radians(10),math.radians(20)])
            assert joint_state.name==poses[-1]['joint_names']
            assert joint_state.header.stamp==eef.call_args.args[0].header.stamp
            assert [t.header.stamp.sec+t.header.stamp.nanosec/1e9 for t in transforms] == pytest.approx([now-.25,now-.05],abs=1e-6)
            node.feedback();assert send.call_count==1
            assert node.last_feedback_stamp==poses[-1]['stamp_unix_s']
            poses[:]=[pose(now-2)]
            node.feedback()
            assert not node.feedback_available
            assert 'Stale' in node.feedback_error
            assert send.call_count==1
    finally:
        node.destroy_node();rclpy.shutdown()

@pytest.mark.parametrize('stop_during_start',[False,True])
def test_slow_arm_service_keeps_detection_callbacks_and_stop_responsive(stop_during_start):
    import time
    import threading
    import rclpy
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.executors import SingleThreadedExecutor
    from std_srvs.srv import Trigger
    from object_tracking_ros2.node import ObjectFollower
    rclpy.init();node=make_node();driver=Node('slow_arm_test_client')
    entered,release=threading.Event(),threading.Event()
    node.set_parameters([Parameter('mount_configured',value=True),
        Parameter('frame_verified',value=True),Parameter('scene_verified',value=True)])
    node.session.dry_run=False
    # Isolate startup scheduling from the independently tested planning worker.
    node.session.tick=lambda:None
    def api(route,body=None):
        if route.endswith('/start'):
            entered.set();assert release.wait(3)
            return {'session_id':'owned','ik':{'mouse_aim_session_id':'owned',
                'mouse_aim_client_id':'ros2_object_follow','mouse_executor':'encoder_feedback_tracking',
                'mouse_feedback_tracking_protocol':1,'mouse_tracking_region_protocol':1,
                'mouse_session_owner_protocol':1,'mouse_pause_protocol':1,
                'tracking_region_settings':{'mode':'cone_disk','distance_mm':350,'half_angle_deg':30},
                'visibility_camera':node.visibility_camera.configuration(),
                'velocity_shape':{'kind':'smooth_trapezoid','rise_percent':5,'fall_percent':10}}}
        if route=='/api/feedback':
            return {'feedback_pose':{'stamp_unix_s':node.get_clock().now().nanoseconds/1e9,
                'frame_id':'g_base','child_frame_id':'joint6_flange',
                'translation_m':[0,0,.4],'quaternion_xyzw':[0,0,0,1]}}
        return {}
    node.api.call=api
    def detection():node.session.submit([0,0,350],node.get_clock().now().nanoseconds/1e9)
    node.create_timer(.03,detection)  # Same callback group as real detections.
    detection();node.feedback()
    executor=SingleThreadedExecutor();executor.add_node(node);executor.add_node(driver)
    worker=threading.Thread(target=executor.spin,daemon=True);worker.start()
    def complete(future,timeout=2):
        end=time.monotonic()+timeout
        while not future.done() and time.monotonic()<end:time.sleep(.01)
        assert future.done()
        return future.result()
    try:
        arm=driver.create_client(Trigger,'/object_follow/arm');assert arm.wait_for_service(timeout_sec=2)
        f=arm.call_async(Trigger.Request());assert entered.wait(2)
        first=node.session.latest[1];time.sleep(.8)
        assert node.session.latest[1]>first+.6
        if stop_during_start:
            stop=driver.create_client(Trigger,'/object_follow/stop');assert stop.wait_for_service(timeout_sec=1)
            assert complete(stop.call_async(Trigger.Request()),timeout=.5).success
        release.set();response=complete(f)
        assert response.success is not stop_during_start
        assert node.session.armed is not stop_during_start
        if stop_during_start:assert 'cancelled' in response.message
    finally:
        release.set();node.session.stop()
        for thread in node.session.stop_threads:thread.join(1)
        for timer in node.timers:timer.cancel()
        executor.remove_node(node);executor.remove_node(driver)
        executor.shutdown(timeout_sec=2);worker.join(2)
        if node.arm_thread is not None:node.arm_thread.join(1.5)
        if node.feedback_thread is not None:node.feedback_thread.join(1.5)
        node.destroy_node();driver.destroy_node();rclpy.try_shutdown()


def test_direct_node_selects_in_process_backend_without_connecting_by_default(monkeypatch):
    import sys
    import types
    import rclpy
    from rclpy.parameter import Parameter
    from object_tracking_ros2.node import ObjectFollower
    calls=[]
    class Backend:
        timeout=.2
        def __init__(self, *, camera): self.camera=camera
        def call(self,route,body=None):
            calls.append((route,body))
            return {'connected':False,'feedback_pose':None}
        def shutdown(self):pass
    module=types.ModuleType('object_tracking_ros2.direct');module.DirectBackend=Backend
    monkeypatch.setitem(sys.modules,'object_tracking_ros2.direct',module)
    rclpy.init();node=make_node(parameter_overrides=[Parameter('backend',value='direct')])
    try:
        assert isinstance(node.api.backend,Backend)
        assert node.api.backend.camera.configuration()==node.session.visibility_camera
        assert calls==[]
        node.feedback()
        assert calls==[('/api/feedback',None)]
        assert not node.feedback_available and not node.session.armed
    finally:
        node.destroy_node();rclpy.try_shutdown()


def test_timing_topic_records_detection_age_and_tf_wait_without_motion():
    import json
    import time
    import rclpy
    from rclpy.time import Time
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from std_msgs.msg import String
    rclpy.init();node=make_node()
    reader=rclpy.create_node('timing_test_reader')
    messages=[]
    subscription=reader.create_subscription(String, '/object_follow/timing',
        lambda message: messages.append(json.loads(message.data)), 10)
    try:
        node.detector_tracking=True
        stamp=node.get_clock().now().nanoseconds/1e9-.1
        tf=TransformStamped();tf.header.stamp=Time(nanoseconds=int(stamp*1e9)).to_msg()
        tf.header.frame_id='g_base';tf.child_frame_id='camera_color_optical_frame'
        tf.transform.rotation.w=1.
        pose=PoseStamped();pose.header.stamp=tf.header.stamp
        pose.header.frame_id='camera_color_optical_frame';pose.pose.position.z=.35
        node.tf.set_transform(tf,'timing-test')
        node.last_feedback_stamp=stamp+.001
        node.pose(pose);node.process_pose()
        assert node.sequence==1
        deadline=time.monotonic()+3.
        while not messages and time.monotonic()<deadline:
            node.publish_timing()
            rclpy.spin_once(reader, timeout_sec=.05)
        assert messages
        rows=messages[-1]['stages']
        assert rows['detection.capture_to_callback']['last_ms']>=90.
        assert rows['detection.capture_to_target']['last_ms']>=90.
        assert rows['detection.callback_to_target']['count']==1
        assert not messages[-1]['armed']
        assert messages[-1]['dry_run']
    finally:
        node.destroy_node();reader.destroy_node();rclpy.try_shutdown()


def test_async_connect_keeps_stop_responsive_and_disconnect_cancels_pending_connect():
    import time
    import threading
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from std_srvs.srv import Trigger
    from object_tracking_ros2.node import ObjectFollower
    rclpy.init();node=make_node();driver=Node('connection_test_client')
    entered,release=threading.Event(),threading.Event();calls=[]
    def api(route,body=None):
        calls.append(route)
        if route=='/api/connect':
            entered.set();assert release.wait(3)
        return {'connected':False,'feedback_pose':None}
    node.api.call=api
    ticks=[];node.create_timer(.02,lambda:ticks.append(time.monotonic()))
    executor=SingleThreadedExecutor();executor.add_node(node);executor.add_node(driver)
    worker=threading.Thread(target=executor.spin,daemon=True);worker.start()
    def complete(future,timeout=2):
        end=time.monotonic()+timeout
        while not future.done() and time.monotonic()<end:time.sleep(.01)
        assert future.done()
        return future.result()
    try:
        connect=driver.create_client(Trigger,'/object_follow/connect')
        disconnect=driver.create_client(Trigger,'/object_follow/disconnect')
        stop=driver.create_client(Trigger,'/arm/stop')
        for client in (connect,disconnect,stop):assert client.wait_for_service(timeout_sec=2)
        pending=connect.call_async(Trigger.Request());assert entered.wait(2)
        first=len(ticks);time.sleep(.15);assert len(ticks)>first+3
        assert complete(stop.call_async(Trigger.Request()),.5).success
        off=disconnect.call_async(Trigger.Request())
        time.sleep(.05);release.set()
        assert not complete(pending).success
        assert complete(off).success
        assert calls[-1]=='/api/disconnect' or '/api/disconnect' in calls
        assert not node.session.armed and not node.feedback_available
    finally:
        release.set();node.session.stop();node.commands.shutdown()
        for timer in node.timers:timer.cancel()
        executor.remove_node(node);executor.remove_node(driver)
        executor.shutdown(timeout_sec=2);worker.join(2)
        for thread in node.connection_threads:thread.join(1)
        if node.feedback_thread:node.feedback_thread.join(1)
        node.destroy_node();driver.destroy_node();rclpy.try_shutdown()


def test_generic_goals_preserve_ros_units_stamp_and_frame_and_block_velocity():
    import rclpy
    from unittest.mock import Mock
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import PoseStamped
    from object_tracking_ros2.node import ObjectFollower
    rclpy.init();node=make_node()
    try:
        node.commands=Mock()
        joints=JointState();joints.header.stamp.sec=123
        joints.name=['j'+str(i) for i in range(6)];joints.position=[.1]*6
        node.command_joints(joints)
        node.commands.submit_joints.assert_called_once_with(list(joints.name),list(joints.position),123.)
        joints.velocity=[.2]*6;node.command_joints(joints)
        assert node.commands.submit_joints.call_count==1
        pose=PoseStamped();pose.header.frame_id='g_base';pose.header.stamp.sec=456
        pose.pose.position.z=.35;pose.pose.orientation.w=1.
        node.command_pose(pose)
        node.commands.submit_pose.assert_called_once_with('g_base',[0.,0.,.35],[0.,0.,0.,1.],456.)
    finally:
        node.destroy_node();rclpy.try_shutdown()
