"""Native ROS -> actual HTTP/controller/planner/executor -> fake encoder plant.

Synthetic optical detections are computed from timestamped measured feedback.
No camera inference, physical transport, or motor dynamics are simulated here.
The production planner and all downstream validation remain enabled.
"""
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

pytest.importorskip('rclpy')
from test_direct_lifecycle import EncoderRobot, RobotController, ARM, v2_mouse, real_tracking_plan
from server import Handler
from simulation.camera_model import configured_camera

ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT/'output/object_follow_ros2/install/object_tracking_cpp/lib/object_tracking_cpp/object_follow_cpp'


def test_native_full_approach_loss_resume_retarget_stop_and_faults(tmp_path, monkeypatch):
    if not EXE.is_file():
        pytest.skip('Build object_tracking_cpp first')
    import rclpy
    from geometry_msgs.msg import PoseStamped, PointStamped
    from std_msgs.msg import String
    from sensor_msgs.msg import JointState
    from std_srvs.srv import Trigger
    from rclpy.time import Time
    from rcl_interfaces.srv import SetParameters
    from rclpy.parameter import Parameter
    from visualization_msgs.msg import MarkerArray, Marker
    from object_tracking_ros2.visualization import TrackingVisualization

    monkeypatch.setenv('ROS_DOMAIN_ID', '185')
    camera = configured_camera()
    robot = EncoderRobot()
    initial_angles = robot.angles[:]
    # This factory is the only motor transport; even connect() creates no socket.
    controller = RobotController(lambda *_: robot)
    controller.visibility_camera = camera
    plans, states, targets, joints, events, detections = [], [], [], [], [], []
    planning_updates, marker_checks, planning_samples = [], [], {}
    target_stamps = []
    plans_lock = threading.Lock()
    child = server = worker = driver = logfile = view = None
    final_status = None
    start_time = time.monotonic()

    def record(event):
        events.append({'event': event, 'elapsed_s': time.monotonic()-start_time,
                       'encoder_packets': sum(c[0]=='encoder_drag' for c in robot.calls)})
        (tmp_path/'events.json').write_text(json.dumps(events, indent=2)+'\n')

    def actual_plan(body):
        result = real_tracking_plan(body)
        with plans_lock:
            plans.append({'request': copy.deepcopy({k:v for k,v in body.items() if not k.startswith('_')}),
                          'result': copy.deepcopy(result)})
            (tmp_path/'latest_plan.json').write_text(json.dumps(plans[-1], indent=2)+'\n')
        return result

    monkeypatch.setattr(v2_mouse.classical_tracking, 'plan', actual_plan)
    try:
        controller.connect('127.0.0.1', 19000)
        transform = camera.optical_transform(ARM.transform(np.deg2rad(robot.angles)))
        cursor = transform[:3, 3] + .350*transform[:3, 2]
        cls = type('PipelineHandler', (Handler,), {
            'controller': controller, 'token': 'simulation-only', 'hardware_enabled': True,
            'mouse_aim_lock': threading.Lock(), 'log_message': lambda *args: None})
        server = ThreadingHTTPServer(('127.0.0.1', 0), cls)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        logfile = (tmp_path/'native.log').open('w')
        child = subprocess.Popen([str(EXE), '--ros-args', '--params-file',
            str(ROOT/'object_tracking_ros2/config/object_follow.yaml'),
            '-p', f'project_root:={ROOT}', '-p', 'backend:=http',
            '-p', f'api_url:=http://127.0.0.1:{server.server_port}',
            '-p', 'dry_run:=false', '-p', 'auto_connect:=false',
            '-p', 'frame_verified:=true', '-p', 'scene_verified:=true'],
            env=dict(os.environ, OPENBLAS_NUM_THREADS='1'), stdout=logfile, stderr=logfile)
        rclpy.init()
        driver = rclpy.create_node('full_pipeline_simulation')
        view = TrackingVisualization()
        def planning(message):
            value = json.loads(message.data)
            planning_updates.append({k:value[k] for k in ('enabled','epoch','attempt','status','rewires')})
            planning_updates[-1].update(nodes=len(value['nodes']), route_points=len(value['route_deg']))
            if len(value['nodes']) > 2 and 'tree' not in planning_samples:
                planning_samples['tree'] = value
            if value['route_deg'] and 'accepted' not in planning_samples:
                planning_samples['accepted'] = value
        def markers(message):
            tree = next((m for m in message.markers if m.ns=='rrt_search'), None)
            route = next((m for m in message.markers if m.ns=='rrt_route'), None)
            if tree and route:
                marker_checks.append({'tree_action':tree.action, 'tree_points':len(tree.points),
                    'tree_width':tree.scale.x, 'tree_r':tree.color.r,
                    'route_action':route.action, 'route_points':len(route.points),
                    'route_width':route.scale.x, 'route_b':route.color.b})
        def accepted_target(message):
            targets.append([message.point.x, message.point.y, message.point.z])
            target_stamps.append(message.header.stamp.sec+message.header.stamp.nanosec/1e9)
        subscriptions = [
            driver.create_subscription(String, '/object_follow/planning_visualization', planning, 50),
            driver.create_subscription(MarkerArray, '/arm/planning_visualization', markers, 10),
            driver.create_subscription(String, '/object_follow/status', lambda m: states.append(json.loads(m.data)), 50),
            driver.create_subscription(PointStamped, '/object_follow/target', accepted_target, 50),
            driver.create_subscription(JointState, '/joint_states',
                lambda m: joints.append({'stamp': m.header.stamp.sec+m.header.stamp.nanosec/1e9,
                                         'q': list(m.position)}), 50)]
        tracking = driver.create_publisher(String, '/object_pose/tracking_state', 10)
        poses = driver.create_publisher(PoseStamped, '/object_pose/filtered', 10)
        last_sample = [0.]

        def publish(stamp, optical, frame='camera_color_optical_frame'):
            msg = PoseStamped()
            msg.header.frame_id = frame
            msg.header.stamp = Time(nanoseconds=int(stamp*1e9)).to_msg()
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, optical)
            msg.pose.orientation.w = 1.
            poses.publish(msg)
            detections.append({'stamp':stamp, 'frame':frame, 'optical_m':list(map(float,optical))})

        def feed(offset=None):
            sample = controller.feedback_snapshot()['feedback_pose']
            if sample and sample['stamp_unix_s'] > last_sample[0]:
                last_sample[0] = sample['stamp_unix_s']
                t = camera.optical_transform(ARM.transform(np.deg2rad(sample['angles_deg'])))
                object_point = cursor if offset is None else cursor+offset
                optical = t[:3,:3].T @ (object_point-t[:3,3])
                publish(last_sample[0], optical)

        def until(predicate, timeout=12., *, detection=False, offset=None):
            deadline = time.monotonic()+timeout
            while time.monotonic() < deadline:
                assert child.poll() is None, (tmp_path/'native.log').read_text()
                if detection:
                    feed(offset)
                rclpy.spin_once(driver, timeout_sec=.02)
                rclpy.spin_once(view, timeout_sec=0.)
                if predicate():
                    return
            pytest.fail(f'Timed out: last status={states[-1] if states else None}; events={events}')

        def settle(seconds, **kwargs):
            deadline = time.monotonic()+seconds
            until(lambda:time.monotonic() >= deadline, timeout=seconds+2., **kwargs)

        def trigger(name):
            client = driver.create_client(Trigger, name)
            try:
                assert client.wait_for_service(timeout_sec=5.)
                future = client.call_async(Trigger.Request())
                until(future.done)
                response = future.result()
                assert response.success, response.message
            finally:
                driver.destroy_client(client)

        def show_rrt(enabled):
            client = driver.create_client(SetParameters, '/object_follow/set_parameters')
            try:
                assert client.wait_for_service(timeout_sec=5.)
                request = SetParameters.Request(parameters=[Parameter('show_rrt', value=enabled).to_parameter_msg()])
                future = client.call_async(request)
                until(future.done, detection=True)
                assert all(r.successful for r in future.result().results)
            finally:
                driver.destroy_client(client)

        until(lambda:states and states[-1]['api_connected'] and joints and poses.get_subscription_count())
        trigger('/object_follow/arm')
        until(lambda:states[-1]['armed'] and states[-1]['paused'])
        assert not any(c[0]=='encoder_drag' for c in robot.calls)
        record('armed_waiting_for_detection_without_motion')
        tracking.publish(String(data='TRACKING'))
        until(lambda:states[-1]['detector_tracking'])
        until(lambda:sum(c[0]=='encoder_drag' for c in robot.calls) >= 5, timeout=75., detection=True)
        owner = controller.ik.mouse_session
        assert owner
        np.testing.assert_allclose(targets[-1], cursor, atol=1e-8)
        record('optical_detection_transformed_and_real_encoder_execution_started')
        until(lambda:any(p['route_points'] > 1 for p in planning_updates), detection=True)
        until(lambda:any(m['route_action']==Marker.ADD and m['route_width']==.006 and m['route_b']==1.
            for m in marker_checks), detection=True)
        show_rrt(False)
        until(lambda:planning_updates and not planning_updates[-1]['enabled'], detection=True)
        until(lambda:marker_checks and marker_checks[-1]['tree_action']==Marker.DELETE, detection=True)
        assert controller.ik.mouse_session == owner
        record('rrt_toggle_off_clears_display_and_retains_motion_owner')

        tracking.publish(String(data='LOST'))
        until(lambda:states[-1]['armed'] and states[-1]['paused'] and controller.snapshot()['ik']['state']=='paused')
        before = sum(c[0]=='encoder_drag' for c in robot.calls)
        settle(.7)
        assert sum(c[0]=='encoder_drag' for c in robot.calls) == before
        assert controller.ik.mouse_session == owner
        record('lost_holds_motion_and_retains_owner')

        resumed_plan_index = len(plans)
        measured = robot.angles[:]
        tracking.publish(String(data='TRACKING'))
        until(lambda:states[-1]['detector_tracking'])
        until(lambda:len(plans)>resumed_plan_index, timeout=45., detection=True)
        np.testing.assert_allclose(plans[resumed_plan_index]['request']['start_deg'], measured, atol=1e-9)
        assert controller.ik.mouse_session == owner
        record('reacquisition_reanchors_from_measured_joints')
        assert planning_updates[-1]['enabled'] is False
        show_rrt(True)
        record('rrt_toggle_on_waits_for_next_current_search')

        until(lambda:states[-1]['approach']['phase']=='holding', timeout=240., detection=True)
        final_status = copy.deepcopy(controller.snapshot()['ik'])
        assert final_status['final_goal_accepted'] and final_status['state']=='completed'
        completed = [p for p in plans if p['result'].get('playable')]
        assert {p['request']['tracking_region']['distance_mm'] for p in completed} == set(range(150,351,20))
        assert all(p['result']['visibility_model']['camera']==camera.configuration() for p in completed)
        assert all(p['result']['visibility']['passed'] for p in completed)
        assert all(p['request']['speed_rad_s']==pytest.approx(np.deg2rad(70/1.5)) for p in completed)
        np.testing.assert_allclose(np.asarray(targets), np.broadcast_to(cursor, (len(targets),3)), atol=1e-8)
        record('all_eleven_stages_complete_at_deployed_speed')
        assert any(p['enabled'] and p['nodes'] > 2 for p in planning_updates)
        assert any(m['tree_action']==Marker.ADD and m['tree_points'] > 2 and m['tree_width']==.004
            and m['tree_r']==1. for m in marker_checks)
        # Preserve a replayable completed approach before subsequent fault tests.
        (tmp_path/'approach_trace.json').write_text(json.dumps({
            'physical_motion':False, 'camera':camera.configuration(), 'plans':plans,
            'object_mm':(cursor*1000).tolist(),
            'encoder_path_deg':[initial_angles]+[
                ((np.asarray(c[1])-2048)/(v2_mouse.encoder_drag.STEPS_PER_DEG*robot.directions)).tolist()
                for c in robot.calls if c[0]=='encoder_drag'],
            'final_status':final_status}, indent=2)+'\n')

        before = len(plans)
        settle(1.1, detection=True, offset=np.array([.005,0.,0.]))
        assert len(plans)==before
        np.testing.assert_allclose(targets[-1], cursor, atol=1e-8)
        record('five_mm_detection_jitter_stays_inside_thirty_mm_deadband')

        # Replaying an older sample must not replace the accepted base target.
        # Drain the final valid detection by timestamp before counting. Target
        # and status use different topics and can be delivered in either order.
        until(lambda:target_stamps and target_stamps[-1] >= last_sample[0]-1e-6)
        old_target_count = len(targets)
        publish(last_sample[0]-.1, [0.,0.,.9])
        settle(.15)
        assert len(targets)==old_target_count
        record('out_of_order_detection_ignored')

        # Starve detections while feedback remains healthy.
        until(lambda:states[-1]['paused'], timeout=3.)
        assert states[-1]['armed'] and controller.ik.mouse_session==owner
        record('expired_detection_pauses_without_releasing_owner')
        until(lambda:not states[-1]['paused'], detection=True)
        record('fresh_detection_resumes_after_expiry')

        cursor = cursor + np.array([.035,0.,0.])
        until(lambda:targets and np.linalg.norm(np.array(targets[-1])-cursor)<1e-8, detection=True)
        record('thirty_five_mm_retarget_updates_base_target')
        trigger('/object_follow/stop')
        until(lambda:not states[-1]['armed'] and controller.ik.mouse_session is None)
        before = sum(c[0]=='encoder_drag' for c in robot.calls)
        settle(1.2, detection=True)
        assert sum(c[0]=='encoder_drag' for c in robot.calls)==before
        record('explicit_stop_releases_owner_and_prevents_late_execution')
        until(lambda:not planning_updates[-1]['enabled'])
        assert not planning_updates[-1]['nodes'] and not planning_updates[-1]['route_points']

        trigger('/object_follow/arm')
        until(lambda:states[-1]['armed'])
        publish(time.time()-.01, [0.,0.,.15], frame='wrong_frame')
        until(lambda:not states[-1]['armed'] and controller.ik.mouse_session is None)
        assert 'Unexpected detection frame' in states[-1]['reason']
        record('wrong_frame_fault_disarms')
        controller.disconnect()
        until(lambda:'disconnected' in states[-1]['feedback_error'].lower())
        record('simulated_disconnect_reported')
    finally:
        if view is not None:
            view.destroy_node()
        if driver is not None:
            driver.destroy_node()
        rclpy.try_shutdown()
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=10.)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
        if logfile is not None:
            logfile.close()
        if server is not None:
            server.shutdown(); server.server_close()
        if worker is not None:
            worker.join(2.)
        controller.shutdown()
        (tmp_path/'pipeline_trace.json').write_text(json.dumps({
            'physical_motion':False, 'camera':camera.configuration(), 'events':events,
            'plans':plans, 'statuses':states, 'targets_m':targets, 'joint_feedback':joints,
            'target_stamps':target_stamps,
            'optical_detections':detections, 'robot_calls':robot.calls,
            'planning_updates':planning_updates, 'planning_samples':planning_samples,
            'marker_checks':marker_checks,
            'child_returncode':None if child is None else child.returncode,
            'limitations':'Synthetic detections, instantaneous fake encoder plant, assumed camera FOV. No physical hardware.'},
            indent=2)+'\n')
    assert child.returncode==0, (tmp_path/'native.log').read_text()
