"""Native ROS integration with simulated feedback; no physical hardware.

Requires a built object_tracking_cpp executable and ROS Jazzy environment.
"""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pytest
pytest.importorskip('rclpy')

ROOT=Path(__file__).parents[2]
EXE=ROOT/'output/object_follow_ros2/install/object_tracking_cpp/lib/object_tracking_cpp/object_follow_cpp'


def test_native_detection_feedback_wait_loss_stop_and_clean_shutdown(tmp_path,monkeypatch):
    if not EXE.is_file():pytest.skip('Build object_tracking_cpp first')
    monkeypatch.setenv('ROS_DOMAIN_ID','184')
    import rclpy
    from geometry_msgs.msg import PoseStamped, PointStamped
    from std_msgs.msg import String
    from sensor_msgs.msg import JointState
    from std_srvs.srv import Trigger
    from rclpy.time import Time
    calls=[]
    connection_started=threading.Event()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            calls.append(self.path)
            now=time.time()
            def sample(stamp):
                return {'stamp_unix_s':stamp,'frame_id':'g_base','child_frame_id':'joint6_flange',
                    'translation_m':[.1,.2,.3],'quaternion_xyzw':[0.,0.,0.,1.],
                    'joint_names':[f'joint{i}' for i in range(1,7)],'angles_deg':[0.]*6}
            if self.path=='/api/feedback':
                result={'feedback_pose':sample(now), 'feedback_poses':[sample(now-.15),sample(now)]}
            else:result={'connected':False,'token':'native-test'}
            payload=json.dumps(result).encode()
            self.send_response(200);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
        def do_POST(self):
            calls.append(self.path)
            assert self.path=='/api/connect','No simulated motion command is allowed'
            self.rfile.read(int(self.headers.get('Content-Length',0)))
            connection_started.set();time.sleep(1.5)
            payload=b'{"connected": true}'
            self.send_response(200);self.send_header('Content-Length',str(len(payload)))
            self.end_headers();self.wfile.write(payload)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    logfile=(tmp_path/'native.log').open('w')
    child=subprocess.Popen([str(EXE),'--ros-args','--params-file',str(ROOT/'object_tracking_ros2/config/object_follow.yaml'),
        '-p',f'project_root:={ROOT}','-p','backend:=http','-p',f'api_url:=http://127.0.0.1:{server.server_port}',
        '-p','dry_run:=true','-p','auto_connect:=false'],env=dict(os.environ,OPENBLAS_NUM_THREADS='1'),stdout=logfile,stderr=logfile)
    rclpy.init();driver=rclpy.create_node('native_test_driver')
    states=[];targets=[];timings=[];joints=[]
    subscriptions=[driver.create_subscription(String,'/object_follow/status',lambda m:states.append(json.loads(m.data)),10),
        driver.create_subscription(String,'/object_follow/timing',lambda m:timings.append(json.loads(m.data)),10),
        driver.create_subscription(PointStamped,'/object_follow/target',lambda m:targets.append(m),10),
        driver.create_subscription(JointState,'/joint_states',lambda m:joints.append(m),10)]
    tracking=driver.create_publisher(String,'/object_pose/tracking_state',10)
    poses=driver.create_publisher(PoseStamped,'/object_pose/filtered',10)
    def until(predicate,timeout=8):
        deadline=time.monotonic()+timeout
        while not predicate() and time.monotonic()<deadline:
            assert child.poll() is None,(tmp_path/'native.log').read_text()
            rclpy.spin_once(driver,timeout_sec=.03)
        assert predicate(),(tmp_path/'native.log').read_text()
    def trigger(name):
        client=driver.create_client(Trigger,name)
        assert client.wait_for_service(timeout_sec=5)
        future=client.call_async(Trigger.Request());until(future.done)
        driver.destroy_client(client);return future.result()
    def publish_pose(offset=0.,frame='camera_color_optical_frame'):
        pose=PoseStamped();pose.header.frame_id=frame
        pose.header.stamp=Time(nanoseconds=int((time.time()-.05)*1e9)).to_msg()
        pose.pose.position.x=offset;pose.pose.position.z=.35
        poses.publish(pose)
    try:
        until(lambda:states and states[-1]['api_connected'] and joints)
        assert states[-1]['implementation']=='cpp' and not states[-1]['armed']
        assert states[-1]['orientation_policy']=='strict_then_fov'
        assert states[-1]['visibility_camera']['horizontal_fov_deg']==82.
        assert states[-1]['visibility_camera']['vertical_fov_deg']==68.
        assert trigger('/object_follow/arm').success
        until(lambda:states[-1]['armed'] and states[-1]['paused'])
        tracking.publish(String(data='TRACKING'))
        until(lambda:states[-1]['detector_tracking'])
        publish_pose()
        until(lambda:targets and not states[-1]['paused'])
        assert targets[-1].header.frame_id=='g_base'
        assert [targets[-1].point.x,targets[-1].point.y,targets[-1].point.z]==pytest.approx([.1,.2,.65])
        old=len(targets);publish_pose(.005);until(lambda:len(targets)>old)
        assert targets[-1].point.x==pytest.approx(.1) # 30 mm jitter deadband
        tracking.publish(String(data='LOST'))
        until(lambda:states[-1]['armed'] and states[-1]['paused'])
        assert trigger('/object_follow/stop').success
        until(lambda:not states[-1]['armed'])
        until(lambda:timings)
        assert timings[-1]['implementation']=='cpp'
        assert all(path in ('/api/feedback','/api/state') for path in calls)
        # A blocking backend connect must not starve the native Stop callback.
        client=driver.create_client(Trigger,'/object_follow/connect')
        assert client.wait_for_service(timeout_sec=3)
        pending=client.call_async(Trigger.Request())
        until(connection_started.is_set)
        assert trigger('/object_follow/stop').success
        assert not pending.done()
        until(pending.done)
        assert pending.result().success
        driver.destroy_client(client)
        assert not trigger('/arm/enable_motion').success # dry run never enables hardware
        # Wrong detection frames remain a fault, rather than tracking a guess.
        assert trigger('/object_follow/arm').success
        tracking.publish(String(data='TRACKING'))
        until(lambda:states[-1]['detector_tracking'] and states[-1]['armed'])
        publish_pose(frame='wrong_frame')
        until(lambda:not states[-1]['armed'])
        assert 'Unexpected detection frame' in states[-1]['reason']
    finally:
        driver.destroy_node();rclpy.try_shutdown()
        child.send_signal(signal.SIGINT)
        try:child.wait(timeout=10)
        except subprocess.TimeoutExpired:child.kill();child.wait();pytest.fail('Native node did not shut down cleanly')
        logfile.close();server.shutdown();server.server_close();worker.join(2)
    assert child.returncode==0,(tmp_path/'native.log').read_text()
