"""Real HTTP Handler + RobotController + encoder executor, fake motors only.

Only planner proposal generation is stubbed; backend curve/collision/live-join,
exclusive ownership, motor ACK, heartbeat and completion are real.
"""
import sys
import time
import threading
from pathlib import Path
from http.server import ThreadingHTTPServer
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'webui/tests_python'))
sys.path.insert(0, str(ROOT/'webui'))
import test_v2_mouse as backend_tests
from server import Handler
from simulation.kinematics import ARM
from simulation.tracking_geometry import region_for, region_metrics
from object_tracking_ros2.core import FollowSession, LocalAPI


class Harness(backend_tests.V2MouseTest):
    __test__ = False
    ENGINE = 'learned_rrt'
    PROFILE = {'kind':'smooth_trapezoid','rise_percent':5,'fall_percent':10}
    def proposal(self, body):
        result = super().proposal(body)
        result['tracking_region'] = region_for(np.asarray(body['cursor_mm']),
            np.asarray(body['target_axis']), body['tracking_region'])
        return result

@pytest.fixture
def harness():
    h = Harness(methodName='runTest'); h.setUp()
    h.controller.ik.stop_mouse_aim(h.session)
    h.start[3] = np.deg2rad(-40.)  # 350 mm object must remain above the table.
    h.robot.angles = np.rad2deg(h.start).tolist()
    with h.controller.lock:
        h.controller._read()
    h.goal = h.start.copy(); h.goal[0] += np.deg2rad(1.)
    cls = type('QuietHandler', (Handler,), {'controller':h.controller,'token':'test-token',
        'hardware_enabled':True, 'mouse_aim_lock':threading.Lock(),
        'log_message':lambda *args:None})
    server = ThreadingHTTPServer(('127.0.0.1',0),cls)
    thread = threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    api = LocalAPI('http://127.0.0.1:'+str(server.server_port), timeout=2.)
    h.follower = FollowSession(api,time.time,max_age=.5,dry_run=False,speed=30,reference_speed=.35,
        target_deadband_mm=1.)
    t=ARM.transform(h.start)
    h.cursor=(t[:3,3]*1000+350*t[:3,2]).tolist()
    try:
        yield h
    finally:
        h.follower.stop()
        for t in h.follower.stop_threads:t.join(2)
        server.shutdown(); server.server_close();thread.join(2)
        h.tearDown();h.doCleanups()

def drive(h, condition, timeout=6):
    until=time.monotonic()+timeout
    while time.monotonic()<until:
        h.follower.submit(h.cursor,time.time())
        h.follower.tick()
        if condition():return
        time.sleep(.02)
    raise AssertionError((h.follower.last_error,h.controller.snapshot()['ik']))

def test_real_http_moving_replacement_ack_completion_and_explicit_stop(harness):
    h=harness; f=h.follower
    f.submit(h.cursor,time.time());f.arm()
    drive(h,lambda:f.execution is not None and h.controller.snapshot()['ik'].get('curve_time_s',0)>.06)
    first=f.execution
    # New cursor forces a real moving candidate; retain root execution owner.
    h.cursor[0]+=2.
    drive(h,lambda:f.plan is not None and f.plan!=first and h.controller.snapshot()['ik'].get('active_plan_id')==f.plan)
    assert f.execution==first
    assert h.controller.ik.mouse_pending_plan is None or h.controller.ik.mouse_pending_plan['confirmed']
    drive(h,lambda:h.controller.snapshot()['ik']['state']=='completed')
    drive(h,lambda:f.execution is None)
    assert f.armed, f.last_error
    assert any(call[0]=='encoder_drag' for call in h.robot.calls)
    assert not any(call[0]=='move' for call in h.robot.calls)
    f.stop('Explicit Stop')
    for thread in f.stop_threads:thread.join(2)
    assert h.controller.ik.mouse_session is None
    assert not f.armed

def test_real_http_busy409_retains_session_then_retries(harness):
    h=harness;f=h.follower
    f.submit(h.cursor,time.time());f.arm()
    original=h.planner.side_effect
    from webui.mouse_motion import StaleMousePlan
    count=[0]
    def stale_once(body):
        count[0]+=1
        if count[0]==1:raise StaleMousePlan('Live cursor anchor changed; retry')
        return original(body)
    h.planner.side_effect=stale_once
    drive(h,lambda:count[0]>=2 and f.execution is not None)
    assert f.armed
    assert h.controller.ik.mouse_session==f.session

def test_real_http_stationary_partial_prefix_replans_after_completion(harness):
    h=harness;f=h.follower
    original=h.planner.side_effect
    count=[0]
    def partial_once(body):
        count[0]+=1
        result=original(body)
        if count[0]==1:
            result.update(converged=False,replan_required=True,status='chunk_ready')
        return result
    h.planner.side_effect=partial_once
    f.submit(h.cursor,time.time());f.arm()
    drive(h,lambda:count[0]>=2 and f.execution is not None)
    assert f.armed, f.last_error
    assert f.accepted_target==h.cursor
    drive(h,lambda:h.controller.snapshot()['ik']['state']=='completed')
    drive(h,lambda:f.execution is None)
    assert f.armed

def test_real_http_slow_plan_keeps_lease_with_fresh_stationary_detections(harness):
    h=harness;f=h.follower
    original=h.planner.side_effect
    entered=threading.Event()
    def cold(body):
        entered.set();time.sleep(1.8)
        return original(body)
    h.planner.side_effect=cold
    f.submit(h.cursor,time.time());f.arm()
    f.tick();assert entered.wait(1)
    tick_times=[]
    until=time.monotonic()+1.5
    while time.monotonic()<until:
        f.submit(h.cursor,time.time())
        begin=time.monotonic();f.tick();tick_times.append(time.monotonic()-begin)
        time.sleep(.02)
    assert max(tick_times)<.05
    assert h.controller.ik.mouse_session==f.session
    drive(h,lambda:f.execution is not None)
    assert f.armed, f.last_error

def test_real_http_stale_accepted_pose_scoped_pause(harness):
    h=harness;f=h.follower
    f.submit(h.cursor,time.time());f.arm()
    drive(h,lambda:f.execution is not None)
    time.sleep(.55)
    f.tick()
    for thread in f.stop_threads:thread.join(2)
    assert f.armed and f.paused
    assert h.controller.ik.mouse_session == f.session
    assert h.controller.snapshot()['ik']['state']=='paused'
    assert 'expired' in f.last_error

def test_real_http_lost_holds_lease_and_resumes_from_measured_joints(harness):
    h=harness;f=h.follower
    f.submit(h.cursor,time.time());f.arm()
    drive(h,lambda:f.execution is not None and h.controller.snapshot()['ik'].get('curve_time_s',0)>.06)
    owner,execution=f.session,f.execution
    f.pause('Detector LOST')
    for thread in f.stop_threads:thread.join(2)
    assert f.armed and f.paused and not f.pause_pending
    measured=h.robot.angles[:]
    count=len(h.robot.calls)
    # Longer than the backend's 5-second lease timeout, with no detections.
    until=time.monotonic()+5.3
    while time.monotonic()<until:
        f.tick();time.sleep(.02)
    assert f.armed and f.paused and f.latest is None
    assert h.controller.ik.mouse_session==owner
    assert len(h.robot.calls)==count  # No encoder movement or repeated Stop.
    assert h.controller.snapshot()['ik']['state']=='paused'
    # The resumed stub must still meet the region's one-degree gaze
    # tolerance. Another full degree on top of the pre-pause movement does not.
    h.goal=np.deg2rad(measured);h.goal[0]+=np.deg2rad(.2)
    resumed_region = region_for(np.asarray(h.cursor),
        np.asarray(h.cursor)-ARM.fk(np.deg2rad(measured))*1000,
        h.controller.ik.v2_mouse.region_settings)
    assert region_metrics(h.goal, resumed_region, 1., 1.)['accepted']
    drive(h,lambda:f.execution is not None and f.execution!=execution)
    assert f.armed and not f.paused and f.session==owner
    request=h.planner.call_args.args[0]
    np.testing.assert_allclose(request['start_deg'],measured)
    np.testing.assert_allclose(request['start_velocity_rad_s'],np.zeros(6))
    assert f.execution!=execution

def test_real_http_control_waits_for_valid_hardware_cycle_without_disarming(harness):
    h=harness;f=h.follower
    # Use the production control deadline, rather than the harness override.
    f.api=LocalAPI(f.api.url)
    f.submit(h.cursor,time.time());f.arm()
    drive(h,lambda:f.execution is not None and h.controller.snapshot()['ik'].get('curve_time_s',0)>.06)
    entered,release=threading.Event(),threading.Event()
    def serialized_hardware_work():
        with h.controller.lock:
            entered.set();release.wait(2)
    thread=threading.Thread(target=serialized_hardware_work);thread.start()
    assert entered.wait(1)
    until=time.monotonic()+1.25
    try:
        while time.monotonic()<until:
            f.submit(h.cursor,time.time());f.tick();time.sleep(.02)
    finally:
        release.set();thread.join(2)
    drive(h,lambda:not f.control_busy and h.controller.snapshot()['ik']['state']=='completed')
    assert f.armed, f.last_error
    assert h.controller.ik.mouse_session==f.session
