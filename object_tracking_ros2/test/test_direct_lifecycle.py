"""Real direct controller/planner validation/executor; fake motors, no HTTP.

Only the proposal generator and Pi transport are replaced. These tests exercise
FollowSession against DirectBackend without importing any WebUI server or its
fixtures. Collision checks, profiles, encoder conversion and ownership are real.
"""
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hardware.controller import RobotController
from object_tracking_ros2.core import FollowSession
from object_tracking_ros2.direct import DirectBackend, DirectBackendError
from pymycobot.common import ProtocolCode
from simulation.continuous import frames_for
from simulation.diffusion.env import ArmEnv
from simulation.diffusion.tracking import SceneSpace
from simulation.diffusion.classical_tracking import plan as real_tracking_plan
from simulation.kinematics import ARM
from simulation.tracking_geometry import region_for
from webui import encoder_drag, v2_mouse


class EncoderRobot:
    """Instantaneous fake encoder actuator with real packet conversion."""
    directions = np.array([-1, -1, 1, -1, -1, -1])

    def __init__(self):
        # Tool/cursor are above the table for the 350 mm visible approach.
        self.angles = [1., -40., -130., -40., 1., 50.]
        self.calls = []
        self.fresh_mode = 0

    def get_angles(self): return self.angles[:]
    def is_moving(self): return 0
    def get_system_version(self): return 6.5
    def get_fresh_mode(self): return self.fresh_mode
    def is_power_on(self): return 1
    def is_all_servo_enable(self): return 1
    def is_paused(self): return 0
    def get_error_information(self): return 0
    def get_encoders(self):
        return np.rint(2048 + np.asarray(self.angles) *
            encoder_drag.STEPS_PER_DEG * self.directions).astype(int).tolist()

    def _mesg(self, genre, *args, **kwargs):
        if genre == ProtocolCode.SET_ENCODERS_DRAG:
            self.calls.append(('encoder_drag', args[0][:], args[1][:], kwargs))
            self.angles = ((np.asarray(args[0])-2048) /
                (encoder_drag.STEPS_PER_DEG*self.directions)).tolist()
        elif genre == ProtocolCode.SET_FRESH_MODE:
            self.fresh_mode = args[0]
            self.calls.append(('mode', args[0], kwargs))
        else:
            self.calls.append(('stop', genre, kwargs))

    def send_angles(self, *args, **kwargs):
        raise AssertionError('Tracking must use the checked encoder executor')

    def close(self): self.calls.append(('close',))


@pytest.fixture
def harness(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError('Direct backend attempted network/HTTP I/O with fake motors')
    monkeypatch.setattr(urllib.request, 'urlopen', no_network)
    monkeypatch.setattr(socket, 'create_connection', no_network)
    monkeypatch.setattr(socket.socket, 'connect', no_network)
    monkeypatch.setattr(socket.socket, 'bind', no_network)
    h = SimpleNamespace(robot=EncoderRobot())
    h.controller = RobotController(lambda *_: h.robot)
    h.api = DirectBackend(controller=h.controller)
    h.api.call('/api/connect', {'host': '192.168.77.225', 'port': 9000})
    h.env = ArmEnv(); h.env.set_obstacle(None)
    h.space = SceneSpace(h.env)
    h.start = np.deg2rad(h.robot.angles)
    h.goal = h.start.copy(); h.goal[0] += np.deg2rad(1.)
    transform = ARM.transform(h.start)
    h.cursor = (transform[:3, 3]*1000+350*transform[:3, 2]).tolist()

    def proposal(body):
        start = np.deg2rad(body['start_deg'])
        transform = ARM.transform(h.goal)
        target, axis = transform[:3, 3], transform[:3, 2]
        states, times, smoothing = v2_mouse.profile_tracking.prepare(
            h.space, [start, h.goal], np.array(body['start_velocity_rad_s']),
            np.array(body['start_acceleration_rad_s2']), body['speed_rad_s'],
            velocity_shape=body['velocity_shape'])
        return {'method': 'learned_rrt', 'task': 'pose5', 'mouse_aim': True,
            'simulation_only': True, 'continuous_motion': True,
            'tracking_strategy': 'learned_ik_rrt', 'tracking_chunk': True,
            'tracking_source': 'classical_rrt', 'status': 'target_reached',
            'playable': True, 'converged': True, 'replan_required': False,
            'gap_mm': 350., 'coordinate_frame': 'g_base',
            'tool_frame': 'joint6_flange', 'checkpoint_sha256': v2_mouse.IK_SHA256,
            'validation': 'joint_limits_self_table_block', 'obstacle_mm': None,
            'scene_mode': 'no_block', 'cursor_mm': body['cursor_mm'],
            'target_mm': (target*1000).tolist(), 'target_axis': axis.tolist(),
            'goal_deg': np.rad2deg(h.goal).tolist(),
            'goal_cursor_acceptance': {'accepted': True},
            'cursor_acceptance': {'accepted': True}, 'final_error_mm': 0.,
            'final_orientation_error_deg': 0., 'smoothing': smoothing,
            'speed_rad_s': body['speed_rad_s'], 'acceleration_rad_s2': 3.,
            'frames': frames_for(states, times, target, axis),
            'tracking_region': region_for(np.asarray(body['cursor_mm']),
                np.asarray(body['target_axis']), body['tracking_region'])}

    h.proposal = proposal
    h.follower = FollowSession(h.api, time.time, max_age=.5, dry_run=False,
        speed=30, reference_speed=.35, target_deadband_mm=1.)
    with patch.object(v2_mouse.classical_tracking, 'plan', side_effect=proposal) as planner:
        h.planner = planner
        try:
            yield h
        finally:
            h.follower.stop('Test shutdown')
            for worker in h.follower.stop_threads: worker.join(2)
            h.api.shutdown()
            h.env.close()


def drive(h, condition, timeout=6., *, detection=True):
    until = time.monotonic()+timeout
    while time.monotonic() < until:
        if detection: h.follower.submit(h.cursor, time.time())
        h.follower.tick()
        if condition(): return
        time.sleep(.02)
    raise AssertionError((h.follower.last_error, h.controller.snapshot()['ik']))


def arm(h):
    h.follower.submit(h.cursor, time.time())
    h.follower.arm()


def test_deployed_speed_reaches_tracking_executor_and_encoder_profile(harness):
    import yaml
    h = harness; f = h.follower
    settings = yaml.safe_load((ROOT/'object_tracking_ros2/config/object_follow.yaml').read_text())['object_follow']['ros__parameters']
    f.speed = settings['sdk_speed']
    f.reference_speed = settings['reference_speed_rad_s']
    arm(h)
    mouse = h.controller.ik.v2_mouse
    assert mouse.speed == 70
    assert mouse.speed_rad_s == pytest.approx(np.deg2rad(46.6666666667))
    drive(h, lambda: f.execution is not None and
          any(call[0] == 'encoder_drag' for call in h.robot.calls))
    assert h.planner.call_args.args[0]['speed_rad_s'] == pytest.approx(mouse.speed_rad_s)
    assert h.controller.snapshot()['ik']['state'] in ('running', 'completed')


def test_direct_arm_without_object_waits_then_tracks_first_detection(harness):
    h = harness; f = h.follower
    f.arm()
    owner = f.session
    drive(h, lambda: f.paused and not f.pause_pending, detection=False)
    deadline = time.monotonic()+2
    while time.monotonic() < deadline:
        f.pause('Detector LOST')
        f.tick()
        time.sleep(.02)
    assert f.armed and f.paused and f.session == owner
    assert h.controller.snapshot()['ik']['state'] == 'paused'
    assert not h.planner.called
    assert not any(call[0] == 'encoder_drag' for call in h.robot.calls)
    drive(h, lambda: f.execution is not None and
          any(call[0] == 'encoder_drag' for call in h.robot.calls))
    assert f.armed and not f.paused and f.session == owner


def test_approach_backend_requires_arrival_and_does_not_shrink_twice_on_failed_plan(harness):
    h = harness; f = h.follower
    f.approach = {'final_distance_mm':150.,'step_mm':20.,'distance_tolerance_mm':10.}
    arm(h)
    request={'session_id':f.session,'cursor_mm':h.cursor,
             'tracking_region':{'mode':'cone_disk','distance_mm':330.,'half_angle_deg':30.}}
    with pytest.raises(DirectBackendError,match='current target region'):
        h.api.call('/api/ik/mouse-aim/plan',request)
    assert h.controller.ik.v2_mouse.region_settings['distance_mm']==350.
    drive(h,lambda:h.controller.snapshot()['ik']['state']=='completed')
    assert h.controller.snapshot()['ik']['final_goal_accepted']
    old_region=h.controller.ik.v2_mouse.active_region.copy()
    # Off-center measured arrival changes the camera-to-object line, even
    # though the accepted object's coordinates have not changed.
    h.robot.angles[0] += 1.
    measured=h.robot.angles[:]
    expected_axis=np.asarray(h.cursor)-ARM.fk(np.deg2rad(measured))*1000
    expected_axis/=np.linalg.norm(expected_axis)
    assert np.linalg.norm(expected_axis-np.asarray(old_region['axis'])) > .001
    def no_path(body):
        result=h.proposal(body)
        result.update(playable=False,converged=False,frames=[])
        return result
    h.planner.side_effect=no_path
    with pytest.raises(DirectBackendError,match='next bounded distance'):
        h.api.call('/api/ik/mouse-aim/plan',{**request,'tracking_region':{**request['tracking_region'],'distance_mm':310.}})
    response=h.api.call('/api/ik/mouse-aim/plan',request)
    assert not response['result']['playable']
    assert response['result']['tracking_region']['radius_mm']==pytest.approx(330*np.tan(np.deg2rad(30)))
    mouse=h.controller.ik.v2_mouse
    assert mouse.region_settings['distance_mm']==330.
    assert mouse.active_region['distance_mm']==350.  # Failed route never replaces the checked region.
    new_region=response['result']['tracking_region']
    np.testing.assert_allclose(new_region['axis'],expected_axis,atol=1e-12)
    np.testing.assert_allclose(new_region['center_mm'],np.asarray(h.cursor)-330*expected_axis,atol=1e-10)
    np.testing.assert_allclose(mouse.active_region['axis'],old_region['axis'],atol=1e-12)
    h.robot.angles[0] += .3
    retry=h.api.call('/api/ik/mouse-aim/plan',request)  # Same stage may retry.
    np.testing.assert_allclose(retry['result']['tracking_region']['axis'],expected_axis,atol=1e-12)
    with pytest.raises(DirectBackendError,match='current target region'):
        h.api.call('/api/ik/mouse-aim/plan',{**request,'tracking_region':{**request['tracking_region'],'distance_mm':310.}})


def test_real_planner_approaches_one_stage_with_fake_motors(harness):
    h = harness; f = h.follower
    f.approach = {'final_distance_mm':150.,'step_mm':20.,'distance_tolerance_mm':10.}
    arm(h)
    drive(h,lambda:h.controller.snapshot()['ik']['state']=='completed')
    from object_tracking_ros2.planning_visualization import PlanningVisualization
    display = PlanningVisualization()
    display.set_enabled(True)
    f.planning_visualization = display
    h.planner.side_effect=real_tracking_plan
    drive(h,lambda:h.controller.snapshot()['ik'].get('tracking_region',{}).get('distance_mm')==330.
          and h.controller.snapshot()['ik']['state']=='completed',timeout=20.)
    state=h.controller.snapshot()['ik']
    assert state['final_goal_accepted']
    assert state['tracking_region_acceptance']['accepted']
    assert state['tracking_region_acceptance']['axial_error_mm'] <= 10.
    assert state['approach_completed_with_allowance']
    assert state['curve_time_s'] < state['curve_duration_s']
    assert state['tracking_region']['radius_mm']==pytest.approx(330*np.tan(np.deg2rad(30)))
    import json
    drawing = json.loads(display.message())
    assert drawing['nodes'] and len(drawing['route_deg']) > 1
    assert drawing['status'] in ('direct_checked', 'checked_route')


def test_direct_tracking_retargets_encoder_curve_and_releases_on_stop(harness):
    h = harness; f = h.follower
    assert not f.armed and h.robot.calls == []
    arm(h)
    drive(h, lambda: f.execution is not None and
        h.controller.snapshot()['ik'].get('curve_time_s', 0) > .06)
    execution = f.execution
    h.cursor[0] += 2.
    drive(h, lambda: f.plan is not None and f.plan != execution and
        h.controller.snapshot()['ik'].get('active_plan_id') == f.plan)
    assert f.execution == execution
    drive(h, lambda: h.controller.snapshot()['ik']['state'] == 'completed')
    assert f.armed, f.last_error
    assert any(call[0] == 'encoder_drag' for call in h.robot.calls)
    f.stop('Explicit Stop')
    for worker in f.stop_threads: worker.join(2)
    count = len(h.robot.calls)
    time.sleep(.15)
    assert len(h.robot.calls) == count
    assert not f.armed and h.controller.ik.mouse_session is None


def test_direct_lost_retains_lease_and_fresh_target_resumes_measured_pose(harness):
    h = harness; f = h.follower; arm(h)
    drive(h, lambda: f.execution is not None and
        h.controller.snapshot()['ik'].get('curve_time_s', 0) > .06)
    owner, execution = f.session, f.execution
    f.pause('Detector LOST')
    for worker in f.stop_threads: worker.join(2)
    measured, count = h.robot.angles[:], len(h.robot.calls)
    assert f.armed and f.paused and not f.pause_pending
    # Keep the direct backend lease alive beyond its five-second expiry.
    deadline = time.monotonic()+5.3
    while time.monotonic() < deadline:
        f.tick(); time.sleep(.02)
    assert h.controller.ik.mouse_session == owner
    assert f.armed and f.paused and f.latest is None
    assert len(h.robot.calls) == count
    assert h.controller.snapshot()['ik']['state'] == 'paused'
    h.goal = np.deg2rad(measured); h.goal[0] += np.deg2rad(.2)
    drive(h, lambda: f.execution is not None and f.execution != execution)
    assert f.armed and not f.paused and f.session == owner
    request = h.planner.call_args.args[0]
    np.testing.assert_allclose(request['start_deg'], measured)
    np.testing.assert_allclose(request['start_velocity_rad_s'], np.zeros(6))


def test_direct_stale_detection_pauses_without_disarming(harness):
    h = harness; f = h.follower; arm(h)
    drive(h, lambda: f.execution is not None)
    time.sleep(.55); f.tick()
    for worker in f.stop_threads: worker.join(2)
    assert f.armed and f.paused
    assert h.controller.ik.mouse_session == f.session
    assert h.controller.snapshot()['ik']['state'] == 'paused'
    assert 'expired' in f.last_error


def test_direct_stop_invalidates_inflight_plan_before_any_motion(harness):
    h = harness; f = h.follower
    entered, release = threading.Event(), threading.Event()
    def slow(body):
        entered.set()
        assert release.wait(3)
        return h.proposal(body)
    h.planner.side_effect = slow
    arm(h); f.tick()
    assert entered.wait(2)
    try:
        f.stop('Explicit Stop during planning')
        for worker in f.stop_threads: worker.join(2)
    finally:
        release.set()
    deadline = time.monotonic()+3
    while f.busy and time.monotonic() < deadline: time.sleep(.02)
    assert not f.busy and not f.armed
    assert h.controller.ik.mouse_session is None
    assert not any(call[0] == 'encoder_drag' for call in h.robot.calls)


def test_direct_stale_join_is_retryable_without_losing_owner(harness):
    from webui.mouse_motion import StaleMousePlan
    h = harness; f = h.follower
    count = [0]
    def stale_once(body):
        count[0] += 1
        if count[0] == 1: raise StaleMousePlan('Live anchor changed; retry')
        return h.proposal(body)
    h.planner.side_effect = stale_once
    arm(h)
    drive(h, lambda: count[0] >= 2 and f.execution is not None)
    assert f.armed and h.controller.ik.mouse_session == f.session


def test_direct_hardware_feedback_failure_disarms(harness):
    h = harness; f = h.follower; arm(h)
    drive(h, lambda: f.execution is not None and
        h.controller.snapshot()['ik'].get('curve_time_s', 0) > .06)
    def disconnected(): raise OSError('Test hardware transport lost')
    h.robot.get_angles = disconnected
    # Execution disarms immediately; the idle feedback poll then closes the
    # failed transport. Wait for both asynchronous lifecycle steps.
    drive(h, lambda: not f.armed and not h.controller.snapshot()['connected'])
    assert not h.controller.snapshot()['connected']
    assert any(call[0] == 'stop' for call in h.robot.calls)


@pytest.mark.parametrize('kind', ['joints', 'pose'])
def test_direct_general_commands_use_real_classical_planner_and_encoder_executor(harness, kind):
    from object_tracking_ros2.commands import CheckedCommands
    from scipy.spatial.transform import Rotation
    from simulation.diffusion import classical_service
    h = harness
    # Preserve the original exact-pose fixture; this command has no view task.
    h.robot.angles[3] = 80.
    with h.controller.lock:
        h.controller._read()
    h.start = np.deg2rad(h.robot.angles)
    settings = {'dry_run': False, 'frame_verified': True, 'scene_verified': True,
        'sdk_speed': 5, 'max_detection_age_s': .75, 'base_frame': 'g_base'}
    commands = CheckedCommands(h.api, settings.__getitem__, h.follower)
    # All six joints move, including tool roll, so a pose-only replacement
    # cannot accidentally satisfy the exact-joint contract.
    goal = h.start + np.deg2rad([1.2, .5, -.6, .7, .8, -1.1])
    target = ARM.transform(goal)
    try:
        with patch.object(classical_service, 'plan', wraps=classical_service.plan) as planner:
            commands.enable()
            assert h.robot.calls == []
            if kind == 'joints':
                # Names, rather than array order, identify the requested joints.
                commands.submit_joints(list(reversed(ARM.names)), goal[::-1].tolist(), time.time())
            else:
                commands.submit_pose('g_base', target[:3, 3].tolist(),
                    Rotation.from_matrix(target[:3, :3]).as_quat().tolist(), time.time())
            deadline = time.monotonic()+20
            while time.monotonic() < deadline:
                commands.tick()
                if commands.status['state'] in ('completed', 'stopped', 'rejected'):
                    break
                time.sleep(.02)
            assert commands.status['state'] == 'completed', commands.status
            assert commands.enabled
            assert planner.call_count == 1
            request = planner.call_args.args[0]
            assert request['obstacle_mm'] is None
            assert any(call[0] == 'encoder_drag' for call in h.robot.calls)
            state = h.controller.snapshot()['ik']
            assert state['state'] == 'completed'
            if kind == 'joints':
                np.testing.assert_allclose(request['goal_deg'], np.rad2deg(goal), atol=1e-10)
                np.testing.assert_allclose(h.robot.angles, np.rad2deg(goal), atol=.05)
            else:
                assert 'goal_deg' not in request
                actual = ARM.transform(np.deg2rad(h.robot.angles))
                assert np.linalg.norm(actual[:3, 3]-target[:3, 3]) < .001
                angle = np.rad2deg(np.arccos(np.clip(actual[:3, 2] @ target[:3, 2], -1., 1.)))
                assert angle < 1.
    finally:
        commands.shutdown()
    assert not commands.enabled
    count = len(h.robot.calls)
    time.sleep(.15)
    assert len(h.robot.calls) == count
