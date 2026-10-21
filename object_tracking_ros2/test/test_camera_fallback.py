"""ROS approach policy through real planning/encoder gates, fake motors only."""
import copy
import json
import time
import threading
from unittest.mock import patch
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from object_tracking_ros2.core import FollowSession
from simulation.camera_model import CameraModel
from simulation.diffusion.orientation_policy import effective_orientation_mode, plan_with_orientation
from simulation.kinematics import ARM
from simulation.tracking_geometry import region_for
from webui.tracking_scene import region_acceptance
from test_direct_lifecycle import harness, arm, drive, real_tracking_plan


@pytest.mark.parametrize('protocol,mode,accepted',[(None,None,False),(1,'point_at_object',False),(1,'strict_then_fov',True)])
def test_hybrid_session_requires_backend_policy_handshake(protocol,mode,accepted):
    from test_core import API
    api=API(); original=api.call
    def call(route,body=None):
        result=original(route,body)
        if route.endswith('/start'):
            result['ik'].update(mouse_orientation_policy_protocol=protocol,requested_orientation_mode=mode)
        return result
    api.call=call
    follower=FollowSession(api,lambda:10.,dry_run=False,approach_final_distance_mm=150.,
        approach_camera_fov_fallback=True)
    if accepted:
        follower.arm();assert follower.armed
        follower.stop()
        for worker in follower.stop_threads:worker.join(1.)
    else:
        with pytest.raises(ValueError,match='strict-first camera orientation protocol'):
            follower.arm()
        assert not follower.armed
        assert api.calls[-1]==('/api/ik/mouse-aim/stop',{'session_id':'owned'})
    assert not any(route=='/api/ik/execute' for route,_ in api.calls)


def test_relaxed_acceptance_keeps_disk_and_rectangular_camera_bounds():
    q = np.deg2rad([1., -40., -130., -40., 1., 50.])
    t = ARM.transform(q)
    # 20 degrees from tool Z: valid image bounds, invalid strict pointing.
    delta = .15*t[:3,2] + .15*np.tan(np.deg2rad(20))*t[:3,0]
    obj = t[:3,3]+delta
    plan = {'tracking_region':region_for(obj*1000,delta,{'distance_mm':np.linalg.norm(delta)*1000}),
        'orientation_mode':'camera_fov', 'visibility_object_m':obj.tolist(),
        'visibility_camera':CameraModel(horizontal_fov_deg=82.,vertical_fov_deg=68.).configuration()}
    assert region_acceptance(q,plan,1.,3.)['accepted']
    assert not region_acceptance(q,{**plan,'orientation_mode':'point_at_object'},1.,3.)['accepted']
    # Same geometry cannot pass a narrower rectangular horizontal field.
    narrow = CameraModel(horizontal_fov_deg=20.,vertical_fov_deg=68.).configuration()
    assert not region_acceptance(q,{**plan,'visibility_camera':narrow},1.,3.)['accepted']
    changed = copy.deepcopy(plan)
    changed['tracking_region']['center_mm'] = (np.asarray(changed['tracking_region']['center_mm'])+10*delta/np.linalg.norm(delta)).tolist()
    assert not region_acceptance(q,changed,1.,3.)['accepted']
    with pytest.raises(ValueError,match='pinned camera/object'):
        region_acceptance(q,{**plan,'visibility_camera':None},1.,3.)


def test_hybrid_report_cannot_relax_a_playable_strict_result():
    attempt = Mock(side_effect=[{'status':'ik_failed','playable':False,'reason':'search exhausted'},
        {'status':'target_reached','playable':True,'reason':'checked'}])
    result = plan_with_orientation(attempt,'strict_then_fov')
    assert effective_orientation_mode(result,'strict_then_fov') == 'camera_fov'
    with pytest.raises(ValueError,match='changed the requested'):
        effective_orientation_mode(result,'point_at_object')
    result['orientation_selection']['strict']['playable'] = True
    with pytest.raises(ValueError,match='failed strict search'):
        effective_orientation_mode(result,'strict_then_fov')
    with pytest.raises(ValueError,match='strict-first'):
        effective_orientation_mode({'orientation_mode':'camera_fov'},'strict_then_fov')


@pytest.mark.parametrize('final_distance',[150.,130.])
def test_real_hybrid_approach_survives_encoder_rounding_and_measured_arrival(harness, tmp_path, final_distance):
    h = harness
    fixture = Path(__file__).parents[2]/'simulation/diffusion/tests/fixtures/ik_failed_reachable_150.json'
    case = json.loads(fixture.read_text())
    h.robot.angles = case['request']['start_deg'][:]
    h.cursor = case['request']['cursor_mm'][:]
    camera = CameraModel(**case['camera'])
    h.controller.visibility_camera = camera
    with h.controller.lock:
        h.controller._read()
    f = h.follower = FollowSession(h.api, time.time, max_age=.75,
        dry_run=False, speed=70, reference_speed=1., target_deadband_mm=1.,
        cone_distance_mm=170., approach_final_distance_mm=final_distance,
        visibility_camera=camera.configuration(), approach_camera_fov_fallback=True)
    captured = []
    def actual(body):
        if body['tracking_region']['distance_mm'] == 130.:
            # Prove fresh arrival from the relaxed stage allows the next stage
            # through the server gate, without executing an additional route.
            from webui.mouse_motion import StaleMousePlan
            captured.append((copy.deepcopy(body),{'status':'test_hold','playable':False}))
            raise StaleMousePlan('Test holds after the next measured stage is authorized')
        result = real_tracking_plan(body)
        captured.append((copy.deepcopy(body),copy.deepcopy(result)))
        return result
    h.planner.side_effect = actual
    arm(h)
    assert h.controller.snapshot()['ik']['requested_orientation_mode'] == 'strict_then_fov'
    drive(h,lambda:f.approach_status['phase']=='holding' if final_distance==150. else
        any(b['tracking_region']['distance_mm']==130. for b,_ in captured),timeout=90.)
    relaxed = [(body,result) for body,result in captured if result.get('orientation_mode')=='camera_fov' and result.get('playable')]
    assert relaxed, [(r.get('status'),r.get('orientation_mode')) for _,r in captured]
    for body,result in relaxed:
        selection = result['orientation_selection']
        assert selection['strict']['status'] in ('ik_failed','no_path')
        assert selection['strict']['playable'] is False
        assert result['visibility']['passed']
        np.testing.assert_allclose(result['frames'][0]['q'],body['start_deg'],atol=1e-10)
    status = h.controller.snapshot()['ik']
    assert status['orientation_mode']=='camera_fov'
    assert status['final_goal_accepted'] and status['cursor_acceptance']['camera_projection']['in_view']
    assert any(c[0]=='encoder_drag' for c in h.robot.calls)
    (tmp_path/'camera_fallback_trace.json').write_text(json.dumps({
        'physical_motion':False,'plans':[{'request':{k:v for k,v in b.items() if not k.startswith('_')},'result':r}
                                       for b,r in captured], 'final_status':status},indent=2)+'\n')


def test_hybrid_session_rejects_unchecked_result_before_any_motor_packet(harness):
    h=harness
    h.follower=FollowSession(h.api,time.time,dry_run=False,approach_final_distance_mm=150.,
        approach_camera_fov_fallback=True)
    # The deterministic proposal has valid strict geometry but no policy report.
    arm(h)
    h.follower._plan(h.follower.latest,h.follower.generation,h.follower.session)
    assert not h.follower.armed
    assert 'strict-first orientation selection' in h.follower.last_error
    assert not any(c[0]=='encoder_drag' for c in h.robot.calls)


def test_stop_cancels_inflight_endpoint_search_without_waiting_for_its_budget(harness):
    from simulation.diffusion.env import ArmEnv
    from webui.mouse_motion import StaleMousePlan
    h=harness; f=h.follower
    f.initial_distance_mm=f.region_distance_mm=330.
    h.planner.side_effect=real_tracking_plan
    entered,release=threading.Event(),threading.Event()
    def solve(*args,cancel_check=None,**kwargs):
        assert callable(cancel_check)
        entered.set()
        while not release.wait(.01):
            cancel_check()
        raise StaleMousePlan('Test releases unfinished solver')
    with patch.object(ArmEnv,'inverse_kinematics',side_effect=solve):
        arm(h); f.tick()
        try:
            assert entered.wait(5.)
            began=time.monotonic();f.stop()
            while f.busy and time.monotonic()-began<2.:
                time.sleep(.01)
            assert not f.busy
            assert not any(c[0]=='encoder_drag' for c in h.robot.calls)
        finally:
            release.set()
