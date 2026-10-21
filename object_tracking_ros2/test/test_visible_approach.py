"""Real candidate scheduler + follower/controller/executor, fake motors only."""
import copy
import json
import threading
import time
import numpy as np
from test_direct_lifecycle import harness, drive, arm, real_tracking_plan


def test_real_visible_planner_completes_full_350_to_150_approach(harness, tmp_path):
    h = harness; f = h.follower; captured = []
    f.approach = {'final_distance_mm':150.,'step_mm':20.,'distance_tolerance_mm':10.}
    def actual(body):
        result = real_tracking_plan(body)
        captured.append((copy.deepcopy(body),result))
        return result
    h.planner.side_effect = actual
    arm(h)
    drive(h,lambda:f.approach_status['phase']=='holding',timeout=180.)
    completed = [(b,r) for b,r in captured if r['playable']]
    assert {b['tracking_region']['distance_mm'] for b,r in completed} == set(range(150,351,20))
    assert all(r['visibility']['passed'] and r['modeled_visibility_checked'] for b,r in completed)
    assert all(r['endpoint_selection']['candidate_configuration']['ik_seconds']==.025 for b,r in completed)
    from simulation.camera_model import configured_camera
    assert all(r['visibility_model']['camera']==configured_camera().configuration() for b,r in completed)
    assert all(not r['visibility_model']['calibrated'] for b,r in completed)
    for body,result in completed:
        np.testing.assert_allclose(result['frames'][0]['q'],body['start_deg'],atol=1e-10)
        assert result['smoothing']['ends_at_rest']
    assert f.armed and not f.paused
    assert h.controller.snapshot()['ik']['final_goal_accepted']
    assert any(c[0]=='encoder_drag' for c in h.robot.calls)
    (tmp_path/'approach_trace.json').write_text(json.dumps({
        'camera':configured_camera().configuration(), 'physical_motion':False,
        'plans':[{'request':{k:v for k,v in body.items() if not k.startswith('_')},'result':result}
                 for body,result in captured],
        'final_status':h.controller.snapshot()['ik']}, indent=2)+'\n')


def test_lost_discards_real_inflight_plan_then_resumes_from_measured_joints(harness):
    h = harness; f = h.follower
    f.initial_distance_mm = f.region_distance_mm = 330.
    entered,release = threading.Event(),threading.Event(); captured=[]
    def actual(body):
        result=real_tracking_plan(body);captured.append(copy.deepcopy(body))
        if len(captured)==1:
            entered.set(); assert release.wait(8.)
        return result
    h.planner.side_effect=actual
    arm(h);owner=f.session;f.tick()
    assert entered.wait(8.)
    try:
        f.pause('Detector LOST during real planning')
        for t in f.stop_threads:t.join(2.)
        assert f.armed and f.paused and h.controller.ik.mouse_session==owner
    finally:
        release.set()
    until=time.monotonic()+8
    while f.busy and time.monotonic()<until:time.sleep(.02)
    assert not f.busy
    assert not any(c[0]=='encoder_drag' for c in h.robot.calls)
    h.robot.angles[0]+=.3;measured=h.robot.angles[:]
    drive(h,lambda:h.controller.snapshot()['ik']['state']=='completed',timeout=30.)
    assert len(captured)>=2
    np.testing.assert_allclose(captured[-1]['start_deg'],measured,atol=1e-10)
    np.testing.assert_array_equal(captured[-1]['start_velocity_rad_s'],np.zeros(6))
    assert f.armed and not f.paused and f.session==owner


def test_changed_object_discards_completed_old_proposal_before_execution(harness):
    h=harness;f=h.follower;f.initial_distance_mm=f.region_distance_mm=330.
    entered=[threading.Event(),threading.Event()];release=[threading.Event(),threading.Event()]
    captured=[]
    def actual(body):
        index=len(captured);captured.append(copy.deepcopy(body))
        result=real_tracking_plan(body)
        if index<2:
            entered[index].set();assert release[index].wait(8.)
        return result
    h.planner.side_effect=actual
    arm(h);f.tick();assert entered[0].wait(8.)
    try:
        h.cursor[0]+=2.
        f.submit(h.cursor,time.time());release[0].set()
        drive(h,lambda:entered[1].is_set(),timeout=12.)
        assert not any(c[0]=='encoder_drag' for c in h.robot.calls)
        assert captured[0]['cursor_mm']!=captured[1]['cursor_mm']
    finally:
        for event in release:event.set()
    drive(h,lambda:h.controller.snapshot()['ik']['state']=='completed',timeout=30.)
    assert f.armed
    assert h.controller.snapshot()['ik']['tracking_region']['cursor_mm']==h.cursor
