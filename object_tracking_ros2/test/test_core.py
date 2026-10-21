import threading
import time
import pytest
from object_tracking_ros2.core import FollowSession, validate_detection

class API:
    def __init__(self):
        self.calls = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.execution = None
    def call(self, route, body=None):
        self.calls.append((route, body))
        if route.endswith('/start'):
            return {'session_id': 'owned', 'ik': {'mouse_aim_session_id':'owned',
                'mouse_aim_client_id':'ros2_object_follow','mouse_executor':'encoder_feedback_tracking',
                'mouse_feedback_tracking_protocol':1,'mouse_tracking_region_protocol':1,
                'mouse_session_owner_protocol':1,'mouse_pause_protocol':1,
                'mouse_approach_protocol':1,'approach_settings':body.get('approach'),
                'tracking_region_settings': body['tracking_region'],
                'velocity_shape':{'kind':'smooth_trapezoid','rise_percent':5,'fall_percent':10}}}
        if route.endswith('/plan'):
            self.entered.set()
            self.release.wait(2)
            return {'session_id':'owned', 'plan': {'id': 'checked',
                'execution_mode':'encoder_feedback_tracking','joint_command':'encoder_drag_profile'},
                'result': {'playable': True}}
        if route == '/api/ik/execute':
            self.execution = body['plan_id']
            return {'ik':{'mouse_aim_session_id':'owned','execution_id':self.execution,
                'accepted_plan_id':body['plan_id']}}
        if route == '/api/state':
            return {'connected':True,'ik':{'mouse_aim_session_id':'owned','state':'running' if self.execution else 'idle',
                'execution_id':self.execution}}
        if route.endswith('/pause'):
            self.execution = None
            return {'connected':True,'ik':{'mouse_aim_session_id':'owned','state':'paused'}}
        return {}


@pytest.mark.parametrize('status,expected_seeds', [
    ('no_path', [42,43,44]), ('ik_failed', [42,43,44]),
    ('object_not_visible', [42,42,42])])
def test_bounded_search_retry_changes_seed_without_changing_region_or_executing(status, expected_seeds):
    api = API(); original = api.call
    proposals = []
    def response(route, body=None):
        if route.endswith('/plan'):
            proposals.append(body.copy())
            return {'session_id':'owned', 'plan':None, 'result':{'playable':False, 'status':status}}
        return original(route, body)
    api.call = response
    follower = FollowSession(api, lambda:10., dry_run=False, approach_final_distance_mm=150.)
    follower.submit([100.,0.,300.], 10.)
    follower.arm()
    try:
        for _ in range(3):
            follower._plan(follower.latest, follower.generation, follower.session)
        assert [p['seed'] for p in proposals] == expected_seeds
        assert all(p['tracking_region']==proposals[0]['tracking_region'] and
                   p['cursor_mm']==proposals[0]['cursor_mm'] for p in proposals)
        assert follower.armed and follower.session=='owned'
        assert not any(route=='/api/ik/execute' for route, _ in api.calls)
        follower.submit([135.,0.,300.], 10.1)
        assert follower.planning_seed == 42
    finally:
        follower.stop()
        for worker in follower.stop_threads: worker.join(1.)

@pytest.mark.parametrize('matching', [True, False])
def test_ros_camera_contract_is_checked_before_arming(matching):
    from simulation.camera_model import CameraModel
    camera = CameraModel().configuration()
    api = API(); original = api.call
    def response(route, body=None):
        result = original(route, body)
        if route.endswith('/start'):
            result['ik']['visibility_camera'] = (camera if matching else
                CameraModel(eef_to_optical_translation_m=(.01,0.,0.)).configuration())
        return result
    api.call = response
    session = FollowSession(api, lambda:10., dry_run=False, visibility_camera=camera)
    if matching:
        session.arm()
        assert session.armed
        session.stop()
        for thread in session.stop_threads: thread.join(1.)
    else:
        with pytest.raises(ValueError, match='ROS optical mount differs'):
            session.arm()
        assert not session.armed
        assert api.calls[-1] == ('/api/ik/mouse-aim/stop', {'session_id':'owned'})
    assert not any(route == '/api/ik/execute' for route, _ in api.calls)

@pytest.mark.parametrize('point,stamp,now,frame,previous', [
    ([0,0,float('nan')],10,10,'opt',0), ([0,0,-1],10,10,'opt',0),
    ([0,0,500],10,10,'opt',0), ([0,0,1],0,10,'opt',0),
    ([0,0,1],9,10,'opt',0), ([0,0,1],11,10,'opt',0),
    ([0,0,1],10,10,'wrong',0), ([0,0,1],10,10,'opt',10)])
def test_detection_rejected(point, stamp, now, frame, previous):
    with pytest.raises(ValueError):
        validate_detection(point,stamp,now,frame,'opt',previous)

def test_stale_detection_pauses_without_releasing_owner():
    api = API(); now = [10.]
    session = FollowSession(api, lambda: now[0], dry_run=False)
    session.submit([1,2,3],10); session.arm()
    now[0] = 10.6; session.tick()
    assert session.armed and session.paused
    for thread in session.stop_threads: thread.join(1)
    assert ('/api/ik/mouse-aim/pause', {'session_id':'owned'}) in api.calls
    assert not any(route.endswith('/stop') for route,_ in api.calls)
    session._control(session.generation,session.session)
    assert api.calls[-1] == ('/api/ik/mouse-aim/heartbeat', {'session_id':'owned'})
    assert not any(route == '/api/stop' for route,_ in api.calls)

def test_stop_invalidates_inflight_plan_and_latest_targets_coalesce():
    api = API(); session = FollowSession(api, lambda:10., dry_run=False)
    session.submit([1,2,3],9.9); session.arm(); session.tick()
    assert api.entered.wait(1)
    session.submit([4,5,6],9.95); session.tick()
    assert len([r for r,b in api.calls if r.endswith('/plan')]) == 1
    session.stop(); api.release.set()
    for _ in range(100):
        if not session.busy: break
        time.sleep(.005)
    assert not any(route == '/api/ik/execute' for route,_ in api.calls)

def test_execute_has_owned_session_and_ack_is_owned():
    api = API(); api.release.set()
    session = FollowSession(api, lambda:10., dry_run=False)
    session.submit([1,2,3],9.9); session.arm(); session.tick()
    for _ in range(100):
        if not session.busy: break
        time.sleep(.005)
    execute = next(b for r,b in api.calls if r == '/api/ik/execute')
    assert execute['session_id'] == 'owned'
    assert execute['plan_id'] == 'checked'
    session._control(session.generation, session.session)
    assert any(b.get('execution_id') == 'checked' and b.get('session_id') == 'owned'
        for r,b in api.calls if r.endswith('/heartbeat'))

def test_dryrun_never_calls_http():
    api = API(); session = FollowSession(api, lambda:10.)
    session.submit([1,2,3],9.9); session.arm(); session.tick(); session.stop()
    assert api.calls == []


def wait_idle(session):
    deadline=time.monotonic()+2
    while session.busy and time.monotonic()<deadline:time.sleep(.005)
    assert not session.busy

def test_candidate_discarded_when_latest_target_moves_over_30mm():
    api=API();session=FollowSession(api,lambda:10.,dry_run=False)
    session.submit([0,0,350],9.9);session.arm();session.tick()
    assert api.entered.wait(1)
    session.submit([31,0,350],9.95);api.release.set();wait_idle(session)
    assert not any(r=='/api/ik/execute' for r,b in api.calls)
    assert session.armed

def test_target_threshold_and_minimum_accept_interval():
    api=API();api.release.set();now=[10.]
    session=FollowSession(api,lambda:now[0],dry_run=False)
    session.submit([0,0,350],9.9);session.arm();session.tick();wait_idle(session)
    initial=sum(r.endswith('/plan') for r,b in api.calls)
    original_plan=session.plan
    now[0]=10.2;session.submit([.9,0,350],10.2);session.tick();wait_idle(session)
    assert sum(r.endswith('/plan') for r,b in api.calls)==initial
    now[0]=10.205;session.submit([18.,24.,350],10.205);session.tick();wait_idle(session)
    assert sum(r.endswith('/plan') for r,b in api.calls)==initial
    assert session.plan==original_plan and session.latest[0]==[0,0,350]
    assert sum(r=='/api/ik/execute' for r,b in api.calls)==1
    now[0]=10.21;session.submit([30.01,0,350],10.21);session.tick();wait_idle(session)
    assert sum(r.endswith('/plan') for r,b in api.calls)==initial+1
    now[0]=10.22;session.submit([65.,0,350],10.22);session.tick();wait_idle(session)
    assert sum(r.endswith('/plan') for r,b in api.calls)==initial+1

def test_deadband_holds_position_refreshes_age_and_allows_accumulated_motion():
    now=[10.];session=FollowSession(API(),lambda:now[0])
    session.submit([0,0,350],10.);session.arm()
    for stamp,point in [(10.2,[18,24,350]),(10.4,[0,0,379]),(10.6,[-29,0,350])]:
        now[0]=stamp;session.submit(point,stamp);session.tick()
        assert session.latest==([0,0,350],stamp)
        assert session.armed
    now[0]=10.8;session.submit([30.01,0,350],10.8);session.tick()
    assert session.latest==([30.01,0,350],10.8)
    now[0]=11.4;session.tick()
    assert session.armed and session.paused and session.last_error=='Accepted detection expired'

def test_jitter_during_planning_keeps_original_candidate():
    api=API();session=FollowSession(api,lambda:10.,dry_run=False)
    session.submit([0,0,350],9.9);session.arm();session.tick()
    assert api.entered.wait(1)
    session.submit([18,24,350],9.95)
    api.release.set();wait_idle(session)
    assert session.plan=='checked' and session.latest==([0,0,350],9.95)
    assert sum(r.endswith('/plan') for r,b in api.calls)==1
    assert sum(r=='/api/ik/execute' for r,b in api.calls)==1

def test_mismatched_execution_owner_is_never_acknowledged():
    class WrongOwner(API):
        def call(self,route,body=None):
            result=super().call(route,body)
            if route=='/api/ik/execute':result['ik']['mouse_aim_session_id']='another-owner'
            return result
    api=WrongOwner();api.release.set()
    session=FollowSession(api,lambda:10.,dry_run=False)
    session.submit([0,0,350],9.9);session.arm();session.tick();wait_idle(session)
    assert not session.armed
    assert not any(r.endswith('/heartbeat') and b.get('execution_id') for r,b in api.calls)

def test_stop_during_start_reports_cancelled_not_armed():
    class CancelStart(API):
        def call(self,route,body=None):
            result=super().call(route,body)
            if route.endswith('/start'):session.stop('Cancelled')
            return result
    api=CancelStart();session=FollowSession(api,lambda:10.,dry_run=False)
    session.submit([0,0,350],9.9)
    with pytest.raises(ValueError,match='cancelled'):session.arm()
    assert not session.armed
    for thread in session.stop_threads:thread.join(1)
    assert ('/api/ik/mouse-aim/stop',{'session_id':'owned'}) in api.calls

def test_lost_invalidates_inflight_plan_then_same_target_resumes():
    api=API();now=[10.];session=FollowSession(api,lambda:now[0],dry_run=False)
    session.submit([0,0,350],9.9);session.arm();session.tick()
    assert api.entered.wait(1)
    session.pause('Detector LOST')
    for thread in session.stop_threads:thread.join(1)
    session.pause('Detector LOST')  # Repeated LOST is idempotent.
    assert session.armed and session.paused and session.latest is None
    assert sum(r.endswith('/pause') for r,b in api.calls)==1
    api.release.set();wait_idle(session)
    assert not any(r=='/api/ik/execute' for r,b in api.calls)
    now[0]=10.1;session.submit([0,0,350],now[0]);session.tick();wait_idle(session)
    assert session.armed and not session.paused and session.execution=='checked'

def test_recovered_target_waits_for_hold_ack_and_explicit_stop_wins():
    class SlowPause(API):
        def call(self,route,body=None):
            if route.endswith('/pause'):
                entered.set();assert release.wait(2)
            return super().call(route,body)
    entered,release=threading.Event(),threading.Event()
    api=SlowPause();now=[10.];session=FollowSession(api,lambda:now[0],dry_run=False)
    session.submit([0,0,350],9.9);session.arm();session.pause()
    assert entered.wait(1)
    now[0]=10.1;session.submit([0,0,350],10.1);session.tick()
    assert session.armed and session.paused and session.pause_pending
    assert not any(r.endswith('/plan') for r,b in api.calls)
    session.stop();release.set()
    for thread in session.stop_threads:thread.join(1)
    session.submit([0,0,350],10.2);session.tick()
    assert not session.armed
    assert not any(r=='/api/ik/execute' for r,b in api.calls)
    assert api.calls[-1]==('/api/ik/mouse-aim/stop',{'session_id':'owned'})

def test_hold_failure_and_hardware_failure_still_disarm():
    class BadPause(API):
        def call(self,route,body=None):
            if route.endswith('/pause'):raise OSError('Transport lost')
            return super().call(route,body)
    session=FollowSession(BadPause(),lambda:10.,dry_run=False)
    session.submit([0,0,350],9.9);session.arm();session.pause()
    for thread in session.stop_threads:thread.join(1)
    assert not session.armed and 'Transport lost' in session.last_error
    session=FollowSession(API(),lambda:10.,dry_run=False)
    session.submit([0,0,350],9.9);session.arm();session.pause()
    for thread in session.stop_threads:thread.join(1)
    session.observe({'connected':True,'ik':{'mouse_aim_session_id':'owned',
        'state':'failed','message':'Self collision'}},session.generation,session.session)
    assert not session.armed and session.last_error=='Self collision'

@pytest.mark.parametrize('recover', [False, True])
def test_lost_during_arm_waits_or_uses_recovered_detection(recover):
    class LostStart(API):
        def call(self,route,body=None):
            result=super().call(route,body)
            if route.endswith('/start'):
                session.pause()
                session.pause()  # Repeated LOST while hardware startup waits.
                if recover:session.submit([0,0,350],9.95)
            return result
    session=FollowSession(LostStart(),lambda:10.,dry_run=False)
    session.submit([0,0,350],9.9)
    session.arm()
    assert session.armed and session.paused == (not recover)
    for thread in session.stop_threads:thread.join(1)
    assert not any(r.endswith('/stop') for r,b in session.api.calls)

@pytest.mark.parametrize('old_stamp', [None, 9.0])
def test_arm_without_fresh_detection_holds_lease_and_resumes_on_first_target(old_stamp):
    api=API();api.release.set();now=[10.]
    session=FollowSession(api,lambda:now[0],dry_run=False)
    if old_stamp is not None:session.submit([0,0,350],old_stamp)
    session.arm()
    for thread in session.stop_threads:thread.join(1)
    assert session.armed and session.paused and session.latest is None
    assert not session.pause_pending
    session._control(session.generation,session.session)
    assert any(r.endswith('/heartbeat') for r,b in api.calls)
    assert not any(r.endswith('/plan') or r=='/api/ik/execute' for r,b in api.calls)
    now[0]=10.1;session.submit([0,0,350],10.1);session.tick();wait_idle(session)
    assert session.armed and not session.paused and session.execution=='checked'

def test_detection_expiring_during_arm_becomes_waiting_not_failed():
    now=[10.]
    class SlowStart(API):
        def call(self,route,body=None):
            result=super().call(route,body)
            if route.endswith('/start'):now[0]=11.
            return result
    session=FollowSession(SlowStart(),lambda:now[0],dry_run=False)
    session.submit([0,0,350],10.);session.arm()
    for thread in session.stop_threads:thread.join(1)
    assert session.armed and session.paused and session.latest is None

def test_lost_does_not_erase_disarmed_fault_reason():
    session=FollowSession(API(),lambda:10.)
    session.stop('Hardware fault');session.pause('Detector LOST')
    assert not session.armed and session.last_error=='Hardware fault'

def completed_stage(session, distance, *, accepted=True, cursor=None):
    return {'connected':True,'ik':{'mouse_aim_session_id':session.session,
        'state':'completed','execution_id':session.execution,
        'final_goal_accepted':accepted,'replan_required':not accepted,
        'tracking_region':{'distance_mm':distance,
            'cursor_mm':cursor if cursor is not None else session.latest[0]}}}

def test_approach_shrinks_only_after_each_measured_stage_and_stops_at_150():
    session=FollowSession(API(),lambda:10.,approach_final_distance_mm=150.)
    session.submit([0,0,350],10.);session.arm()
    for distance in range(350,150,-20):
        failed=completed_stage(session,distance,accepted=False)
        session.observe(failed,session.generation,session.session)
        assert session.region_distance_mm==distance
        completed=completed_stage(session,distance)
        session.observe(completed,session.generation,session.session)
        assert session.region_distance_mm==distance-20
        session.observe(completed,session.generation,session.session)
        assert session.region_distance_mm==distance-20  # no duplicate advance
    assert session.approach_status['phase']=='approaching'
    session.observe(completed_stage(session,150),session.generation,session.session)
    assert session.region_distance_mm==150 and session.approach_status['phase']=='holding'

def test_approach_pauses_for_lost_or_changed_target():
    session=FollowSession(API(),lambda:10.,approach_final_distance_mm=150.)
    session.submit([0,0,350],9.9);session.arm()
    old=completed_stage(session,350)
    session.pause();session.observe(old,session.generation,session.session)
    assert session.region_distance_mm==350
    session.submit([40,0,350],10.);session.tick()
    session.observe(old,session.generation,session.session)
    assert session.region_distance_mm==350
    session.observe(completed_stage(session,350),session.generation,session.session)
    assert session.region_distance_mm==330

def test_lost_before_stage_request_resumes_last_backend_distance():
    session=FollowSession(API(),lambda:10.,approach_final_distance_mm=150.)
    session.submit([0,0,350],10.);session.arm()
    session.observe(completed_stage(session,350),session.generation,session.session)
    assert session.region_distance_mm==330
    session.pause()
    session.observe({'connected':True,'ik':{'mouse_aim_session_id':None,'state':'paused',
        'tracking_region_settings':{'distance_mm':350.}}},session.generation,session.session)
    assert session.region_distance_mm==350
    session.submit([0,0,350],10.01);session.tick()
    assert session.armed and not session.paused

def test_approach_stationary_target_requests_smaller_region_without_target_jitter():
    api=API();api.release.set();now=[10.]
    session=FollowSession(api,lambda:now[0],dry_run=False,approach_final_distance_mm=150.)
    session.submit([0,0,350],10.);session.arm();session.tick();wait_idle(session)
    session.observe(completed_stage(session,350),session.generation,session.session)
    now[0]=10.2
    session.tick();wait_idle(session)
    regions=[b['tracking_region']['distance_mm'] for r,b in api.calls if r.endswith('/plan')]
    assert regions==[350.,330.]
    session.submit([18,24,350],10.01);session.tick();wait_idle(session)
    assert len([r for r,b in api.calls if r.endswith('/plan')])==2

@pytest.mark.parametrize('final,step',[(0,20),(351,20),(float('nan'),20),(150,0),(150,51)])
def test_invalid_approach_limits_are_rejected(final,step):
    with pytest.raises(ValueError):
        FollowSession(API(),lambda:10.,approach_final_distance_mm=final,approach_step_mm=step)

@pytest.mark.parametrize('tolerance',[0.,-1.,10.01,float('nan'),True])
def test_invalid_approach_allowance_is_rejected(tolerance):
    with pytest.raises(ValueError):
        FollowSession(API(),lambda:10.,approach_final_distance_mm=150.,
                      approach_distance_tolerance_mm=tolerance)

def test_custom_cone_reaches_backend_and_rearming_resets_custom_initial_distance():
    api=API();session=FollowSession(api,lambda:10.,dry_run=False,
        approach_final_distance_mm=120.,cone_distance_mm=300.,cone_half_angle_deg=20.)
    session.submit([0,0,350],10.);session.arm()
    start=next(body for route,body in api.calls if route.endswith('/start'))
    assert start['tracking_region']=={'mode':'cone_disk','distance_mm':300.,'half_angle_deg':20.}
    session.observe(completed_stage(session,300),session.generation,session.session)
    assert session.region_distance_mm==280.
    session.stop()
    for thread in session.stop_threads:thread.join(1)
    session.arm()
    assert session.region_settings['distance_mm']==300.
    assert session.approach_status['initial_distance_mm']==300.
    for thread in session.stop_threads:thread.join(1)
