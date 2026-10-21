"""General command validation, ownership and Stop cancellation without motors."""
import threading
import time
from types import SimpleNamespace
import numpy as np
import pytest
from simulation.kinematics import ARM
from object_tracking_ros2.commands import CheckedCommands


class API:
    def __init__(self):
        self.calls=[]
        self.entered=threading.Event();self.release=threading.Event();self.release.set()
        self.execution=None;self.state='idle';self.connected=True
    def call(self,route,body=None):
        self.calls.append((route,body))
        if route=='/api/state':
            return {'connected':self.connected,'motion':None,'moving':False,
                'ik':{'state':self.state,'execution_id':self.execution}}
        if route.endswith('/plan'):
            self.entered.set();assert self.release.wait(2)
            return {'plan':{'id':'checked','planner':'learned_rrt',
                'joint_command':'encoder_drag_profile','execution_mode':'encoder_profile'},
                'result':{'converged':True}}
        if route=='/api/ik/execute':
            self.execution=body['plan_id'];self.state='running'
            return {'ik':{'execution_id':self.execution}}
        if route=='/api/stop':self.state='stopped';self.execution=None
        return {}


@pytest.fixture
def command():
    api=API();tracking=SimpleNamespace(armed=False,session=None)
    settings={'dry_run':False,'frame_verified':True,'scene_verified':True,
        'sdk_speed':5,'max_detection_age_s':.75,'base_frame':'g_base'}
    command=CheckedCommands(api,settings.__getitem__,tracking)
    yield command,api,tracking
    api.release.set();command.shutdown()


def wait(command):
    until=time.monotonic()+2
    while command.busy and time.monotonic()<until:time.sleep(.005)
    assert not command.busy


def joint_goal(command):
    command.submit_joints(list(ARM.names),np.deg2rad([1,-40,-130,80,1,50]).tolist(),time.time())


def test_goal_requires_enable_and_exclusive_tracking_owner(command):
    c,api,tracking=command
    with pytest.raises(ValueError,match='Enable'):joint_goal(c)
    assert not any(r.endswith('/plan') for r,b in api.calls)
    tracking.armed=True
    with pytest.raises(ValueError,match='Stop object'):c.enable()
    tracking.armed=False;tracking.session='owner'
    with pytest.raises(ValueError,match='Stop object'):c.enable()


def test_named_radians_and_pose_units_use_only_selected_stack(command):
    c,api,_=command;c.enable()
    degrees=[1,-40,-130,80,1,50]
    c.submit_joints(list(reversed(ARM.names)),np.deg2rad(degrees[::-1]).tolist(),time.time())
    wait(c)
    body=next(b for r,b in api.calls if r.endswith('/plan'))
    np.testing.assert_allclose(body['goal_deg'],degrees)
    assert body['planner']=='learned_rrt' and body['joint_command']=='encoder_drag_profile'
    assert body['velocity_shape']=={'kind':'smooth_trapezoid','rise_percent':5,'fall_percent':10}
    assert c.execution=='checked'
    api.state='completed';c.tick()
    for thread in list(c.threads):thread.join(1)
    assert c.execution is None and c.status['state']=='completed'
    c.submit_pose('g_base',[.1,.2,.3],[0,0,0,1],time.time());wait(c)
    body=[b for r,b in api.calls if r.endswith('/plan')][-1]
    assert body['target_mm']==[100,200,300] and body['target_axis']==[0,0,1]
    assert 'goal_deg' not in body


@pytest.mark.parametrize('fault',['names','radians','limits','stamp','frame','quaternion'])
def test_invalid_goal_cannot_reach_planner(command,fault):
    c,api,_=command;c.enable()
    names=list(ARM.names);q=np.deg2rad([1,-40,-130,80,1,50]).tolist();stamp=time.time()
    if fault=='names':names[0]=names[1]
    if fault=='radians':q[0]=float('nan')
    if fault=='limits':q[0]=100
    if fault=='stamp':stamp-=2
    with pytest.raises(ValueError):
        if fault in ('frame','quaternion'):
            c.submit_pose('camera' if fault=='frame' else 'g_base',[.1,.2,.3],
                [0,0,0,2] if fault=='quaternion' else [0,0,0,1],stamp)
        else:c.submit_joints(names,q,stamp)
    assert not any(r.endswith('/plan') for r,b in api.calls)


def test_stop_cancels_inflight_planning_and_later_execute(command):
    c,api,_=command;c.enable();api.release.clear();joint_goal(c)
    assert api.entered.wait(1)
    c.stop();api.release.set();wait(c)
    for thread in list(c.threads):thread.join(1)
    assert not c.enabled
    assert not any(r=='/api/ik/execute' for r,b in api.calls)


def test_second_goal_rejected_while_command_active(command):
    c,api,_=command;c.enable();joint_goal(c);wait(c)
    with pytest.raises(ValueError,match='already active'):joint_goal(c)
    assert sum(r.endswith('/plan') for r,b in api.calls)==1


def test_hardware_disconnect_disarms_general_commands(command):
    c,api,_=command;c.enable();joint_goal(c);wait(c)
    api.connected=False;c.tick()
    for thread in list(c.threads):thread.join(1)
    assert not c.enabled and c.status['state']=='stopped'
