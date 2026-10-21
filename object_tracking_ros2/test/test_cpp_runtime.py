"""Embedded runtime behavior without ROS or a physical robot connection."""
from pathlib import Path
import pytest
import yaml
from unittest.mock import patch
from object_tracking_ros2.cpp_runtime import Runtime


class Backend:
    timeout=.1
    def __init__(self):self.calls=[]
    def call(self,route,body=None):
        self.calls.append((route,body))
        return {'connected':False,'feedback_pose':None}


def runtime(**settings):
    config=yaml.safe_load((Path(__file__).parents[1]/'config/object_follow.yaml').read_text())['object_follow']['ros__parameters']
    config.update(backend='http',**settings)
    return Runtime(config,lambda:100.,Backend())


def test_arm_without_object_waits_lost_remains_armed_and_stop_disarms():
    r=runtime(dry_run=True)
    assert r.api.backend.calls==[]
    assert r.arm()=='Armed (dry run); waiting for object'
    r.set_detector_state('LOST')
    assert r.session.armed and r.session.paused
    r.set_detector_state('TRACKING')
    assert r.submit([0.,0.,350.],99.9)==[0.,0.,350.]
    r.tick()
    assert r.session.armed and not r.session.paused
    assert r.submit([5.,0.,350.],99.95)==[0.,0.,350.]
    r.set_detector_state('LOST')
    assert r.session.armed and r.session.paused and r.session.latest is None
    r.stop()
    assert not r.session.armed
    for thread in r.commands.threads:thread.join(1.)


def test_cpp_runtime_keeps_fresh_feedback_and_sim_time_hardware_guards():
    r=runtime(dry_run=False,frame_verified=True,scene_verified=True)
    with pytest.raises(ValueError,match='Fresh measured'):r.arm()
    with pytest.raises(ValueError,match='wall-clock'):r.arm(use_sim_time=True)
    assert not r.session.armed and not r.api.backend.calls


def test_lost_data_errors_pause_but_wrong_frames_disarm():
    r=runtime(dry_run=True)
    r.arm();r.submit([0.,0.,350.],99.9);r.tick()
    r.detection_error('Stale or future detection',expired=True)
    assert r.session.armed and r.session.paused
    r.detection_error('Unexpected detection frame')
    assert not r.session.armed


def test_runtime_forwards_ros_mount_to_direct_controller_and_contract():
    config=yaml.safe_load((Path(__file__).parents[1]/'config/object_follow.yaml').read_text())['object_follow']['ros__parameters']
    config['eef_to_optical_translation_m'] = [.01,.02,.03]
    backend = Backend()
    with patch('object_tracking_ros2.direct.DirectBackend', return_value=backend) as factory:
        r = Runtime(config, lambda:100.)
    camera = factory.call_args.kwargs['camera']
    assert camera.eef_to_optical_translation_m == (.01,.02,.03)
    assert r.session.visibility_camera == camera.configuration()
    assert backend.calls == []

