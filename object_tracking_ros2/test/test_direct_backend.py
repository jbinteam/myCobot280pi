"""Standalone backend contract and hardware ownership, without a ROS graph."""
import json
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch

from hardware.controller import RobotController
from object_tracking_ros2.direct import DirectBackend, DirectBackendError
from webui.mouse_motion import StaleMousePlan


class FakeRobot:
    def __init__(self, *args):
        self.calls = []
    def get_angles(self): return [0.] * 6
    def is_moving(self): return 0
    def get_system_version(self): return 6.5
    def get_fresh_mode(self): return 1
    def close(self): self.calls.append('close')


class DirectBackendTests(unittest.TestCase):
    def setUp(self):
        self.c = RobotController(factory=FakeRobot)
        self.backend = DirectBackend(controller=self.c)
        self.addCleanup(self.backend.shutdown)

    def connect(self):
        return self.backend.call('/api/connect', {'host': '127.0.0.1', 'port': 9000})

    def test_construct_disconnected_shutdown_idempotent(self):
        self.assertFalse(self.backend.call('/api/state')['connected'])
        self.assertIsNone(self.c.robot)
        self.backend.shutdown()
        self.backend.shutdown()
        self.assertFalse(self.c.poller.is_alive())
        with self.assertRaises(DirectBackendError) as caught:
            self.backend.call('/api/connect', {'host': '127.0.0.1'})
        self.assertEqual(caught.exception.code, 503)

    def test_measured_feedback_has_matching_joint_names_and_is_detached(self):
        self.connect()
        state = self.backend.call('/api/state')
        self.assertEqual(state['backend'], 'direct')
        self.assertTrue(state['hardware_enabled'])
        sample = self.backend.call('/api/feedback')['feedback_pose']
        self.assertEqual(len(sample['joint_names']), 6)
        self.assertEqual(sample['angles_deg'], [0.] * 6)
        stamp = sample['stamp_unix_s']
        sample['angles_deg'][0] = 100.
        actual = self.backend.call('/api/feedback')['feedback_pose']
        self.assertEqual(actual['angles_deg'][0], 0.)
        self.assertEqual(actual['stamp_unix_s'], stamp)
        self.backend.call('/api/disconnect', {})
        self.assertIsNone(self.backend.call('/api/feedback')['feedback_pose'])

    def test_feedback_does_not_wait_for_command_lock(self):
        self.connect()
        result = []
        with self.c.lock:
            thread = threading.Thread(target=lambda: result.append(self.backend.call('/api/feedback')))
            thread.start()
            thread.join(.25)
            self.assertFalse(thread.is_alive())
        self.assertTrue(result[0]['connected'])

    def test_only_selected_planner_and_executor_can_be_requested(self):
        with patch.object(self.c.ik, 'prepare_diffusion', return_value={'plan': {'id': 'checked'}}) as prepare:
            self.backend.call('/api/ik/diffusion/plan', {'target_mm': [1, 2, 3]})
            settings = prepare.call_args.args[0]
            self.assertEqual(settings['planner'], 'learned_rrt')
            self.assertEqual(settings['joint_command'], 'encoder_drag_profile')
            self.assertEqual(settings['velocity_shape'], self.backend.velocity_shape)
            for wrong in ({'planner': 'diffusion_v2'}, {'joint_command': 'send_angles'},
                          {'velocity_shape': {'kind': 'quintic'}}):
                with self.assertRaises(DirectBackendError) as caught:
                    self.backend.call('/api/ik/diffusion/plan', wrong)
                self.assertEqual(caught.exception.code, 400)
            self.assertEqual(prepare.call_count, 1)

    def test_sandbox_and_raw_motion_routes_unavailable(self):
        for route in ('/api/move', '/api/jog', '/api/ik/plan', '/api/sim/solve',
                      '/api/ik/joint-diagnostic/plan', '/api/ik/jog-qualification/plan'):
            with self.assertRaises(DirectBackendError) as caught:
                self.backend.call(route, {})
            self.assertEqual(caught.exception.code, 404)
        self.assertIsNone(self.c.robot)

    def test_stale_plans_and_busy_planner_are_retryable(self):
        with patch.object(self.c.ik, 'prepare_mouse_aim', side_effect=StaleMousePlan('Old live anchor')):
            with self.assertRaises(DirectBackendError) as caught:
                self.backend.call('/api/ik/mouse-aim/plan', {'session_id': 'ours'})
            self.assertEqual(caught.exception.code, 409)

        with self.backend.mouse_aim_lock:
            with self.assertRaises(DirectBackendError) as caught:
                self.backend.call('/api/ik/mouse-aim/plan', {'session_id': 'ours'})
            self.assertEqual(caught.exception.code, 409)

    def test_observed_plan_keeps_request_and_exclusive_planner_gate(self):
        events = []
        body = {'session_id':'ours', 'cursor_mm':[1,2,3]}
        def prepare(actual, observer=None):
            self.assertEqual(actual, body)
            observer({'kind':'begin'})
            return {'plan':None}
        with patch.object(self.c.ik, 'prepare_mouse_aim', side_effect=prepare) as planner:
            self.backend.plan_with_observer(body, events.append)
            self.assertEqual(events, [{'kind':'begin'}])
            with self.backend.mouse_aim_lock:
                with self.assertRaises(DirectBackendError) as caught:
                    self.backend.plan_with_observer(body, events.append)
                self.assertEqual(caught.exception.code, 409)
            self.assertEqual(planner.call_count, 1)

    def test_plan_releases_controller_lock_and_stop_remains_responsive(self):
        entered, release = threading.Event(), threading.Event()
        def prepare(body):
            entered.set()
            release.wait(2)
            return {'plan': None}
        with patch.object(self.c.ik, 'prepare_mouse_aim', side_effect=prepare), \
                patch.object(self.c.ik, 'pause_mouse_aim', return_value={'ik': {'state': 'paused'}}) as pause:
            worker = threading.Thread(target=lambda: self.backend.call('/api/ik/mouse-aim/plan', {}))
            worker.start()
            try:
                self.assertTrue(entered.wait(.5))
                self.assertEqual(self.backend.call('/api/ik/mouse-aim/pause', {'session_id': 'ours'})['ik']['state'], 'paused')
                pause.assert_called_once_with('ours')
            finally:
                release.set()
                worker.join(2)

    def test_execute_rejects_embedded_paths_and_excess_speed(self):
        with patch.object(self.c.ik, 'execute') as execute:
            for body in ({'plan_id': 'checked', 'angles': [0]*6}, {'plan_id': 'checked', 'speed': 99}):
                with self.assertRaises(DirectBackendError) as caught:
                    self.backend.call('/api/ik/execute', body)
                self.assertEqual(caught.exception.code, 400)
            execute.assert_not_called()

    def test_import_does_not_load_http_server(self):
        code = "from object_tracking_ros2.direct import DirectBackend; import sys,json; print(json.dumps([n for n in sys.modules if n in ('webui.server','server','http.server')]))"
        output = subprocess.check_output([sys.executable, '-c', code], text=True)
        self.assertEqual(json.loads(output), [])


class ControllerOwnershipTests(unittest.TestCase):
    def test_real_controller_excludes_second_process_then_releases(self):
        c = RobotController()
        self.addCleanup(c.shutdown)
        c._acquire_ownership('127.0.0.1', 65530)
        code = '''from hardware.controller import RobotController
c=RobotController()
try:
    c._acquire_ownership('127.0.0.1',65530)
except RuntimeError as e:
    print(str(e))
else:
    print('acquired')
finally:
    c.shutdown()
'''
        output = subprocess.check_output([sys.executable, '-c', code], text=True)
        self.assertIn('already owned', output)
        c.disconnect()
        output = subprocess.check_output([sys.executable, '-c', code], text=True)
        self.assertEqual(output.strip(), 'acquired')

    def test_failed_connect_releases_ownership(self):
        def failed(host, port):
            raise OSError('fake refused connection')
        with patch('hardware.controller.connect_robot', failed):
            c = RobotController(factory=failed)
            self.addCleanup(c.shutdown)
            with self.assertRaisesRegex(RuntimeError, 'fake refused'):
                c.connect('127.0.0.1', 65530)
            self.assertIsNone(c._ownership_fd)
            other = RobotController(factory=failed)
            self.addCleanup(other.shutdown)
            other._acquire_ownership('127.0.0.1', 65530)


if __name__ == '__main__':
    unittest.main()
