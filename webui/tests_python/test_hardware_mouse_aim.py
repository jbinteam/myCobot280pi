import sys
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import RobotController
from test_hardware_ik import TrackingRobot
from simulation import pose5, seed5, service
from simulation.kinematics import ARM
from simulation.prm import Roadmap, planning_space


class HardwareMouseAimTest(unittest.TestCase):
    def setUp(self):
        self.robot = TrackingRobot()
        self.controller = RobotController(lambda *_: self.robot)
        self.controller.connect('192.168.77.225', 9000)
        self.road = Roadmap(planning_space('self', 1.))
        start = np.deg2rad(self.robot.angles); t = ARM.transform(start)
        self.cursor = ((t[:3, 3]+.054*t[:3, 2])*1000).tolist()
        solved = pose5.solve(start, t[:3, 3]+.004*t[:3, 2], target_axis=t[:3, 2])
        self.assertTrue(solved['converged'])
        self.goal = np.deg2rad(solved['frames'][-1]['q'])
        self.session = self.controller.ik.start_mouse_aim({'armed': True, 'frame_verified': True})['session_id']

    def tearDown(self):
        self.controller.shutdown()

    def prepare(self, **changes):
        with patch.object(service, 'load_roadmap', return_value=self.road), \
             patch.object(seed5, 'propose', return_value=([self.goal], {'status': 'ready'}, 0.)):
            return self.controller.ik.prepare_mouse_aim({'session_id': self.session, 'cursor_mm': self.cursor,
                                                        'guided_route_repair': True, **changes})

    def wait(self, states):
        until = time.monotonic()+6
        while time.monotonic() < until:
            status = self.controller.snapshot()['ik']
            if status['state'] in states:
                return status
            time.sleep(.025)
        self.fail(str(self.controller.snapshot()['ik']))

    def execute(self, prepared, confirm=True):
        self.controller.ik.execute(prepared['plan']['id'], 10, True, True)
        if confirm:
            self.controller.ik.mouse_aim_heartbeat(self.session, prepared['plan']['id'])

    def test_start_requires_both_confirmations_and_sends_no_motion(self):
        self.controller.ik.stop_mouse_aim(self.session)
        for body in ({'armed': False, 'frame_verified': True}, {'armed': True, 'frame_verified': False},
                     {'armed': True, 'frame_verified': True, 'angles': [0]*6}):
            with self.assertRaises(ValueError): self.controller.ik.start_mouse_aim(body)
        self.assertEqual(self.robot.calls, [])

    def test_prepare_reads_live_start_preserves_route_vertices_and_bounds_commands(self):
        data = self.prepare()
        self.assertFalse(data['simulation_only'])
        self.assertTrue(data['hardware_mouse_aim'])
        self.assertTrue(data['result']['cursor_acceptance']['accepted'])
        self.assertEqual(data['result']['validation'], 'joint_limits_and_self_collision')
        self.assertEqual(data['start_deg'], self.robot.angles)
        self.assertEqual(self.robot.calls, [])
        self.assertGreater(len(self.controller.ik.plan['waypoints']), 0)
        commands = self.controller.ik.plan['waypoints']
        np.testing.assert_allclose(commands[-1], data['result']['frames'][-1]['q'])
        for vertex in data['result']['waypoints_deg'][1:]:
            self.assertTrue(any(np.allclose(vertex, command, atol=1e-8) for command in commands))
        self.assertEqual(data['plan']['execution_mode'], 'velocity_feedback')
        points = [data['start_deg'], *self.controller.ik.plan['waypoints']]
        self.assertLessEqual(np.max(np.abs(np.diff(points, axis=0))), 3.)

    def test_local_preparation_reads_current_joints_and_skips_models_and_roadmap(self):
        self.robot.angles[0] += 1.
        with patch.object(service, 'load_roadmap', side_effect=AssertionError('Local plan must skip roadmap')), \
             patch.object(seed5, 'propose', side_effect=AssertionError('Local plan must skip NN')):
            data = self.controller.ik.prepare_mouse_aim({'session_id': self.session, 'cursor_mm': self.cursor})
        self.assertTrue(data['result']['converged'], data['result'].get('reason'))
        self.assertEqual(data['result']['tracking_source'], 'local_ik')
        self.assertEqual(data['start_deg'], self.robot.angles)
        np.testing.assert_allclose(data['result']['frames'][0]['q'], self.robot.angles)
        self.assertIsNone(data['result']['ik']['nn_checkpoint'])
        self.assertEqual(self.robot.calls, [])

    def test_client_cannot_supply_start_trajectory_or_unrelated_target(self):
        for extra in ({'start_deg': [0]*6}, {'frames': []}, {'angles': [0]*6}, {'target_mm': [0, 0, 0]}):
            with self.assertRaises(ValueError): self.prepare(**extra)
        self.assertEqual(self.robot.calls, [])

    def test_rejects_bad_sessions_and_excessive_tolerances(self):
        for extra in ({'session_id': 'foreign'}, {'tolerance_mm': 6}, {'orientation_tolerance_deg': 4},
                      {'tolerance_mm': True}, {'cursor_mm': [float('nan'), 0, 0]}):
            with self.assertRaises(ValueError): self.prepare(**extra)
        self.assertEqual(self.robot.calls, [])

    def test_session_reserves_motion_and_manual_planning(self):
        # Isolate the mouse-session exclusion on a firmware without the
        # independent Atom 6.5 directional-jog qualification block.
        self.controller.firmware = 6.4
        with self.assertRaises(RuntimeError): self.controller.move(self.robot.angles, 10, True)
        with self.assertRaises(RuntimeError): self.controller.jog(1, 1, 10, .1, True)
        with self.assertRaises(RuntimeError): self.controller.ik.prepare({'target_mm': [0, 0, 0]})
        with self.assertRaises(RuntimeError): self.controller.ik.start_mouse_aim({'armed': True, 'frame_verified': True})
        self.assertEqual(self.robot.calls, [])

    def test_no_motor_command_before_received_execution_is_acknowledged(self):
        data = self.prepare(); self.execute(data, False)
        self.controller.ik.mouse_aim_heartbeat(self.session)
        time.sleep(.4)
        self.assertEqual(self.robot.calls, [])
        self.controller.ik.mouse_aim_heartbeat(self.session, data['plan']['id'])
        self.assertEqual(self.wait({'completed', 'failed'})['state'], 'completed')
        self.assertTrue(any(c[0] == 'move' for c in self.robot.calls))
        self.assertTrue(all(c[2] == 10 and c[3] == {'_async': True} for c in self.robot.calls if c[0] == 'move'))

    def test_foreign_execution_cannot_be_acknowledged(self):
        data = self.prepare(); self.execute(data, False)
        with self.assertRaises(ValueError): self.controller.ik.mouse_aim_heartbeat(self.session, 'foreign')
        self.assertEqual(self.robot.calls, [])

    def test_stop_invalidates_prepared_plan_and_session(self):
        data = self.prepare(); self.controller.ik.stop_mouse_aim(self.session)
        with self.assertRaises(ValueError): self.controller.ik.execute(data['plan']['id'], 10, True, True)
        self.assertIsNone(self.controller.ik.mouse_session)
        self.assertEqual(self.robot.calls, [])

    def test_stop_cancels_running_queue(self):
        data = self.prepare(); self.execute(data, False)
        self.controller.ik.stop_mouse_aim(self.session)
        time.sleep(.3)
        self.assertEqual([c[0] for c in self.robot.calls], ['stop'])
        self.assertIsNone(self.controller.snapshot()['motion'])

    def test_idle_lease_expiry_cancels_without_a_motor_command(self):
        self.controller.ik.mouse_heartbeat_at -= 6
        with self.controller.lock: self.controller.ik.expire_mouse_aim()
        with self.assertRaises(ValueError): self.prepare()
        self.assertEqual(self.robot.calls, [])

    def test_running_lease_expiry_stops_before_any_motor_command(self):
        data = self.prepare(); self.execute(data, False)
        with self.controller.lock:
            self.controller.ik.mouse_heartbeat_at -= 6
            self.controller.ik.expire_mouse_aim()
        self.assertEqual([c[0] for c in self.robot.calls], ['stop'])

    def test_old_session_stop_cannot_cancel_a_new_session(self):
        old = self.session
        self.controller.ik.stop_mouse_aim(old)
        self.session = self.controller.ik.start_mouse_aim({'armed': True, 'frame_verified': True})['session_id']
        self.controller.ik.stop_mouse_aim(old)
        self.assertEqual(self.controller.ik.mouse_session, self.session)
        self.assertEqual(self.robot.calls, [])

    def test_stop_while_planning_prevents_a_late_plan_from_becoming_executable(self):
        data = self.prepare()['result']; entered = threading.Event(); release = threading.Event(); errors = []
        def solve(*args, **kwargs):
            entered.set(); release.wait(3); return data
        def prepare():
            try: self.prepare()
            except Exception as exc: errors.append(exc)
        with patch.object(service, 'mouse_aim_plan', side_effect=solve):
            thread = threading.Thread(target=prepare); thread.start()
            self.assertTrue(entered.wait(2))
            self.controller.ik.stop_mouse_aim(self.session); release.set(); thread.join(3)
        self.assertTrue(errors)
        self.assertIsNone(self.controller.ik.plan)
        self.assertEqual(self.robot.calls, [])

    def test_live_collision_or_limit_drift_is_checked_before_each_command(self):
        data = self.prepare()
        with patch.object(self.controller.ik.plan['space'], 'edge_valid', return_value=False):
            self.execute(data)
            status = self.wait({'failed'})
        self.assertIn('self-collision', status['message'])
        self.assertEqual([c[0] for c in self.robot.calls], ['stop'])

    def test_disconnect_cancels_owner_and_queued_execution(self):
        data = self.prepare(); self.execute(data, False); self.controller.disconnect()
        time.sleep(.3)
        self.assertIsNone(self.controller.ik.mouse_session)
        self.assertEqual([c[0] for c in self.robot.calls], ['stop', 'close'])


class HardwareMouseAimHTTPTest(unittest.TestCase):
    def test_hardware_api_reads_live_joints_and_uses_owned_stop_and_execution_ids(self):
        import json, urllib.error, urllib.request
        from server import Handler, ThreadingHTTPServer
        robot = TrackingRobot(); controller = RobotController(lambda *_: robot)
        controller.connect('192.168.77.225', 9000)
        handler = type('HardwareMouseHandler', (Handler,), {'controller': controller, 'hardware_enabled': True,
                        'token': 'mouse-test', 'log_message': lambda *args: None})
        http = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        threading.Thread(target=http.serve_forever, daemon=True).start()
        def post(path, body):
            req = urllib.request.Request(f'http://127.0.0.1:{http.server_port}/api/ik/{path}',
                  data=json.dumps(body).encode(), headers={'Content-Type': 'application/json', 'X-Robot-Token': 'mouse-test'})
            with urllib.request.urlopen(req, timeout=10) as response: return json.load(response)
        try:
            session = post('mouse-aim/start', {'armed': True, 'frame_verified': True})['session_id']
            self.assertEqual(robot.calls, [])
            t = ARM.transform(np.deg2rad(robot.angles)); cursor = (t[:3, 3]*1000+50*t[:3, 2]).tolist()
            with self.assertRaises(urllib.error.HTTPError) as caught:
                post('mouse-aim/plan', {'session_id': session, 'cursor_mm': cursor, 'start_deg': [999]*6})
            self.assertEqual(caught.exception.code, 400)
            with patch.object(service, 'load_roadmap', return_value=Roadmap(planning_space('self', 1.))):
                data = post('mouse-aim/plan', {'session_id': session, 'cursor_mm': cursor, 'guided_route_repair': True})
            self.assertTrue(data['result']['converged'])
            self.assertEqual(data['start_deg'], robot.angles)
            self.assertEqual(robot.calls, [])
            next = post('execute', {'plan_id': data['plan']['id'], 'speed': 10, 'armed': True, 'frame_verified': True})
            self.assertEqual(next['ik']['mouse_aim_session_id'], session)
            self.assertTrue(post('mouse-aim/heartbeat', {'session_id': session, 'execution_id': data['plan']['id']})['active'])
            next = post('mouse-aim/stop', {'session_id': session})
            self.assertIsNone(next['motion'])
            self.assertIsNone(next['ik']['mouse_aim_session_id'])
            self.assertFalse(any(call[0] == 'move' for call in robot.calls))
            with self.assertRaises(urllib.error.HTTPError): post('execute', {'plan_id': data['plan']['id'], 'speed': 10, 'armed': True, 'frame_verified': True})
        finally: http.shutdown(); http.server_close(); controller.shutdown()

    def test_simulation_server_refuses_all_hardware_mouse_routes(self):
        import json, urllib.error, urllib.request
        from server import Handler, ThreadingHTTPServer
        controller = RobotController(lambda *_: TrackingRobot())
        handler = type('SimulationHandler', (Handler,), {'controller': controller, 'hardware_enabled': False,
                        'token': 'mouse-test', 'log_message': lambda *args: None})
        http = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        threading.Thread(target=http.serve_forever, daemon=True).start()
        try:
            for path in ('start', 'plan', 'heartbeat', 'stop'):
                req = urllib.request.Request(f'http://127.0.0.1:{http.server_port}/api/ik/mouse-aim/{path}',
                      data=json.dumps({'armed': True, 'frame_verified': True}).encode(),
                      headers={'Content-Type': 'application/json', 'X-Robot-Token': 'mouse-test'})
                with self.assertRaises(urllib.error.HTTPError) as caught: urllib.request.urlopen(req, timeout=5)
                self.assertEqual(caught.exception.code, 403)
        finally: http.shutdown(); http.server_close(); controller.shutdown()


if __name__ == '__main__': unittest.main()
