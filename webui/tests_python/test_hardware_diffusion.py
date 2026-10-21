"""No physical transport: checked planner adapter and fake-arm execution gates."""
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import RobotController
import test_hardware_ik as helpers
from simulation.kinematics import ARM
from simulation.diffusion.env import ArmEnv
from webui import hardware_diffusion as adapter


class HardwareDiffusionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deployment = adapter.verified_status()

    def setUp(self):
        self.robot = helpers.TrackingRobot()
        self.controller = RobotController(lambda *_: self.robot)
        self.controller.connect('192.168.77.225', 9000)
        self.start = self.robot.angles[:]
        self.end = self.start[:]; self.end[0] += 3.6
        t = ARM.transform(np.deg2rad(self.end))
        self.body = {'target_mm': (t[:3, 3]*1000).tolist(), 'target_axis': t[:3, 2].tolist(), 'joint_command': 'send_angles'}
        self.result = {'method': 'diffusion', 'task': 'pose5', 'converged': True,
                       'status': 'success', 'simulation_only': True,
                       'validation': 'joint_limits_self_table_block', 'obstacle_mm': None,
                       'checkpoint_sha256': adapter.V2_SHA256, 'environment': self.deployment['environment'],
                       'motion_conditioning': {'schema': 2}, **self.body,
                       'final_error_mm': 0., 'final_orientation_error_deg': 0.,
                       'frames': [{'q': self.start}, {'q': self.end}],
                       'smoothing': {'segments': [{'start_deg': self.start, 'end_deg': self.end}]}}
        patcher = patch.object(adapter.service, 'plan', return_value=self.result)
        self.mock_plan = patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.controller.shutdown()

    def plan(self):
        return self.controller.ik.prepare_diffusion(self.body)

    def set_path(self, corners):
        self.end = corners[-1][:]
        t = ARM.transform(np.deg2rad(self.end))
        self.body = {'target_mm': (t[:3, 3]*1000).tolist(), 'target_axis': t[:3, 2].tolist(), 'joint_command': 'send_angles'}
        self.result.update(self.body)
        self.result['frames'] = [{'q': q} for q in [self.start, *corners]]
        self.result['smoothing']['segments'] = [
            {'start_deg': a, 'end_deg': b} for a, b in zip([self.start, *corners[:-1]], corners)]

    def execute(self, data, **kwargs):
        return self.controller.ik.execute(data['plan']['id'], kwargs.get('speed', 5),
                                          kwargs.get('armed', True), kwargs.get('frame_verified', True),
                                          kwargs.get('scene_verified', True))

    def wait_state(self):
        deadline = time.monotonic()+4
        while time.monotonic() < deadline:
            state = self.controller.snapshot()['ik']
            if state['state'] in ('completed', 'failed'):
                return state
            time.sleep(.025)
        self.fail('Fake-arm execution did not finish')

    def test_reads_live_start_no_motion_and_retains_bounded_checked_path(self):
        data = self.plan()
        sent = self.mock_plan.call_args.args[0]
        self.assertEqual(self.mock_plan.call_args.kwargs,{})
        self.assertEqual(sent['start_deg'], self.start)
        self.assertIsNone(sent['obstacle_mm'])
        self.assertNotIn('goal_deg', sent)
        self.assertEqual(self.robot.calls, [])
        self.assertTrue(data['plan']['hardware_diffusion'])
        self.assertEqual(data['plan']['checkpoint_sha256'], adapter.V2_SHA256)
        self.assertAlmostEqual(data['plan']['table_top_mm'], -35.)
        self.assertEqual(data['plan']['arrival_deg'], 1.5)
        self.assertEqual(data['plan']['path_arrival_deg'], 3.)
        self.assertEqual(data['plan']['max_live_command_deg'], 4.)
        self.assertEqual(self.controller.ik.plan['stop_indices'], {1})
        points = np.array([self.start, *self.controller.ik.plan['waypoints']])
        self.assertEqual(data['plan']['max_step_deg'], 2.)
        self.assertLessEqual(np.abs(np.diff(points, axis=0)).max(), 2.+1e-8)

    def test_client_cannot_supply_start_goal_or_scene(self):
        for key, value in [('start_deg', [0]*6), ('goal_deg', self.end), ('obstacle_mm', None), ('frames', [])]:
            with self.assertRaises(ValueError):
                self.controller.ik.prepare_diffusion({**self.body, key: value})
        self.assertEqual(self.robot.calls, [])

    def test_local_offset_uses_fresh_live_flange_rotation(self):
        offset = np.array([2., -1., 3.])
        rotation = ARM.transform(np.deg2rad(self.start))[:3, :3]
        self.body['target_mm'] = (np.asarray(self.result['target_mm'])-rotation@offset).tolist()
        self.body['eef_offset_mm'] = offset.tolist()
        data = self.plan()
        np.testing.assert_allclose(data['target_mm'], self.result['target_mm'], atol=1e-9)

    def test_failed_rollout_returns_no_executable_plan(self):
        self.result.update(converged=False, frames=[], status='no_safe_chunk')
        data = self.plan()
        self.assertIsNone(data['plan'])
        self.assertIsNone(self.controller.ik.plan)
        self.assertEqual(self.robot.calls, [])

    def test_rejects_unqualified_checkpoint_and_deployment_changes(self):
        wrong = {**self.deployment, 'model_name': 'diffusion_cartesian_relative_v4'}
        with patch.object(adapter.service, 'info', return_value=wrong):
            with self.assertRaisesRegex(RuntimeError, 'pinned'):
                self.plan()
        self.mock_plan.assert_not_called()
        data = self.plan()
        with patch.object(adapter.service, 'info', return_value=wrong):
            with self.assertRaisesRegex(RuntimeError, 'pinned'):
                self.execute(data)
        self.assertEqual(self.robot.calls, [])

    def test_collision_and_discontinuous_corners_fail_closed(self):
        with patch.object(ArmEnv, 'path_valid', return_value=False):
            with self.assertRaisesRegex(ValueError, 'collision'):
                self.plan()
        self.result['smoothing']['segments'][0]['start_deg'] = [0]*6
        with self.assertRaisesRegex(ValueError, 'discontinuous'):
            self.plan()
        self.assertEqual(self.robot.calls, [])

    def test_rejects_endpoint_outside_strict_tolerance(self):
        self.result['frames'][-1]['q'] = self.start
        self.result['smoothing']['segments'][0]['end_deg'] = self.start
        with self.assertRaisesRegex(ValueError, 'strict'):
            self.plan()

    def test_live_session_change_during_planning_discards_result(self):
        def changed(_, **kwargs):
            with self.controller.lock:
                self.controller._cancel_timer()
            return self.result
        self.mock_plan.side_effect = changed
        with self.assertRaisesRegex(RuntimeError, 'session changed'):
            self.plan()
        self.assertIsNone(self.controller.ik.plan)

    def test_fixed_target_accepts_80_percent_with_existing_profile_checks(self):
        self.body['speed']=80
        data=self.plan()
        self.assertEqual(data['plan']['max_sdk_speed'],80)
        self.execute(data,speed=80);self.controller.ik.heartbeat(data['plan']['id'])
        state=self.wait_state()
        self.assertEqual(state['state'],'completed',state.get('message'))
        self.assertTrue(any(c[0]=='move' and c[2]==80 for c in self.robot.calls))

    def test_speed_change_requires_replanning_before_any_motion(self):
        data=self.plan()
        self.assertEqual(data['plan']['planned_sdk_speed'],5)
        self.assertAlmostEqual(data['result']['joint_velocity_allowance_deg_s'],7.5)
        with self.assertRaisesRegex(ValueError,'plan again'):
            self.execute(data,speed=80)
        self.assertEqual(self.robot.calls,[])

    def test_sdk_sends_do_not_run_ahead_of_reviewed_profile(self):
        data=self.plan()
        intervals=self.controller.ik.plan['waypoint_intervals_s'][:]
        sends=[];original=self.robot.send_angles
        def sent(*args,**kwargs):
            sends.append(time.monotonic());return original(*args,**kwargs)
        self.robot.send_angles=sent
        self.execute(data);acknowledged=time.monotonic()
        self.controller.ik.heartbeat(data['plan']['id'])
        state=self.wait_state()
        self.assertEqual(state['state'],'completed',state.get('message'))
        self.assertEqual(len(sends),len(intervals))
        for elapsed,planned in zip(np.diff([acknowledged,*sends]),intervals):
            self.assertGreaterEqual(elapsed,planned-1e-6)
        for sample in state['samples']:
            if 'command_velocity_deg_s' in sample:
                self.assertLessEqual(max(map(abs,sample['command_velocity_deg_s'])),7.5+1e-8)

    def test_scene_frame_arm_speed_expiry_and_pose_drift_gates(self):
        data = self.plan()
        for kwargs in ({'scene_verified': False}, {'frame_verified': False}, {'armed': False}, {'speed': 81}):
            with self.assertRaises(ValueError):
                self.execute(data, **kwargs)
        self.controller.ik.plan['created'] -= 61
        with self.assertRaisesRegex(ValueError, 'expired'):
            self.execute(data)
        data = self.plan(); self.robot.angles[0] += 2.
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.execute(data)
        self.assertEqual(self.robot.calls, [])

    def test_fake_arm_waits_for_heartbeat_executes_once_and_stops_at_waypoints(self):
        data = self.plan(); self.execute(data)
        time.sleep(.35)
        self.assertEqual(self.robot.calls, [])
        self.controller.ik.heartbeat(data['plan']['id'])
        state = self.wait_state()
        self.assertEqual(state['state'], 'completed')
        self.assertTrue(state['hardware_diffusion'])
        self.assertEqual(state['completed_waypoints'], 2)
        self.assertTrue(all(call[0] == 'move' and call[2] == 5 for call in self.robot.calls))
        with self.assertRaisesRegex(ValueError, 'matching'):
            self.execute(data)

    def test_live_segment_collision_stops_before_any_motor_target(self):
        data = self.plan()
        with patch.object(ArmEnv, 'edge_valid', return_value=False):
            self.execute(data); self.controller.ik.heartbeat(data['plan']['id'])
            state = self.wait_state()
        self.assertEqual(state['state'], 'failed')
        self.assertIn('collision', state['message'])
        self.assertEqual([c[0] for c in self.robot.calls], ['stop'])

    def test_stopped_subdegree_residual_advances_from_actual_telemetry(self):
        send = self.robot.send_angles
        def undershoot(angles, speed, **kwargs):
            send(angles, speed, **kwargs)
            self.robot.angles[0] -= .65
        self.robot.send_angles = undershoot
        data = self.plan()
        with patch.object(ArmEnv, 'edge_valid', autospec=True, return_value=True) as edge:
            self.execute(data); self.controller.ik.heartbeat(data['plan']['id'])
            state = self.wait_state()
        self.assertEqual(state['state'], 'completed', state.get('message'))
        self.assertEqual(state['completed_waypoints'], 2)
        self.assertEqual(len(self.robot.calls), 2)
        # The second command is 2.45 degrees from measured joints, exceeding
        # the old 1.5-degree bound. Validate that actual segment, not the plan start.
        np.testing.assert_allclose(np.rad2deg(edge.call_args.args[1]), [1.15, *self.start[1:]])
        self.assertLessEqual(state['estimated_error_mm'], 5.)

    def test_endpoint_residual_above_deadband_allowance_still_times_out(self):
        send = self.robot.send_angles
        def undershoot(angles, speed, **kwargs):
            send(angles, speed, **kwargs)
            self.robot.angles[0] -= 1.6
        self.robot.send_angles = undershoot
        with patch.object(helpers.hardware, 'SEGMENT_TIMEOUT', .05):
            data = self.plan(); self.execute(data)
            self.controller.ik.heartbeat(data['plan']['id'])
            state = self.wait_state()
        self.assertEqual(state['state'], 'failed')
        self.assertIn('settle', state['message'])
        self.assertEqual(state['completed_waypoints'], 1)
        self.assertEqual([c[0] for c in self.robot.calls], ['move', 'move', 'stop'])

    def test_slow_joint_crosses_servo_deadband_over_several_waypoints(self):
        self.end = self.start[:]
        self.end[0] += 4.5
        self.end[2] += 2.
        t = ARM.transform(np.deg2rad(self.end))
        self.body = {'target_mm': (t[:3, 3]*1000).tolist(), 'target_axis': t[:3, 2].tolist(), 'joint_command': 'send_angles'}
        self.result.update(self.body)
        self.result['frames'][-1]['q'] = self.end
        self.result['smoothing']['segments'][0]['end_deg'] = self.end
        send = self.robot.send_angles
        measured = []
        def deadband(angles, speed, **kwargs):
            live = self.robot.angles[:]
            send(angles, speed, **kwargs)
            # Ignore each small joint change until its absolute target crosses
            # a 1.4-degree deadband. Joint 3 stays still for two commands,
            # then moves on the final command after crossing the deadband.
            self.robot.angles = [target if abs(target-actual) > 1.4 else actual
                                 for target, actual in zip(angles, live)]
            measured.append(self.robot.angles[:])
        self.robot.send_angles = deadband
        data = self.plan(); self.execute(data)
        self.controller.ik.heartbeat(data['plan']['id'])
        state = self.wait_state()
        self.assertEqual(state['state'], 'completed', state.get('message'))
        self.assertEqual(state['completed_waypoints'], 3)
        self.assertEqual(len(self.robot.calls), 3)
        self.assertEqual([q[2] for q in measured[:2]], [self.start[2]]*2)
        self.assertAlmostEqual(measured[2][2], self.start[2]+2.)
        self.assertLessEqual(state['estimated_error_mm'], 5.)
        self.assertLessEqual(state['estimated_orientation_error_deg'], 3.)

    def test_fixed_and_continuous_v2_share_deadband_allowance(self):
        from webui.v2_mouse import ARRIVAL_DEG
        self.assertEqual(adapter.ARRIVAL_DEG, ARRIVAL_DEG)

    def test_moving_interior_advances_but_moving_endpoint_still_waits_and_stops(self):
        data = self.plan()
        self.robot.is_moving = lambda: int(bool(self.robot.calls) and self.robot.calls[-1][0] == 'move')
        with patch.object(helpers.hardware, 'SEGMENT_TIMEOUT', .05):
            self.execute(data); self.controller.ik.heartbeat(data['plan']['id'])
            state = self.wait_state()
        self.assertEqual(state['state'], 'failed')
        self.assertIn('settle', state['message'])
        self.assertEqual(state['completed_waypoints'], 1)
        self.assertEqual([c[0] for c in self.robot.calls], ['move', 'move', 'stop'])

    def test_relaxed_interior_lag_while_moving_reaches_strict_endpoint(self):
        end = self.start[:]; end[0] += 5.
        self.set_path([end])
        send = self.robot.send_angles
        def lag(angles, speed, **kwargs):
            send(angles, speed, **kwargs)
            if angles != end:
                self.robot.angles[0] = max(self.start[0], angles[0]-2.2)
        self.robot.send_angles = lag
        self.robot.is_moving = lambda: int(bool(self.robot.calls) and self.robot.calls[-1][0] == 'move'
                                         and self.robot.calls[-1][1] != end)
        data = self.plan(); self.execute(data)
        self.controller.ik.heartbeat(data['plan']['id'])
        state = self.wait_state()
        self.assertEqual(state['state'], 'completed', state.get('message'))
        self.assertEqual(state['completed_waypoints'], 3)
        self.assertTrue(any(s.get('moving') and s.get('max_joint_error_deg', 0) > 1.5
                            for s in state['samples']))
        self.assertLessEqual(state['estimated_error_mm'], 5.)

    def test_command_timing_wait_does_not_misclassify_owned_interior_motion(self):
        from webui.command_profile import CommandProfile
        original = CommandProfile.candidate
        delayed = []
        def wait_once(profile, target, timestamp):
            sample = original(profile, target, timestamp)
            if profile.interval and not delayed:
                delayed.append(True)
                return {**sample, 'accepted': False}
            return sample
        self.robot.is_moving = lambda: int(bool(self.robot.calls) and self.robot.calls[-1][0] == 'move'
                                         and self.robot.calls[-1][1] != self.end)
        data = self.plan()
        with patch.object(CommandProfile, 'candidate', wait_once):
            self.execute(data); self.controller.ik.heartbeat(data['plan']['id'])
            state = self.wait_state()
        self.assertTrue(delayed)
        self.assertEqual(state['state'], 'completed', state.get('message'))
        self.assertEqual([c[0] for c in self.robot.calls], ['move', 'move'])

    def test_stalled_arm_cannot_accumulate_unbounded_interior_commands(self):
        end = self.start[:]; end[0] += 6.
        self.set_path([end]); self.robot.track = False
        with patch.object(helpers.hardware, 'SEGMENT_TIMEOUT', .05):
            data = self.plan(); self.execute(data)
            self.controller.ik.heartbeat(data['plan']['id'])
            state = self.wait_state()
        self.assertEqual(state['state'], 'failed')
        targets = [call[1] for call in self.robot.calls if call[0] == 'move']
        self.assertEqual(len(targets), 2)
        self.assertLessEqual(np.abs(np.array(targets)-self.start).max(), 4.)
        self.assertEqual(self.robot.calls[-1][0], 'stop')

    def test_retained_corner_stops_before_turning_with_relaxed_interiors(self):
        import threading
        corner = self.start[:]; corner[0] += 3.
        end = corner[:]; end[1] += 2.
        self.set_path([corner, end])
        corner_seen, release = threading.Event(), threading.Event()
        send = self.robot.send_angles
        def observe(angles, speed, **kwargs):
            send(angles, speed, **kwargs)
            if angles == corner: corner_seen.set()
        self.robot.send_angles = observe
        self.robot.is_moving = lambda: int(bool(self.robot.calls) and not release.is_set())
        data = self.plan()
        self.assertEqual(self.controller.ik.plan['stop_indices'], {1, 2})
        self.execute(data); self.controller.ik.heartbeat(data['plan']['id'])
        self.assertTrue(corner_seen.wait(2))
        time.sleep(.3)
        self.assertEqual(len(self.robot.calls), 2)
        self.assertEqual(self.controller.snapshot()['ik']['completed_waypoints'], 1)
        release.set()
        state = self.wait_state()
        self.assertEqual(state['state'], 'completed', state.get('message'))
        self.assertEqual(len(self.robot.calls), 3)

    def test_refresh_mode_is_verified_and_restored_after_path(self):
        self.robot.fresh_mode = 0
        data = self.plan(); self.execute(data)
        self.assertEqual(self.robot.fresh_mode, 1)
        self.controller.ik.heartbeat(data['plan']['id'])
        self.assertEqual(self.wait_state()['state'], 'completed')
        self.assertEqual(self.robot.fresh_mode, 0)
        self.assertEqual([c[0] for c in self.robot.calls], ['mode', 'move', 'move', 'mode'])

    def test_refresh_mode_loss_stops_before_another_target(self):
        send = self.robot.send_angles
        def changed(angles, speed, **kwargs):
            send(angles, speed, **kwargs)
            self.robot.fresh_mode = 0
        self.robot.send_angles = changed
        data = self.plan(); self.execute(data)
        self.controller.ik.heartbeat(data['plan']['id'])
        state = self.wait_state()
        self.assertEqual(state['state'], 'failed')
        self.assertIn('refresh mode', state['message'])
        self.assertEqual(len([c for c in self.robot.calls if c[0] == 'move']), 1)
        self.assertEqual(self.robot.fresh_mode, 1)

    def test_joint_settling_does_not_bypass_final_cartesian_error(self):
        send = self.robot.send_angles
        def residual(angles, speed, **kwargs):
            send(angles, speed, **kwargs)
            # Each joint settles within 1 degree, but their combined FK error
            # exceeds 5 mm at the final target.
            for j in (1, 2, 3):
                self.robot.angles[j] -= .9
        self.robot.send_angles = residual
        data = self.plan(); self.execute(data)
        self.controller.ik.heartbeat(data['plan']['id'])
        state = self.wait_state()
        self.assertEqual(state['completed_waypoints'], 2)
        self.assertEqual(state['state'], 'failed')
        self.assertIn('Final URDF FK error', state['message'])
        self.assertEqual([c[0] for c in self.robot.calls], ['move', 'move', 'stop'])

    def test_lost_heartbeat_stops_without_motor_target(self):
        data = self.plan(); self.execute(data)
        with self.controller.lock:
            self.controller.ik.heartbeat_at -= 6
        state = self.wait_state()
        self.assertEqual(state['state'], 'failed')
        self.assertIn('heartbeat', state['message'])
        self.assertEqual([c[0] for c in self.robot.calls], ['stop'])


class HardwareDiffusionHTTPTest(unittest.TestCase):
    setUp = helpers.HardwareIKHTTPTest.setUp
    tearDown = helpers.HardwareIKHTTPTest.tearDown
    post = helpers.HardwareIKHTTPTest.post

    def test_new_endpoint_is_hardware_gated_and_uses_live_start(self):
        import urllib.error
        body = {'target_mm': [160., -64., 83.], 'target_axis': [0., 0., 1.], 'joint_command':'send_angles'}
        self.http.RequestHandlerClass.hardware_enabled = False
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post('/api/ik/diffusion/plan', body)
        self.assertEqual(caught.exception.code, 403)
        self.http.RequestHandlerClass.hardware_enabled = True
        with patch('webui.hardware_diffusion.prepare', side_effect=ValueError('checked capture')) as prepare:
            with self.assertRaises(urllib.error.HTTPError) as caught:
                self.post('/api/ik/diffusion/plan', body)
            self.assertEqual(caught.exception.code, 400)
            self.assertEqual(prepare.call_args.args, (self.robot.angles, body))
        self.assertEqual(self.robot.calls, [])
