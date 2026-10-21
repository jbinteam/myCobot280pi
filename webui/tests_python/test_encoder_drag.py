"""Encoder trials on fake hardware only; no physical transport."""
import sys
import time
from pathlib import Path
from unittest.mock import patch
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import RobotController
from test_hardware_ik import TrackingRobot
from webui import hardware_diffusion as adapter, encoder_drag
from simulation.kinematics import ARM
from pymycobot.common import ProtocolCode


class EncoderRobot(TrackingRobot):
    directions = np.array([-1, -1, 1, -1, -1, -1])

    def get_encoders(self):
        return np.rint(2048+np.asarray(self.angles)*encoder_drag.STEPS_PER_DEG*self.directions).astype(int).tolist()

    def _mesg(self, genre, *args, **kwargs):
        if genre == ProtocolCode.SET_ENCODERS_DRAG:
            self.calls.append(('encoder_drag', args[0][:], args[1][:], kwargs))
            if self.track:
                self.angles = ((np.asarray(args[0])-2048)/(encoder_drag.STEPS_PER_DEG*self.directions)).tolist()
        else:
            return super()._mesg(genre, *args, **kwargs)


class EncoderDragTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.deployment = adapter.verified_status()

    def setUp(self):
        self.robot = EncoderRobot()
        self.robot.angles[0] = 1.
        self.controller = RobotController(lambda *_: self.robot)
        self.controller.connect('192.168.77.225', 9000)
        self.start = self.robot.angles[:]
        self.end = self.start[:]
        self.end[0] += 1.8
        self.end[3] -= .9
        t = ARM.transform(np.deg2rad(self.end))
        self.body = {'target_mm': (t[:3, 3]*1000).tolist(), 'target_axis': t[:3, 2].tolist(),
                     'joint_command': 'encoder_drag', 'speed': 5}
        self.result = {'method': 'diffusion', 'task': 'pose5', 'converged': True,
            'validation': 'joint_limits_self_table_block', 'obstacle_mm': None,
            'checkpoint_sha256': adapter.V2_SHA256, 'environment': self.deployment['environment'],
            'motion_conditioning': {'schema': 2}, 'target_mm': self.body['target_mm'],
            'target_axis': self.body['target_axis'], 'final_error_mm': 0., 'final_orientation_error_deg': 0.,
            'frames': [{'q': self.start}, {'q': self.end}],
            'smoothing': {'segments': [{'start_deg': self.start, 'end_deg': self.end}]}}
        patcher = patch.object(adapter.service, 'plan', return_value=self.result)
        self.mock_plan = patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.controller.shutdown()

    def plan(self):
        return self.controller.ik.prepare_diffusion(self.body)

    def run_plan(self, data):
        self.controller.ik.execute(data['plan']['id'], 5, True, True, True)
        self.controller.ik.heartbeat(data['plan']['id'])
        deadline = time.monotonic()+5
        while time.monotonic() < deadline:
            state = self.controller.snapshot()['ik']
            if state['state'] != 'running':
                return state
            time.sleep(.025)
        self.fail('Fake encoder trial timed out')

    def test_preflight_and_plan_read_only_nominal_speed_vectors_and_quantization(self):
        preflight = self.controller.ik.encoder_preflight({})
        self.assertFalse(preflight['motion_sent'])
        self.assertEqual(preflight['encoder_drag']['encoder_directions'], [-1, -1, 1, -1, 0, -1])
        data = self.plan()
        self.assertEqual(self.robot.calls, [])
        self.assertEqual(data['plan']['joint_command'], 'encoder_drag')
        self.assertEqual(data['plan']['max_sdk_speed'], 5)
        self.assertNotIn('joint_command', self.mock_plan.call_args.args[0])
        packet = data['result']['encoder_drag']['packets'][0]
        self.assertEqual(len(packet['encoders']), 6)
        self.assertEqual(len(packet['speeds']), 6)
        self.assertGreater(packet['speeds'][0], packet['speeds'][3])
        self.assertEqual(packet['speeds'][1], 1)
        self.assertGreaterEqual(min(packet['speeds']), 1)
        quantized = np.array(data['result']['encoder_drag']['quantized_waypoints_deg'])
        np.testing.assert_allclose(quantized, self.controller.ik.plan['waypoints'], atol=.045)

    def test_executes_six_joint_packet_without_angle_commands_and_reports_raw_words(self):
        data = self.plan()
        state = self.run_plan(data)
        self.assertEqual(state['state'], 'completed', state.get('message'))
        sends = [x for x in self.robot.calls if x[0] == 'encoder_drag']
        self.assertEqual(len(sends), data['plan']['waypoints'])
        self.assertTrue(all(x[3] == {'_async': True} for x in sends))
        self.assertFalse(any(x[0] == 'move' for x in self.robot.calls))
        self.assertTrue(any('sent_encoder_speeds' in x for x in state['samples']))
        self.assertLess(self.robot.angles[3], self.start[3])
        self.assertGreater(self.robot.angles[0], self.start[0])

    def test_incompatible_or_malformed_encoder_telemetry_never_creates_motion(self):
        for bad in (-1, [2048]*5, [True]*6, [0]*6, [2048]*6):
            with self.subTest(bad=bad):
                self.robot.get_encoders = lambda: bad
                with self.assertRaises((ValueError, RuntimeError)):
                    self.plan()
                self.assertIsNone(self.controller.ik.plan)
                self.assertEqual(self.robot.calls, [])

    def test_mapping_change_after_review_refuses_before_motion(self):
        data = self.plan()
        original = self.robot.get_encoders
        self.robot.get_encoders = lambda: [e+4 for e in original()]
        with self.assertRaisesRegex(RuntimeError, 'mapping changed'):
            self.controller.ik.execute(data['plan']['id'], 5, True, True, True)
        self.assertEqual(self.robot.calls, [])

    def test_near_zero_direction_is_not_guessed_for_a_moving_joint(self):
        self.robot.angles[0] = 0.
        self.start[0] = 0.
        with self.assertRaisesRegex(ValueError, 'ambiguous encoder direction'):
            self.plan()
        self.assertEqual(self.robot.calls, [])

    def test_browser_confirmation_required_and_stop_cancels_before_any_packet(self):
        data = self.plan()
        self.controller.ik.execute(data['plan']['id'], 5, True, True, True)
        time.sleep(.3)
        self.assertEqual(self.robot.calls, [])
        self.controller.stop()
        time.sleep(.3)
        self.assertEqual([x[0] for x in self.robot.calls], ['stop'])

    def test_failed_confirmation_does_not_send_encoder_packet(self):
        data = self.plan()
        for armed, frame, scene in [(False, True, True), (True, False, True), (True, True, False)]:
            with self.assertRaises(ValueError):
                self.controller.ik.execute(data['plan']['id'], 5, armed, frame, scene)
        self.assertEqual(self.robot.calls, [])

    def test_speed_and_excursion_caps_rejected(self):
        self.body['speed'] = 6
        with self.assertRaisesRegex(ValueError, '5%'):
            self.plan()
        self.body['speed'] = 5
        data = self.plan()
        outside = self.start[:]; outside[0] += 4
        prepared = {'result': data['result'], 'waypoints': [outside],
                    'waypoint_intervals_s': [2.]}
        with self.assertRaisesRegex(ValueError, '3 degrees'):
            encoder_drag.prepare(prepared, data['plan']['encoder_drag'], 5)
        self.assertEqual(self.robot.calls, [])

    def test_quantized_path_collision_rejected_before_review(self):
        with patch('simulation.diffusion.env.ArmEnv.path_valid', side_effect=[True, True, False]):
            with self.assertRaisesRegex(ValueError, 'Quantized encoder path'):
                self.plan()
        self.assertEqual(self.robot.calls, [])

    def test_send_failure_stops_without_falling_back_or_retrying(self):
        original = self.robot._mesg
        def fail(genre, *args, **kwargs):
            if genre == ProtocolCode.SET_ENCODERS_DRAG:
                self.robot.calls.append(('encoder_attempt',))
                raise OSError('encoder delivery failure')
            return original(genre, *args, **kwargs)
        self.robot._mesg = fail
        state = self.run_plan(self.plan())
        self.assertEqual(state['state'], 'failed')
        self.assertFalse(self.controller.snapshot()['connected'])
        self.assertEqual([x[0] for x in self.robot.calls], ['encoder_attempt', 'stop', 'close'])

    def test_slow_read_cycle_aborts_before_first_encoder_command(self):
        data = self.plan()
        original = self.controller._health
        calls = 0
        def slow():
            nonlocal calls
            calls += 1
            original()
            if calls > 1:
                time.sleep(2.01)
        self.controller._health = slow
        state = self.run_plan(data)
        self.assertEqual(state['state'], 'failed')
        self.assertIn('two seconds', state['message'])
        self.assertFalse(any(x[0] == 'encoder_drag' for x in self.robot.calls))

class EncoderStreamTest(EncoderDragTest):
    """Run the same transport/mapping/confirmation guards on streaming too."""
    def setUp(self):
        super().setUp()
        self.body['joint_command'] = 'encoder_drag_stream'

    def run_plan(self, data):
        self.controller.ik.execute(data['plan']['id'], 5, True, True, True)
        deadline = time.monotonic()+12
        while time.monotonic() < deadline:
            with self.controller.lock:
                state = self.controller.snapshot()['ik']
                if state['state'] != 'running':
                    return state
                self.controller.ik.heartbeat(data['plan']['id'])
            time.sleep(.025)
        self.fail('Fake encoder stream timed out')

    def test_preflight_and_plan_read_only_nominal_speed_vectors_and_quantization(self):
        # Override the single-target speed-ratio assertion: quantization can
        # suppress a joint in the first tiny sample of a continuous approach.
        data = self.plan()
        self.assertEqual(self.robot.calls, [])
        self.assertEqual(data['plan']['execution_mode'], 'encoder_stream')
        self.assertEqual(data['plan']['joint_command'], 'encoder_drag_stream')
        self.assertEqual(data['plan']['max_sdk_speed'], 5)
        self.assertGreater(data['plan']['waypoints'], 3)
        trial = data['result']['encoder_drag']
        self.assertTrue(trial['streaming'])
        packets = trial['packets']
        self.assertTrue(all(p['reference_interval_s'] >= .2-1e-9 for p in packets))
        self.assertEqual(packets[-1]['reference_velocity_deg_s'], [0.]*6)
        self.assertAlmostEqual(packets[-1]['reference_phase_s'], data['result']['smoothing']['duration_s'])
        self.assertTrue(all(len(p['encoders']) == len(p['speeds']) == 6 for p in packets))
        self.assertTrue(all(1 <= s <= 85 for p in packets for s in p['speeds']))
        np.testing.assert_allclose(trial['quantized_waypoints_deg'], self.controller.ik.plan['waypoints'])
        for segment in data['result']['smoothing']['segments']:
            self.assertGreaterEqual(segment['duration_s'], 2.-1e-9)

    def test_replaces_interior_targets_while_moving_without_skipping_and_settles_endpoint(self):
        self.robot.fresh_mode = 0
        data = self.plan()
        packets = data['result']['encoder_drag']['packets']
        def moving():
            sends = [x for x in self.robot.calls if x[0] == 'encoder_drag']
            return int(bool(sends) and sends[-1][1] != packets[-1]['encoders'])
        self.robot.is_moving = moving
        state = self.run_plan(data)
        self.assertEqual(state['state'], 'completed', state.get('message'))
        sent = [x[1] for x in self.robot.calls if x[0] == 'encoder_drag']
        self.assertEqual(sent, [p['encoders'] for p in packets])
        samples = [s for s in state['samples'] if 'sent_encoder_targets' in s]
        self.assertGreater(sum(bool(s.get('moving')) for s in samples), 1)
        intervals = np.diff([s['command_send_time_s'] for s in samples])
        self.assertTrue(np.all(intervals >= .2-1e-6))
        self.assertTrue(all(s['command_profile_checked'] for s in samples))
        self.assertGreaterEqual(state['samples'][-1]['time']-samples[-1]['command_send_time_s'], 1.)
        self.assertEqual(state['completed_waypoints'], len(packets))
        self.assertEqual(self.robot.get_fresh_mode(), 0)

    def test_stalled_feedback_stops_before_runaway_reference(self):
        import webui.hardware_ik as hardware
        self.robot.track = False
        data = self.plan()
        with patch.object(hardware, 'SEGMENT_TIMEOUT', .25):
            state = self.run_plan(data)
        self.assertEqual(state['state'], 'failed', state.get('message'))
        sends = [c for c in self.robot.calls if c[0] == 'encoder_drag']
        self.assertLess(len(sends), data['plan']['waypoints'])
        commands = [s['sent_command_deg'] for s in state['samples'] if 'sent_command_deg' in s]
        self.assertLessEqual(np.max(np.abs(np.asarray(commands)-self.start)), 1.)
        self.assertIn('stop', [c[0] for c in self.robot.calls])

    def test_late_cycle_never_raises_speed_words_to_catch_up(self):
        from webui.encoder_stream import paced_speeds
        packet = {'encoders':[101,110,95,100,100,100], 'speeds':[5,50,25,1,1,1]}
        self.assertEqual(paced_speeds(packet, np.array([100]*6), 1.), [1,10,5,1,1,1])
        self.assertEqual(paced_speeds(packet, np.array([100]*6), .01), packet['speeds'])

    def test_lost_lease_during_stream_stops_without_later_packets(self):
        self.robot.fresh_mode = 0
        data = self.plan()
        self.controller.ik.execute(data['plan']['id'], 5, True, True, True)
        self.controller.ik.heartbeat(data['plan']['id'])
        deadline = time.monotonic()+3
        while not any(c[0]=='encoder_drag' for c in self.robot.calls) and time.monotonic()<deadline:
            time.sleep(.02)
        before = sum(c[0]=='encoder_drag' for c in self.robot.calls)
        self.assertGreater(before, 0)
        with self.controller.lock:
            self.controller.ik.heartbeat_at -= 6
        time.sleep(.5)
        state = self.controller.snapshot()['ik']
        self.assertEqual(state['state'], 'failed')
        self.assertIn('heartbeat', state['message'])
        self.assertEqual(sum(c[0]=='encoder_drag' for c in self.robot.calls), before)
        self.assertEqual(self.robot.get_fresh_mode(), 0)

    def test_measured_excursion_stops_before_another_target(self):
        data = self.plan()
        original = self.robot._mesg
        def displaced(genre, *args, **kwargs):
            result = original(genre, *args, **kwargs)
            if genre == ProtocolCode.SET_ENCODERS_DRAG:
                self.robot.angles[0] = self.start[0]+3.4
            return result
        self.robot._mesg = displaced
        state = self.run_plan(data)
        self.assertEqual(state['state'], 'failed')
        self.assertIn('measured excursion', state['message'])
        self.assertEqual(sum(c[0]=='encoder_drag' for c in self.robot.calls), 1)

    def test_retains_and_observes_a_corner_before_next_leg(self):
        middle = ((np.asarray(self.start)+self.end)/2).tolist()
        self.result['frames'] = [{'q':self.start}, {'q':middle}, {'q':self.end}]
        self.result['smoothing']['segments'] = [
            {'start_deg':self.start, 'end_deg':middle},
            {'start_deg':middle, 'end_deg':self.end}]
        data = self.plan()
        stop_indices = sorted(self.controller.ik.plan['stop_indices'])
        self.assertEqual(len(stop_indices), 2)
        state = self.run_plan(data)
        self.assertEqual(state['state'], 'completed', state.get('message'))
        samples = [s for s in state['samples'] if 'sent_encoder_targets' in s]
        self.assertEqual(len(samples), data['plan']['waypoints'])
        corner = stop_indices[0]
        self.assertGreaterEqual(samples[corner+1]['command_send_time_s']-
                                samples[corner]['command_send_time_s'], 1.)
        packets = data['result']['encoder_drag']['packets']
        self.assertEqual(packets[corner]['reference_velocity_deg_s'], [0.]*6)

    def test_health_fault_during_stream_stops_before_another_packet(self):
        data = self.plan()
        original = self.controller._health
        def fault():
            if any(c[0]=='encoder_drag' for c in self.robot.calls):
                raise RuntimeError('Robot servos are not enabled')
            original()
        self.controller._health = fault
        state = self.run_plan(data)
        self.assertEqual(state['state'], 'failed')
        self.assertIn('servos', state['message'])
        self.assertEqual(sum(c[0]=='encoder_drag' for c in self.robot.calls), 1)

    def test_false_motion_flag_does_not_hide_endpoint_drift(self):
        data = self.plan()
        final = data['result']['encoder_drag']['packets'][-1]['encoders']
        target = data['result']['encoder_drag']['quantized_waypoints_deg'][-1]
        original = self.robot.get_angles
        reads = 0
        def drift():
            nonlocal reads
            sends = [c for c in self.robot.calls if c[0]=='encoder_drag']
            if len(sends) == data['plan']['waypoints'] and reads < 6:
                self.robot.angles[0] = target[0]-.5+.1*reads
                reads += 1
            return original()
        self.robot.get_angles = drift
        state = self.run_plan(data)
        self.assertEqual(state['state'], 'completed', state.get('message'))
        samples = [s for s in state['samples'] if 'sent_encoder_targets' in s]
        self.assertEqual(reads, 6)
        self.assertGreater(state['samples'][-1]['time']-samples[-1]['command_send_time_s'], 1.8)


if __name__ == '__main__':
    unittest.main()
