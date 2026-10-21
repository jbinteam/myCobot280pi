"""Larger encoder routes on fake hardware; no physical motor transport."""
import time
import numpy as np
import test_encoder_drag as helpers
from simulation.kinematics import ARM


class EncoderStagedTest(helpers.EncoderDragTest):
    def setUp(self):
        super().setUp()
        self.body['joint_command'] = 'encoder_drag_staged'

    def run_plan(self, data):
        self.controller.ik.execute(data['plan']['id'], 5, True, True, True)
        deadline = time.monotonic()+35
        while time.monotonic() < deadline:
            with self.controller.lock:
                state = self.controller.snapshot()['ik']
                if state['state'] != 'running':
                    return state
                self.controller.ik.heartbeat(data['plan']['id'])
            time.sleep(.025)
        self.fail('Fake staged encoder route timed out')

    def larger_goal(self, displacement=6.):
        self.end[0] = self.start[0]+displacement
        pose = ARM.transform(np.deg2rad(self.end))
        self.body.update(target_mm=(pose[:3,3]*1000).tolist(), target_axis=pose[:3,2].tolist())
        self.result.update(target_mm=self.body['target_mm'],target_axis=self.body['target_axis'])

    def test_preflight_and_plan_read_only_nominal_speed_vectors_and_quantization(self):
        self.larger_goal()
        data = self.plan()
        trial = data['result']['encoder_drag']
        self.assertEqual(self.robot.calls, [])
        self.assertTrue(trial['staged'])
        self.assertIsNone(trial['maximum_excursion_deg'])
        self.assertEqual(trial['maximum_stage_step_deg'], 2.)
        self.assertGreaterEqual(trial['stage_count'], 3)
        self.assertEqual(data['plan']['joint_command'], 'encoder_drag_staged')
        segments = data['result']['smoothing']['segments']
        self.assertTrue(all(np.max(np.abs(np.asarray(s['end_deg'])-s['start_deg'])) <= 2.+1e-9 for s in segments))
        self.assertGreater(np.max(np.abs(np.asarray(self.controller.ik.plan['waypoints'])-self.start)), 3.)
        self.assertEqual(data['plan']['max_sdk_speed'],5)
        self.assertEqual(data['plan']['max_live_command_deg'],1.)

    def test_completes_larger_route_and_checks_every_stage_mapping(self):
        self.larger_goal()
        reads = 0
        original = self.robot.get_encoders
        def count():
            nonlocal reads
            reads += 1
            return original()
        self.robot.get_encoders = count
        data = self.plan()
        from unittest.mock import patch
        with patch('webui.hardware_ik.EXECUTION_TIMEOUT', .01):
            state = self.run_plan(data)
        self.assertEqual(state['state'], 'completed', state.get('message'))
        self.assertEqual(state['completed_stages'], data['plan']['encoder_drag']['stage_count'])
        self.assertEqual(state['completed_waypoints'],data['plan']['waypoints'])
        self.assertEqual(reads, 2+state['completed_stages'])
        self.assertFalse(any(c[0]=='move' for c in self.robot.calls))
        samples = [s for s in state['samples'] if 'stage_error_mm' in s]
        self.assertEqual(len(samples),state['completed_stages'])
        self.assertTrue(all(s['stage_error_mm']<=5. for s in samples))

    def test_mapping_fault_at_first_stage_stops_before_next_stage(self):
        self.larger_goal()
        reads = 0
        original = self.robot.get_encoders
        def fault():
            nonlocal reads
            reads += 1
            encoders = original()
            return [e+4 for e in encoders] if reads >= 3 else encoders
        self.robot.get_encoders = fault
        data = self.plan()
        first_stop = data['result']['encoder_drag']['stage_end_indices'][0]
        state = self.run_plan(data)
        self.assertEqual(state['state'],'failed')
        self.assertIn('mapping changed',state['message'])
        self.assertEqual(sum(c[0]=='encoder_drag' for c in self.robot.calls),first_stop+1)
        self.assertEqual(state['completed_stages'],0)

    def test_unspecified_driver_defaults_to_velocity_profile(self):
        self.larger_goal()
        self.body.pop('joint_command')
        data = self.plan()
        self.assertEqual(data['plan']['joint_command'], 'encoder_drag_profile')
        self.assertTrue(data['plan']['encoder_drag']['velocity_profile'])
        self.assertEqual(data['plan']['max_sdk_speed'], 5)
        self.assertEqual(self.robot.calls, [])

    def test_default_does_not_fall_back_if_encoder_telemetry_is_invalid(self):
        self.body.pop('joint_command')
        self.robot.get_encoders = lambda: -1
        with self.assertRaises(ValueError):
            self.plan()
        self.assertIsNone(self.controller.ik.plan)
        self.mock_plan.assert_not_called()
        self.assertEqual(self.robot.calls, [])

    def test_local_measured_excursion_stops_without_a_global_cap(self):
        from pymycobot.common import ProtocolCode
        self.larger_goal()
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
        self.assertIn('stage measured excursion', state['message'])
        self.assertEqual(sum(c[0]=='encoder_drag' for c in self.robot.calls), 1)
        self.assertIn('stop', [c[0] for c in self.robot.calls])

    def test_full_joint_range_passes_old_distance_packet_and_duration_caps(self):
        self.robot.angles[0] = self.start[0] = -160.
        self.larger_goal(320.)
        data = self.plan()
        trial = data['result']['encoder_drag']
        self.assertIsNone(trial['maximum_excursion_deg'])
        self.assertIsNone(trial['maximum_packets'])
        self.assertIsNone(trial['maximum_reference_duration_s'])
        self.assertGreater(trial['stage_count'], 100)
        self.assertGreater(len(trial['packets']), 200)
        self.assertGreater(data['result']['smoothing']['duration_s'], 60.)
        self.assertGreater(data['plan']['execution_timeout_s'], 180.)
        quantized = np.asarray(trial['quantized_waypoints_deg'])
        self.assertGreater(np.max(np.abs(quantized-self.start)), 300.)
        self.assertTrue(np.all(np.deg2rad(quantized)>=ARM.lower))
        self.assertTrue(np.all(np.deg2rad(quantized)<=ARM.upper))
        self.assertEqual(self.robot.calls, [])
        self.body['joint_command'] = 'encoder_drag_stream'
        self.larger_goal(6.)
        with self.assertRaisesRegex(ValueError,'3 degrees'):
            self.plan()

    def test_longer_review_window_still_expires_and_never_sends_without_heartbeat(self):
        from webui.hardware_ik import PLAN_TTL
        self.larger_goal()
        data = self.plan()
        self.assertGreater(data['plan']['expires_in'], PLAN_TTL)
        self.controller.ik.plan['created'] -= PLAN_TTL+1
        self.controller.ik.execute(data['plan']['id'], 5, True, True, True)
        self.assertEqual(self.robot.calls, [])
        self.controller.stop()
        data = self.plan()
        self.controller.ik.plan['created'] -= self.controller.ik.plan['review_ttl_s']+1
        with self.assertRaisesRegex(ValueError,'expired'):
            self.controller.ik.execute(data['plan']['id'], 5, True, True, True)
        self.assertFalse(any(c[0]=='encoder_drag' for c in self.robot.calls))

    def test_checked_detour_can_exceed_old_200_source_waypoint_budget(self):
        self.larger_goal(79.)
        middle = self.start[:]; middle[0] = -80.
        corners = [self.start, middle, self.end, middle, self.end]
        self.result['frames'] = [{'q':q} for q in corners]
        self.result['smoothing']['segments'] = [
            {'start_deg':a,'end_deg':b} for a,b in zip(corners,corners[1:])]
        data = self.plan()
        self.assertGreater(data['plan']['encoder_drag']['stage_count'], 200)
        self.assertEqual(self.robot.calls, [])

    def test_unlimited_route_still_rejects_quantized_joint_limit_overflow(self):
        from webui import encoder_drag
        mapping = encoder_drag.capture(self.robot,self.start)
        outside = self.start[:]
        outside[0] = float(np.rad2deg(ARM.upper[0])+1.)
        prepared = {'result':{'converged':True}, 'waypoints':[outside],
                    'waypoint_intervals_s':[1000.]}
        with self.assertRaisesRegex(ValueError,'joint limit'):
            encoder_drag.prepare(prepared,mapping,5,maximum_waypoints=None,maximum_excursion_deg=None)
        self.assertEqual(self.robot.calls, [])
