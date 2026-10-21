"""Timed derivative driving, tested with fake hardware only."""
import threading
import time
import unittest
from unittest.mock import patch
import numpy as np
import test_encoder_drag as fixtures
from webui import encoder_profile as driver, encoder_drag
from webui.fixed_profile import retime_fixed
from simulation.kinematics import ARM


class ProfileMathTest(unittest.TestCase):
    def setUp(self):
        self.start = [1., -40., -130., 80., 1., 50.]
        self.end = [7., -40., -130., 77., 1., 50.]
        self.curve = retime_fixed({'converged': True, 'target_mm': [100,0,100],
            'frames': [{'q':self.start}, {'q':self.end}]}, 5)['smoothing']
        self.mapping = {'start_deg':self.start, 'start_encoders':[2000]*6,
                        'encoder_directions':[-1,-1,1,-1,-1,-1]}

    def test_derivative_matches_numeric_differentiation_and_signed_ratios(self):
        duration = self.curve['duration_s']
        for fraction in (.1, .25, .5, .8):
            t = fraction*duration; h = 1e-5
            q, v, a, _ = driver.motion(self.curve,t)
            before = driver.motion(self.curve,t-h); after = driver.motion(self.curve,t+h)
            np.testing.assert_allclose((after[0]-before[0])/(2*h),v,atol=1e-7)
            np.testing.assert_allclose((after[1]-before[1])/(2*h),a,atol=1e-7)
            self.assertGreater(v[0],0); self.assertAlmostEqual(v[3],-.5*v[0])
        for t in (0.,duration):
            np.testing.assert_allclose(driver.motion(self.curve,t)[1],np.zeros(6),atol=1e-8)
            np.testing.assert_allclose(driver.motion(self.curve,t)[2],np.zeros(6),atol=1e-8)

    def test_speed_words_use_instantaneous_derivative_not_average_position_steps(self):
        p = driver.packet(self.curve,self.curve['duration_s']/2,self.mapping,5)
        expected = np.maximum(1,np.floor(np.abs(p['reference_velocity_deg_s'])*encoder_drag.STEPS_PER_DEG+1e-9)).astype(int)
        self.assertEqual(p['speeds'],expected.tolist())
        # Opposite joint directions are encoded by targets, not negative words.
        self.assertLess(p['encoders'][0],2000); self.assertGreater(p['encoders'][3],2000)
        self.assertTrue(all(x>=1 for x in p['speeds']))
        self.assertEqual(p['encoders'][1],2000); self.assertEqual(p['speeds'][1],1)

    def test_live_lead_shortens_target_while_preserving_velocity_and_clock(self):
        end = self.start.copy(); end[0] += 40.
        curve = retime_fixed({'converged': True, 'target_mm': [100,0,100],
            'frames': [{'q': self.start}, {'q': end}]}, 50)['smoothing']
        phase = curve['duration_s']/2
        reference = driver.motion(curve, phase)[0]
        live = reference.copy(); live[0] -= 4.8
        original = driver.packet(curve, phase, self.mapping, 50)
        self.assertGreater(np.max(np.abs(np.asarray(original['target_deg'])-live)), driver.LEAD_DEG)
        bounded = driver.packet(curve, phase, self.mapping, 50, live_deg=live)
        self.assertLessEqual(bounded['command_lead_deg'], driver.LEAD_DEG)
        self.assertGreater(bounded['target_phase_s'], phase)
        self.assertLess(bounded['lookahead_s_actual'], driver.LOOKAHEAD_S)
        for key in ('reference_phase_s', 'reference_deg', 'reference_velocity_deg_s',
                    'reference_acceleration_deg_s2', 'speeds'):
            self.assertEqual(bounded[key], original[key])
        # Encoder rounding is also included in the bound.
        quantized = np.asarray(bounded['target_deg'])
        expected = driver.motion(curve, bounded['target_phase_s']-1e-10)[0]
        np.testing.assert_allclose(quantized, expected, atol=.5/encoder_drag.STEPS_PER_DEG)
        self.assertGreater(quantized[0], reference[0])
        np.testing.assert_allclose(quantized[1:], self.start[1:])

    def test_healthy_lookahead_is_unchanged_and_stalled_reference_is_not_hidden(self):
        phase = self.curve['duration_s']/2
        reference = driver.motion(self.curve, phase)[0]
        original = driver.packet(self.curve, phase, self.mapping, 5)
        healthy = driver.packet(self.curve, phase, self.mapping, 5, live_deg=reference)
        self.assertEqual(healthy['encoders'], original['encoders'])
        self.assertEqual(healthy['speeds'], original['speeds'])
        stalled = reference.copy(); stalled[0] -= 7.
        bad = driver.packet(self.curve, phase, self.mapping, 5, live_deg=stalled)
        self.assertGreater(bad['command_lead_deg'], driver.LEAD_DEG)
        self.assertEqual(bad['reference_deg'], original['reference_deg'])

    def test_lookahead_does_not_shortcut_a_reversing_corner(self):
        result=retime_fixed({'converged':True,'target_mm':[100,0,100],
            'frames':[{'q':self.start},{'q':self.end},{'q':self.start}]},5,
            path=np.deg2rad([self.start,self.end,self.start]))
        curve=result['smoothing']; boundary=curve['segments'][0]['duration_s']
        before=driver.packet(curve,boundary-.05,self.mapping,5)
        after=driver.packet(curve,boundary+.05,self.mapping,5)
        np.testing.assert_allclose(before['target_deg'],self.end,atol=.045)
        self.assertGreater(before['reference_velocity_deg_s'][0],0)
        self.assertLess(after['reference_velocity_deg_s'][0],0)
        self.assertEqual(before['segment_index'],0); self.assertEqual(after['segment_index'],1)


class EncoderProfileTest(unittest.TestCase):
    setUpClass = classmethod(fixtures.EncoderDragTest.setUpClass.__func__)
    tearDown = fixtures.EncoderDragTest.tearDown
    plan = fixtures.EncoderDragTest.plan
    run_plan = fixtures.EncoderDragTest.run_plan

    def setUp(self):
        fixtures.EncoderDragTest.setUp(self)
        self.body['joint_command']='encoder_drag_profile'

    def larger_goal(self):
        self.end[0]=7.
        t=ARM.transform(np.deg2rad(self.end))
        self.body.update(target_mm=(t[:3,3]*1000).tolist(), target_axis=t[:3,2].tolist())
        self.result.update(target_mm=self.body['target_mm'],target_axis=self.body['target_axis'])

    def test_custom_reference_shape_is_reviewed_and_executes_with_same_derivatives(self):
        self.body['velocity_shape'] = {'kind':'smooth_trapezoid','rise_percent':20,'fall_percent':40}
        data = self.plan()
        curve = data['result']['smoothing']
        self.assertEqual(curve['method'], 'specification_checked_velocity_shape')
        self.assertEqual(curve['reference_shape'], self.body['velocity_shape'])
        self.assertNotIn('velocity_shape', self.mock_plan.call_args.args[0])
        self.assertEqual(self.robot.calls, [])
        state = self.run_plan(data)
        self.assertEqual(state['state'], 'completed', state['message'])
        for sample in state['samples']:
            expected = driver.motion(curve, sample['reference_phase_s'])
            np.testing.assert_allclose(sample['reference_velocity_deg_s'], expected[1], atol=1e-12)
            np.testing.assert_allclose(sample['reference_deg'], expected[0], atol=1e-12)
        self.assertFalse(any(c[0]=='move' for c in self.robot.calls))

    def test_custom_reference_shape_refuses_legacy_encoder_driver(self):
        self.body.update(joint_command='encoder_drag', velocity_shape={
            'kind':'smooth_trapezoid','rise_percent':20,'fall_percent':40})
        with self.assertRaisesRegex(ValueError, 'velocity-profile mode'):
            self.plan()
        self.assertEqual(self.robot.calls, [])

    def test_planning_timings_cover_live_capture_model_validation_and_profile(self):
        data=self.plan(); timings=data['result']['planning_timing_ms']
        stages=['initial_lock_wait','live_joint_read','encoder_mapping_capture',
                'trajectory_prepare','encoder_profile_prepare','final_lock_wait','live_start_recheck']
        self.assertTrue(all(np.isfinite(timings[k]) and timings[k]>=0 for k in stages))
        self.assertGreaterEqual(timings['server_total'],sum(timings[k] for k in stages))
        v2=timings['v2_prepare']
        self.assertAlmostEqual(v2['total'],sum(v2[k] for k in (
            'deployment_and_request','trajectory_generation','hardware_route_validation','velocity_retiming')),places=6)
        self.assertEqual(data['plan']['planning_timing_ms'],timings)
        self.assertEqual(self.robot.calls,[])
        state=self.run_plan(data)
        self.assertEqual(state['planning_timing_ms'],timings)

    def test_preserves_v2_corners_and_preview_profile_without_added_stages(self):
        self.larger_goal()
        data=self.plan()
        self.assertEqual(self.robot.calls,[])
        self.assertEqual(data['plan']['execution_mode'],'encoder_profile')
        self.assertEqual(data['plan']['waypoints'],1)
        self.assertTrue(data['plan']['encoder_drag']['velocity_profile'])
        self.assertIsNone(data['plan']['encoder_drag']['maximum_excursion_deg'])
        self.assertFalse(data['plan']['encoder_drag']['staged'])
        segments=data['result']['smoothing']['segments']
        self.assertEqual(len(segments),1)
        np.testing.assert_allclose(segments[0]['end_deg'],self.end)
        self.assertGreaterEqual(segments[0]['duration_s'],4*driver.PERIOD_S)
        self.assertGreater(data['plan']['execution_timeout_s'],data['result']['smoothing']['duration_s'])

    def test_profile_accepts_80_percent_and_rejects_81_without_motion(self):
        self.body['speed'] = 81
        with self.assertRaises(ValueError):
            self.plan()
        self.assertEqual(self.robot.calls, [])
        self.body['speed'] = 80
        data = self.plan()
        self.assertEqual(data['plan']['max_sdk_speed'], 80)
        self.assertEqual(data['plan']['encoder_drag']['maximum_speed_percent'], 80)
        self.assertEqual(data['result']['planned_sdk_speed'], 80)
        self.assertAlmostEqual(data['result']['joint_velocity_allowance_deg_s'], 120.)
        curve = data['result']['smoothing']
        for phase in np.linspace(0., curve['duration_s'], 101):
            packet = driver.packet(curve, phase, self.controller.ik.plan['encoder_mapping'], 80)
            self.assertLessEqual(max(packet['speeds']), int(120*encoder_drag.STEPS_PER_DEG))
        self.assertEqual(self.robot.calls, [])

    def test_streams_profile_speeds_with_no_interior_arrival_wait_or_angle_fallback(self):
        data=self.plan(); state=self.run_plan(data)
        self.assertEqual(state['state'],'completed',state['message'])
        sends=[x for x in self.robot.calls if x[0]=='encoder_drag']
        self.assertGreater(len(sends),1)
        self.assertFalse(any(x[0]=='move' for x in self.robot.calls))
        speeds=[]
        for sample in state['samples']:
            if 'sent_encoder_speeds' in sample:
                expected=np.maximum(1,np.floor(np.abs(sample['reference_velocity_deg_s'])*encoder_drag.STEPS_PER_DEG+1e-9)).astype(int)
                self.assertEqual(sample['sent_encoder_speeds'],expected.tolist())
                self.assertAlmostEqual(sample['reference_phase_s'],min(sample['time'],data['result']['smoothing']['duration_s']))
                speeds.append(sample['sent_encoder_speeds'][0])
        self.assertGreater(max(speeds),min(speeds))
        self.assertEqual(state['completed_waypoints'],1)
        self.assertEqual(state['completed_stages'],0)

    def test_fixed_target_endpoint_miss_still_stops_instead_of_requesting_gaze_correction(self):
        data = self.plan()
        self.controller.ik.plan['target_mm'][0] += 50.
        state = self.run_plan(data)
        self.assertEqual(state['state'], 'failed', state)
        self.assertIn('Cartesian/direction tolerance', state['message'])
        self.assertTrue(any(call[0] == 'stop' for call in self.robot.calls))

    def run_with_query_delay(self, delays):
        data=self.plan()
        self.controller.ik.execute(data['plan']['id'],5,True,True,True)
        original=self.robot.get_angles
        query_count=0
        def delayed():
            nonlocal query_count
            delay=delays[query_count % len(delays)]; query_count+=1
            time.sleep(delay)
            return original()
        with patch.object(self.robot,'get_angles',delayed):
            self.controller.ik.heartbeat(data['plan']['id'])
            deadline=time.monotonic()+5
            while time.monotonic()<deadline:
                state=self.controller.snapshot()['ik']
                if state['state']!='running':
                    return state
                time.sleep(.01)
        self.fail('Delayed-query fake profile did not finish')

    def test_two_second_cycle_still_stops_and_records_failed_timing(self):
        state = self.run_with_query_delay([2.05])
        self.assertEqual(state['state'], 'failed')
        self.assertIn('telemetry/send cycle exceeded two seconds', state['message'])
        self.assertGreater(state['failed_cycle_timing']['cycle_elapsed_s'], 2.)
        self.assertFalse(any(c[0] == 'encoder_drag' for c in self.robot.calls))

    def test_measured_query_work_is_inside_the_200ms_command_period(self):
        state=self.run_with_query_delay([.095])
        self.assertEqual(state['state'],'completed',state['message'])
        samples=[s for s in state['samples'] if s.get('actual_send_interval_s') is not None]
        intervals=[s['actual_send_interval_s'] for s in samples]
        self.assertGreaterEqual(len(intervals),4)
        self.assertGreaterEqual(min(intervals),driver.PERIOD_S)
        self.assertLess(float(np.median(intervals)),.235)
        self.assertTrue(all(s['telemetry_cycle_s']>=.095 for s in samples))
        self.assertLess(float(np.median([s['send_deadline_lateness_s'] for s in samples])),.035)

    def test_stop_can_cancel_between_early_telemetry_and_delivery(self):
        data=self.plan()
        original_get=self.robot.get_angles
        original_wait=self.controller.closed.wait
        query_count=0; executor=None; cancelled=[]
        def queried():
            nonlocal query_count,executor
            query_count+=1; executor=threading.get_ident()
            if query_count==1:
                time.sleep(.095)
            return original_get()
        def wait(seconds):
            if threading.get_ident()==executor and query_count>=2 and seconds>.01 and not cancelled:
                self.assertFalse(self.controller.lock._is_owned())
                cancelled.append(True)
                self.controller.stop()
            return original_wait(seconds)
        self.controller.ik.execute(data['plan']['id'],5,True,True,True)
        with patch.object(self.robot,'get_angles',queried), patch.object(self.controller.closed,'wait',wait):
            self.controller.ik.heartbeat(data['plan']['id'])
            deadline=time.monotonic()+3
            while self.controller.ik.status['state']=='running' and time.monotonic()<deadline:
                time.sleep(.01)
            time.sleep(.15)
        self.assertTrue(cancelled)
        self.assertEqual(self.controller.ik.status['state'],'stopped')
        self.assertEqual(sum(c[0]=='encoder_drag' for c in self.robot.calls),1)
        self.assertTrue(any(c[0]=='stop' for c in self.robot.calls))

    def test_variable_query_duration_does_not_send_early_or_catch_up(self):
        state=self.run_with_query_delay([.04,.11,.26,.03])
        self.assertEqual(state['state'],'completed',state['message'])
        samples=[s for s in state['samples'] if s.get('actual_send_interval_s') is not None]
        self.assertTrue(all(s['actual_send_interval_s']>=driver.PERIOD_S for s in samples))
        self.assertTrue(any(s['telemetry_cycle_s']>=.26 for s in samples))
        self.assertTrue(any(s['deadline_wait_s']>.1 for s in samples))
        for previous,current in zip(samples,samples[1:]):
            if previous['telemetry_cycle_s']>=.26 and current['telemetry_cycle_s']<.05:
                self.assertLess(current['actual_send_interval_s'],.235)

    def test_stalled_joint_stops_on_reference_error_instead_of_stretching_clock(self):
        self.larger_goal();self.robot.track=False
        data=self.plan(); state=self.run_plan(data)
        self.assertEqual(state['state'],'failed')
        self.assertIn('tracking error',state['message'])
        self.assertTrue(any(c[0]=='stop' for c in self.robot.calls))
        self.assertLess(state['reference_phase_s'],data['result']['smoothing']['duration_s'])

    def run_with_initial_tracking_error(self, lag):
        data = self.plan()
        self.assertEqual(data['plan']['encoder_drag']['maximum_tracking_error_deg'], 5.)
        self.assertEqual(data['plan']['encoder_drag']['maximum_command_lead_deg'], 6.)
        self.assertEqual(data['plan']['encoder_drag']['arrival_deg'], .75)
        self.controller.ik.execute(data['plan']['id'], 5, True, True, True)
        original = self.robot.get_angles
        first = True
        def lagged():
            nonlocal first
            angles = original()
            if first:
                first = False
                angles[0] -= lag
            return angles
        with patch.object(self.robot, 'get_angles', lagged):
            self.controller.ik.heartbeat(data['plan']['id'])
            deadline = time.monotonic()+5
            while time.monotonic() < deadline:
                state = self.controller.snapshot()['ik']
                if state['state'] != 'running':
                    return state
                time.sleep(.01)
        self.fail('Fake tracking-boundary trial did not finish')

    def test_five_degree_tracking_error_is_allowed_without_changing_reference(self):
        state = self.run_with_initial_tracking_error(5.)
        self.assertEqual(state['state'], 'completed', state['message'])
        self.assertAlmostEqual(state['samples'][0]['tracking_error_deg'], 5.)
        self.assertEqual(state['samples'][0]['reference_phase_s'], 0.)
        self.assertTrue(any(c[0]=='encoder_drag' for c in self.robot.calls))

    def test_tracking_error_just_above_five_degrees_stops_before_sending(self):
        state = self.run_with_initial_tracking_error(5.01)
        self.assertEqual(state['state'], 'failed')
        self.assertIn('5.01° exceeds 5°', state['message'])
        self.assertFalse(any(c[0]=='encoder_drag' for c in self.robot.calls))
        self.assertTrue(any(c[0]=='stop' for c in self.robot.calls))

    def test_expired_heartbeat_after_health_query_stops_before_sending(self):
        data=self.plan()
        self.controller.ik.execute(data['plan']['id'],5,True,True,True)
        original=self.controller._health
        def expired():
            original();self.controller.ik.heartbeat_at=time.monotonic()-10.
        with patch.object(self.controller,'_health',expired):
            self.controller.ik.heartbeat(data['plan']['id'])
            deadline=time.monotonic()+3
            while self.controller.ik.status['state']=='running' and time.monotonic()<deadline:
                time.sleep(.025)
        self.assertEqual(self.controller.ik.status['state'],'failed')
        self.assertFalse(any(c[0]=='encoder_drag' for c in self.robot.calls))
        self.assertTrue(any(c[0]=='stop' for c in self.robot.calls))

    def test_slow_query_stops_before_first_packet(self):
        data=self.plan()
        self.controller.ik.execute(data['plan']['id'],5,True,True,True)
        original=self.robot.get_angles
        def slow():
            time.sleep(driver.MAX_CYCLE_S+.05);return original()
        with patch.object(self.robot,'get_angles',slow):
            self.controller.ik.heartbeat(data['plan']['id'])
            deadline=time.monotonic()+4
            while self.controller.ik.status['state']=='running' and time.monotonic()<deadline:
                time.sleep(.025)
        self.assertEqual(self.controller.ik.status['state'],'failed')
        self.assertIn('cycle exceeded',self.controller.ik.status['message'])
        self.assertFalse(any(c[0]=='encoder_drag' for c in self.robot.calls))

    test_browser_confirmation_required_and_stop_cancels_before_any_packet = fixtures.EncoderDragTest.test_browser_confirmation_required_and_stop_cancels_before_any_packet
    test_mapping_change_after_review_refuses_before_motion = fixtures.EncoderDragTest.test_mapping_change_after_review_refuses_before_motion
    test_near_zero_direction_is_not_guessed_for_a_moving_joint = fixtures.EncoderDragTest.test_near_zero_direction_is_not_guessed_for_a_moving_joint
    test_incompatible_or_malformed_encoder_telemetry_never_creates_motion = fixtures.EncoderDragTest.test_incompatible_or_malformed_encoder_telemetry_never_creates_motion


if __name__=='__main__':unittest.main()
