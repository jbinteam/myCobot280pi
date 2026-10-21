"""Continuous v2 hardware lifecycle with in-memory robot transport only."""
import sys
import json
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_encoder_drag import EncoderRobot
from webui import encoder_profile, encoder_drag
from server import RobotController
from simulation.kinematics import ARM
from simulation.continuous import prepare, frames_for, sample_motion
from simulation.diffusion.env import ArmEnv
from simulation.diffusion.tracking import SceneSpace
from webui import v2_mouse
from webui.mouse_motion import StaleMousePlan


class V2MouseTest(unittest.TestCase):
    ENGINE = "diffusion_v2"
    PROFILE = None
    def setUp(self):
        self.robot = EncoderRobot()
        self.robot.angles[0] = self.robot.angles[4] = 1.
        self.robot.fresh_mode = 0
        self.controller = RobotController(lambda *_: self.robot)
        self.controller.connect('192.168.77.225', 9000)
        self.env = ArmEnv(); self.env.set_obstacle(None)
        self.space = SceneSpace(self.env)
        self.start = np.deg2rad(self.robot.angles)
        self.goal = self.start.copy(); self.goal[0] += np.deg2rad(12.)
        self.session = self.controller.ik.start_mouse_aim({'armed': True, 'frame_verified': True,
            'tracking_engine': self.ENGINE, 'scene_verified': True, 'speed': 30,
            **({'velocity_shape':self.PROFILE} if self.PROFILE else {})})['session_id']
        patcher = patch.object(v2_mouse.classical_tracking if self.ENGINE == 'learned_rrt' else v2_mouse.tracking, 'plan', side_effect=self.proposal)
        self.planner = patcher.start(); self.addCleanup(patcher.stop)

    def tearDown(self):
        self.controller.shutdown(); self.env.close()

    def proposal(self, body):
        start = np.deg2rad(body['start_deg'])
        t = ARM.transform(self.goal); target = t[:3, 3]; axis = t[:3, 2]
        if 'velocity_shape' in body:
            states, times, smoothing = v2_mouse.profile_tracking.prepare(self.space, [start, self.goal],
                np.array(body['start_velocity_rad_s']), np.array(body['start_acceleration_rad_s2']), body['speed_rad_s'], velocity_shape=body['velocity_shape'])
        else:
            states, times, smoothing = prepare(self.space, [start, self.goal],
                np.array(body['start_velocity_rad_s']), np.array(body['start_acceleration_rad_s2']), body['speed_rad_s'])
        return {'method': 'learned_rrt' if self.ENGINE == 'learned_rrt' else 'local_diffusion', 'task': 'pose5', 'mouse_aim': True,
            'simulation_only': True, 'continuous_motion': True, 'tracking_strategy': 'learned_ik_rrt' if self.ENGINE == 'learned_rrt' else 'local_then_diffusion',
            'tracking_chunk': True, 'tracking_source': 'diffusion_fallback', 'status': 'target_reached',
            'playable': True, 'converged': True, 'replan_required': False, 'gap_mm': 50.,
            'coordinate_frame': 'g_base', 'tool_frame': 'joint6_flange',
            'checkpoint_sha256': v2_mouse.IK_SHA256 if self.ENGINE == 'learned_rrt' else v2_mouse.V2_SHA256, 'validation': 'joint_limits_self_table_block',
            'obstacle_mm': None, 'scene_mode': 'no_block', 'cursor_mm': body['cursor_mm'],
            'target_mm': (target*1000).tolist(), 'target_axis': axis.tolist(),
            'goal_deg': np.rad2deg(self.goal).tolist(), 'goal_cursor_acceptance': {'accepted': True},
            'cursor_acceptance': {'accepted': True}, 'final_error_mm': 0., 'final_orientation_error_deg': 0.,
            'smoothing': smoothing, 'speed_rad_s': body['speed_rad_s'], 'acceleration_rad_s2': 3.,
            'frames': frames_for(states, times, target, axis)}

    def plan(self):
        t = ARM.transform(self.goal)
        return self.controller.ik.prepare_mouse_aim({'session_id': self.session,
            'cursor_mm': (t[:3, 3]*1000+50*t[:3, 2]).tolist(), 'target_axis': t[:3, 2].tolist()})

    def execute(self, data, acknowledge=True):
        result = self.controller.ik.execute(data['plan']['id'], 30, True, True, True)
        if acknowledge:
            self.controller.ik.mouse_aim_heartbeat(self.session, result['ik']['execution_id'],
                data['plan']['id'] if data['plan']['id'] != result['ik']['execution_id'] else None)
        return result

    def wait(self, wanted, seconds=5):
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            s = self.controller.snapshot()['ik']
            if s['state'] in wanted: return s
            if self.controller.ik.mouse_session:
                try:
                    self.controller.ik.mouse_aim_heartbeat(self.session, s.get('execution_id'))
                except ValueError:
                    # Completion can race the preceding snapshot.
                    final = self.controller.snapshot()['ik']
                    if final['state'] in wanted:
                        return final
                    raise
            time.sleep(.02)
        self.fail(str(self.controller.snapshot()['ik']))

    def wait_motion(self):
        end = time.monotonic()+2
        while time.monotonic() < end:
            if self.controller.snapshot()['ik'].get('curve_time_s', 0) > .08: return
            time.sleep(.01)
        self.fail('No curve progression')

    def test_plan_is_read_only_server_owns_boundary_and_uses_same_c2_curve(self):
        data = self.plan()
        self.assertEqual(self.robot.calls, [])
        self.assertTrue(data['plan']['hardware_v2_mouse'])
        self.assertEqual(data['plan']['execution_mode'], 'encoder_feedback_tracking')
        request = self.planner.call_args.args[0]
        np.testing.assert_allclose(request['start_deg'], self.robot.angles)
        np.testing.assert_allclose(request['start_velocity_rad_s'], np.zeros(6))
        self.assertAlmostEqual(request['speed_rad_s'], np.deg2rad(20))
        for extra in ({'start_deg': [0]*6}, {'start_velocity_rad_s': [0]*6}, {'goal_hint_deg': [0]*6}, {'obstacle_mm': [0]*6}):
            with self.assertRaises(ValueError):
                self.controller.ik.prepare_mouse_aim({'session_id': self.session, 'cursor_mm': [100, 0, 200], **extra})

    def test_integrated_velocity_preserves_checked_curve_and_nonzero_boundary_motion(self):
        goal = self.start.copy(); goal[0] -= np.deg2rad(4)
        states, times, curve = prepare(self.space, [self.start, goal],
            np.array([-.03, 0, 0, 0, 0, 0]), np.array([-.02, 0, 0, 0, 0, 0]), .35)
        for phase in [curve['duration_s'], .07, 0., .31, .07, *times]:
            actual = v2_mouse.motion(curve, phase)
            expected = sample_motion(curve, phase)
            for a, b in zip(actual, expected):
                np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)
        self.assertLess(v2_mouse.motion(curve, .07)[1][0], 0)
        for q, t in zip(states, times):
            np.testing.assert_allclose(v2_mouse.motion(curve, t)[0], q, atol=1e-12, rtol=0)

    def test_first_command_waits_for_ack_completes_and_restore_mode(self):
        data = self.plan(); self.execute(data, acknowledge=False)
        time.sleep(.15); self.assertEqual(self.robot.calls, [])
        self.controller.ik.mouse_aim_heartbeat(self.session, data['plan']['id'])
        status = self.wait({'completed', 'failed'})
        self.assertEqual(status['state'], 'completed', status)
        self.assertGreater(len([c for c in self.robot.calls if c[0] == 'encoder_drag']), 3)
        self.assertLessEqual(status['estimated_error_mm'], 5)
        self.assertTrue(all(len(s['reference_velocity_deg_s']) == 6 for s in status['samples']))
        self.assertGreater(max(s['reference_velocity_deg_s'][0] for s in status['samples']), 0)
        self.assertFalse(any(c[0] == 'move' for c in self.robot.calls))
        self.controller.ik.stop_mouse_aim(self.session)
        self.assertEqual(self.robot.fresh_mode, 0)
        with self.assertRaises(ValueError): self.execute(data)

    def test_live_retarget_preserves_measured_motion_and_execution_owner(self):
        first = self.plan(); self.execute(first); self.wait_motion()
        json.dumps(self.controller.snapshot(), allow_nan=False)
        self.goal[0] += np.deg2rad(4)
        new = self.plan(); root = first['plan']['id']
        self.assertGreater(np.linalg.norm(self.planner.call_args.args[0]['start_velocity_rad_s']), 0)
        response = self.execute(new)
        self.assertEqual(response['ik']['execution_id'], root)
        status = self.wait({'completed', 'failed'})
        self.assertEqual(status['state'], 'completed', status)
        self.assertEqual(status['active_plan_id'], new['plan']['id'])
        for key in ('join_position_error_rad', 'join_velocity_error_rad_s', 'join_acceleration_error_rad_s2'):
            self.assertLessEqual(status[key], 1e-8)
        self.assertFalse(any(call[0] == 'stop' for call in self.robot.calls))

    def test_unacknowledged_replacement_never_changes_active_path(self):
        first = self.plan(); self.execute(first); self.wait_motion()
        json.dumps(self.controller.snapshot(), allow_nan=False)
        self.goal[0] += np.deg2rad(4)
        new = self.plan(); self.execute(new, acknowledge=False)
        self.controller.ik.mouse_aim_heartbeat(self.session, first['plan']['id'])
        time.sleep(.2)
        self.assertEqual(self.controller.snapshot()['ik']['active_plan_id'], first['plan']['id'])
        self.controller.ik.stop_mouse_aim(self.session)

    def test_late_join_is_rejected_without_interrupting_checked_tail(self):
        first = self.plan(); self.execute(first); self.wait_motion()
        self.goal[0] += np.deg2rad(3)
        new = self.plan()
        with self.controller.lock:
            self.controller.ik.plan['join']['curve_id'] = 'superseded-curve'
        with self.assertRaises(StaleMousePlan): self.execute(new)
        self.assertEqual(self.controller.snapshot()['ik']['active_plan_id'], first['plan']['id'])
        self.assertFalse(any(call[0] == 'stop' for call in self.robot.calls))

    def test_partial_neural_prefix_executes_and_requests_continuation(self):
        real = self.proposal
        def partial(body):
            r = real(body); r.update(converged=False, replan_required=True, status='chunk_ready')
            return r
        self.planner.side_effect = partial
        data = self.plan(); self.assertFalse(data['plan']['converged'])
        self.assertTrue(data['plan']['playable']); self.execute(data)
        status = self.wait({'completed', 'failed'})
        self.assertEqual(status['state'], 'completed', status)
        self.assertTrue(status['replan_required'])

    def test_failed_proposal_does_not_create_or_replace_plan(self):
        real = self.proposal
        def rejected(body):
            r = real(body); r.update(playable=False, converged=False, frames=[], status='no_safe_chunk')
            return r
        self.planner.side_effect = rejected
        self.assertIsNone(self.plan()['plan'])
        self.assertEqual(self.robot.calls, [])

    def test_speed_scene_checkpoint_and_derivative_gates(self):
        data = self.plan()
        for speed, scene in [(30, False), (10, True)]:
            with self.assertRaises(ValueError):
                self.controller.ik.execute(data['plan']['id'], speed, True, True, scene)
        with patch.object(v2_mouse.classical_tracking if self.ENGINE == 'learned_rrt' else v2_mouse, 'verified_status', side_effect=RuntimeError('pinned checkpoint changed')):
            with self.assertRaises(RuntimeError): self.execute(data)
        real = self.proposal
        def wrong_boundary(body):
            r = real(body)
            if r['smoothing']['method'] == v2_mouse.profile_tracking.METHOD:
                r['smoothing']['segments'][0]['start_velocity_rad_s'][0] += .1
            else:
                r['smoothing']['segments'][0]['controls_deg'][1][0] += 3
            return r
        self.planner.side_effect = wrong_boundary
        with self.assertRaisesRegex(ValueError, 'join'):
            self.plan()
        self.assertEqual(self.robot.calls, [])

    def test_stop_cancels_pending_continuation_and_no_late_motion(self):
        first = self.plan(); self.execute(first); self.wait_motion()
        json.dumps(self.controller.snapshot(), allow_nan=False)
        self.goal[0] += np.deg2rad(4)
        new = self.plan(); self.execute(new, acknowledge=False)
        self.controller.ik.stop_mouse_aim(self.session)
        count = len(self.robot.calls); time.sleep(.1)
        self.assertEqual(len(self.robot.calls), count)
        with self.assertRaises(ValueError):
            self.controller.ik.mouse_aim_heartbeat(self.session, first['plan']['id'], new['plan']['id'])

    def test_live_collision_and_heartbeat_fail_before_motion(self):
        data = self.plan()
        with patch.object(ArmEnv, 'path_valid', return_value=False):
            self.execute(data)
            status = self.wait({'failed'})
        self.assertIn('collision', status['message'])
        self.assertFalse(any(c[0] == 'encoder_drag' for c in self.robot.calls))

    def test_lost_heartbeat_stops_before_first_command(self):
        data = self.plan(); self.execute(data, acknowledge=False)
        with self.controller.lock: self.controller.ik.heartbeat_at -= 6
        end = time.monotonic()+2
        while time.monotonic() < end and self.controller.snapshot()['ik']['state'] != 'failed': time.sleep(.02)
        self.assertEqual(self.controller.snapshot()['ik']['state'], 'failed')
        self.assertFalse(any(c[0] == 'encoder_drag' for c in self.robot.calls))

    def test_final_gaze_miss_requests_fresh_live_correction_without_stop(self):
        data = self.plan()
        self.controller.ik.plan['cursor_mm'][0] += 50.
        self.execute(data); status = self.wait({'completed', 'failed'})
        self.assertEqual(status['state'], 'completed', status)
        self.assertFalse(status['final_goal_accepted'])
        self.assertFalse(status['cursor_acceptance']['accepted'])
        self.assertTrue(status['replan_required'])
        self.assertTrue(status['feedback_replan_required'])
        self.assertIn('correction', status['message'])
        self.assertFalse(any(call[0] == 'stop' for call in self.robot.calls))

    def test_scene_confirmation_required_before_session_start(self):
        self.controller.ik.stop_mouse_aim(self.session)
        with self.assertRaisesRegex(ValueError, 'table scene'):
            self.controller.ik.start_mouse_aim({'armed': True, 'frame_verified': True, 'tracking_engine': self.ENGINE})

    def test_mode_refusal_and_read_failure_stop_tracking(self):
        data = self.plan(); self.robot.get_fresh_mode = lambda: 0
        self.execute(data); status = self.wait({'failed'})
        self.assertIn('mode', status['message'])
        self.assertFalse(any(c[0] == 'encoder_drag' for c in self.robot.calls))

    def test_encoder_endpoint_within_shared_arrival_tolerance_completes(self):
        self.goal = self.start.copy(); self.goal[5] += np.deg2rad(3)
        data = self.plan(); original = self.robot._mesg
        def offset(genre, *args, **kwargs):
            result = original(genre, *args, **kwargs)
            from pymycobot.common import ProtocolCode
            if genre == ProtocolCode.SET_ENCODERS_DRAG:
                self.robot.angles[5] -= .6
            return result
        self.robot._mesg = offset
        self.execute(data); status = self.wait({'completed', 'failed'})
        self.assertEqual(status['state'], 'completed', status)
        self.assertAlmostEqual(status['samples'][-1]['max_joint_error_deg'], .6, delta=.05)
        self.assertLess(status['estimated_error_mm'], .5)

    def test_stalled_motor_waits_instead_of_aborting_on_nominal_tracking_error(self):
        data = self.plan(); self.robot.track = False; self.execute(data)
        end = time.monotonic()+2.5
        while time.monotonic() < end:
            self.controller.ik.mouse_aim_heartbeat(self.session, data['plan']['id'])
            time.sleep(.02)
        status = self.controller.snapshot()['ik']
        self.assertEqual(status['state'], 'running', status)
        self.assertTrue(status['tracking_paused'])
        self.assertGreater(status['nominal_tracking_error_deg'], 5.)
        self.assertGreater(status['reference_lag_s'], .5)
        self.assertLessEqual(status['command_lead_deg'], encoder_profile.MOUSE_LEAD_DEG)
        self.assertFalse(any(call[0] == 'stop' for call in self.robot.calls))
        self.robot.track = True
        status = self.wait({'completed', 'failed'}, seconds=12)
        self.assertEqual(status['state'], 'completed', status)

    def test_external_drift_stops_without_recovery_command(self):
        data = self.plan(); self.execute(data); self.wait_motion()
        with self.controller.lock:
            self.robot.angles[0] += 10
            count = len([c for c in self.robot.calls if c[0] == 'encoder_drag'])
        status = self.wait({'failed'})
        self.assertIn('command lead', status['message'])
        self.assertEqual(len([c for c in self.robot.calls if c[0] == 'encoder_drag']), count)

    def test_new_cursor_retargets_from_live_joints_even_after_large_nominal_lag(self):
        first = self.plan(); self.robot.track = False; self.execute(first)
        end = time.monotonic()+1.6
        while time.monotonic() < end:
            self.controller.ik.mouse_aim_heartbeat(self.session, first['plan']['id'])
            time.sleep(.02)
        old = self.controller.snapshot()['ik']
        self.assertEqual(old['state'], 'running')
        self.assertGreater(old['nominal_tracking_error_deg'], 5.)
        self.goal[0] += np.deg2rad(3)
        new = self.plan()
        np.testing.assert_allclose(self.planner.call_args.args[0]['start_deg'], self.robot.angles)
        self.assertEqual(self.controller.ik.plan['join']['mode'], 'live_feedback')
        self.robot.track = True
        response = self.execute(new)
        self.assertEqual(response['ik']['execution_id'], first['plan']['id'])
        status = self.wait({'completed', 'failed'}, seconds=12)
        self.assertEqual(status['state'], 'completed', status)
        self.assertEqual(status['active_plan_id'], new['plan']['id'])
        self.assertTrue(all(s['command_lead_deg'] <= encoder_profile.MOUSE_LEAD_DEG+1e-8 for s in status['samples']))
        self.assertFalse(any(c[0] == 'stop' for c in self.robot.calls))

    def test_slow_motor_continues_and_reaches_target_without_clock_accuracy_stop(self):
        data = self.plan(); self.robot.track = False
        original = self.robot.get_angles
        def slow_read():
            sends = [call for call in self.robot.calls if call[0] == 'encoder_drag']
            if sends:
                target = (np.asarray(sends[-1][1])-2048)/(encoder_drag.STEPS_PER_DEG*self.robot.directions)
                self.robot.angles = (np.asarray(self.robot.angles)+np.clip(target-self.robot.angles, -.4, .4)).tolist()
            return original()
        self.robot.get_angles = slow_read
        self.execute(data)
        status = self.wait({'completed', 'failed'}, seconds=16)
        self.assertEqual(status['state'], 'completed', status)
        self.assertGreater(max(s['nominal_tracking_error_deg'] for s in status['samples']), 5.)
        self.assertGreater(max(s['reference_lag_s'] for s in status['samples']), .5)
        self.assertTrue(all(s['command_lead_deg'] <= encoder_profile.MOUSE_LEAD_DEG+1e-8 for s in status['samples']))
        self.assertFalse(any(call[0] == 'stop' for call in self.robot.calls))

    def test_feedback_phase_waits_at_checked_corner_until_live_arrival(self):
        goal = self.start.copy(); goal[0] += np.deg2rad(3)
        final = goal.copy(); final[5] += np.deg2rad(2)
        # Preserve this valid two-leg fixture so it exercises an actual retained
        # corner rather than the planner's collision-free direct shortcut.
        with patch.object(v2_mouse.profile_tracking, 'simplify_path', side_effect=lambda path, _: path):
            states, times, curve = v2_mouse.profile_tracking.prepare(self.space, [self.start, goal, final],
                np.zeros(6), np.zeros(6), .35, minimum_duration_s=.8)
        corner = curve['segments'][0]['duration_s']
        self.assertEqual(encoder_profile.feedback_phase(curve, corner, .2, np.rad2deg(self.start)), corner)
        self.assertEqual(encoder_profile.feedback_phase(curve, corner, .2, np.rad2deg(goal), moving=True), corner)
        self.assertGreater(encoder_profile.feedback_phase(curve, corner, .2, np.rad2deg(goal)), corner)

    def test_unsafe_retarget_connection_keeps_checked_path_and_owner(self):
        from types import SimpleNamespace
        first = self.plan(); active = self.controller.ik.plan
        self.execute(first); self.wait_motion()
        self.goal[0] += np.deg2rad(3)
        new = self.plan(); self.execute(new, acknowledge=False)
        with self.controller.lock:
            mouse = self.controller.ik.v2_mouse
            self.controller.ik.mouse_pending_plan['confirmed'] = True
            kept, _ = mouse.advance_pending(active, mouse.clock_origin, time.monotonic(),
                SimpleNamespace(path_valid=lambda *args: False))
            self.assertEqual(kept['id'], first['plan']['id'])
            self.assertEqual(self.controller.motion, 'ik')
            self.assertIsNone(self.controller.ik.mouse_pending_plan)
            self.assertTrue(self.controller.ik.status['feedback_replan_required'])
            self.assertIn('collision', self.controller.ik.status['retarget_rejected'])
        self.assertFalse(any(c[0] == 'stop' for c in self.robot.calls))

    def test_unreadable_telemetry_stops_before_first_motor_target(self):
        data = self.plan(); self.execute(data, acknowledge=False)
        self.robot.get_angles = lambda: []
        self.controller.ik.mouse_aim_heartbeat(self.session, data['plan']['id'])
        status = self.wait({'failed'})
        self.assertFalse(any(c[0] == 'encoder_drag' for c in self.robot.calls), status)

    def test_80_percent_session_executes_and_81_is_rejected(self):
        self.controller.ik.stop_mouse_aim(self.session)
        body={'armed':True,'frame_verified':True,'scene_verified':True,
              'tracking_engine':self.ENGINE,'speed':81}
        with self.assertRaises(ValueError):self.controller.ik.start_mouse_aim(body)
        self.session=self.controller.ik.start_mouse_aim({**body,'speed':80})['session_id']
        data=self.plan()
        self.assertEqual(data['plan']['max_sdk_speed'],80)
        result=self.controller.ik.execute(data['plan']['id'],80,True,True,True)
        self.controller.ik.mouse_aim_heartbeat(self.session,result['ik']['execution_id'])
        status=self.wait({'completed','failed'})
        self.assertEqual(status['state'],'completed',status.get('message'))
        self.assertTrue(any(c[0]=='encoder_drag' for c in self.robot.calls))
        self.assertFalse(any(c[0]=='move' for c in self.robot.calls))
        self.assertTrue(all(max(c[2])<=int(120*encoder_drag.STEPS_PER_DEG) for c in self.robot.calls if c[0]=='encoder_drag'))

    def test_reference_slider_can_lower_session_speed(self):
        self.controller.ik.stop_mouse_aim(self.session)
        self.session = self.controller.ik.start_mouse_aim({'armed': True, 'frame_verified': True,
            'scene_verified': True, 'tracking_engine': self.ENGINE, 'speed': 30,
            'speed_rad_s': .1})['session_id']
        self.plan()
        self.assertAlmostEqual(self.planner.call_args.args[0]['speed_rad_s'], .1)

    def test_interior_preview_must_match_the_executed_curve(self):
        real = self.proposal
        def changed_preview(body):
            result = real(body); result['frames'][2]['q'][0] += .1
            return result
        self.planner.side_effect = changed_preview
        with self.assertRaisesRegex(ValueError, 'preview frame'):
            self.plan()
        self.assertEqual(self.robot.calls, [])


    def test_mouse_packets_follow_profile_velocity_with_feedback_paced_clock(self):
        data = self.plan(); self.execute(data)
        status = self.wait({'completed', 'failed'}, seconds=8)
        self.assertEqual(status['state'], 'completed', status.get('message'))
        curve = data['result']['smoothing']
        for sample in status['samples']:
            if 'sent_encoder_speeds' not in sample: continue
            expected = encoder_profile.motion(curve, sample['reference_phase_s'])
            np.testing.assert_allclose(sample['reference_velocity_deg_s'], expected[1], atol=1e-9)
            words = np.maximum(1, np.floor(np.abs(expected[1])*encoder_drag.STEPS_PER_DEG+1e-9)).astype(int)
            self.assertEqual(sample['sent_encoder_speeds'], words.tolist())
            self.assertLessEqual(sample['reference_phase_s'], min(sample['trajectory_elapsed_s'], curve['duration_s'])+1e-8)
            self.assertGreaterEqual(sample['reference_lag_s'], 0.)
            self.assertLessEqual(sample['command_lead_deg'], encoder_profile.MOUSE_LEAD_DEG+1e-8)
            if sample['actual_send_interval_s'] is not None:
                self.assertGreaterEqual(sample['actual_send_interval_s'], encoder_profile.PERIOD_S)
        json.dumps(self.controller.snapshot(), allow_nan=False)

    def test_encoder_mapping_is_reverified_before_first_motion(self):
        data = self.plan()
        original = self.robot.get_encoders
        self.robot.get_encoders = lambda: [v+20 for v in original()]
        with self.assertRaises((ValueError, RuntimeError)):
            self.execute(data)
        self.assertFalse(any(c[0] in ('move', 'encoder_drag') for c in self.robot.calls))
        self.assertIsNone(self.controller.motion)

    def test_unknown_encoder_direction_cannot_move_a_joint(self):
        self.controller.ik.stop_mouse_aim(self.session)
        self.controller.encoder_calibration.forget(self.controller.host, self.controller.port, self.controller.firmware)
        self.robot.angles[0] = 0.
        self.session = self.controller.ik.start_mouse_aim({'armed':True, 'frame_verified':True,
            'scene_verified':True, 'tracking_engine':self.ENGINE, 'speed':30})['session_id']
        with self.assertRaisesRegex(ValueError, 'ambiguous encoder direction'):
            self.plan()
        self.assertFalse(any(c[0] in ('move', 'encoder_drag') for c in self.robot.calls))

    def test_late_replacement_ack_keeps_original_encoder_tail(self):
        self.goal[0] += np.deg2rad(12)
        first = self.plan(); self.execute(first); self.wait_motion()
        self.goal[0] += np.deg2rad(4)
        new = self.plan(); self.execute(new, acknowledge=False)
        mouse = self.controller.ik.v2_mouse
        with self.controller.lock:
            deadline = self.controller.ik.mouse_pending_plan['created']+2.03
        while time.monotonic() < deadline:
            self.controller.ik.mouse_aim_heartbeat(self.session, first['plan']['id'])
            time.sleep(.02)
        with self.assertRaises(StaleMousePlan):
            self.controller.ik.mouse_aim_heartbeat(self.session, first['plan']['id'], new['plan']['id'])
        status = self.wait({'completed', 'failed'}, seconds=12)
        self.assertEqual(status['state'], 'completed', status)
        self.assertEqual(status['active_plan_id'], first['plan']['id'])
        self.assertIn('checked path', status['retarget_rejected'])
        self.assertFalse(any(c[0] == 'stop' for c in self.robot.calls))

    def test_encoder_lookahead_stays_before_a_moving_join_reversal(self):
        self.goal[0] = self.start[0]-np.deg2rad(2.)
        transform = ARM.transform(self.goal); target = transform[:3, 3]; axis = transform[:3, 2]
        states, times, curve = v2_mouse.profile_tracking.prepare(self.space, [self.start, self.goal],
            np.array([.1,0,0,0,0,0]), np.zeros(6), np.deg2rad(20),
            velocity_shape=self.PROFILE, minimum_duration_s=.8)
        result = {'frames':frames_for(states, times, target, axis), 'smoothing':curve}
        plan = {'result':result, 'curve':curve, 'environment':self.env.metadata()}
        mapping = self.controller.ik.v2_mouse.encoder_mapping
        encoder_profile.prepare_mouse(plan, mapping, 30)
        self.assertTrue(curve['encoder_turns_s'])
        turn = curve['encoder_turns_s'][0]; phase = max(0., turn-.1)
        packet = encoder_profile.packet(curve, phase, mapping, 30)
        self.assertLessEqual(packet['target_phase_s'], turn+1e-9)
        self.assertGreater(packet['reference_velocity_deg_s'][0], 0)
        np.testing.assert_allclose(packet['reference_velocity_deg_s'], encoder_profile.motion(curve, phase)[1])


if __name__ == '__main__': unittest.main()
