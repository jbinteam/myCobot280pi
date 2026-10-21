"""Actual learned endpoint/scene checks and continuous cursor replacement gates."""
import unittest
from unittest.mock import patch
import numpy as np
from simulation.kinematics import ARM
from simulation.continuous import sample_motion
from simulation.diffusion import classical_tracking as tracker
from simulation.diffusion import service
from simulation.diffusion.env import ArmEnv


class ClassicalTrackingTest(unittest.TestCase):
    def setUp(self):
        self.start = [0, -40, -130, 80, 0, 50]
        goal = np.deg2rad([3, -42, -125, 78, 2, 50])
        t = ARM.transform(goal)
        self.body = {'start_deg': self.start, 'cursor_mm': (t[:3, 3]*1000+50*t[:3, 2]).tolist(),
                     'target_axis': t[:3, 2].tolist(), 'obstacle_mm': None, 'continuous_motion': True}

    def test_actual_endpoint_uses_learned_refinement_and_rrt_without_diffusion(self):
        with patch.object(ArmEnv, 'inverse_kinematics', autospec=True,
                side_effect=ArmEnv.inverse_kinematics) as solver, \
                patch.object(service, 'load_policy', side_effect=AssertionError('No diffusion')):
            r = tracker.plan(self.body)
        self.assertTrue(r['playable'], r)
        self.assertTrue(r['converged'])
        self.assertTrue(solver.call_args.kwargs['learned'])
        self.assertTrue(solver.call_args.kwargs['robust'])
        self.assertEqual(solver.call_args.kwargs['attempts'], 32)
        self.assertIn(r['rrt_diagnostics']['status'], ['direct_optimal', 'rrt_star'])
        self.assertTrue(r['goal_cursor_acceptance']['accepted'])
        self.assertLessEqual(r['final_error_mm'], 1.)
        self.assertLessEqual(r['final_orientation_error_deg'], 1.)
        with ArmEnv() as env:
            env.set_obstacle(None)
            self.assertTrue(env.path_valid(np.deg2rad([f['q'] for f in r['frames']]), .01))

    def test_actual_nonzero_motion_replacement_preserves_q_v_a_and_stops(self):
        first = tracker.plan(self.body)
        q, v, a = sample_motion(first['smoothing'], .12)
        goal = np.deg2rad([4, -42, -125, 78, 2, 50]); t = ARM.transform(goal)
        second = tracker.plan({**self.body, 'start_deg': np.rad2deg(q).tolist(),
            'cursor_mm': (t[:3, 3]*1000+50*t[:3, 2]).tolist(), 'target_axis': t[:3, 2].tolist(),
            'start_velocity_rad_s': v.tolist(), 'start_acceleration_rad_s2': a.tolist()})
        self.assertTrue(second['playable'], second)
        for actual, expected in zip(sample_motion(second['smoothing'], 0), (q, v, a)):
            np.testing.assert_allclose(actual, expected, atol=1e-10)
        self.assertTrue(second['smoothing']['ends_at_rest'])
        self.assertLessEqual(second['max_velocity_rad_s'], 1.+1e-8)
        self.assertLessEqual(second['max_acceleration_rad_s2'], 3.+1e-8)

    def test_resting_chunk_is_bounded_and_keeps_verified_goal(self):
        goal = np.deg2rad([30, -42, -125, 78, 2, 50]); t = ARM.transform(goal)
        r = tracker.plan({**self.body, 'continuous_motion': False,
            'cursor_mm': (t[:3, 3]*1000+50*t[:3, 2]).tolist(), 'target_axis': t[:3, 2].tolist(),
            'speed_rad_s': .2})
        self.assertTrue(r['playable'], r)
        self.assertLessEqual(r['frames'][-1]['time'], 1.3)
        self.assertTrue(r['goal_cursor_acceptance']['accepted'])
        self.assertFalse(r['converged']); self.assertTrue(r['replan_required'])
        self.assertTrue(r['smoothing']['starts_and_ends_at_rest'])

    def test_failed_or_unsafe_route_never_returns_motion(self):
        with patch.object(tracker, 'rrt_plan', return_value=(None, {'status': 'iteration_limit'})):
            r = tracker.plan(self.body)
        self.assertFalse(r['playable']); self.assertEqual(r['frames'], [])
        with patch.object(tracker, 'continuous_prepare', side_effect=ValueError('unsafe join')):
            r = tracker.plan(self.body)
        self.assertEqual(r['status'], 'smoothing_validation_failed')
        self.assertFalse(r['playable']); self.assertEqual(r['frames'], [])

    def test_slow_shared_profile_startup_uses_checked_prefix(self):
        goal = np.deg2rad([30, -42, -125, 78, 2, 50]); t = ARM.transform(goal)
        shape = {'kind':'smooth_trapezoid', 'rise_percent':5, 'fall_percent':10}
        r = tracker.plan({**self.body, 'speed_rad_s':.03, 'velocity_shape':shape,
            'cursor_mm':(t[:3, 3]*1000+50*t[:3, 2]).tolist(), 'target_axis':t[:3, 2].tolist()})
        self.assertTrue(r['playable'], r)
        self.assertFalse(r['converged']); self.assertTrue(r['replan_required'])
        self.assertTrue(r['goal_cursor_acceptance']['accepted'])
        self.assertTrue(r['smoothing']['tracking_prefix_limited'])
        self.assertEqual(r['smoothing']['reference_shape'], shape)
        self.assertLessEqual(r['frames'][-1]['time'], 12.)
        self.assertLessEqual(r['max_velocity_rad_s'], .03+1e-8)
        self.assertGreater(np.linalg.norm(np.asarray(r['frames'][-1]['q'])-self.start), .1)
        from simulation.profile_tracking import sample_motion, validate
        with ArmEnv() as env:
            env.set_obstacle(None)
            validate(r['smoothing'], r['frames'], (np.deg2rad(self.start), np.zeros(6), np.zeros(6)), .03, env)
        np.testing.assert_allclose(sample_motion(r['smoothing'], r['frames'][-1]['time'])[1:], np.zeros((2, 6)), atol=1e-10)

    def test_invalid_scene_and_untrusted_goal_do_not_bypass_checks(self):
        with patch.object(ArmEnv, 'inverse_kinematics', side_effect=AssertionError('Invalid start')):
            r = tracker.plan({**self.body, 'obstacle_mm': [0, 0, 15, 250, 250, 50]})
        self.assertEqual(r['status'], 'invalid_start'); self.assertEqual(r['frames'], [])
        with patch.object(ArmEnv, 'inverse_kinematics', autospec=True,
                side_effect=ArmEnv.inverse_kinematics) as solver:
            r = tracker.plan({**self.body, 'goal_deg': [0]*6, 'goal_hint_deg': [0]*6})
        self.assertTrue(r['playable']); self.assertTrue(solver.called)

    def test_failed_moving_profile_does_not_fall_back_to_resting_prefix(self):
        shape = {'kind':'smooth_trapezoid', 'rise_percent':5, 'fall_percent':10}
        with patch('simulation.profile_tracking.prepare', side_effect=ValueError('unsafe moving join')), \
                patch('simulation.profile_tracking.prepare_tracking', side_effect=AssertionError('Cannot reset moving boundary')) as prefix:
            r = tracker.plan({**self.body, 'velocity_shape':shape,
                'start_velocity_rad_s':[.01, 0, 0, 0, 0, 0]})
        prefix.assert_not_called()
        self.assertFalse(r['playable']); self.assertEqual(r['frames'], [])
        self.assertEqual(r['status'], 'smoothing_validation_failed')

    def test_changed_checkpoint_and_invalid_inputs_fail_closed(self):
        with patch.object(tracker.classical_service, 'info', return_value={'status': 'unavailable', 'reason': 'changed'}):
            with self.assertRaises(RuntimeError): tracker.plan(self.body)
        for extra in ({'speed_rad_s': True}, {'tolerance_mm': float('nan')},
                      {'continuous_motion': False, 'start_velocity_rad_s': [.01]*6}):
            with self.assertRaises(ValueError): tracker.plan({**self.body, **extra})
        no_scene = dict(self.body); del no_scene['obstacle_mm']
        with self.assertRaises(ValueError): tracker.plan(no_scene)


if __name__ == '__main__': unittest.main()
