"""Exercise the deployed endpoint solver and final path gates with real geometry."""
import unittest
from unittest.mock import patch
import numpy as np
from simulation.kinematics import ARM
from simulation.diffusion import classical_service as service
from simulation.diffusion.env import ArmEnv
from simulation.diffusion.smoothing import motion_bounds


class ClassicalServiceTest(unittest.TestCase):
    def test_outside_workspace_skips_endpoint_ik_and_rrt(self):
        with patch.object(ArmEnv, 'inverse_kinematics', side_effect=AssertionError('Outside workspace')), \
                patch.object(service, 'rrt_plan', side_effect=AssertionError('Outside workspace')):
            result = service.plan({**self.body, 'target_mm': [500., 0., 139.56]})
        self.assertEqual(result['status'], 'outside_workspace')
        self.assertFalse(result['physical_reach_precheck']['possible'])
        self.assertFalse(result['converged'])
        self.assertEqual(result['frames'], [])

    def setUp(self):
        self.start = [0, -40, -130, 80, 0, 50]
        goal = np.deg2rad([3, -42, -125, 78, 2, 50])
        transform = ARM.transform(goal)
        self.body = {'start_deg': self.start, 'target_mm': (transform[:3, 3]*1000).tolist(),
                     'target_axis': transform[:3, 2].tolist(), 'obstacle_mm': None}

    def test_actual_learned_numerical_endpoint_checked_path_and_serving_reproducibility(self):
        with patch.object(ArmEnv, 'inverse_kinematics', autospec=True,
                          side_effect=ArmEnv.inverse_kinematics) as solver:
            result = service.plan(self.body)
        self.assertTrue(solver.call_args.kwargs['learned'])
        self.assertEqual(solver.call_args.kwargs['attempts'], 32)
        self.assertTrue(result['converged'])
        self.assertEqual(result['frames'], service.plan(self.body)['frames'])
        q = np.deg2rad([f['q'] for f in result['frames']])
        times = np.array([f['time'] for f in result['frames']])
        with ArmEnv() as env:
            env.set_obstacle(None)
            self.assertTrue(env.path_valid(q, .01))
        np.testing.assert_allclose(q[0], np.deg2rad(self.start), atol=1e-12)
        self.assertLessEqual(result['final_error_mm'], 1.)
        self.assertLessEqual(result['final_orientation_error_deg'], 1.)
        speed, acceleration = motion_bounds(q, times)
        self.assertLessEqual(speed, 1.+1e-8)
        self.assertLessEqual(acceleration, 3.+1e-8)

    def test_failed_smoothing_never_returns_partial_playback(self):
        with patch('simulation.diffusion.smoothing.prepare', side_effect=ValueError('unsafe timed path')):
            result = service.plan(self.body)
        self.assertEqual(result['status'], 'smoothing_validation_failed')
        self.assertFalse(result['converged']); self.assertEqual(result['frames'], [])

    def test_wrong_numerical_endpoint_is_rejected_before_path_search(self):
        with patch.object(ArmEnv, 'inverse_kinematics', return_value=np.deg2rad(self.start)), \
                patch.object(service, 'rrt_plan', side_effect=AssertionError('Must reject invalid goal')):
            result = service.plan(self.body)
        self.assertFalse(result['converged']); self.assertEqual(result['frames'], [])

    def test_missing_or_changed_evaluated_checkpoint_is_unavailable(self):
        with patch.object(service, 'checkpoint_status', return_value={'status': 'ready', 'checkpoint_sha256': 'changed'}):
            self.assertEqual(service.info()['status'], 'unavailable')
            with self.assertRaises(RuntimeError): service.plan(self.body)

    def test_invalid_scene_and_start_never_reach_solver(self):
        with patch.object(ArmEnv, 'inverse_kinematics', side_effect=AssertionError('Unsafe start')):
            result = service.plan({**self.body, 'obstacle_mm': [0, 0, 15, 250, 250, 50]})
        self.assertEqual(result['status'], 'invalid_start')
        self.assertEqual(result['frames'], [])


if __name__ == '__main__': unittest.main()
