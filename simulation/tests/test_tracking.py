import unittest
from unittest.mock import patch
import numpy as np
from simulation import pose5, tracking
from simulation.kinematics import ARM
from simulation.planning import IKConfig
from simulation.prm import JointSpace


class LocalTrackingPathTest(unittest.TestCase):
    def setUp(self):
        self.start = np.deg2rad([0., -40., -130., 80., 0., 50.])
        self.goal = self.start.copy(); self.goal[0] += .12
        self.pose = ARM.transform(self.goal)
        self.config = IKConfig(task='pose5', nn='none', pose5_method='dls')
        # Endpoints are valid, but the direct edge crosses a modeled obstacle.
        def validity(q):
            x, y = q[:2]-self.start[:2]
            return not (.035 < x < .085 and y < .04)
        self.space = JointSpace(validity, {'checker': 'tracking-fixture', 'environment': 'unit-test'})

    def run_path(self, path, **extra):
        solved = {'converged': True, 'frames': [{'q': np.rad2deg(q).tolist()} for q in path],
                  'elapsed_ms': 0., 'iterations': len(path)-1, 'stop_reason': 'target_reached',
                  'final_error_mm': 0., 'final_orientation_error_deg': 0.}
        with patch.object(pose5, 'solve', return_value=solved):
            return tracking.local_pose_plan(self.start, self.pose[:3, 3], self.pose[:3, 2],
                                            self.space, self.config, **extra)

    def test_valid_endpoints_cannot_hide_an_invalid_intermediate_state(self):
        bad = self.start.copy(); bad[0] += .06
        result = self.run_path([self.start, bad, self.goal])
        self.assertFalse(result['converged'])
        self.assertEqual(result['local_ik']['status'], 'path_invalid')
        self.assertNotIn('frames', result)

    def test_valid_endpoints_cannot_hide_an_invalid_segment(self):
        self.assertTrue(self.space.valid(self.start))
        self.assertTrue(self.space.valid(self.goal))
        result = self.run_path([self.start, self.goal])
        self.assertFalse(result['converged'])
        self.assertEqual(result['local_ik']['status'], 'path_invalid')
        self.assertIn('segment', result['local_ik']['reason'])

    def test_simplification_preserves_safe_turns_and_rechecks_playback(self):
        a = self.start.copy(); a[1] += .1
        b = self.goal.copy(); b[1] += .1
        dense = [self.start]
        for left, right in zip([self.start, a, b], [a, b, self.goal]):
            dense.extend(left+(right-left)*fraction for fraction in np.linspace(.1, 1., 10))
        result = self.run_path(dense)
        self.assertTrue(result['converged'])
        vertices = np.deg2rad(result['waypoints_deg'])
        self.assertGreater(len(vertices), 2)
        self.assertLess(len(vertices), len(dense))
        np.testing.assert_allclose(vertices[-1], self.goal)
        frames = result['frames']
        for left, right in zip(frames, frames[1:]):
            qa, qb = np.deg2rad(left['q']), np.deg2rad(right['q'])
            self.assertTrue(self.space.edge_valid(qa, qb, .04))
            self.assertLessEqual(np.linalg.norm(qb-qa), self.config.speed_rad_s*(right['time']-left['time'])+1e-9)

    def test_joint_limit_violation_is_rejected_before_simplification(self):
        result = self.run_path([self.start, ARM.upper+.01, self.goal])
        self.assertFalse(result['converged'])
        self.assertEqual(result['local_ik']['status'], 'path_invalid')

    def test_cursor_acceptance_is_checked_even_when_solver_claims_success(self):
        self.space = JointSpace()
        result = self.run_path([self.start, self.goal], goal_constraint=lambda q: False)
        self.assertFalse(result['converged'])
        self.assertEqual(result['local_ik']['status'], 'goal_rejected')

    def test_a_different_start_cannot_be_admitted_as_local_motion(self):
        result = self.run_path([self.goal])
        self.assertFalse(result['converged'])
        self.assertEqual(result['local_ik']['status'], 'invalid_path')


if __name__ == '__main__': unittest.main()
