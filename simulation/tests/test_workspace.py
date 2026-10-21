import unittest
import numpy as np
from simulation.workspace import geometric_estimate as estimate, transforms
from simulation.kinematics import ARM


class WorkspaceTests(unittest.TestCase):
    def test_batch_matches_robot_fk(self):
        q = np.random.default_rng(8).uniform(ARM.lower, ARM.upper, (12, 6))
        t, axes = transforms(q, True)
        for i, angles in enumerate(q):
            expected, jacobian = ARM.spatial_jacobian(angles)
            np.testing.assert_allclose(t[i], expected, atol=1e-12)
            np.testing.assert_allclose(axes[i], jacobian[3:], atol=1e-12)

    def test_direction_and_tolerance_envelope(self):
        result = estimate({'samples':200, 'orientation_xy_deg':[0, 0], 'orientation_tolerance_deg':1})
        self.assertGreater(result['accepted'], 0)
        self.assertLessEqual(result['max_direction_error_deg'], 1)
        cells = set(map(tuple, result['voxels']))
        for p in result['points_mm']:
            self.assertIn(tuple(np.floor(np.array(p)/result['voxel_mm']).astype(int)), cells)
        wider = estimate({'samples':200, 'orientation_xy_deg':[0, 0], 'tolerance_mm':10})
        self.assertTrue(cells.issubset(set(map(tuple, wider['voxels']))))
        self.assertEqual(result['points_mm'], wider['points_mm'])

    def test_position_and_validation(self):
        result = estimate({'samples':100, 'task':'position'})
        self.assertEqual(result['accepted'], 100)
        for body in [{'samples':100000}, {'voxel_mm':0}, {'tolerance_mm':float('nan')}, {'orientation_xy_deg':[None, 0]}, {'task':'bad'}]:
            with self.assertRaises(ValueError): estimate(body)


class SolvableWorkspaceTests(unittest.TestCase):
    def test_successful_targets_match_production_solver(self):
        from simulation.workspace import estimate as solvable
        from simulation.service import run
        body = {'samples':100, 'targets':8, 'start_deg':[0,-40,-130,80,0,50],
                'task':'pose5', 'orientation_xy_deg':[0,180], 'method':'bounded_dls'}
        result = solvable(body)
        self.assertEqual(result['tested'], len(result['outcomes']))
        self.assertGreater(result['accepted'], 0)
        for point in result['points_mm']:
            checked = run({**body, 'target_mm':point, 'methods':['bounded_dls']})['results'][0]
            self.assertTrue(checked['converged'])
            self.assertLessEqual(checked['final_error_mm'], 1)
            self.assertLessEqual(checked['final_orientation_error_deg'], 1)

    def test_same_target_depends_on_start(self):
        from unittest.mock import patch
        from simulation.workspace import estimate as solvable
        near = [0,-40,-130,80,0,50]
        far = [60,0,0,0,0,0]
        target = (ARM.fk(np.deg2rad(near))*1000).tolist()
        pool = estimate({'samples':100, 'task':'position'})
        pool['points_mm'] = [target]
        body = {'task':'position', 'samples':100, 'targets':2, 'steps':1, 'method':'dls'}
        with patch('simulation.workspace.geometric_estimate', return_value=pool.copy()):
            a = solvable({**body, 'start_deg':near})
        with patch('simulation.workspace.geometric_estimate', return_value=pool.copy()):
            b = solvable({**body, 'start_deg':far})
        self.assertEqual(a['outcomes'][-1]['target_mm'], b['outcomes'][-1]['target_mm'])
        self.assertTrue(a['outcomes'][-1]['converged'])
        self.assertFalse(b['outcomes'][-1]['converged'])

    def test_default_task_is_used_for_solver_rollouts(self):
        from simulation.workspace import estimate as solvable
        from simulation.pose5 import xy_from_axis
        start = [0,-40,-130,80,0,50]
        tilt = xy_from_axis(ARM.transform(np.deg2rad(start))[:3,2])
        result = solvable({'start_deg':start, 'orientation_xy_deg':tilt, 'samples':100, 'targets':1})
        self.assertEqual(result['task'], 'pose5')
        self.assertEqual(result['accepted'], 1)
        self.assertIsNotNone(result['outcomes'][0]['final_orientation_error_deg'])

    def test_start_and_solver_validation(self):
        from simulation.workspace import estimate as solvable
        for body in [{}, {'start_deg':[999]*6}, {'start_deg':[0]*6, 'targets':513},
                     {'start_deg':[0]*6, 'method':'seed_dls', 'task':'pose5'}]:
            with self.assertRaises(ValueError): solvable(body)


if __name__ == '__main__': unittest.main()
