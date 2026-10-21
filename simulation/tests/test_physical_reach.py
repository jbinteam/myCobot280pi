"""Physical exclusion must never discard actual FK poses or overlapping disks."""
import unittest
import numpy as np
from simulation.kinematics import ARM
from simulation.physical_reach import ENVELOPE, ReachEnvelope


class PhysicalReachTest(unittest.TestCase):
    def test_envelope_contains_limit_corners_and_random_fk_positions(self):
        corners = np.array(np.meshgrid(*[[lo, hi] for lo, hi in zip(ARM.lower, ARM.upper)])).reshape(6, -1).T
        random = np.random.default_rng(42).uniform(ARM.lower, ARM.upper, (2000, 6))
        for q in np.vstack((corners, random)):
            self.assertTrue(ENVELOPE.point(ARM.fk(q)*1000)['possible'])

    def test_disk_overlap_is_not_decided_by_its_center(self):
        envelope = ReachEnvelope((0., 0., 0.), 10.)
        region = {'center_mm': [12., 0., 0.], 'axis': [0., 0., 1.], 'radius_mm': 3.}
        self.assertFalse(envelope.point(region['center_mm'])['possible'])
        check = envelope.disk(region)
        self.assertTrue(check['possible'])
        self.assertAlmostEqual(check['minimum_distance_mm'], 9.)
        region['radius_mm'] = 1.
        self.assertFalse(envelope.disk(region)['possible'])

    def test_plane_allowance_and_point_allowance_preserve_boundary_goals(self):
        envelope = ReachEnvelope((0., 0., 0.), 10.)
        region = {'center_mm': [0., 0., 11.], 'axis': [0., 0., 1.], 'radius_mm': 3.}
        self.assertFalse(envelope.disk(region)['possible'])
        self.assertTrue(envelope.disk(region, 1.)['possible'])
        self.assertFalse(envelope.point([11., 0., 0.])['possible'])
        self.assertTrue(envelope.point([11., 0., 0.], 1.)['possible'])

    def test_disk_nearest_distance_combines_axial_and_radial_separation(self):
        envelope = ReachEnvelope((0., 0., 0.), 10.)
        region = {'center_mm': [12., 0., 8.], 'axis': [0., 0., 1.], 'radius_mm': 6.}
        check = envelope.disk(region)
        self.assertAlmostEqual(check['minimum_distance_mm'], 10.)
        self.assertTrue(check['possible'])
        region['center_mm'][2] = 9.
        self.assertFalse(envelope.disk(region)['possible'])

    def test_inner_envelope_is_not_a_reachability_certificate(self):
        check = ENVELOPE.point(ENVELOPE.center_mm)
        self.assertTrue(check['possible'])
        self.assertEqual(check['reason'], 'not_excluded_by_outer_bound')

    def test_invalid_inputs_do_not_enter_the_reach_check(self):
        for point in ([float('nan'), 0., 0.], [0., 0.], [0., float('inf'), 0.]):
            with self.assertRaises(ValueError):
                ENVELOPE.point(point)
        for allowance in (-1., float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                ENVELOPE.point([0., 0., 0.], allowance)


if __name__ == '__main__':
    unittest.main()
