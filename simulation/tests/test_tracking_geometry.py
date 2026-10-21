import unittest
import numpy as np
from simulation.kinematics import ARM
from simulation.tracking_geometry import region_for, position_metrics, candidate_poses, gaze_score, annotate_gaze


class TrackingGeometryTest(unittest.TestCase):
    def setUp(self):
        self.region = region_for([0, 0, 100], [0, 0, 1], {})

    def test_approach_distance_band_keeps_radial_and_pointing_checks(self):
        region=region_for([0,0,350],[0,0,1],{'distance_mm':150.,'half_angle_deg':30.})
        for axial in (-10.,0.,10.):
            self.assertTrue(position_metrics([0,0,200+axial],[0,0,1],region,10.,3.)['accepted'])
        for axial in (-10.01,10.01):
            self.assertFalse(position_metrics([0,0,200+axial],[0,0,1],region,10.,3.)['accepted'])
        point=np.array([region['radius_mm']+.01,0,200])
        self.assertFalse(position_metrics(point,np.array(region['cursor_mm'])-point,region,10.,3.)['accepted'])
        self.assertFalse(position_metrics([0,0,200],[0,0,-1],region,10.,3.)['accepted'])

    def test_default_and_derived_values_are_server_owned(self):
        self.assertEqual(self.region['distance_mm'], 100.)
        self.assertAlmostEqual(self.region['radius_mm'], 36.39702342662024)
        np.testing.assert_allclose(self.region['center_mm'], [0, 0, 0])
        for extra in ({'radius_mm': 1000}, {'center_mm': [0, 0, 0]},
                      {'half_angle_deg': True}, {'distance_mm': 0}, {'mode': 'sphere'}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                region_for([0, 0, 100], [0, 0, 1], extra)
        self.assertIsNone(region_for([0, 0, 100], [0, 0, 1], None))

    def test_off_center_point_with_longer_slant_distance_is_accepted(self):
        point = np.array([16., 0., 0.]); aimed = np.array(self.region['cursor_mm'])-point
        metrics = position_metrics(point, aimed, self.region, 1., 1.)
        self.assertTrue(metrics['accepted'])
        self.assertGreater(metrics['distance_mm'], 100.)
        self.assertEqual(metrics['axial_error_mm'], 0.)
        self.assertEqual(metrics['radial_distance_mm'], 16.)

    def test_oblique_disk_is_perpendicular_to_object_to_camera_and_not_cone_volume(self):
        obj=np.array([160.,-80.,380.]);camera=np.array([-20.,130.,210.])
        normal=(camera-obj)/np.linalg.norm(camera-obj)
        region=region_for(obj,-normal,{'distance_mm':350.,'half_angle_deg':30.})
        center=obj+350.*normal
        np.testing.assert_allclose(region['center_mm'],center)
        tangent=np.cross(normal,[0.,0.,1.]);tangent/=np.linalg.norm(tangent)
        point=center+.6*region['radius_mm']*tangent
        self.assertAlmostEqual(float((point-center)@normal),0.)
        self.assertTrue(position_metrics(point,obj-point,region,1.,1.)['accepted'])
        # A point along the cone axis can be inside the cone but off the plane.
        off_plane=center-50.*normal
        self.assertFalse(position_metrics(off_plane,obj-off_plane,region,1.,1.)['accepted'])

    def test_axial_radial_and_pointing_gates_are_independent(self):
        for point, direction in [([16, 0, 2], [-16, 0, 98]),
                                 ([38, 0, 0], [-38, 0, 100]),
                                 ([16, 0, 0], [0, 0, 1]),
                                 ([0, 0, 0], [0, 0, -1])]:
            self.assertFalse(position_metrics(point, direction, self.region, 1., 1.)['accepted'])

    def test_axial_tolerance_accepts_near_plane_without_exact_projection(self):
        for axial, expected in [(.5, True), (1.1, False)]:
            point = np.array([16., 0., axial])
            direction = np.array(self.region['cursor_mm'])-point
            metrics = position_metrics(point, direction, self.region, 1., 1.)
            self.assertEqual(metrics['accepted'], expected)
            self.assertAlmostEqual(metrics['axial_error_mm'], axial)

    def test_candidate_planes_and_pointing_remain_frozen(self):
        poses = list(candidate_poses(self.region, np.deg2rad([0, -40, -130, 80, 0, 50])))
        self.assertGreater(len(poses), 8)
        for target, direction in poses:
            self.assertTrue(position_metrics(target*1000, direction, self.region, .001, .001)['accepted'])
            self.assertAlmostEqual(target[2], 0.)

    def off_center(self):
        q = np.deg2rad([0, -40, -130, 80, 0, 50])
        pose = ARM.transform(q)
        axis = np.cos(np.deg2rad(8))*pose[:3, 2]+np.sin(np.deg2rad(8))*pose[:3, 0]
        cursor = pose[:3, 3]*1000+100/np.cos(np.deg2rad(8))*pose[:3, 2]
        return q, region_for(cursor, axis, {})

    def test_gaze_ranking_uses_cursor_line_of_sight_and_full_chunk(self):
        q, region = self.off_center()
        wrong = q.copy(); wrong[4] += .15
        # A route that looks away in its middle loses to a stationary gaze,
        # even though both terminal states have the same perfect pointing.
        self.assertLess(gaze_score([q, q], region), gaze_score([q, wrong, q], region))
        self.assertLess(gaze_score([q, q], region), gaze_score([q, wrong], region))
        # Free flange-Z rotation has no gaze benefit; travel breaks the tie.
        spin = q.copy(); spin[5] += .15
        self.assertLess(gaze_score([q, q], region), gaze_score([q, spin], region))

    def test_gaze_diagnostics_do_not_confuse_disk_axis_with_cursor_aim(self):
        q, region = self.off_center()
        pose = ARM.transform(q)
        frame = {'position': (pose[:3, 3]*1000).tolist(), 'tool_axis': pose[:3, 2].tolist()}
        result = annotate_gaze([frame], region['cursor_mm'], region)
        self.assertLess(result['final_direction_error_deg'], 1e-10)
        self.assertGreater(frame['cursor_radial_distance_mm'], 10.)
        self.assertEqual(result['free_rotation'], 'local flange Z')
        self.assertIsNone(annotate_gaze([], region['cursor_mm'], region))

    def test_gaze_undefined_at_object_is_not_reported_as_perfect(self):
        frame = {'position': [0., 0., 0.], 'tool_axis': [0., 0., 1.]}
        result = annotate_gaze([frame], [0., 0., 0.])
        self.assertEqual(result['final_direction_error_deg'], 180.)


if __name__ == '__main__':
    unittest.main()
