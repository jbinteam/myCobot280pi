import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from simulation.kinematics import ARM
from simulation.diffusion.visibility import CameraModel, VisibilityEnv
from simulation.camera_model import configured_camera


class CameraTest(unittest.TestCase):
    def test_configured_orbbec_color_fov_reaches_the_rectangular_visibility_gate(self):
        camera = configured_camera()
        self.assertEqual((camera.horizontal_fov_deg,camera.vertical_fov_deg),(82.,68.))
        self.assertTrue(camera.project([np.tan(np.deg2rad(40.)),0.,1.])['in_view'])
        self.assertFalse(camera.project([np.tan(np.deg2rad(42.)),0.,1.])['in_view'])
        self.assertTrue(camera.project([0.,np.tan(np.deg2rad(33.)),1.])['in_view'])
        self.assertFalse(camera.project([0.,np.tan(np.deg2rad(35.)),1.])['in_view'])
        alternate = CameraModel.from_ros_settings({'horizontal_fov_deg':94.,'vertical_fov_deg':62.,'angular_margin_deg':2.,'near_m':.01})
        self.assertEqual((alternate.horizontal_fov_deg,alternate.vertical_fov_deg),(94.,62.))
        self.assertFalse(alternate.project([np.tan(np.deg2rad(46.)),0.,1.])['in_view'])

    def test_mount_composes_flange_translation_and_rotation(self):
        camera = CameraModel(eef_to_optical_translation_m=[.01,0.,0.],
            eef_to_optical_quaternion_xyzw=[0.,np.sqrt(.5),0.,np.sqrt(.5)])
        flange = np.array([[0.,-1.,0.,.1],[1.,0.,0.,.2],[0.,0.,1.,.3],[0.,0.,0.,1.]])
        optical = camera.optical_transform(flange)
        np.testing.assert_allclose(optical[:3,3], [.1,.21,.3], atol=1e-12)
        np.testing.assert_allclose(optical[:3,2], [0.,1.,0.], atol=1e-12)
        self.assertEqual(CameraModel(**camera.configuration()), camera)

    def test_mount_validation_and_input_copy(self):
        translation = [.01,0.,0.]
        camera = CameraModel(eef_to_optical_translation_m=translation)
        translation[0] = 5.
        self.assertEqual(camera.eef_to_optical_translation_m, (.01,0.,0.))
        for kwargs in ({'eef_to_optical_translation_m':[0.,0.]},
                {'eef_to_optical_translation_m':[0.,0.,float('nan')]},
                {'eef_to_optical_quaternion_xyzw':[0.,0.,0.,2.]},
                {'mount_assumed':True,'mount_calibrated':True}):
            with self.assertRaises(ValueError): CameraModel(**kwargs)

    def test_front_behind_near_and_rectangular_view(self):
        camera = CameraModel(horizontal_fov_deg=90, vertical_fov_deg=60)
        self.assertTrue(camera.project([0., 0., 1.])['in_view'])
        self.assertFalse(camera.project([0., 0., -1.])['in_view'])
        self.assertFalse(camera.project([0., 0., .004])['in_view'])
        self.assertTrue(camera.project([.9, 0., 1.])['in_view'])
        self.assertFalse(camera.project([0., .9, 1.])['in_view'])

    def test_margin_rejects_edge_and_invalid_parameters(self):
        point = [np.tan(np.deg2rad(39.)), 0., 1.]
        self.assertTrue(CameraModel().project(point)['in_view'])
        self.assertFalse(CameraModel(angular_margin_deg=2).project(point)['in_view'])
        for kwargs in ({'near_m': 0}, {'angular_margin_deg': 40}, {'horizontal_fov_deg': 180}):
            with self.assertRaises(ValueError):
                CameraModel(**kwargs)


class VisibilityIntegrationTest(unittest.TestCase):
    def test_projection_and_occlusion_ray_use_optical_mount(self):
        q = np.deg2rad([1.,-40.,-130.,-40.,1.,50.])
        camera = CameraModel(eef_to_optical_translation_m=(.02,0.,.04),
            eef_to_optical_quaternion_xyzw=(0.,np.sqrt(.5),0.,np.sqrt(.5)))
        optical = camera.optical_transform(ARM.transform(q))
        obj = optical[:3,3]+.2*optical[:3,2]
        with VisibilityEnv(obj, camera=camera) as env:
            env.set_obstacle(None); env.reset(q)
            with patch('simulation.diffusion.visibility.p.rayTest', return_value=[(-1,-1)]) as ray:
                self.assertTrue(env.inspect(q)['visible'])
            np.testing.assert_allclose(ray.call_args.args[0], optical[:3,3]+.005*optical[:3,2], atol=1e-12)
            np.testing.assert_allclose(ray.call_args.args[1], obj, atol=1e-12)
            np.testing.assert_array_equal(env.current(), q)
            self.assertEqual(env.visibility_metadata()['camera'], camera.configuration())
            self.assertFalse(env.visibility_metadata()['calibrated'])

    def test_scene_obstruction_is_detected_and_removal_invalidates_cache(self):
        fixture = Path(__file__).resolve().parents[1]/'scenarios/reachable_failures.json'
        case = json.loads(fixture.read_text())['cases'][1]
        goal = np.array(case['goal_witness_rad'])
        t = ARM.transform(goal)
        with VisibilityEnv(case['object_m']) as env:
            env.set_obstacle(None); env.reset(goal)
            self.assertTrue(env.inspect(goal)['visible'])
            blocker = np.r_[t[:3, 3]+.075*t[:3, 2], [.01, .01, .01]]
            env.set_obstacle(blocker)
            result = env.inspect(goal)
            self.assertFalse(result['visible'])
            self.assertTrue(result['occluded'])
            self.assertEqual(result['occluding_body'], env.box)
            np.testing.assert_array_equal(env.current(), goal)
            env.set_obstacle(None)
            self.assertTrue(env.inspect(goal)['visible'])

    def test_edge_checks_interior_and_target_changes_invalidate_cache(self):
        fixture = Path(__file__).resolve().parents[1]/'scenarios/reachable_failures.json'
        case = json.loads(fixture.read_text())['cases'][1]
        start, goal = np.array(case['start_rad']), np.array(case['goal_witness_rad'])
        with VisibilityEnv(case['object_m']) as env:
            env.set_obstacle(None); env.reset(start)
            self.assertTrue(env.edge_valid(start, goal, .01))
            # Collision endpoints and edges were cached above. A visibility
            # obstruction only in the middle must still reject the edge.
            def synthetic_occlusion(q):
                fraction = (q[0]-start[0])/(goal[0]-start[0])
                return {'visible': not .4 < fraction < .6}
            with patch.object(env, 'inspect', side_effect=synthetic_occlusion):
                self.assertTrue(env.valid(start)); self.assertTrue(env.valid(goal))
                self.assertFalse(env.edge_valid(start, goal, .01))
            np.testing.assert_array_equal(env.current(), start)
            t = ARM.transform(start)
            env.set_object(t[:3, 3] - .15*t[:3, 2])
            self.assertFalse(env.valid(start))
            env.set_object(case['object_m'])
            self.assertTrue(env.valid(start))


if __name__ == '__main__':
    unittest.main()
