import unittest
from unittest.mock import patch
import numpy as np
from simulation.kinematics import ARM
from webui.server import RobotController
from test_encoder_drag import EncoderRobot


class HardwareClassicalTest(unittest.TestCase):
    def setUp(self):
        self.robot = EncoderRobot()
        # Nonzero joints make the existing encoder direction calibration observable.
        self.robot.angles[0] = 5.
        self.robot.angles[4] = 5.
        self.controller = RobotController(lambda *_: self.robot)
        self.controller.connect('192.168.77.225', 9000)
        q = np.deg2rad(self.robot.angles)
        q[0] += .02
        transform = ARM.transform(q)
        self.body = {'target_mm': (transform[:3, 3]*1000).tolist(),
            'target_axis': transform[:3, 2].tolist(), 'planner': 'learned_rrt',
            'joint_command': 'send_angles', 'speed': 5}
        self.addCleanup(self.controller.shutdown)

    def test_live_start_prepare_sends_no_motion_and_requires_scene_review(self):
        data = self.controller.ik.prepare_diffusion(self.body)
        self.assertTrue(data['result']['converged'])
        self.assertEqual(data['result']['method'], 'learned_rrt')
        self.assertEqual(data['plan']['start_deg'], self.robot.angles)
        self.assertEqual(self.robot.calls, [])
        with self.assertRaisesRegex(ValueError, 'Verify the table'):
            self.controller.ik.execute(data['plan']['id'], 5, True, True, False)
        self.assertEqual(self.robot.calls, [])

    def test_changed_ik_checkpoint_after_review_refuses_execution(self):
        data = self.controller.ik.prepare_diffusion(self.body)
        with patch('simulation.diffusion.classical_service.info', return_value={'status': 'unavailable'}):
            with self.assertRaisesRegex(ValueError, 'deployment changed'):
                self.controller.ik.execute(data['plan']['id'], 5, True, True, True)
        self.assertEqual(self.robot.calls, [])

    def test_classical_path_prepares_the_default_encoder_profile_and_ramps_without_motion(self):
        data = self.controller.ik.prepare_diffusion({**self.body,
            'joint_command': 'encoder_drag_profile', 'speed': 70,
            'velocity_shape': {'kind': 'smooth_trapezoid', 'rise_percent': 5, 'fall_percent': 10}})
        self.assertTrue(data['result']['converged'])
        self.assertEqual(data['plan']['execution_mode'], 'encoder_profile')
        self.assertEqual(data['result']['planner'], 'learned_rrt')
        self.assertEqual(data['result']['smoothing']['reference_shape']['rise_percent'], 5)
        self.assertEqual(self.robot.calls, [])


if __name__ == '__main__': unittest.main()
