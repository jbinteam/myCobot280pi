"""Live-derived target selection reads fake hardware and sends no motion."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import RobotController
from test_hardware_ik import TrackingRobot
from webui.small_targets import PRESETS, STAGED_PRESETS
from simulation.kinematics import ARM
from simulation.pose5 import axis_from_xy


class SmallTargetTest(unittest.TestCase):
    def setUp(self):
        self.robot = TrackingRobot()
        self.controller = RobotController(lambda *_: self.robot)
        self.controller.connect('192.168.77.225', 9000)

    def tearDown(self):
        self.controller.shutdown()

    def test_all_presets_fill_corresponding_pose_without_motion_or_plan(self):
        for preset, (joint, delta) in PRESETS.items():
            with self.subTest(preset=preset):
                data = self.controller.ik.small_target({'preset':preset})
                q = np.array(self.robot.angles, dtype=float); q[joint] += delta
                pose = ARM.transform(np.deg2rad(q))
                np.testing.assert_allclose(data['target_mm'], pose[:3,3]*1000, atol=.00051)
                np.testing.assert_allclose(axis_from_xy(data['orientation_xy_deg']), pose[:3,2], atol=2e-6)
                self.assertEqual(data['eef_offset_mm'], [0,0,0])
                self.assertTrue(data['requires_planning'])
                self.assertFalse(data['motion_sent'])
                self.assertEqual(data['maximum_speed_percent'], 5)
                self.assertEqual(data['staged'], preset in STAGED_PRESETS)
                self.assertEqual(data['recommended_joint_command'], 'encoder_drag_staged' if preset in STAGED_PRESETS else None)
        self.assertIsNone(self.controller.ik.plan)
        self.assertEqual(self.robot.calls, [])

    def test_repeated_selection_uses_fresh_live_joints(self):
        first = self.controller.ik.small_target({'preset':'j4_minus_2'})
        self.robot.angles[3] += 1.
        second = self.controller.ik.small_target({'preset':'j4_minus_2'})
        self.assertNotEqual(first['target_mm'], second['target_mm'])
        self.assertEqual(second['start_deg'], self.robot.angles)
        self.assertEqual(self.robot.calls, [])

    def test_only_listed_presets_are_accepted(self):
        for body in ({}, {'preset':'other'}, {'preset':[]},
                     {'preset':'j4_minus_2','angles':[0]*6}):
            with self.subTest(body=body), self.assertRaises(ValueError):
                self.controller.ik.small_target(body)
        self.assertEqual(self.robot.calls, [])

    def test_moving_and_mouse_sessions_refused(self):
        self.robot.is_moving = lambda: 1
        with self.assertRaisesRegex(RuntimeError, 'Stop the arm'):
            self.controller.ik.small_target({'preset':'j4_minus_2'})
        self.robot.is_moving = lambda: 0
        self.controller.ik.mouse_session = 'active'
        with self.assertRaises(RuntimeError):
            self.controller.ik.small_target({'preset':'j4_minus_2'})
        self.assertEqual(self.robot.calls, [])

    def test_joint_limit_overflow_refused(self):
        self.robot.angles[3] = float(np.rad2deg(ARM.upper[3])-.25)
        with self.assertRaisesRegex(ValueError, 'joint limit'):
            self.controller.ik.small_target({'preset':'j4_plus_2'})
        self.assertEqual(self.robot.calls, [])

    def test_disconnected_selection_refused(self):
        self.controller.disconnect()
        with self.assertRaises(RuntimeError):
            self.controller.ik.small_target({'preset':'j4_minus_2'})


if __name__ == '__main__':
    unittest.main()
