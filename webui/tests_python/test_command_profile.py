import unittest
import numpy as np
from webui.command_profile import CommandProfile
from webui.mouse_motion import JointServo
from simulation.motion_limits import envelope, scale_needed


class CommandProfileTest(unittest.TestCase):
    def test_selected_allowance_rejects_commands_below_manufacturer_maximum(self):
        target=[0,0,0,0,0,10]
        self.assertTrue(CommandProfile([0]*6,0.).candidate(target,1.)['accepted'])
        self.assertFalse(CommandProfile([0]*6,0.,np.deg2rad(7.5)).candidate(target,1.)['accepted'])
        for limit in (0.,-1.,float('nan')):
            with self.assertRaises(ValueError):CommandProfile([0]*6,0.,limit)
    def test_actual_rounded_target_and_elapsed_time_are_checked(self):
        monitor = CommandProfile([0]*6, 10.)
        bad = monitor.candidate([10, 0, 0, 0, 0, 0], 10.02)
        self.assertFalse(bad['accepted'])
        with self.assertRaises(RuntimeError): monitor.commit(bad)
        self.assertEqual(monitor.timestamp, 10.)
        safe = monitor.candidate([10, 0, 0, 0, 0, 0], 11.)
        self.assertTrue(safe['accepted'])
        result = monitor.commit(safe)
        self.assertAlmostEqual(result['command_velocity_deg_s'][0], 10.)
        self.assertAlmostEqual(result['command_acceleration_deg_s2'][0], 20.)

    def test_cartesian_and_acceleration_violations_are_rejected_below_joint_speed_limit(self):
        monitor = CommandProfile([0]*6, 0.)
        # Six 15 degree/s joints pass the per-joint cap, but exceed the
        # conservative flange angular/translation envelope.
        self.assertFalse(monitor.candidate([15]*6, 1.)['accepted'])
        monitor.commit(monitor.candidate([10, 0, 0, 0, 0, 0], 1.))
        self.assertFalse(monitor.candidate([9, 0, 0, 0, 0, 0], 1.02)['accepted'])
        self.assertEqual(monitor.timestamp, 1.)

    def test_invalid_timestamps_and_joint_positions_fail_closed(self):
        monitor = CommandProfile([0]*6, 0.)
        for q, t in (([0]*6, 0.), ([0]*6, -1.), ([float('nan')]*6, 1.), ([180]*6, 1.)):
            with self.assertRaises(ValueError): monitor.candidate(q, t)

    def test_classical_servo_limits_combined_flange_motion_and_acceleration(self):
        servo = JointServo([0]*6, 30)
        target = np.array([20, -20, 15, 10, -10, 20.])
        for _ in range(100):
            old = servo.velocity.copy()
            servo.step(servo.reference.copy(), target, .05)
            bounds = envelope(np.deg2rad(np.maximum(np.abs(old), np.abs(servo.velocity))),
                              np.deg2rad(np.abs(servo.velocity-old)/.05))
            self.assertLessEqual(scale_needed(bounds), 1.+1e-7)
        self.assertGreater(np.linalg.norm(servo.reference), 0.)
