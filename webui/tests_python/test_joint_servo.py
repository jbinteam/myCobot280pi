import unittest
from unittest.mock import patch
import numpy as np
from webui.mouse_motion import JointServo, rebase_path, StaleMousePlan
from simulation.prm import JointSpace


class JointServoTest(unittest.TestCase):
    def test_velocity_acceleration_and_reversal_are_bounded_for_variable_cycles(self):
        servo = JointServo([0]*6,30)
        previous = np.zeros(6)
        for i in range(150):
            dt = (.05,.08,.12)[i%3]
            goal = np.array([12,6,0,0,0,0] if i<40 else [-8,-4,0,0,0,0])
            before = servo.reference.copy()
            command = servo.step(before,goal,dt)
            self.assertLessEqual(np.max(np.abs(servo.velocity)),20+1e-8)
            self.assertLessEqual(np.max(np.abs(servo.velocity-previous))/dt,60+1e-8)
            np.testing.assert_allclose(command-before,servo.velocity*dt,atol=1e-12)
            previous = servo.velocity.copy()
        self.assertTrue(servo.at_target(goal))

    def test_stalled_plant_accumulates_deadband_but_keeps_a_bounded_reference(self):
        servo = JointServo([0]*6,30)
        live = np.zeros(6)
        for _ in range(150):
            servo.step(live,[30,0,0,0,0,0],.05)
            self.assertLessEqual(np.max(np.abs(servo.reference-live)),4)
        self.assertGreater(servo.reference[0],.5)
        self.assertLessEqual(servo.reference[0],3+1e-6)
        self.assertLess(abs(servo.velocity[0]),.1)
        with self.assertRaisesRegex(RuntimeError,'drift'):
            servo.step([-10,0,0,0,0,0],[30,0,0,0,0,0],.05)

    def test_invalid_measurements_and_stale_time_steps_are_rejected(self):
        servo = JointServo([0]*6)
        for live,dt in (([0]*5,.05),([float('nan')]*6,.05),([0]*6,0),([0]*6,.5)):
            with self.assertRaises(ValueError): servo.step(live,[0]*6,dt)

    def test_rebase_continues_along_the_route_and_retains_future_turns(self):
        start=np.array([0,-40,-130,80,0,50.]); corner=start.copy();corner[0]+=6
        goal=corner.copy();goal[5]+=6
        reference=start.copy();reference[0]+=2;live=reference.copy();live[0]-=.5
        targets,stops,distance=rebase_path(reference,live,[start,corner,goal],JointSpace())
        np.testing.assert_allclose(targets,[corner,goal])
        self.assertEqual(stops,{0,1});self.assertAlmostEqual(distance,0)
        far=reference.copy();far[1]+=3
        with self.assertRaises(StaleMousePlan): rebase_path(far,far,[start,corner,goal],JointSpace())
        space=JointSpace()
        with patch.object(space,'edge_valid',return_value=False):
            with self.assertRaises(StaleMousePlan): rebase_path(reference,live,[start,corner,goal],space)


if __name__=='__main__': unittest.main()
