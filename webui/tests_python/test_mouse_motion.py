"""Motion regressions using a robot that actually traverses each fake command."""
import sys
from pathlib import Path
import time
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import RobotController
from test_hardware_ik import TrackingRobot
from webui.mouse_motion import command_path
from simulation import service
from simulation.kinematics import ARM
from simulation.prm import Roadmap, planning_space
from simulation.planning import trajectory


class MovingRobot(TrackingRobot):
    def __init__(self):
        super().__init__()
        self.track = False
        self.command = None
        self.sends = []
        self.settle_reads = 0

    def get_angles(self):
        if self.fail_read:
            raise OSError('link lost')
        if self.command is not None:
            delta = np.asarray(self.command)-self.angles
            distance = max(abs(delta))
            if distance > 1e-8:
                self.angles = (self.angles+delta*min(1, .25/distance)).tolist()
                self.settle_reads = 2
            elif self.settle_reads:
                self.settle_reads -= 1
        return self.angles[:]

    def is_moving(self):
        return int(self.command is not None and (self.settle_reads > 0 or
                   max(abs(a-b) for a,b in zip(self.angles,self.command)) > 1e-8))

    def send_angles(self, angles, speed, **kwargs):
        self.sends.append({'live': self.angles[:], 'target': angles[:], 'moving': bool(self.is_moving())})
        super().send_angles(angles, speed, **kwargs)
        self.command = angles[:]

    def _mesg(self, genre, *args, **kwargs):
        from pymycobot.common import ProtocolCode
        if genre == ProtocolCode.STOP:
            self.command = None
            self.settle_reads = 0
        super()._mesg(genre, *args, **kwargs)


class MouseMotionTest(unittest.TestCase):
    def setUp(self):
        self.robot = MovingRobot()
        self.controller = RobotController(lambda *_: self.robot)
        self.controller.connect('192.168.77.225', 9000)
        self.space = planning_space('self', 1.)
        self.start = np.asarray(self.robot.angles)
        self.session = self.controller.ik.start_mouse_aim({'armed':True, 'frame_verified':True})['session_id']

    def tearDown(self):
        self.controller.shutdown()

    def prepare(self, vertices):
        final = np.deg2rad(vertices[-1])
        transform = ARM.transform(final)
        cursor = (transform[:3,3]*1000+50*transform[:3,2]).tolist()
        # Real collision-checked interpolation, deliberately densely sampled.
        frames = trajectory(np.deg2rad(vertices), transform[:3,3], self.space,
                            .04, .12, target_axis=transform[:3,2])
        result = {'method':'self_checked_prm', 'task':'pose5', 'converged':True,
                  'frames':frames, 'waypoints_deg':np.asarray(vertices).tolist(),
                  'target_mm':(transform[:3,3]*1000).tolist(), 'target_axis':transform[:3,2].tolist(),
                  'final_error_mm':0., 'final_orientation_error_deg':0.,
                  'validation':'joint_limits_and_self_collision', 'cursor_acceptance':{'accepted':True}}
        with patch.object(service, 'load_roadmap', return_value=Roadmap(self.space)), \
             patch.object(service, 'mouse_aim_plan', return_value=result):
            return self.controller.ik.prepare_mouse_aim({'session_id':self.session, 'cursor_mm':cursor})

    def execute(self, data):
        self.controller.ik.execute(data['plan']['id'], 10, True, True)
        self.controller.ik.mouse_aim_heartbeat(self.session, data['plan']['id'])

    def wait(self, state, seconds=4):
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            status = self.controller.snapshot()['ik']
            if status['state'] == state:
                return status
            if status['state'] == 'failed' and state != 'failed':
                self.fail(status['message'])
            time.sleep(.01)
        self.fail(str(self.controller.snapshot()['ik']))

    def test_dense_preview_ticks_do_not_each_become_motor_commands(self):
        goal = self.start.copy(); goal[0] += 9
        data = self.prepare([self.start, goal])
        self.assertGreater(len(data['result']['frames']), 25)
        self.assertEqual(data['plan']['waypoints'], 3)
        np.testing.assert_allclose(self.controller.ik.plan['waypoints'][-1], goal)

    def test_advances_in_motion_on_straight_legs_but_stops_at_the_turn_and_goal(self):
        corner = self.start.copy(); corner[0] += 6
        goal = corner.copy(); goal[5] += 6
        data = self.prepare([self.start, corner, goal])
        self.execute(data)
        status = self.wait('completed')
        self.assertGreater(len(self.robot.sends), 4)
        self.assertTrue(any(send['moving'] for send in self.robot.sends))
        turn = next(send for send in self.robot.sends if abs(send['target'][5]-self.start[5]) > .02)
        self.assertFalse(turn['moving'])
        np.testing.assert_allclose(turn['live'], corner, atol=.02)
        self.assertFalse(self.robot.is_moving())
        self.assertEqual(status['completed_waypoints'], status['total_waypoints'])
        for send in self.robot.sends:
            self.assertLessEqual(max(abs(a-b) for a,b in zip(send['live'],send['target'])), 4+1e-8)
            self.assertTrue(self.space.edge_valid(np.deg2rad(send['live']),np.deg2rad(send['target']),.04))

    def test_stop_during_continued_motion_drops_all_remaining_commands(self):
        goal = self.start.copy(); goal[0] += 15
        self.execute(self.prepare([self.start, goal]))
        deadline = time.monotonic()+2
        while len(self.robot.sends) < 2 and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertGreaterEqual(len(self.robot.sends), 2)
        self.controller.ik.stop_mouse_aim(self.session)
        count = len(self.robot.sends)
        time.sleep(.15)
        self.assertEqual(len(self.robot.sends), count)
        self.assertIsNone(self.controller.snapshot()['motion'])

    def test_read_failure_during_worker_owned_polling_stops(self):
        goal = self.start.copy(); goal[0] += 9
        self.execute(self.prepare([self.start, goal]))
        deadline = time.monotonic()+2
        while not self.robot.sends and time.monotonic() < deadline:
            time.sleep(.01)
        self.robot.fail_read = True
        status = self.wait('failed')
        self.assertIn('link lost', status['message'])
        self.assertTrue(any(call[0]=='stop' for call in self.robot.calls))

    def stopped_with_offset(self, offset):
        original = self.robot.send_angles
        def send(angles, speed, **kwargs):
            original(angles, speed, **kwargs)
            self.robot.angles = (np.asarray(angles)+offset).tolist()
            self.robot.command = self.robot.angles[:]
            self.robot.settle_reads = 0
        self.robot.send_angles = send

    def test_subdegree_joint_deadband_can_complete_without_widening_cartesian_acceptance(self):
        goal = self.start.copy(); goal[5] += 6
        self.stopped_with_offset(np.array([0,0,0,0,0,.75]))
        self.execute(self.prepare([self.start,goal]))
        status = self.wait('completed')
        self.assertLessEqual(max(abs(a-b) for a,b in zip(self.robot.angles,goal)), .02)
        self.assertAlmostEqual(max(abs(a-b) for a,b in zip(self.robot.angles,status['command_deg'])), .75)
        self.assertLess(status['estimated_error_mm'], 1e-6)
        self.assertLess(status['estimated_orientation_error_deg'], 1e-6)

    def test_near_arrival_does_not_bypass_final_five_mm_acceptance(self):
        goal = self.start.copy(); goal[0] += 6
        self.stopped_with_offset(np.full(6,.9))
        self.execute(self.prepare([self.start,goal]))
        status = self.wait('failed')
        self.assertIn('exceeds 5 mm',status['message'])
        self.assertGreater(status['estimated_error_mm'],5)
        self.assertTrue(any(call[0]=='stop' for call in self.robot.calls))

    def test_stalled_command_reports_actual_joint_error_and_target(self):
        goal = self.start.copy(); goal[0] += 3
        data = self.prepare([self.start,goal])
        self.robot.get_angles = lambda: self.start.tolist()
        self.robot.is_moving = lambda: 0
        with patch('webui.hardware_ik.SEGMENT_TIMEOUT', .15):
            self.execute(data)
            status = self.wait('failed')
        self.assertIn('route joint error 3.00°',status['message'])
        np.testing.assert_allclose(status['route_target_deg'],goal)
        self.assertAlmostEqual(status['route_joint_error_deg'],3)
        self.assertLess(status['max_joint_error_deg'],1)
        self.assertEqual(status['samples'][-1]['moving'],False)

    def test_rejects_changed_start_missing_vertices_and_unchecked_shortcuts(self):
        corner = self.start.copy(); corner[0] += 6
        goal = corner.copy(); goal[5] += 6
        with self.assertRaises(ValueError): command_path(self.start, [], self.space)
        with self.assertRaises(ValueError): command_path(self.start, [corner,goal], self.space)
        with patch.object(self.space, 'edge_valid', return_value=False):
            with self.assertRaisesRegex(ValueError,'self-collision'):
                command_path(self.start, [self.start,goal], self.space)
        commands, stops = command_path(self.start,[self.start,corner,goal],self.space)
        np.testing.assert_allclose(commands[1],corner)
        np.testing.assert_allclose(commands[-1],goal)
        self.assertEqual(stops,{1,3})


if __name__ == '__main__':
    unittest.main()
