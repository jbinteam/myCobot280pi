import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_hardware_ik import TrackingRobot
from server import RobotController
from webui.hardware_diagnostic import prepare
from simulation.diffusion.env import ArmEnv


class JointDiagnosticTest(unittest.TestCase):
    def setUp(self):
        self.robot=TrackingRobot();self.c=RobotController(lambda *_:self.robot)
        self.c.connect('192.168.77.225',9000)
        self.start=self.robot.angles[:]
    def tearDown(self):self.c.shutdown()
    def body(self,variant='single_target'):
        return {'joint':4,'target_deg':self.start[3]+2,'variant':variant,'speed':70}
    def test_plan_checks_scene_and_changes_only_one_joint_without_motor_commands(self):
        with patch.object(ArmEnv,'path_valid',autospec=True,return_value=True) as checked:
            for variant,count in [('single_target',1),('small_waypoints',100)]:
                data=self.c.ik.prepare_joint_diagnostic(self.body(variant))
                self.assertTrue(data['plan']['hardware_joint_diagnostic'])
                self.assertFalse(data['plan']['hardware_diffusion'])
                p=self.c.ik.plan
                self.assertEqual(len(p['waypoints']),count)
                self.assertEqual(p['stop_indices'],{count-1})
                points=np.array(p['waypoints'])
                np.testing.assert_allclose(np.delete(points,3,axis=1),np.tile(np.delete(self.start,3),(count,1)))
                self.assertTrue((np.diff(points[:,3])>0).all())
                if count>1:self.assertLessEqual(np.diff(points[:,3]).max(),.02000001)
                self.assertEqual(p['waypoint_intervals_s'],[.3]*count)
                self.assertAlmostEqual(data['plan']['max_step_deg'],2. if count==1 else .02)
            self.assertEqual(checked.call_count,2)
        self.assertEqual(self.robot.calls,[])
    def test_rejects_large_moves_bad_settings_and_collisions(self):
        for body in ({**self.body(),'target_deg':self.start[3]+3.1},
                     {**self.body(),'joint':True},{**self.body(),'variant':'unknown'},
                     {**self.body(),'speed':71},{**self.body(),'target_deg':float('nan')},
                     {**self.body(),'start_deg':[0]*6}):
            with self.assertRaises(ValueError):prepare(self.start,body)
        with patch.object(ArmEnv,'path_valid',return_value=False):
            with self.assertRaisesRegex(ValueError,'validation'):self.c.ik.prepare_joint_diagnostic(self.body())
        self.assertEqual(self.robot.calls,[])
    def test_execution_keeps_heartbeat_scene_and_shared_controller_ownership(self):
        with patch.object(ArmEnv,'path_valid',return_value=True),patch.object(ArmEnv,'valid',return_value=True),patch.object(ArmEnv,'edge_valid',return_value=True):
            data=self.c.ik.prepare_joint_diagnostic(self.body());id=data['plan']['id']
            with self.assertRaisesRegex(ValueError,'table'):self.c.ik.execute(id,70,True,True,False)
            self.c.ik.execute(id,70,True,True,True)
            time.sleep(.35);self.assertEqual(self.robot.calls,[])
            self.c.ik.heartbeat(id)
            deadline=time.monotonic()+4
            while time.monotonic()<deadline:
                state=self.c.snapshot()['ik']
                if state['state'] in ('completed','failed'):break
                self.c.ik.heartbeat(id);time.sleep(.05)
            self.assertEqual(state['state'],'completed',state.get('message'))
            self.assertEqual(len([x for x in self.robot.calls if x[0]=='move']),1)
            self.assertTrue(state['hardware_joint_diagnostic'])
            self.assertGreaterEqual(state['samples'][-1]['time']-state['samples'][0]['time'],2.)
