import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_hardware_ik import TrackingRobot
from server import RobotController, ProtocolCode
from simulation.diffusion.env import ArmEnv
from webui.hardware_jog import prepare


class JogRobot(TrackingRobot):
    def __init__(self,replace=False):
        super().__init__();self.active={};self.replace=replace;self.last=time.monotonic()
    def integrate(self):
        now=time.monotonic();dt=now-self.last;self.last=now
        for j,sign in self.active.items():self.angles[j-1]+=sign*.6*dt
    def get_angles(self):
        self.integrate();return super().get_angles()
    def is_moving(self):return int(bool(self.active))
    def jog_angle(self,joint,direction,speed,**kwargs):
        self.integrate()
        if self.replace:self.active.clear()
        self.active[joint]=1 if direction else -1
        super().jog_angle(joint,direction,speed,**kwargs)
    def _mesg(self,genre,*args,**kwargs):
        if genre==ProtocolCode.STOP:self.integrate();self.active.clear()
        return super()._mesg(genre,*args,**kwargs)


class JogQualificationTest(unittest.TestCase):
    def setUp(self):
        self.robot=JogRobot();self.c=RobotController(lambda *_:self.robot);self.c.connect('192.168.77.225',9000)
        self.valid=patch.object(ArmEnv,'valid',return_value=True);self.valid.start()
        self.execution_gate=patch('webui.hardware_jog.verify_execution_ready');self.execution_gate.start()
    def tearDown(self):self.c.shutdown();self.valid.stop();self.execution_gate.stop()
    def start(self):
        p=self.c.ik.prepare_jog_qualification({})
        self.c.ik.execute(p['plan']['id'],1,True,True,True)
        self.c.ik.heartbeat(p['plan']['id']);return p['plan']['id']
    def terminal(self,id,renew=True):
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            s=self.c.snapshot()['ik']
            if s['state']!='running':return s
            if renew:
                try:self.c.ik.heartbeat(id)
                except ValueError:pass
            time.sleep(.04)
        self.fail('Jog worker did not finish')
    def test_read_only_plan_rejects_overrides_and_requires_scene_confirmation(self):
        p=self.c.ik.prepare_jog_qualification({})
        self.assertEqual(p['plan']['execution_mode'],'jog_qualification')
        self.assertEqual(p['plan']['planned_sdk_speed'],1)
        self.assertEqual(self.robot.calls,[])
        with self.assertRaises(ValueError):prepare(self.robot.angles,{'speed':50})
        with patch.object(ArmEnv,'valid',return_value=False):
            with self.assertRaisesRegex(ValueError,'validation'):prepare(self.robot.angles,{})
        with self.assertRaisesRegex(ValueError,'table'):self.c.ik.execute(p['plan']['id'],1,True,True,False)
        self.assertEqual(self.robot.calls,[])
    def test_independent_jogs_are_observed_and_stopped_without_position_packets(self):
        id=self.start();s=self.terminal(id)
        self.assertEqual(s['state'],'completed',s['message'])
        self.assertEqual(s['qualification']['conclusion'],'both_joints_observed_moving_after_second_jog')
        self.assertEqual([x[0] for x in self.robot.calls],['jog','jog','stop'])
        self.assertEqual([x[1] for x in self.robot.calls[:2]],[4,6])
        self.assertTrue(all('command_velocity_deg_s' not in x for x in s['samples']))
        self.assertFalse(self.robot.active)
    def test_second_packet_replacing_first_is_detected(self):
        self.robot.replace=True;id=self.start();s=self.terminal(id)
        self.assertEqual(s['state'],'completed',s['message'])
        self.assertEqual(s['qualification']['conclusion'],'first_joint_stopped_after_second_jog')
        self.assertFalse(self.robot.active)
    def test_lost_heartbeat_stops_before_first_jog(self):
        id=self.start()
        with self.c.lock:self.c.ik.heartbeat_at-=6
        s=self.terminal(id,renew=False)
        self.assertEqual(s['state'],'failed')
        self.assertEqual([x[0] for x in self.robot.calls],['stop'])
    def test_stop_cancels_active_jog_without_next_packet(self):
        id=self.start();deadline=time.monotonic()+1
        while not self.robot.active and time.monotonic()<deadline:time.sleep(.01)
        self.assertTrue(self.robot.active);self.c.stop();count=len(self.robot.calls);time.sleep(.2)
        self.assertEqual(len(self.robot.calls),count);self.assertFalse(self.robot.active)
        self.assertEqual(self.c.snapshot()['ik']['state'],'stopped')
    def test_excursion_guard_stops_active_jog(self):
        id=self.start();deadline=time.monotonic()+1
        while not self.robot.active and time.monotonic()<deadline:time.sleep(.01)
        with self.c.lock:self.robot.angles[3]+=3
        s=self.terminal(id)
        self.assertEqual(s['state'],'failed');self.assertIn('excursion',s['message'])
        self.assertFalse(self.robot.active)
    def test_production_execution_is_quarantined_before_any_robot_command(self):
        p=self.c.ik.prepare_jog_qualification({});self.execution_gate.stop()
        with self.assertRaisesRegex(ValueError,'disabled'):
            self.c.ik.execute(p['plan']['id'],1,True,True,True)
        self.assertEqual(self.robot.calls,[])
    def test_delayed_health_check_cannot_send_second_jog_after_deadline(self):
        original=self.c._health;checks=0
        def delayed():
            nonlocal checks
            checks+=1
            if checks==3:time.sleep(2.6)
            original()
        with patch.object(self.c,'_health',side_effect=delayed):
            id=self.start();s=self.terminal(id)
        self.assertIn(s['state'],('failed','stopped'))
        self.assertEqual([x[0] for x in self.robot.calls],['jog','stop'])
        self.assertFalse(self.robot.active)
