import sys
from pathlib import Path
import time
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from server import RobotController
import test_mouse_motion as motion
from simulation import service
from simulation.kinematics import ARM
from simulation.prm import Roadmap,planning_space
from simulation.planning import trajectory
from webui.mouse_motion import StaleMousePlan


class VelocityTrackingTest(unittest.TestCase):
    prepare=motion.MouseMotionTest.prepare
    execute=motion.MouseMotionTest.execute
    wait=motion.MouseMotionTest.wait

    def setUp(self):
        self.robot=motion.MovingRobot();self.robot.fresh_mode=0
        self.controller=RobotController(lambda *_:self.robot)
        self.controller.connect('192.168.77.225',9000)
        self.space=planning_space('self',1.)
        self.start=np.asarray(self.robot.angles)
        self.session=self.controller.ik.start_mouse_aim({'armed':True,'frame_verified':True})['session_id']

    def tearDown(self):self.controller.shutdown()

    def wait_moves(self,count=1):
        deadline=time.monotonic()+3
        while len(self.robot.sends)<count and time.monotonic()<deadline:time.sleep(.01)
        self.assertGreaterEqual(len(self.robot.sends),count)

    def long_route(self):
        goal=self.start.copy();goal[0]+=15
        return self.prepare([self.start,goal])

    def live_plan(self,goal,delay=0):
        transform=ARM.transform(np.deg2rad(goal));cursor=(transform[:3,3]*1000+50*transform[:3,2]).tolist()
        def solve(body,*args,**kwargs):
            start=body['start_deg']
            time.sleep(delay)
            frames=trajectory(np.deg2rad([start,goal]),transform[:3,3],self.space,.04,.3,target_axis=transform[:3,2])
            return {'method':'self_checked_prm','task':'pose5','converged':True,'frames':frames,
                    'waypoints_deg':[start,goal.tolist()],'target_mm':(transform[:3,3]*1000).tolist(),
                    'target_axis':transform[:3,2].tolist(),'final_error_mm':0.,'final_orientation_error_deg':0.,
                    'validation':'joint_limits_and_self_collision','cursor_acceptance':{'accepted':True}}
        with patch.object(service,'load_roadmap',return_value=Roadmap(self.space)),patch.object(service,'mouse_aim_plan',side_effect=solve):
            return self.controller.ik.prepare_mouse_aim({'session_id':self.session,'cursor_mm':cursor})

    def test_mode_change_waits_for_ack_and_stop_restores_the_original_mode_once(self):
        data=self.long_route()
        self.controller.ik.execute(data['plan']['id'],10,True,True)
        self.controller.ik.mouse_aim_heartbeat(self.session)
        time.sleep(.15)
        self.assertEqual(self.robot.calls,[])
        self.controller.ik.mouse_aim_heartbeat(self.session,data['plan']['id'])
        self.wait_moves()
        self.assertEqual(self.robot.calls[0],('mode',1,{'_async':True}))
        self.assertEqual(self.controller.snapshot()['ik']['refresh_mode'],1)
        self.controller.ik.stop_mouse_aim(self.session)
        self.assertEqual(self.robot.fresh_mode,0)
        self.assertEqual([call[1] for call in self.robot.calls if call[0]=='mode'],[1,0])
        self.assertEqual([call[0] for call in self.robot.calls][-2:],['stop','mode'])

    def test_unreadable_mode_refuses_start_without_config_or_motion(self):
        self.controller.ik.stop_mouse_aim(self.session)
        self.robot.get_fresh_mode=lambda:-1
        with self.assertRaisesRegex(RuntimeError,'verify'):
            self.controller.ik.start_mouse_aim({'armed':True,'frame_verified':True})
        self.assertEqual(self.robot.calls,[])

    def test_mode_refusal_is_fail_closed_before_any_motor_command(self):
        self.robot.get_fresh_mode=lambda:0
        self.execute(self.long_route())
        status=self.wait('failed')
        self.assertIn('refused refresh mode',status['message'])
        self.assertFalse(self.robot.sends)
        self.assertEqual([call[0] for call in self.robot.calls],['mode','stop','mode'])

    def test_external_mode_change_during_motion_stops_tracking(self):
        self.execute(self.long_route());self.wait_moves()
        self.robot.fresh_mode=0
        status=self.wait('failed')
        self.assertIn('mode changed',status['message'])
        self.assertTrue(any(call[0]=='stop' for call in self.robot.calls))

    def test_replacement_plan_uses_same_worker_and_requires_its_own_ack(self):
        first=self.long_route();self.execute(first);self.wait_moves(2)
        goal=self.start.copy();goal[0]+=12
        new=self.live_plan(goal)
        root=first['plan']['id'];plan_id=new['plan']['id']
        response=self.controller.ik.execute(plan_id,10,True,True)
        self.assertEqual(response['ik']['execution_id'],root)
        self.assertEqual(response['ik']['accepted_plan_id'],plan_id)
        self.controller.ik.mouse_aim_heartbeat(self.session,root)
        time.sleep(.15)
        self.assertEqual(self.controller.snapshot()['ik']['active_plan_id'],root)
        self.controller.ik.mouse_aim_heartbeat(self.session,root,plan_id)
        deadline=time.monotonic()+2
        while self.controller.snapshot()['ik']['active_plan_id']!=plan_id and time.monotonic()<deadline:time.sleep(.01)
        self.assertEqual(self.controller.snapshot()['ik']['active_plan_id'],plan_id)
        self.assertFalse(any(call[0]=='stop' for call in self.robot.calls))
        status=self.wait('completed',5)
        self.assertEqual(status['execution_id'],root)
        self.assertLess(status['estimated_error_mm'],5)

    def test_stale_replacement_is_rejected_while_original_route_continues(self):
        first=self.long_route();self.execute(first);self.wait_moves(2)
        goal=self.start.copy();goal[0]-=6
        new=self.live_plan(goal,delay=.45)
        with self.assertRaises(StaleMousePlan):self.controller.ik.execute(new['plan']['id'],10,True,True)
        self.assertEqual(self.controller.snapshot()['ik']['state'],'running')
        self.assertEqual(self.controller.snapshot()['ik']['active_plan_id'],first['plan']['id'])
        self.assertFalse(any(call[0]=='stop' for call in self.robot.calls))

    def test_stop_invalidates_an_unacknowledged_replacement(self):
        first=self.long_route();self.execute(first);self.wait_moves(2)
        goal=self.start.copy();goal[0]+=12
        new=self.live_plan(goal)
        self.controller.ik.execute(new['plan']['id'],10,True,True)
        self.controller.ik.stop_mouse_aim(self.session)
        count=len(self.robot.sends)
        with self.assertRaises(ValueError):self.controller.ik.mouse_aim_heartbeat(self.session,first['plan']['id'],new['plan']['id'])
        time.sleep(.1)
        self.assertEqual(len(self.robot.sends),count)

    def test_endpoint_hold_releases_lock_for_replacement_ack(self):
        goal=self.start.copy();goal[0]+=2
        first=self.prepare([self.start,goal]);self.execute(first);self.wait_moves(2)
        new_goal=self.start.copy();new_goal[0]+=3
        new=self.live_plan(new_goal)
        self.controller.ik.execute(new['plan']['id'],10,True,True)
        time.sleep(.8)
        # Acquiring the controller lock must remain possible at the endpoint,
        # even while the worker holds there for the new response acknowledgement.
        acquired=self.controller.lock.acquire(timeout=.5)
        self.assertTrue(acquired)
        if acquired:
            try:
                self.assertIsNotNone(self.controller.ik.mouse_pending_plan)
                self.assertEqual(self.controller.motion,'ik')
                self.assertLessEqual(np.max(np.abs(np.asarray(self.robot.angles)-goal)),1.)
                self.assertFalse(self.robot.is_moving())
            finally:self.controller.lock.release()
        self.controller.ik.mouse_aim_heartbeat(self.session,first['plan']['id'],new['plan']['id'])
        status=self.wait('completed',3)
        self.assertEqual(status['active_plan_id'],new['plan']['id'])

    def test_disconnect_after_completion_restores_original_mode(self):
        goal=self.start.copy();goal[0]+=2
        self.execute(self.prepare([self.start,goal]));self.wait('completed')
        self.assertEqual(self.robot.fresh_mode,1)
        self.assertIsNone(self.controller.motion)
        self.controller.disconnect()
        self.assertEqual(self.robot.fresh_mode,0)
        self.assertEqual([call[0] for call in self.robot.calls][-3:],['stop','mode','close'])
        self.assertFalse(self.controller.snapshot()['connected'])

    def test_failed_idle_mode_restoration_closes_unverified_connection(self):
        goal=self.start.copy();goal[0]+=2
        self.execute(self.prepare([self.start,goal]));self.wait('completed')
        self.robot.get_fresh_mode=lambda:1
        with self.assertRaisesRegex(RuntimeError,'could not be restored'):
            self.controller.ik.stop_mouse_aim(self.session)
        self.assertFalse(self.controller.snapshot()['connected'])
        self.assertIn('could not be restored',self.controller.error)

    def test_half_second_telemetry_delay_keeps_bounded_tracking_running(self):
        first=self.long_route()
        get_angles=self.robot.get_angles
        def delayed():
            time.sleep(.55)
            return get_angles()
        with patch.object(self.robot,'get_angles',side_effect=delayed):
            self.execute(first);self.wait_moves(2)
            self.assertEqual(self.controller.snapshot()['ik']['state'],'running')
            self.assertGreater(self.controller.snapshot()['ik']['telemetry_read_ms'],500)
            for send in self.robot.sends:
                self.assertLessEqual(np.max(np.abs(np.array(send['target'])-send['live'])),4.)
                self.assertFalse(np.array_equal(np.array(send['target']),np.round(send['live'],2)))

    def test_two_second_telemetry_delay_still_stops_tracking(self):
        first=self.long_route();self.execute(first)
        # Wait until initialization is done before introducing a stale cycle.
        self.wait_moves()
        get_angles=self.robot.get_angles
        def delayed():
            time.sleep(2.05)
            return get_angles()
        with patch.object(self.robot,'get_angles',side_effect=delayed):
            status=self.wait('failed',4)
        self.assertIn('cycle exceeded 2 seconds',status['message'])
        self.assertTrue(any(call[0]=='stop' for call in self.robot.calls))


if __name__=='__main__':unittest.main()
