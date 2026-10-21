import sys
from pathlib import Path
import time
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from server import RobotController
from test_server import FakeRobot
from simulation.kinematics import ARM
import webui.hardware_ik as hardware


class TrackingRobot(FakeRobot):
    def __init__(self):
        super().__init__();self.angles=[0,-40,-130,80,0,50];self.track=True
    def send_angles(self,angles,speed,**kwargs):
        super().send_angles(angles,speed,**kwargs)
        if self.track:self.angles=angles[:]


class HardwareIKTest(unittest.TestCase):
    def setUp(self):
        self.robot=TrackingRobot()
        self.controller=RobotController(lambda *_:self.robot)
        self.controller.connect('192.168.77.225',9000)
        goal=np.deg2rad([3,-38,-130,80,0,50])
        self.body={'target_mm':(ARM.fk(goal)*1000).tolist(),'method':'dls'}
    def tearDown(self):self.controller.shutdown()
    def plan(self):return self.controller.ik.prepare(self.body)
    def execute(self,data):
        result=self.controller.ik.execute(data['plan']['id'],30,True,True)
        self.controller.ik.heartbeat(data['plan']['id'])
        return result
    def wait_state(self,states,seconds=4):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            state=self.controller.snapshot()['ik']
            if state['state'] in states:return state
            time.sleep(.025)
        self.fail(f'IK did not reach {states}: {self.controller.snapshot()}')
    def test_plan_is_read_only_uses_live_start_and_bounded_path(self):
        result=self.plan()
        self.assertEqual(result['start_deg'],self.robot.angles)
        self.assertTrue(result['plan']['converged'])
        self.assertEqual(self.robot.calls,[])
        points=np.array([result['start_deg'],*self.controller.ik.plan['waypoints']])
        self.assertTrue((np.max(np.abs(np.diff(points,axis=0)),axis=1)<=hardware.MAX_STEP_DEG+1e-8).all())
    def test_selected_speed_is_bound_to_the_fixed_target_preview(self):
        self.body['speed']=70
        data=self.plan()
        self.assertEqual(data['plan']['planned_sdk_speed'],70)
        self.assertAlmostEqual(data['result']['joint_velocity_allowance_deg_s'],105.)
        self.assertAlmostEqual(sum(self.controller.ik.plan['waypoint_intervals_s']),
                               data['result']['smoothing']['duration_s'])
        with self.assertRaisesRegex(ValueError,'plan again'):
            self.controller.ik.execute(data['plan']['id'],30,True,True)
        self.assertEqual(self.robot.calls,[])
    def test_local_offset_is_resolved_from_fresh_live_joints(self):
        self.robot.angles[5]+=30
        t=ARM.transform(np.deg2rad(self.robot.angles));offset=np.array([4.,-3.,2.])
        self.body['target_mm']=(t[:3,3]*1000-t[:3,:3]@offset).tolist()
        self.body['eef_offset_mm']=offset.tolist()
        result=self.plan()
        np.testing.assert_allclose(result['target_mm'],t[:3,3]*1000,atol=1e-9)
        np.testing.assert_allclose(result['eef_rotation'],t[:3,:3],atol=1e-12)
        self.assertEqual(result['start_deg'],self.robot.angles)
        self.assertEqual(self.robot.calls,[])
    def test_executes_verified_plan_once_and_reports_telemetry(self):
        result=self.plan();self.execute(result)
        state=self.wait_state({'completed'})
        self.assertEqual(state['completed_waypoints'],state['total_waypoints'])
        self.assertLessEqual(state['estimated_error_mm'],5)
        self.assertTrue(len(state['samples'])>0)
        self.assertTrue(all(c[0]=='move' and c[2]==30 and c[3]=={'_async':True} for c in self.robot.calls))
        with self.assertRaisesRegex(ValueError,'matching'):self.execute(result)
    def test_no_motion_until_browser_receives_execution_and_confirms_heartbeat(self):
        data=self.plan();self.controller.ik.execute(data['plan']['id'],30,True,True)
        time.sleep(.4)
        self.assertEqual(self.robot.calls,[])
        self.controller.ik.heartbeat(data['plan']['id'])
        self.wait_state({'completed'})
        self.assertTrue(any(call[0]=='move' for call in self.robot.calls))

    def test_refuses_expired_changed_failed_and_unverified_plans(self):
        data=self.plan()
        for armed,verified in ((False,True),(True,False)):
            with self.assertRaises(ValueError):self.controller.ik.execute(data['plan']['id'],30,armed,verified)
        self.controller.ik.plan['created']-=61
        with self.assertRaisesRegex(ValueError,'expired'):self.execute(data)
        data=self.plan();self.robot.angles[0]+=2
        with self.assertRaisesRegex(ValueError,'changed'):self.execute(data)
        self.assertEqual(self.robot.calls,[])
        self.body['target_mm']=[800,0,0]
        data=self.plan()
        self.assertFalse(data['plan']['converged'])
        with self.assertRaisesRegex(ValueError,'failed'):self.execute(data)
        self.assertEqual(self.robot.calls,[])
    def test_stop_cancels_queue_before_any_next_waypoint(self):
        data=self.plan();self.execute(data)
        self.controller.stop()
        count=len(self.robot.calls);time.sleep(.4)
        self.assertEqual(len(self.robot.calls),count)
        self.assertEqual(self.robot.calls[-1][0],'stop')
        self.assertEqual(self.controller.snapshot()['ik']['state'],'stopped')
    def test_lost_browser_stops_without_another_motion_command(self):
        data=self.plan();self.execute(data)
        with self.controller.lock:self.controller.ik.heartbeat_at-=6
        state=self.wait_state({'failed'})
        self.assertIn('heartbeat',state['message'])
        self.assertEqual([c[0] for c in self.robot.calls],['stop'])
    def test_stalled_waypoint_stops_and_reports_failure(self):
        self.robot.track=False
        with patch.object(hardware,'SEGMENT_TIMEOUT',.05):
            data=self.plan();self.execute(data)
            state=self.wait_state({'failed'})
        self.assertIn('settle',state['message'])
        self.assertEqual([c[0] for c in self.robot.calls],['move','stop'])
    def test_overlap_and_invalid_speed_refused(self):
        data=self.plan()
        for speed in (0,71,True,3.5):
            with self.assertRaises(ValueError):self.controller.ik.execute(data['plan']['id'],speed,True,True)
        self.execute(data)
        with self.assertRaises(RuntimeError):self.controller.move([0]*6,30,True)
        with self.assertRaises(RuntimeError):self.controller.ik.prepare(self.body)
        with self.assertRaises(ValueError):self.controller.ik.heartbeat('foreign-execution')
    def test_failed_send_stops_disconnects_and_marks_failure(self):
        def fail(*args,**kwargs):raise OSError('send failed')
        self.robot.send_angles=fail
        data=self.plan();self.execute(data)
        state=self.wait_state({'failed'})
        self.assertIn('delivery failed',state['message'])
        self.assertFalse(self.controller.snapshot()['connected'])
        self.assertEqual([c[0] for c in self.robot.calls],['stop','close'])
    def test_unexpected_motion_before_first_command_is_rejected(self):
        data=self.plan();self.execute(data)
        self.robot.is_moving=lambda:1
        state=self.wait_state({'failed'})
        self.assertIn('Unplanned motion',state['message'])
        self.assertEqual([c[0] for c in self.robot.calls],['stop'])
    def test_disconnect_cancels_queued_path(self):
        data=self.plan();self.execute(data)
        self.controller.disconnect();time.sleep(.3)
        self.assertEqual([c[0] for c in self.robot.calls],['stop','close'])
        self.assertIsNone(self.controller.snapshot()['motion'])
        self.assertIsNone(self.controller.snapshot()['ik']['plan'])


class HardwareIKHTTPTest(unittest.TestCase):
    def setUp(self):
        import threading
        from server import Handler,ThreadingHTTPServer
        self.robot=TrackingRobot()
        self.controller=RobotController(lambda *_:self.robot)
        self.controller.connect('192.168.77.225',9000)
        handler=type('HardwareHandler',(Handler,),{'controller':self.controller,'hardware_enabled':True,'token':'http-ik-test','log_message':lambda *args:None})
        self.http=ThreadingHTTPServer(('127.0.0.1',0),handler)
        threading.Thread(target=self.http.serve_forever,daemon=True).start()
        self.base=f'http://127.0.0.1:{self.http.server_port}'
    def tearDown(self):
        self.http.shutdown();self.http.server_close();self.controller.shutdown()
    def post(self,path,body):
        import json,urllib.request
        return urllib.request.urlopen(urllib.request.Request(self.base+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json','X-Robot-Token':'http-ik-test'}),timeout=5)
    def test_prm_dry_run_captures_live_joints_without_preparing_hardware_plan(self):
        import json
        from simulation.prm import Roadmap
        t = ARM.transform(np.deg2rad(self.robot.angles))
        body = {'start_deg':[999]*6, 'target_mm':(t[:3,3]*1000).tolist(), 'nn':'none'}
        with patch('simulation.prm.Roadmap.load', return_value=Roadmap()):
            with self.post('/api/sim/prm', body) as response: result = json.load(response)
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['start_deg'], self.robot.angles)
        self.assertEqual(result['start_source'], 'live robot joints')
        self.assertTrue(result['simulation_only'])
        self.assertIsNone(self.controller.ik.plan)
        self.assertEqual(self.robot.calls, [])
    def test_api_plans_live_start_rejects_client_trajectory_then_executes(self):
        import json,urllib.error
        body={'target_mm':(ARM.fk(np.deg2rad([3,-38,-130,80,0,50]))*1000).tolist(),'method':'dls'}
        with self.assertRaises(urllib.error.HTTPError) as caught:self.post('/api/ik/plan',{**body,'start_deg':[0]*6})
        self.assertEqual(caught.exception.code,400)
        with self.post('/api/ik/plan',body) as response:plan=json.load(response)
        self.assertEqual(plan['start_deg'],self.robot.angles)
        self.assertEqual(self.robot.calls,[])
        execution={'plan_id':plan['plan']['id'],'speed':30,'armed':True,'frame_verified':True}
        for extra in ({'angles':[999]*6},{'target_mm':[800,0,0]}):
            with self.assertRaises(urllib.error.HTTPError) as caught:self.post('/api/ik/execute',{**execution,**extra})
            self.assertEqual(caught.exception.code,400)
        with self.post('/api/ik/execute',execution) as response:state=json.load(response)
        self.assertTrue(state['hardware_enabled'])
        self.assertEqual(state['motion'],'ik')
        with self.post('/api/ik/heartbeat',{'execution_id':plan['plan']['id']}) as response:
            self.assertTrue(json.load(response)['active'])
        with self.post('/api/stop',{}) as response:self.assertIsNone(json.load(response)['motion'])
        count=len(self.robot.calls);time.sleep(.3)
        self.assertEqual(len(self.robot.calls),count)


class HardwarePose5Test(HardwareIKTest):
    def setUp(self):
        super().setUp()
        from simulation.pose5 import xy_from_axis
        t=ARM.transform(np.deg2rad([3,-38,-130,80,2,50]))
        self.body={'task':'pose5','target_mm':(t[:3,3]*1000).tolist(),'orientation_xy_deg':xy_from_axis(t[:3,2]),'method':'dls'}
    def test_plan_and_execution_include_direction_in_telemetry(self):
        data=self.plan()
        self.assertEqual(data['plan']['task'],'pose5')
        self.assertEqual(data['plan']['orientation_xy_deg'],self.body['orientation_xy_deg'])
        self.assertLessEqual(data['plan']['final_orientation_error_deg'],1)
        self.assertEqual(self.robot.calls,[])
        self.execute(data);state=self.wait_state({'completed'})
        self.assertLessEqual(state['estimated_orientation_error_deg'],3)
        self.assertTrue(all('orientation_error_deg' in sample for sample in state['samples']))
    def test_completion_refuses_correct_position_with_wrong_direction(self):
        from simulation.pose5 import xy_from_axis
        t=ARM.transform(np.deg2rad(self.robot.angles))
        self.body={'task':'pose5','target_mm':(t[:3,3]*1000).tolist(),'orientation_xy_deg':xy_from_axis(t[:3,2]),'method':'dls'}
        data=self.plan()
        # Drift still within the start-angle allowance, but beyond a tighter
        # patched direction tolerance. Zero-waypoint execution must check both.
        self.robot.angles[4]+=.5
        with patch.object(hardware,'FINAL_ORIENTATION_DEG',.1):
            self.execute(data);state=self.wait_state({'failed'})
        self.assertIn('pointing error',state['message'])
        self.assertLess(state['estimated_error_mm'],5)
        self.assertEqual([c[0] for c in self.robot.calls],['stop'])
    def test_position_only_goal_is_rejected_when_direction_fails(self):
        t=ARM.transform(np.deg2rad(self.robot.angles))
        self.body.update(target_mm=(t[:3,3]*1000).tolist(),orientation_xy_deg=[0,0],steps=1)
        data=self.plan()
        self.assertFalse(data['plan']['converged'])
        with self.assertRaises(ValueError):self.execute(data)
        self.assertEqual(self.robot.calls,[])
