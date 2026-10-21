import sys
from pathlib import Path
import time
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import RobotController, ProtocolCode, LIMITS


class FakeRobot:
    def __init__(self, *args):
        self.calls = []
        self.angles = [0] * 6
        self.fail_read = False
        self.power = 1
        self.fresh_mode = 1
    def get_angles(self):
        if self.fail_read: raise OSError('link lost')
        return self.angles[:]
    def is_moving(self): return 0
    def get_system_version(self): return 6.5
    def get_fresh_mode(self): return self.fresh_mode
    def is_power_on(self): return self.power
    def is_all_servo_enable(self): return 1
    def is_paused(self): return 0
    def get_error_information(self): return 0
    def send_angles(self, angles, speed, **kwargs): self.calls.append(('move',angles,speed,kwargs))
    def jog_angle(self, joint, direction, speed, **kwargs): self.calls.append(('jog',joint,direction,speed,kwargs))
    def _mesg(self, genre, *args, **kwargs):
        if genre == ProtocolCode.SET_FRESH_MODE:
            self.fresh_mode = args[0]
            self.calls.append(('mode',args[0],kwargs))
        else: self.calls.append(('stop',genre,kwargs))
    def close(self): self.calls.append(('close',))


class ControllerTest(unittest.TestCase):
    def setUp(self):
        self.robot = FakeRobot()
        self.controller = RobotController(lambda *_: self.robot)
        self.controller.connect('192.168.77.225',9000)
    def tearDown(self): self.controller.shutdown()

    def test_connect_does_not_move(self):
        self.assertTrue(self.controller.snapshot()['connected'])
        self.assertEqual(self.robot.calls, [])

    def test_feedback_http_publishes_angle_reply_while_motion_query_blocks(self):
        import json
        import threading
        import urllib.request
        from http.server import ThreadingHTTPServer
        from server import Handler
        entered, release = threading.Event(), threading.Event()
        initial = self.controller.feedback_snapshot()['feedback_pose']['stamp_unix_s']
        def blocked_motion():
            entered.set()
            if not release.wait(2):
                raise RuntimeError('Test motion query was not released')
            return 0
        self.robot.is_moving = blocked_motion
        self.robot.angles = [1]*6
        cls = type('QuietFeedbackHandler', (Handler,), {'controller':self.controller,
            'log_message':lambda *args:None})
        server = ThreadingHTTPServer(('127.0.0.1',0), cls)
        http = threading.Thread(target=server.serve_forever, daemon=True);http.start()
        def read():
            with self.controller.lock:
                self.controller._read()
        worker = threading.Thread(target=read);worker.start()
        try:
            self.assertTrue(entered.wait(1))
            began = time.monotonic()
            data = json.load(urllib.request.urlopen(
                f'http://127.0.0.1:{server.server_port}/api/feedback', timeout=.3))
            self.assertLess(time.monotonic()-began, .3)
            self.assertTrue(data['connected'])
            self.assertGreater(data['feedback_pose']['stamp_unix_s'], initial)
            self.assertGreaterEqual(len(data['feedback_poses']), 2)
            self.assertEqual(data['feedback_pose']['stamp_source'], 'angle_reply_received')
            # The state lock is held and is_moving has not returned, yet the
            # validated angle measurement is already available over HTTP.
            self.assertTrue(worker.is_alive())
            data['feedback_pose']['translation_m'][0]=999
            self.assertNotEqual(self.controller.feedback_snapshot()['feedback_pose']['translation_m'][0],999)
        finally:
            release.set();worker.join(2)
            server.shutdown();server.server_close();http.join(2)

    def test_feedback_cache_does_not_restamp_and_is_invalidated_on_disconnect(self):
        first = self.controller.feedback_snapshot()
        time.sleep(.01)
        self.assertEqual(first, self.controller.feedback_snapshot())
        self.controller.disconnect()
        self.assertEqual(self.controller.feedback_snapshot(), {'connected':False,
            'feedback_protocol':1, 'feedback_pose':None, 'feedback_poses':[]})

    def test_invalid_angle_status_reports_diagnostics_without_sending_motion(self):
        from unittest.mock import Mock
        self.robot.get_angles=lambda:-1
        reader=Mock()
        reader.failure_detail.return_value='query 0x20: 3 failed reply attempts; no complete matching reply'
        self.robot._read=reader
        with self.assertRaisesRegex(RuntimeError,'SDK status -1.*3 failed reply attempts'):
            self.controller._read()
        reader.failure_detail.assert_called_once_with(ProtocolCode.GET_ANGLES)
        self.assertEqual(self.robot.calls,[])

    def test_manual_pose_rejects_old_joint4_limit_before_sending(self):
        pose=[0]*6;pose[3]=146
        with self.assertRaisesRegex(ValueError,'Joint 4'):self.controller.move(pose,30,True)
        self.assertEqual(self.robot.calls,[])
        self.assertEqual(self.controller.snapshot()['limits']['max'][3],143)
        self.assertEqual(LIMITS['angles_min'][1],-133)

    def test_validated_move_and_completion(self):
        self.controller.move([10]*6,30,True)
        self.assertEqual(self.robot.calls[0],('move',[10]*6,30,{'_async':True}))
        with self.assertRaises(RuntimeError): self.controller.move([0]*6,30,True)
        with self.controller.lock:
            self.robot.angles = [10]*6
            self.controller._read()
        self.assertIsNone(self.controller.snapshot()['motion'])
        self.assertIsNone(self.controller.timer)

    def test_manual_controls_accept_80_percent_and_reject_81_before_sending(self):
        self.controller.firmware=6.6  # Exercise legacy jog on a non-quarantined fake firmware.
        for command in (lambda speed:self.controller.move([0]*6,speed,True),
                        lambda speed:self.controller.jog(1,1,speed,.1,True)):
            with self.assertRaises(ValueError):command(81)
        self.assertEqual(self.robot.calls, [])
        self.controller.move([0]*6,80,True)
        self.assertEqual(self.robot.calls[0][2],80)
        self.controller.stop()
        self.controller.jog(1,1,80,.1,True)
        self.assertEqual(self.robot.calls[-1][3],80)

    def test_reject_invalid_commands_before_motion(self):
        for angles,speed,armed in [([0]*6,30,False),([0]*5,30,True),([999]*6,30,True),([float('nan')]*6,30,True),([0]*6,101,True),([0]*6,True,True)]:
            with self.assertRaises(ValueError): self.controller.move(angles,speed,armed)
        for joint,direction,duration in [(0,1,.5),(1,2,.5),(1,1,3),(1,1,float('inf'))]:
            with self.assertRaises(ValueError): self.controller.jog(joint,direction,30,duration,True)
        self.robot.power = 0
        with self.assertRaises(RuntimeError): self.controller.move([0]*6,30,True)
        self.assertEqual(self.robot.calls, [])

    def test_jog_stops_without_browser(self):
        self.controller.firmware=6.6
        self.controller.jog(2,0,30,.1,True)
        time.sleep(.2)
        self.assertEqual(self.robot.calls[-1],('stop',ProtocolCode.STOP,{'_async':True}))
        self.assertIsNone(self.controller.snapshot()['motion'])

    def test_stop_cancels_old_deadline(self):
        self.controller.firmware=6.6
        self.controller.jog(1,1,30,.1,True)
        self.controller.stop()
        self.controller.move([20]*6,30,True)
        time.sleep(.2)
        self.assertEqual([c[0] for c in self.robot.calls],['jog','stop','move'])

    def test_atom_65_jog_is_blocked_before_any_motor_command(self):
        with self.assertRaisesRegex(ValueError,'disabled on Atom 6.5'):
            self.controller.jog(4,0,1,.1,True)
        self.assertEqual(self.robot.calls,[])
        self.assertTrue(self.controller.snapshot()['directional_jog_blocked'])

    def test_connection_loss_attempts_stop_and_disconnects(self):
        self.controller.move([20]*6,30,True)
        self.robot.fail_read = True
        deadline = time.monotonic()+2
        while self.controller.snapshot()['connected'] and time.monotonic()<deadline: time.sleep(.05)
        self.assertFalse(self.controller.snapshot()['connected'])
        self.assertEqual([c[0] for c in self.robot.calls],['move','stop','close'])
        self.assertIn('connection lost',self.controller.snapshot()['error'])

    def test_disconnect_stops_owned_motion(self):
        self.controller.move([20]*6,30,True)
        self.controller.disconnect()
        self.assertEqual([c[0] for c in self.robot.calls],['move','stop','close'])

    def test_failed_motion_delivery_attempts_stop(self):
        def fail(*args, **kwargs): raise OSError('send failed')
        self.robot.send_angles = fail
        with self.assertRaisesRegex(RuntimeError, 'Motion delivery failed'):
            self.controller.move([20]*6,30,True)
        self.assertFalse(self.controller.snapshot()['connected'])
        self.assertEqual([c[0] for c in self.robot.calls],['stop','close'])


class HTTPTest(unittest.TestCase):
    def setUp(self):
        from server import Handler, ThreadingHTTPServer
        import threading
        self.controller = RobotController(FakeRobot)
        self.handler = type('TestHandler',(Handler,),{'controller':self.controller,'token':'test-token', 'hardware_enabled':True, 'log_message':lambda *args:None})
        self.http = ThreadingHTTPServer(('127.0.0.1',0),self.handler)
        threading.Thread(target=self.http.serve_forever,daemon=True).start()
        self.base = f'http://127.0.0.1:{self.http.server_port}'
    def tearDown(self):
        self.http.shutdown()
        self.http.server_close()
        self.controller.shutdown()
    def post(self, headers):
        import urllib.request
        return urllib.request.urlopen(urllib.request.Request(self.base+'/api/connect',data=b'{"host":"192.168.77.225"}',headers={'Content-Type':'application/json',**headers}),timeout=2)
    def test_requires_local_session_token(self):
        import urllib.error
        with self.assertRaises(urllib.error.HTTPError) as caught: self.post({})
        self.assertEqual(caught.exception.code,403)
        self.assertFalse(self.controller.snapshot()['connected'])
        with self.post({'X-Robot-Token':'test-token','Origin':'http://127.0.0.1:5173'}) as response:
            self.assertEqual(response.status,200)
        self.assertTrue(self.controller.snapshot()['connected'])

    def test_approach_stream_delivers_search_before_planner_completion(self):
        import json
        import urllib.request
        import threading
        from unittest.mock import patch
        self.handler.hardware_enabled=False
        released=threading.Event()
        def plan(body):
            body['_rrt_observer']({'kind':'begin','start_deg':[0]*6,'goal_deg':[1]*6})
            if not released.wait(2):raise RuntimeError('Stream was not received while planning')
            return {'playable':False,'status':'no_path'}
        request=urllib.request.Request(self.base+'/api/sim/approach-plan',
            data=json.dumps({'tracking_engine':'learned_rrt','tracking_region':{'distance_mm':350}}).encode(),
            headers={'Content-Type':'application/json','X-Robot-Token':'test-token'})
        try:
            with patch('server.simulation.mouse_aim_plan',side_effect=plan), urllib.request.urlopen(request,timeout=3) as response:
                self.assertEqual(response.headers['Content-Type'],'application/x-ndjson')
                first=json.loads(response.readline());self.assertEqual(first['type'],'rrt')
                released.set()
                self.assertEqual(json.loads(response.readline())['result']['status'],'no_path')
        finally:released.set()

    def test_approach_stream_rejects_hardware_and_missing_local_token(self):
        import json
        import urllib.request
        import urllib.error
        from unittest.mock import patch
        with patch('server.simulation.mouse_aim_plan') as planner:
            for hardware,token in [(True,'test-token'),(False,'wrong-token')]:
                self.handler.hardware_enabled=hardware
                request=urllib.request.Request(self.base+'/api/sim/approach-plan',data=b'{}',
                    headers={'Content-Type':'application/json','X-Robot-Token':token})
                with self.assertRaises(urllib.error.HTTPError) as caught:urllib.request.urlopen(request,timeout=2)
                self.assertEqual(caught.exception.code,403)
            planner.assert_not_called()

    def test_ros_checked_plan_stream_is_live_and_uses_existing_planner(self):
        import threading
        from unittest.mock import patch
        from object_tracking_ros2.core import LocalAPI
        seen = threading.Event()
        def plan(body, observer=None):
            self.assertEqual(body, {'session_id':'owned'})
            observer({'kind':'begin', 'start_deg':[0]*6, 'goal_deg':[1]*6})
            if not seen.wait(2):
                raise RuntimeError('ROS observer did not receive a live batch')
            return {'plan':None, 'result':{'status':'no_path', 'playable':False}}
        def observe(event):
            self.assertEqual(event['kind'], 'begin')
            seen.set()
        with patch.object(self.controller.ik, 'prepare_mouse_aim', side_effect=plan):
            result = LocalAPI(self.base).plan_with_observer({'session_id':'owned'}, observe)
        self.assertFalse(result['result']['playable'])
        self.assertIsNone(self.controller.motion)

    def test_ros_stream_preserves_retry_errors_and_token_mode_gates(self):
        import urllib.error
        from unittest.mock import patch
        from object_tracking_ros2.core import LocalAPI
        from webui.mouse_motion import StaleMousePlan
        api = LocalAPI(self.base)
        with patch.object(self.controller.ik, 'prepare_mouse_aim', side_effect=StaleMousePlan('Old anchor')):
            with self.assertRaises(RuntimeError) as caught:
                api.plan_with_observer({}, lambda event:None)
            self.assertEqual(caught.exception.code, 409)
        with patch.object(self.controller.ik, 'prepare_mouse_aim') as planner:
            api.token = 'invalid'
            with self.assertRaises(urllib.error.HTTPError) as caught:
                api.plan_with_observer({}, lambda event:None)
            self.assertEqual(caught.exception.code, 403)
            api.token = 'test-token'
            self.handler.hardware_enabled = False
            with self.assertRaises(urllib.error.HTTPError) as caught:
                api.plan_with_observer({}, lambda event:None)
            self.assertEqual(caught.exception.code, 403)
            planner.assert_not_called()

    def test_simulation_only_rejects_hardware_routes_without_calling_transport(self):
        import json
        import urllib.request
        import urllib.error
        from unittest.mock import patch
        self.handler.hardware_enabled = False
        routes = ('connect', 'move', 'jog', 'ik/plan', 'ik/diffusion/plan', 'ik/joint-diagnostic/plan', 'ik/jog-qualification/plan',
                  'ik/execute', 'ik/heartbeat', 'ik/mouse-aim/start',
                  'ik/mouse-aim/plan', 'ik/mouse-aim/plan-stream', 'ik/mouse-aim/heartbeat', 'ik/mouse-aim/stop')
        with patch.object(self.controller, 'factory') as transport:
            for route in routes:
                with self.subTest(route=route):
                    request = urllib.request.Request(self.base+'/api/'+route, data=b'{}',
                        headers={'Content-Type':'application/json', 'X-Robot-Token':'test-token'})
                    with self.assertRaises(urllib.error.HTTPError) as caught:
                        urllib.request.urlopen(request, timeout=2)
                    self.assertEqual(caught.exception.code, 403)
                    self.assertIn('hardware commands are disabled', json.load(caught.exception)['error'])
            transport.assert_not_called()
        self.assertFalse(self.controller.snapshot()['connected'])
        self.assertIsNone(self.controller.motion)

    def mode_request(self, mode, token='test-token'):
        import json
        import urllib.request
        request = urllib.request.Request(self.base+'/api/mode', data=json.dumps(mode).encode(),
            headers={'Content-Type':'application/json', 'X-Robot-Token':token})
        with urllib.request.urlopen(request, timeout=4) as response:
            return json.load(response)

    def test_selecting_hardware_updates_shared_mode_without_connecting(self):
        import json
        import urllib.request
        from unittest.mock import patch
        self.handler.hardware_enabled = False
        with patch.object(self.controller, 'factory') as transport:
            result = self.mode_request({'mode':'hardware'})
            transport.assert_not_called()
        self.assertTrue(result['hardware_enabled'])
        self.assertTrue(result['mode_selectable'])
        self.assertEqual(result['control_mode'], 'hardware')
        self.assertFalse(result['connected'])
        with urllib.request.urlopen(self.base+'/api/state') as response:
            self.assertTrue(json.load(response)['hardware_enabled'])

    def test_selecting_simulation_stops_owned_motion_disconnects_and_invalidates_plans(self):
        self.controller.connect('192.168.77.225',9000)
        robot = self.controller.robot
        self.controller.move([20]*6,30,True)
        self.controller.ik.plan = {'id':'stale'}
        self.controller.ik.mouse_session = 'stale-session'
        generation = self.controller.generation
        result = self.mode_request({'mode':'simulation'})
        self.assertFalse(result['hardware_enabled'])
        self.assertFalse(result['connected'])
        self.assertIsNone(result['motion'])
        self.assertIsNone(self.controller.ik.plan)
        self.assertIsNone(self.controller.ik.mouse_session)
        self.assertGreater(self.controller.generation, generation)
        self.assertEqual([call[0] for call in robot.calls], ['move','stop','close'])

    def test_locked_simulation_cannot_enable_hardware(self):
        import urllib.error
        self.handler.hardware_enabled = False
        self.handler.mode_selectable = False
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.mode_request({'mode':'hardware'})
        self.assertEqual(caught.exception.code, 400)
        self.assertFalse(self.handler.hardware_enabled)
        self.assertFalse(self.mode_request({'mode':'simulation'})['hardware_enabled'])

    def test_mode_requires_local_token_and_valid_mode(self):
        import urllib.error
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.mode_request({'mode':'simulation'}, token='invalid')
        self.assertEqual(caught.exception.code,403)
        for body in ({}, {'mode':'unknown'}, {'mode':True}, {'mode':'simulation','armed':True}):
            with self.subTest(body=body), self.assertRaises(urllib.error.HTTPError) as caught:
                self.mode_request(body)
            self.assertEqual(caught.exception.code,400)
        self.assertTrue(self.handler.hardware_enabled)

    def test_mode_change_serializes_with_inflight_connect(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor
        entered, release = threading.Event(), threading.Event()
        robot = FakeRobot()
        def factory(*args):
            entered.set()
            if not release.wait(2): raise RuntimeError('test connect timed out')
            return robot
        self.controller.factory = factory
        def connect():
            with self.post({'X-Robot-Token':'test-token'}) as response:
                return response.status
        with ThreadPoolExecutor(max_workers=2) as pool:
            connecting = pool.submit(connect)
            self.assertTrue(entered.wait(1))
            switching = pool.submit(self.mode_request, {'mode':'simulation'})
            release.set()
            self.assertEqual(connecting.result(timeout=4),200)
            self.assertFalse(switching.result(timeout=4)['connected'])
        self.assertFalse(self.controller.snapshot()['connected'])
        self.assertFalse(self.handler.hardware_enabled)
        self.assertEqual(robot.calls, [('close',)])

    def test_rejects_foreign_origin(self):
        import urllib.error
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post({'X-Robot-Token':'test-token','Origin':'https://foreign.example'})
        self.assertEqual(caught.exception.code,403)
        self.assertFalse(self.controller.snapshot()['connected'])

    def test_workspace_captures_live_start_without_motion(self):
        import json
        import urllib.request
        from unittest.mock import patch
        self.controller.connect('192.168.77.225', 9000)
        self.controller.robot.angles = [10, -20, -30, 40, 5, 6]
        request = urllib.request.Request(self.base+'/api/sim/workspace',
            data=json.dumps({'start_deg':[0]*6}).encode(),
            headers={'Content-Type':'application/json', 'X-Robot-Token':'test-token'})
        with patch('server.estimate_workspace', return_value={'accepted':1}) as estimator:
            with urllib.request.urlopen(request, timeout=2) as response:
                result = json.load(response)
        self.assertEqual(estimator.call_args.args[0]['start_deg'], [10,-20,-30,40,5,6])
        self.assertEqual(result['start_source'], 'live robot joints')
        self.assertEqual(self.controller.robot.calls, [])

    def test_workspace_rejects_moving_start(self):
        import urllib.request
        import urllib.error
        from unittest.mock import patch
        self.controller.connect('192.168.77.225', 9000)
        self.controller.motion = 'ik'
        request = urllib.request.Request(self.base+'/api/sim/workspace', data=b'{}',
            headers={'Content-Type':'application/json', 'X-Robot-Token':'test-token'})
        with patch('server.estimate_workspace') as estimator:
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request, timeout=2)
        self.assertEqual(caught.exception.code, 400)
        estimator.assert_not_called()
        self.controller.motion = None

    def mouse_aim_request(self):
        import urllib.request
        return urllib.request.Request(self.base+'/api/sim/mouse-aim', data=b'{}',
            headers={'Content-Type':'application/json', 'X-Robot-Token':'test-token'})

    def diffusion_request(self):
        import urllib.request
        return urllib.request.Request(self.base+'/api/sim/diffusion',data=b'{}',
            headers={'Content-Type':'application/json','X-Robot-Token':'test-token'})

    def test_diffusion_never_plans_in_hardware_connected_or_motion_mode(self):
        import urllib.request
        import urllib.error
        from unittest.mock import patch
        with patch('server.diffusion.plan') as planner:
            for hardware,connected,motion,moving in [(True,False,None,False),
                    (False,True,None,False),(False,False,'ik',False),(False,False,None,True)]:
                self.handler.hardware_enabled=hardware
                if connected:self.controller.connect('192.168.77.225',9000)
                self.controller.motion=motion;self.controller.moving=moving
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(self.diffusion_request(),timeout=2)
                self.assertEqual(caught.exception.code,403);planner.assert_not_called()
                if connected:
                    self.assertEqual(self.controller.robot.calls,[]);self.controller.disconnect()
                self.controller.motion=None;self.controller.moving=False

    def test_diffusion_preview_and_busy_response_never_read_transport(self):
        import json
        import urllib.request
        import urllib.error
        from unittest.mock import patch
        from simulation.diffusion.service import PlannerBusy
        self.handler.hardware_enabled=False
        with patch.object(self.controller,'_read',side_effect=AssertionError('No transport reads')), \
                patch('server.diffusion.plan',return_value={'simulation_only':True}) as planner:
            with urllib.request.urlopen(self.diffusion_request(),timeout=2) as response:
                self.assertTrue(json.load(response)['simulation_only'])
            planner.assert_called_once_with({})
            planner.side_effect=PlannerBusy('busy')
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(self.diffusion_request(),timeout=2)
            self.assertEqual(caught.exception.code,409)
        self.assertIsNone(self.controller.motion);self.assertIsNone(self.controller.ik.plan)

    def test_mouse_aim_requires_disconnected_hardware_disabled_simulation(self):
        import urllib.request
        import urllib.error
        from unittest.mock import patch
        with patch('server.simulation.mouse_aim_plan') as planner:
            for hardware, connected, motion in [(True, False, None), (False, True, None), (False, False, 'ik')]:
                self.handler.hardware_enabled = hardware
                if connected:self.controller.connect('192.168.77.225', 9000)
                self.controller.motion = motion
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(self.mouse_aim_request(), timeout=2)
                self.assertEqual(caught.exception.code, 403)
                planner.assert_not_called()
                if connected:
                    self.assertEqual(self.controller.robot.calls, [])
                    self.controller.disconnect()
                self.controller.motion = None

    def test_mouse_aim_serializes_work_and_never_calls_transport(self):
        import json
        import urllib.request
        import urllib.error
        from unittest.mock import patch
        self.handler.hardware_enabled = False
        with patch('server.simulation.mouse_aim_plan', return_value={'simulation_only':True}) as planner, \
             patch.object(self.controller, '_read', side_effect=AssertionError('No transport reads')):
            with urllib.request.urlopen(self.mouse_aim_request(), timeout=2) as response:
                self.assertTrue(json.load(response)['simulation_only'])
            planner.assert_called_once_with({})
            self.handler.mouse_aim_lock.acquire()
            try:
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(self.mouse_aim_request(), timeout=2)
                self.assertEqual(caught.exception.code, 409)
                planner.assert_called_once()
            finally:self.handler.mouse_aim_lock.release()

    def test_mouse_aim_releases_busy_lock_after_failure(self):
        import urllib.request
        import urllib.error
        from unittest.mock import patch
        self.handler.hardware_enabled = False
        with patch('server.simulation.mouse_aim_plan', side_effect=ValueError('Invalid cursor')):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(self.mouse_aim_request(), timeout=2)
            self.assertEqual(caught.exception.code, 400)
        self.assertTrue(self.handler.mouse_aim_lock.acquire(blocking=False))
        self.handler.mouse_aim_lock.release()
