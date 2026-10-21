import json
import threading
import unittest
import urllib.request
import urllib.error
import tempfile
from unittest.mock import patch
from pathlib import Path
import numpy as np
from simulation.kinematics import ARM
from simulation.solvers import METHODS, Models, cases, solve
from simulation import service
from webui.server import Handler, RobotController, ThreadingHTTPServer


class KinematicsTest(unittest.TestCase):
    def test_operating_limits_stay_inside_queried_firmware_ranges(self):
        self.assertTrue(np.all(ARM.lower>=ARM.hardware_lower+np.deg2rad(2)-1e-12))
        self.assertTrue(np.all(ARM.upper<=ARM.hardware_upper-np.deg2rad(2)+1e-12))
        self.assertAlmostEqual(np.rad2deg(ARM.hardware_upper[1]),135)
        self.assertAlmostEqual(np.rad2deg(ARM.hardware_upper[3]),145)
        self.assertAlmostEqual(np.rad2deg(ARM.upper[3]),143)

    def test_models_reject_weights_with_different_joint_normalization(self):
        source=Path('simulation/artifacts/models.json')
        data=json.loads(source.read_text());data['limits_rad']['max'][3]+=np.deg2rad(5)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'models.json';path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'joint limits mismatch'):Models(path)

    def test_local_offset_uses_full_starting_eef_rotation_in_both_task_modes(self):
        q=np.deg2rad([10,-40,-110,65,20,50]);offset=np.array([12,-7,25.])
        for mode in ('position','pose5'):
            for spin in (0.,np.pi/2):
                start=q.copy();start[5]+=spin;t=ARM.transform(start)
                # Compensate the entered base target so the resolved target is
                # exactly the starting flange position, independently of spin.
                base=t[:3,3]*1000-t[:3,:3]@offset
                body={'task':mode,'start_deg':np.rad2deg(start).tolist(),'target_mm':base.tolist(),
                      'eef_offset_mm':offset.tolist(),'methods':['dls'],
                      'orientation_xy_deg':[80,-80]}
                result=service.run(body)
                np.testing.assert_allclose(result['target_mm'],t[:3,3]*1000,atol=1e-10)
                self.assertAlmostEqual(result['results'][0]['frames'][0]['error_mm'],0,places=9)
        original=ARM.transform(q);spun=q.copy();spun[5]+=np.pi/2;rotated=ARM.transform(spun)
        np.testing.assert_allclose(original[:3,3],rotated[:3,3],atol=1e-10)
        np.testing.assert_allclose(original[:3,2],rotated[:3,2],atol=1e-10)
        self.assertGreater(np.linalg.norm(original[:3,:3]@offset-rotated[:3,:3]@offset),1)

    def test_rejects_invalid_local_offsets(self):
        body={'start_deg':[0]*6,'target_mm':[100,0,100],'methods':['dls']}
        for offset in ([0,0],[0,0,float('nan')],[0,0,float('inf')],None):
            with self.assertRaises((ValueError,TypeError)):
                service.run({**body,'eef_offset_mm':offset})

    def test_analytic_jacobian_matches_numeric_fk(self):
        for q in (np.zeros(6), np.deg2rad([0,-40,-130,80,0,50]), (ARM.lower+ARM.upper)/3):
            step=1e-6
            numeric=np.column_stack([(ARM.fk(q+np.eye(6)[i]*step)-ARM.fk(q-np.eye(6)[i]*step))/(2*step) for i in range(6)])
            np.testing.assert_allclose(ARM.jacobian(q),numeric,atol=1e-9)

    def test_all_methods_obey_limits_speed_and_report_actual_fk(self):
        fixture=cases(count=1)[0];speed=.8
        for method,_,_ in METHODS:
            with self.subTest(method=method):
                result=solve(np.deg2rad(fixture['start_deg']),np.array(fixture['target_mm'])/1000,method,steps=60,speed=speed)
                q=np.deg2rad([f['q'] for f in result['frames']])
                self.assertTrue(np.isfinite(q).all())
                self.assertTrue(np.all(q>=ARM.lower-1e-9) and np.all(q<=ARM.upper+1e-9))
                self.assertTrue(np.all(np.linalg.norm(np.diff(q,axis=0),axis=1)<=speed*.05+1e-9))
                for angles,f in zip(q,result['frames']):
                    np.testing.assert_allclose(ARM.fk(angles)*1000,f['position'],atol=1e-9)
                    self.assertAlmostEqual(np.linalg.norm(np.array(f['position'])-fixture['target_mm']),f['error_mm'],places=8)
                self.assertEqual(result['converged'],result['final_error_mm']<=1)
                self.assertLessEqual(result['iterations'],60)
                if method.startswith('null_'):self.assertLess(result['null_residual_max'],1e-9)

    def test_unreachable_target_reports_failure(self):
        result=solve(np.zeros(6),np.array([.8,0,0]),steps=60)
        self.assertFalse(result['converged'])
        self.assertGreater(result['final_error_mm'],100)

    def test_invalid_inputs_rejected(self):
        info=service.info()
        base={'start_deg':info['default_start_deg'],'target_mm':info['default_target_mm'],'methods':['dls']}
        for change in ({'start_deg':[0]*5},{'start_deg':[999]*6},{'target_mm':[float('nan'),0,0]}, {'methods':['unknown']},{'methods':['dls','dls']},{'steps':0},{'steps':float('inf')},{'steps':True},{'speed_rad_s':True},{'speed_rad_s':0},{'damping':0}):
            with self.subTest(change=change),self.assertRaises((ValueError,TypeError)):
                service.run({**base,**change})


class SimulationHTTPTest(unittest.TestCase):
    def setUp(self):
        self.hardware_calls=[]
        def forbidden(*args):
            self.hardware_calls.append(args)
            raise AssertionError('Simulation invoked robot transport')
        self.controller=RobotController(forbidden)
        self.handler=type('SimulationHandler',(Handler,),{'controller':self.controller,'token':'sim-test','log_message':lambda *args:None})
        self.http=ThreadingHTTPServer(('127.0.0.1',0),self.handler)
        threading.Thread(target=self.http.serve_forever,daemon=True).start()
        self.base=f'http://127.0.0.1:{self.http.server_port}'
    def tearDown(self):
        self.http.shutdown();self.http.server_close();self.controller.shutdown()
    def post(self,path,body):
        return urllib.request.urlopen(urllib.request.Request(self.base+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json','X-Robot-Token':'sim-test'}),timeout=10)
    def test_simulation_works_with_zero_hardware_calls(self):
        info=json.load(urllib.request.urlopen(self.base+'/api/sim/info'))
        with self.post('/api/sim/solve',{'start_deg':info['default_start_deg'],'target_mm':info['default_target_mm'],'methods':['seed_dls','dls']}) as response:
            result=json.load(response)
        self.assertTrue(result['simulation_only'])
        self.assertEqual(len(result['results']),2)
        self.assertEqual(self.hardware_calls,[])
        self.assertFalse(self.controller.snapshot()['connected'])
    def test_default_server_rejects_every_motion_route(self):
        for path in ('/api/connect','/api/move','/api/jog','/api/ik/plan','/api/ik/execute','/api/ik/heartbeat'):
            with self.subTest(path=path),self.assertRaises(urllib.error.HTTPError) as caught:self.post(path,{})
            self.assertEqual(caught.exception.code,403)
        self.assertEqual(self.hardware_calls,[])

    def test_prm_http_query_is_dry_run_and_has_no_hardware_plan(self):
        from simulation.prm import Roadmap
        angles = np.deg2rad([0,-40,-130,80,0,50])
        body = {'start_deg':np.rad2deg(angles).tolist(), 'target_mm':(ARM.fk(angles)*1000).tolist(), 'nn':'none'}
        with patch('simulation.prm.Roadmap.load', return_value=Roadmap()):
            with self.post('/api/sim/prm', body) as response: result = json.load(response)
        self.assertEqual(result['status'], 'success')
        self.assertTrue(result['simulation_only'])
        self.assertEqual(result['start_source'], 'simulation starting pose')
        self.assertIsNone(self.controller.ik.plan)
        self.assertEqual(self.hardware_calls, [])
