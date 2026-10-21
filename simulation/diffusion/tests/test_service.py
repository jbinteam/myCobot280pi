"""WebUI integration contracts using the actual model and collision scene."""
import unittest
from unittest.mock import patch
import numpy as np
from simulation.kinematics import ARM
from simulation.diffusion import service
from simulation.diffusion.env import ArmEnv


class ServiceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.info=service.info()
        if cls.info['status']!='ready': raise unittest.SkipTest(cls.info.get('reason','Diffusion model not installed'))
        cls.body={k:cls.info['cases'][1][k] for k in
                  ('start_deg','goal_deg','target_mm','target_axis','obstacle_mm')}

    def test_actual_policy_frames_reproduce_and_pass_scene_fk_speed_checks(self):
        result=service.plan(self.body);again=service.plan(self.body)
        self.assertTrue(result['converged']);self.assertTrue(result['simulation_only'])
        self.assertEqual(result['checkpoint_sha256'],self.info['checkpoint_sha256'])
        self.assertEqual(result['frames'],again['frames'])
        q=np.deg2rad([f['q'] for f in result['frames']])
        times=np.asarray([f['time'] for f in result['frames']])
        with ArmEnv() as env:
            env.set_obstacle(np.asarray(self.body['obstacle_mm'])/1000)
            self.assertTrue(env.path_valid(q));self.assertLessEqual(np.linalg.norm(np.diff(q,axis=0)/np.diff(times)[:,None],axis=1).max(),1.+1e-10)
        for frame,joints in zip(result['frames'],q):
            np.testing.assert_allclose(frame['position'],ARM.fk(joints)*1000,atol=1e-9)
        self.assertLessEqual(result['final_error_mm'],1.)
        self.assertLessEqual(result['final_orientation_error_deg'],1.)
        from simulation.diffusion.smoothing import motion_bounds
        self.assertLessEqual(motion_bounds(q,times)[1],3.+1e-8)
        self.assertTrue(result['smoothing']['starts_and_ends_at_rest'])
        self.assertEqual(result['dt'],.02)

    def test_smoothing_failure_returns_no_playable_original_path(self):
        with patch('simulation.diffusion.smoothing.prepare',side_effect=ValueError('recheck rejected')):
            result=service.plan(self.body)
        self.assertFalse(result['converged']);self.assertEqual(result['frames'],[])
        self.assertEqual(result['status'],'smoothing_validation_failed')

    def test_no_block_uses_zero_presence_features_and_holds_when_budget_expires(self):
        from simulation.diffusion.common import observation
        with patch.object(service,'observation',wraps=observation) as encoded:
            result=service.plan({**self.body,'obstacle_mm':None,'max_decisions':1})
        self.assertGreater(encoded.call_count,0)
        for call in encoded.call_args_list:
            self.assertIsNone(call.args[2]);np.testing.assert_array_equal(observation(*call.args)[-7:],np.zeros(7))
        self.assertEqual(result['scene_mode'],'no_block');self.assertIsNone(result['obstacle_mm'])
        self.assertEqual(result['collision_checks'],['joint_limits','self','table'])
        self.assertEqual(result['frames'],[]);self.assertFalse(result['converged'])

    def test_no_block_hold_still_returns_checked_smoothed_references(self):
        pose=ARM.transform(np.deg2rad(self.body['start_deg']))
        result=service.plan({**self.body,'obstacle_mm':None,'goal_deg':self.body['start_deg'],
            'target_mm':(pose[:3,3]*1000).tolist(),'target_axis':pose[:3,2].tolist()})
        self.assertTrue(result['converged']);self.assertEqual(len(result['frames']),1)
        self.assertTrue(result['smoothing']['collision_rechecked']);self.assertEqual(result['scene_mode'],'no_block')

    def test_no_block_does_not_disable_table_clearance_or_accept_missing_scene(self):
        q=np.deg2rad([-48.83593721397369,-99.69114265516853,-45.334503737488916,
                      -60.48369054966126,60.208769635867064,87.33884854371586])
        with ArmEnv() as env:
            env.set_obstacle(None);self.assertTrue(env.space.valid(q));self.assertFalse(env.valid(q))
        with patch('simulation.diffusion.policy.DiffusionPolicy.sample',side_effect=AssertionError('Table collision must hold')):
            result=service.plan({**self.body,'start_deg':np.rad2deg(q).tolist(),'obstacle_mm':None})
        self.assertEqual(result['status'],'invalid_start');self.assertEqual(result['frames'],[])
        missing=dict(self.body);missing.pop('obstacle_mm')
        with self.assertRaises(ValueError):service.plan(missing)

    def test_exhausted_budget_returns_no_partial_playback(self):
        result=service.plan({**self.body,'max_decisions':1})
        self.assertEqual(result['status'],'decision_limit');self.assertFalse(result['converged'])
        self.assertEqual(result['frames'],[])

    def test_unsafe_start_holds_and_never_samples(self):
        with patch('simulation.diffusion.policy.DiffusionPolicy.sample',side_effect=AssertionError('Unsafe scene cannot sample')):
            result=service.plan({**self.body,'obstacle_mm':[0,0,15,250,250,50]})
        self.assertEqual(result['status'],'invalid_start');self.assertEqual(result['frames'],[])

    def test_custom_cartesian_ik_can_hold_current_valid_pose_without_goal_hint(self):
        q=np.deg2rad(self.body['start_deg']);transform=ARM.transform(q)
        body={**self.body,'target_mm':(transform[:3,3]*1000).tolist(),'target_axis':transform[:3,2].tolist()}
        body.pop('goal_deg')
        result=service.plan(body)
        self.assertTrue(result['converged']);self.assertEqual(len(result['frames']),1)

    def test_final_validation_requires_the_requested_cartesian_goal(self):
        body=dict(self.body);body.pop('goal_deg')
        with patch.object(ArmEnv,'inverse_kinematics',return_value=np.deg2rad(body['start_deg'])):
            result=service.plan(body)
        self.assertFalse(result['converged']);self.assertEqual(result['frames'],[])
        self.assertEqual(result['status'],'final_validation_failed')

    def test_endpoint_search_failure_reports_diagnostics_without_sampling(self):
        from simulation.diffusion.endpoint_ik import IKSearchExhausted
        body=dict(self.body);body.pop('goal_deg')
        diagnostic={'method':'exhausted','legacy_attempts':32,'fallback_attempts':32,
                    'best_position_error_mm':23.6,'best_direction_error_deg':1.9,
                    'collision_rejections':0,'elapsed_ms':100}
        def exhausted(env,*args,**kwargs):
            self.assertTrue(kwargs['robust'])
            self.assertEqual(kwargs['attempts'],32)
            env.ik_diagnostics=diagnostic
            raise IKSearchExhausted(diagnostic)
        with patch.object(ArmEnv,'inverse_kinematics',exhausted),patch(
                'simulation.diffusion.policy.DiffusionPolicy.sample',
                side_effect=AssertionError('Failed endpoint must not sample')):
            result=service.plan(body)
        self.assertEqual(result['status'],'ik_failed')
        self.assertEqual(result['ik_diagnostics'],diagnostic)
        self.assertFalse(result['converged']);self.assertEqual(result['frames'],[])

    def test_endpoint_runtime_fault_is_not_reported_as_search_exhaustion(self):
        body=dict(self.body);body.pop('goal_deg')
        with patch.object(ArmEnv,'inverse_kinematics',side_effect=RuntimeError('checker fault')):
            with self.assertRaisesRegex(RuntimeError,'checker fault'):service.plan(body)

    def test_invalid_and_mismatched_inputs_are_rejected_before_inference(self):
        for key,value in [('start_deg',[False]*6),('target_mm',[float('inf'),0,0]),
                          ('target_axis',[0,0,0]),('seed',True),('samples',100),
                          ('goal_deg',[0]*6),('obstacle_mm',[0,0,0,-1,1,1])]:
            with self.subTest(key=key),self.assertRaises(ValueError):service.plan({**self.body,key:value})

    def test_busy_and_failure_release_the_serial_inference_lock(self):
        service._lock.acquire()
        try:
            with self.assertRaises(service.PlannerBusy):service.plan(self.body)
        finally:service._lock.release()
        with patch.object(service,'info',return_value={'status':'unavailable','reason':'missing checkpoint'}):
            with self.assertRaisesRegex(RuntimeError,'missing checkpoint'):service.plan(self.body)
        self.assertTrue(service._lock.acquire(blocking=False));service._lock.release()

    def test_modified_checkpoint_is_unavailable_without_loading_torch_weights(self):
        real_hash=service.file_sha256
        with patch.object(service,'file_sha256',side_effect=lambda path:'modified' if str(path).endswith('.pth') else real_hash(path)):
            result=service.info()
        self.assertEqual(result['status'],'unavailable');self.assertIn('differs',result['reason'])

    def test_modified_active_manifest_is_unavailable(self):
        if not service.ACTIVE.exists():self.skipTest('Default v1 has no active deployment pointer')
        real_hash=service.file_sha256
        with patch.object(service,'file_sha256',side_effect=lambda path:'modified' if str(path).endswith('webui.json') else real_hash(path)):
            result=service.info()
        self.assertEqual(result['status'],'unavailable');self.assertIn('manifest',result['reason'])


if __name__=='__main__':unittest.main()
