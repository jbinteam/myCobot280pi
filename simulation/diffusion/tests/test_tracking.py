"""Tracking chunks must preserve cursor semantics and the shared collision gate."""
import unittest
from unittest.mock import patch
import numpy as np
from simulation import service as simulation
from simulation.kinematics import ARM
from simulation.diffusion import service, tracking
from simulation.diffusion.env import ArmEnv


class TrackingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.info=service.info()
        if cls.info['status']!='ready':raise unittest.SkipTest('Deployed diffusion model unavailable')
        cls.case=cls.info['cases'][1]

    def body(self, offset=(4,2,1)):
        pose=ARM.transform(np.deg2rad(self.case['start_deg']))
        return {'tracking_engine':'local_diffusion','start_deg':self.case['start_deg'],
            'cursor_mm':(pose[:3,3]*1000+50*pose[:3,2]+offset).tolist(),
            'obstacle_mm':self.case['obstacle_mm'],'pose5_method':'bounded_dls'}

    def audit(self, result, body):
        self.assertTrue(result['playable']);self.assertTrue(result['goal_cursor_acceptance']['accepted'])
        self.assertEqual(result['replan_required'],not result['converged'])
        states=np.deg2rad([f['q'] for f in result['frames']]);times=np.array([f['time'] for f in result['frames']])
        np.testing.assert_allclose(result['frames'][0]['q'],body['start_deg'],atol=1e-9)
        self.assertLessEqual(times[-1],1.3+1e-9)
        with ArmEnv() as env:
            env.set_obstacle(None if body['obstacle_mm'] is None else np.array(body['obstacle_mm'])/1000)
            self.assertTrue(env.path_valid(states))
        if len(states)>1:self.assertLessEqual(np.linalg.norm(np.diff(states,axis=0)/np.diff(times)[:,None],axis=1).max(),result['speed_rad_s']+1e-9)
        from simulation.diffusion.smoothing import motion_bounds
        self.assertLessEqual(motion_bounds(states,times)[1],3.+1e-8)
        self.assertTrue(result['smoothing']['collision_rechecked'])
        for frame,q in zip(result['frames'],states):np.testing.assert_allclose(frame['position'],ARM.fk(q)*1000,atol=1e-9)

    def test_partial_local_chunk_is_playable_without_claiming_cursor_reached(self):
        body=self.body()
        with patch('simulation.diffusion.policy.DiffusionPolicy.sample',side_effect=AssertionError('Local IK must run first')):
            result=simulation.mouse_aim_plan(body)
        self.audit(result,body);self.assertEqual(result['tracking_source'],'local_ik')
        self.assertFalse(result['converged']);self.assertFalse(result['cursor_acceptance']['accepted'])

    def test_no_block_tracking_removes_cached_box_and_revalidates_scene_changes(self):
        body={**self.body(),'obstacle_mm':None}
        self.audit(simulation.mouse_aim_plan(body),body)
        self.assertIsNone(tracking._env.box);self.assertIsNone(tracking._env.obstacle)
        result=simulation.mouse_aim_plan({**body,'obstacle_mm':[0,0,15,250,250,50]})
        self.assertFalse(result['playable']);self.assertEqual(result['status'],'invalid_start')
        result=simulation.mouse_aim_plan(body);self.audit(result,body)
        self.assertEqual(result['scene_mode'],'no_block');self.assertIsNone(result['obstacle_mm'])
        self.assertIsNone(tracking._env.box);self.assertEqual(result['collision_checks'],['joint_limits','self','table'])

    def test_no_block_diffusion_fallback_receives_absence_mask(self):
        body={**self.body(),'obstacle_mm':None}
        with patch.object(tracking,'local_pose_plan',return_value={'converged':False,'local_ik':{'status':'path_invalid'}}), \
             patch('simulation.diffusion.policy.DiffusionPolicy.sample',return_value=np.tile(ARM.lower,(1,4,16,1))) as sample:
            result=simulation.mouse_aim_plan(body)
        np.testing.assert_array_equal(sample.call_args.args[0][0,12:19],np.zeros(7))
        self.assertEqual(result['status'],'no_safe_chunk');self.assertEqual(result['frames'],[])

    def test_replanning_current_joints_reaches_a_stationary_cursor(self):
        body=self.body()
        for i in range(30):
            result=simulation.mouse_aim_plan({**body,'seed':i})
            self.audit(result,body)
            if result['converged']:break
            body['start_deg']=result['frames'][-1]['q']
            body['target_axis']=result['target_axis']
            body['goal_hint_deg']=result['goal_deg']
        self.assertTrue(result['converged']);self.assertTrue(result['cursor_acceptance']['accepted'])

    def test_changed_scene_invalidates_cached_collision_answers(self):
        body=self.body();self.assertTrue(simulation.mouse_aim_plan(body)['playable'])
        with patch.object(tracking.pose5,'solve',side_effect=AssertionError('Invalid start must hold before IK')):
            result=simulation.mouse_aim_plan({**body,'obstacle_mm':[0,0,15,250,250,50]})
        self.assertEqual(result['status'],'invalid_start');self.assertEqual(result['frames'],[])
        self.assertFalse(result['playable'])
        self.assertTrue(simulation.mouse_aim_plan(body)['playable'])

    def test_real_diffusion_is_a_single_checked_decision_after_local_failure(self):
        case=self.info['cases'][2];goal=ARM.transform(np.deg2rad(case['goal_deg']))
        body={**self.body(),'start_deg':case['start_deg'],'obstacle_mm':case['obstacle_mm'],
              'cursor_mm':(goal[:3,3]*1000+50*goal[:3,2]).tolist()}
        result=simulation.mouse_aim_plan(body)
        self.audit(result,body);self.assertEqual(result['tracking_source'],'diffusion_fallback')
        self.assertEqual(result['reference_steps'],2);self.assertEqual(result['proposed_chunks'],4)

    def test_all_unsafe_raw_proposals_hold_and_never_return_partial_motion(self):
        body=self.body();unsafe=np.tile(ARM.lower,(1,4,16,1))
        with ArmEnv() as env:self.assertFalse(env.valid(ARM.lower))
        with patch.object(tracking,'local_pose_plan',return_value={'converged':False,'local_ik':{'status':'path_invalid'}}), \
                patch('simulation.diffusion.policy.DiffusionPolicy.sample',return_value=unsafe):
            result=simulation.mouse_aim_plan(body)
        self.assertEqual(result['status'],'no_safe_chunk');self.assertEqual(result['frames'],[])
        self.assertFalse(result['playable']);self.assertEqual(result['rejections']['raw_collision'],4)

    def test_invalid_requests_and_foreign_planning_spaces_cannot_bypass_validation(self):
        for key,value in [('cursor_mm',[True,0,0]),('start_deg',[0]*5),('obstacle_mm',[0,0,0,1,1,1]),
                          ('samples',False),('seed',-1),('pose5_method','invalid'),('speed_rad_s',99)]:
            with self.subTest(key=key),self.assertRaises(ValueError):simulation.mouse_aim_plan({**self.body(),key:value})
        with self.assertRaises(ValueError):simulation.mouse_aim_plan(self.body(),space=object())

    def test_busy_and_exception_release_shared_scheduler_lock(self):
        service._lock.acquire()
        try:
            with self.assertRaises(service.PlannerBusy):simulation.mouse_aim_plan(self.body())
        finally:service._lock.release()
        with patch.object(service,'info',return_value={'status':'unavailable','reason':'unavailable'}):
            with self.assertRaises(RuntimeError):simulation.mouse_aim_plan(self.body())
        self.assertTrue(service._lock.acquire(blocking=False));service._lock.release()


if __name__=='__main__':unittest.main()
