import unittest
import json
from unittest.mock import patch
import numpy as np
from simulation.continuous import prepare, evaluate, initial_motion, bound
from simulation.prm import JointSpace
from simulation.kinematics import ARM
from simulation import service
from simulation.diffusion.env import ArmEnv


class ContinuousTest(unittest.TestCase):
    def space(self,valid=lambda q:True):return JointSpace(valid,{'checker':'continuous test','environment':str(id(valid))})

    def test_carries_nonzero_velocity_and_acceleration_with_proven_bounds(self):
        start=np.zeros(6);goal=start.copy();goal[0]=.3
        v=np.array([.12,0,0,0,0,0]);acc=np.array([.2,0,0,0,0,0])
        states,times,metadata=prepare(self.space(),[start,goal],v,acc)
        c=np.deg2rad(metadata['segments'][0]['controls_deg']);duration=times[-1]
        np.testing.assert_allclose(5*(c[1]-c[0])/duration,v,atol=1e-12)
        np.testing.assert_allclose(20*(c[2]-2*c[1]+c[0])/duration**2,acc,atol=1e-12)
        np.testing.assert_array_equal(states[-1],goal)
        self.assertFalse(metadata['starts_and_ends_at_rest']);self.assertTrue(metadata['ends_at_rest'])
        vc=5*np.diff(c,axis=0)/duration;ac=20*np.diff(c,n=2,axis=0)/duration**2
        self.assertLessEqual(bound(vc),1.+1e-9);self.assertLessEqual(bound(ac),3.+1e-9)
        u=np.linspace(0,1,1001)
        self.assertLessEqual(np.linalg.norm(evaluate(vc,u),axis=1).max(),1.)
        self.assertLessEqual(np.linalg.norm(evaluate(ac,u),axis=1).max(),3.)
        np.testing.assert_allclose(evaluate(vc,1),np.zeros(6),atol=1e-12)
        np.testing.assert_allclose(evaluate(ac,1),np.zeros(6),atol=1e-12)

    def test_safe_straight_route_does_not_authorize_a_colliding_curved_join(self):
        space=self.space(lambda q:q[1]>=-.00001)
        start=np.zeros(6);goal=start.copy();goal[0]=.3
        self.assertTrue(space.edge_valid(start,goal,.04))
        with self.assertRaisesRegex(ValueError,'No collision-free continuous join'):
            prepare(space,[start,goal],np.array([0,-.2,0,0,0,0]),np.zeros(6))

    def test_invalid_motion_is_rejected_and_a_rest_hold_has_no_artificial_motion(self):
        for body in [{'continuous_motion':'yes'}, {'continuous_motion':True,'start_velocity_rad_s':[2,0,0,0,0,0]},
                     {'continuous_motion':True,'start_acceleration_rad_s2':[4,0,0,0,0,0]},
                     {'continuous_motion':False,'start_velocity_rad_s':[.1,0,0,0,0,0]}]:
            with self.assertRaises(ValueError):initial_motion(body,1.)
        states,times,metadata=prepare(self.space(),[np.zeros(6)],np.zeros(6),np.zeros(6))
        self.assertEqual(len(states),1);self.assertEqual(times[-1],0);self.assertEqual(metadata['segments'],[])

    def test_real_arm_both_engines_return_a_complete_checked_local_motion(self):
        case=service.info()['diffusion']['cases'][1];start=np.deg2rad(case['start_deg']);pose=ARM.transform(start)
        body={'start_deg':case['start_deg'],'cursor_mm':(pose[:3,3]*1000+50*pose[:3,2]+[4,2,1]).tolist(),
              'pose5_method':'bounded_dls','obstacle_mm':None,'continuous_motion':True,'speed_rad_s':1.}
        for engine in ['classical','local_diffusion']:
            with patch('simulation.diffusion.policy.DiffusionPolicy.sample',side_effect=AssertionError('Local IK must run first')):
                result=service.mouse_aim_plan({**body,'tracking_engine':engine})
            self.assertTrue(result['converged'],result.get('reason'));self.assertEqual(result['tracking_source'],'local_ik')
            json.dumps(result,allow_nan=False)
            self.assertEqual(result['smoothing']['method'],'collision_checked_bezier_c2')
            self.assertTrue(result['smoothing']['ends_at_rest']);self.assertTrue(result['cursor_acceptance']['accepted'])
            states=np.deg2rad([f['q'] for f in result['frames']])
            with ArmEnv() as env:
                env.set_obstacle(None);self.assertTrue(env.path_valid(states))
            np.testing.assert_allclose(states[0],start,atol=1e-12)

    def test_real_diffusion_fallback_preserves_checked_full_horizon(self):
        case=service.info()['diffusion']['cases'][2];goal=ARM.transform(np.deg2rad(case['goal_deg']))
        body={'start_deg':case['start_deg'],'cursor_mm':(goal[:3,3]*1000+50*goal[:3,2]).tolist(),
              'pose5_method':'bounded_dls','obstacle_mm':case['obstacle_mm'],
              'tracking_engine':'local_diffusion','continuous_motion':True,'seed':42}
        result=service.mouse_aim_plan(body)
        self.assertTrue(result['playable'],result.get('reason'));self.assertEqual(result['tracking_source'],'diffusion_fallback')
        self.assertEqual(result['reference_steps'],16);self.assertEqual(result['proposed_chunks'],4)
        self.assertLessEqual(result['smoothing']['duration_s'],12.)
        self.assertTrue(result['smoothing']['collision_rechecked']);json.dumps(result,allow_nan=False)
        with ArmEnv() as env:
            env.set_obstacle(np.array(case['obstacle_mm'])/1000)
            for segment in result['smoothing']['segments']:
                controls=np.deg2rad(segment['controls_deg'])
                states=evaluate(controls,np.linspace(0,1,int(np.ceil(segment['duration_s']/.005))+1))
                self.assertTrue(env.path_valid(states))


if __name__=='__main__':unittest.main()
