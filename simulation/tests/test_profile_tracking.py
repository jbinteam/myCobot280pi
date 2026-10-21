import json
import unittest
import numpy as np
from simulation import profile_tracking as timing
from simulation.reference_shape import evaluate
from simulation.kinematics import ARM
from simulation.diffusion.env import ArmEnv
from simulation.diffusion import classical_tracking

SHAPE={'kind':'smooth_trapezoid','rise_percent':5,'fall_percent':10}

class ProfileTrackingTest(unittest.TestCase):
    def setUp(self):
        self.env=ArmEnv();self.env.set_obstacle(None)
        self.start=np.deg2rad([0,-40,-130,80,0,50]);self.goal=self.start.copy();self.goal[0]+=.08
    def tearDown(self):self.env.close()
    def curve(self,shape=SHAPE,v=None,a=None):
        return timing.prepare(self.env,[self.start,self.goal],np.zeros(6) if v is None else v,np.zeros(6) if a is None else a,velocity_shape=shape)
    def test_resting_profile_matches_fixed_target_time_law_and_derivatives_exactly(self):
        for shape in (SHAPE,{'kind':'smooth_trapezoid','rise_percent':20,'fall_percent':40},{'kind':'quintic'}):
            states,times,curve=self.curve(shape)
            T=curve['duration_s'];delta=self.goal-self.start
            for u in [0,.025,.05,.1,.4,.9,.95,1]:
                q,v,a=timing.sample_motion(curve,u*T);p,rate,acc=evaluate(u,shape)
                np.testing.assert_allclose(q,self.start+p*delta,atol=1e-12)
                np.testing.assert_allclose(v,rate*delta/T,atol=1e-12)
                np.testing.assert_allclose(a,acc*delta/T**2,atol=1e-12)
            self.assertLessEqual(curve['max_velocity_rad_s'],1+1e-8)
            self.assertLessEqual(curve['max_acceleration_rad_s2'],3+1e-8)
    def test_moving_join_preserves_motion_and_independent_validation_rejects_tampering(self):
        v=np.array([.03,0,0,0,0,0]);a=np.array([.02,0,0,0,0,0])
        states,times,curve=self.curve(v=v,a=a)
        for actual,expected in zip(timing.sample_motion(curve,0),(self.start,v,a)):
            np.testing.assert_allclose(actual,expected,atol=1e-12)
        frames=timing.frames_for(states,times,ARM.fk(self.goal),ARM.transform(self.goal)[:3,2])
        timing.validate(curve,frames,(self.start,v,a),1.,self.env)
        bad=json.loads(json.dumps(curve));bad['segments'][0]['start_velocity_rad_s'][0]+=.02
        with self.assertRaises(ValueError):timing.validate(bad,frames,(self.start,v,a),1.,self.env)
        bad=json.loads(json.dumps(curve));bad['reference_shape']['rise_percent']=20
        with self.assertRaises(ValueError):timing.validate(bad,frames,(self.start,v,a),1.,self.env)
    def test_bounded_resting_prefix_uses_same_shape_and_finishes_at_rest(self):
        states,times,curve=timing.prepare_tracking(self.env,[self.start,self.goal],.1,velocity_shape=SHAPE)
        self.assertLessEqual(times[-1],1.3+1e-9);self.assertTrue(curve['starts_and_ends_at_rest'])
        self.assertEqual(curve['reference_shape'],SHAPE)
        self.assertFalse(np.allclose(states[-1],self.goal))
    def test_actual_classical_tracking_applies_shape_before_preview_and_validates_scene(self):
        t=ARM.transform(np.deg2rad([3,-42,-125,78,2,50]))
        body={'start_deg':np.rad2deg(self.start).tolist(),'cursor_mm':(t[:3,3]*1000+50*t[:3,2]).tolist(),
              'target_axis':t[:3,2].tolist(),'obstacle_mm':None,'continuous_motion':True}
        result=classical_tracking.plan({**body,'velocity_shape':SHAPE})
        self.assertTrue(result['playable'],result)
        self.assertEqual(result['smoothing']['method'],timing.METHOD)
        self.assertEqual(result['smoothing']['reference_shape'],SHAPE)
        timing.validate(result['smoothing'],result['frames'],(np.deg2rad(body['start_deg']),np.zeros(6),np.zeros(6)),1.,self.env)
    def test_invalid_ramps_or_boundary_motion_are_rejected(self):
        with self.assertRaises(ValueError):self.curve({'kind':'smooth_trapezoid','rise_percent':80,'fall_percent':80})
        with self.assertRaises(ValueError):self.curve(v=np.ones(6))

if __name__=='__main__':unittest.main()
