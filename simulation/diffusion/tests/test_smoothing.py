import unittest
from unittest.mock import patch
import numpy as np
from simulation.prm import JointSpace
from simulation.diffusion.smoothing import prepare, prepare_tracking, motion_bounds


class Barrier:
    def __init__(self):self.space=JointSpace(lambda q:not(abs(q[0])<.15 and abs(q[1])<.3),{'checker':'barrier','environment':'test'})
    def edge_valid(self,a,b,step=.04):return self.space.edge_valid(a,b,step)
    def path_valid(self,q,step=.04):return all(self.space.valid(s) for s in q) and all(self.edge_valid(a,b,step) for a,b in zip(q[:-1],q[1:]))


class SmoothingTest(unittest.TestCase):
    def path(self):return np.array([[-.5,0,0,0,0,0],[-.5,.5,0,0,0,0],[.5,.5,0,0,0,0],[.5,0,0,0,0,0]])
    def test_smoothing_keeps_checked_corners_and_bounds_start_stop_acceleration(self):
        env=Barrier();path=self.path();states,times,info=prepare(env,path,speed=.7,acceleration=2.)
        np.testing.assert_array_equal(states[0],path[0]);np.testing.assert_array_equal(states[-1],path[-1])
        self.assertTrue(env.path_valid(states));self.assertGreater(info['path_vertices'],2)
        speed,acceleration=motion_bounds(states,times)
        self.assertLessEqual(speed,.7+1e-8);self.assertLessEqual(acceleration,2+1e-8)
        self.assertTrue(info['starts_and_ends_at_rest']);self.assertTrue(info['collision_rechecked'])
        self.assertTrue(np.allclose(np.diff(times),.02,atol=1e-10))
        self.assertGreater(len(info['segments']),1)
        for segment in info['segments']:
            a,b=segment['start_frame'],segment['end_frame']
            np.testing.assert_array_equal(segment['start_deg'],np.rad2deg(states[a]))
            np.testing.assert_array_equal(segment['end_deg'],np.rad2deg(states[b]))
            self.assertEqual(segment['start_time_s'],times[a])
            duration=segment['duration_s'];u=(times[a:b+1]-times[a])/duration
            expected=states[a]+(10*u**3-15*u**4+6*u**5)[:,None]*(states[b]-states[a])
            np.testing.assert_allclose(states[a:b+1],expected,atol=1e-12)
            distance=np.linalg.norm(states[b]-states[a])
            self.assertLessEqual(1.875*distance/duration,.7+1e-8)
            self.assertLessEqual(10*np.sqrt(3)/3*distance/duration**2,2.+1e-8)

    def test_unsafe_original_path_is_rejected_before_any_shortcut(self):
        path=self.path()[[0,-1]]
        with self.assertRaisesRegex(ValueError,'Input path'):prepare(Barrier(),path)

    def test_grazing_retimed_paths_cannot_fall_back_to_unsmoothed_execution(self):
        env=Barrier()
        with patch.object(env,'path_valid',side_effect=[True,False,False]):
            with self.assertRaisesRegex(ValueError,'Smoothed final trajectory'):prepare(env,self.path())

    def test_unchanged_pose_has_no_motion_or_artificial_frames(self):
        q=self.path()[0];states,times,metadata=prepare(Barrier(),[q])
        np.testing.assert_array_equal(states,[q]);np.testing.assert_array_equal(times,[0])
        self.assertEqual(metadata['max_acceleration_rad_s2'],0)
        self.assertEqual(metadata['segments'],[])

    def test_tracking_budget_shortens_the_checked_route_and_brakes_to_rest(self):
        env = Barrier(); path = self.path()
        states, times, metadata = prepare_tracking(env, path, duration_limit=1.3)
        self.assertLessEqual(times[-1], 1.3+1e-9)
        self.assertTrue(metadata['tracking_prefix_limited'])
        self.assertTrue(metadata['specification_checked'])
        self.assertTrue(metadata['starts_and_ends_at_rest'])
        self.assertTrue(env.path_valid(states))
        self.assertGreater(np.linalg.norm(states[-1]-states[0]), 0.)
        self.assertFalse(np.allclose(states[-1], path[-1]))
        last = metadata['segments'][-1]
        self.assertEqual(last['end_frame'], len(states)-1)
