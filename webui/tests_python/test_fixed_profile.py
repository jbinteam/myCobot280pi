import unittest
import numpy as np
from simulation.motion_limits import scale_needed
from webui.fixed_profile import retime_fixed, waypoint_intervals, selected_speed


class FixedProfileTest(unittest.TestCase):
    def result(self, end):
        return {'target_mm':[100,0,100], 'converged':True,
                'frames':[{'q':[0]*6},{'q':end}]}

    def test_signed_normalization_uses_selected_manufacturer_allowance(self):
        end=[0,0,0,0,-5,20]
        for speed, allowance in ((5,7.5),(30,45.),(80,120.)):
            result=retime_fixed(self.result(end),speed)
            timing=result['smoothing']; segment=timing['segments'][0]
            self.assertAlmostEqual(result['joint_velocity_allowance_deg_s'],allowance)
            self.assertEqual(result['planned_sdk_speed'],speed)
            np.testing.assert_allclose(segment['normalized_joint_direction'],[0,0,0,0,-.25,1])
            self.assertLessEqual(segment['velocity_allowance_scale'],1.)
            self.assertGreater(segment['velocity_allowance_scale'],0.)
            self.assertLessEqual(scale_needed(timing['motion_bounds']),1.+1e-8)
            # At every phase, one shared shape and scale retain signed ratios.
            for frame in result['frames']:
                self.assertAlmostEqual(frame['q'][4],-.25*frame['q'][5])
            np.testing.assert_allclose(result['frames'][-1]['q'],end)

    def test_low_speed_sets_peak_and_inverted_schedule_matches_profile(self):
        end=[0,0,0,0,0,20]
        result=retime_fixed(self.result(end),5)
        segment=result['smoothing']['segments'][0]
        self.assertGreater(segment['velocity_allowance_scale'],.99)
        waypoints=[[0,0,0,0,0,float(q)] for q in (2,8,12,18,20)]
        intervals=waypoint_intervals(result,waypoints)
        self.assertTrue(all(t>0 for t in intervals))
        self.assertAlmostEqual(sum(intervals),result['smoothing']['duration_s'])
        for time,point in zip(np.cumsum(intervals),waypoints):
            u=time/segment['duration_s']
            self.assertAlmostEqual(20*(10*u**3-15*u**4+6*u**5),point[-1],places=8)
        with self.assertRaises(ValueError):waypoint_intervals(result,[[0,0,0,1,0,20]])

    def test_speed_validation_and_retiming_keep_route_vertices(self):
        result=retime_fixed(self.result([0,0,0,0,0,20]),5)
        fast=retime_fixed(result,80)
        self.assertEqual(len(fast['smoothing']['segments']),1)
        self.assertLess(fast['smoothing']['duration_s'],result['smoothing']['duration_s'])
        for value in (True,0,81,5.5):
            with self.assertRaises(ValueError):selected_speed({'speed':value},5)
