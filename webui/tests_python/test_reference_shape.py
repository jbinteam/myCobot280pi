import unittest
import numpy as np
from simulation.reference_shape import settings, evaluate, peaks
from simulation.motion_limits import scale_needed
from webui.fixed_profile import retime_fixed, waypoint_intervals
from webui.encoder_profile import motion


class ReferenceShapeTest(unittest.TestCase):
    def test_rejects_invalid_and_unbounded_shapes(self):
        for value in ([], {}, {'kind':'unknown'}, {'kind':'quintic','rise_percent':25},
                      {'kind':'smooth_trapezoid','rise_percent':True,'fall_percent':25},
                      {'kind':'smooth_trapezoid','rise_percent':4,'fall_percent':25},
                      {'kind':'smooth_trapezoid','rise_percent':80,'fall_percent':25},
                      {'kind':'smooth_trapezoid','rise_percent':float('nan'),'fall_percent':25}):
            with self.subTest(value=value), self.assertRaises(ValueError): settings(value)

    def test_integrated_velocity_has_unit_area_and_continuous_acceleration(self):
        for rise, fall in ((5,5),(20,60),(75,25),(25,25)):
            shape=settings({'kind':'smooth_trapezoid','rise_percent':rise,'fall_percent':fall})
            grid=np.linspace(0,1,20001);position,velocity,acceleration=evaluate(grid,shape)
            self.assertAlmostEqual(np.trapezoid(velocity,grid),1.,places=8)
            self.assertEqual(position[0],0.);self.assertEqual(position[-1],1.)
            self.assertTrue(np.all(np.diff(position)>=-1e-14))
            np.testing.assert_allclose(velocity[[0,-1]],0.,atol=1e-12)
            np.testing.assert_allclose(acceleration[[0,-1]],0.,atol=1e-12)
            h=1e-6
            for u in (.02,.3,.7,.93):
                before=evaluate(u-h,shape);at=evaluate(u,shape);after=evaluate(u+h,shape)
                self.assertAlmostEqual((after[0]-before[0])/(2*h),at[1],places=6)
                self.assertAlmostEqual((after[1]-before[1])/(2*h),at[2],places=5)
            for u in (rise/100,1-fall/100):
                before=evaluate(u-1e-8,shape);after=evaluate(u+1e-8,shape)
                np.testing.assert_allclose(before,after,atol=3e-5)
            peak_rate,peak_acc=peaks(shape)
            self.assertLessEqual(np.max(velocity),peak_rate+1e-12)
            self.assertLessEqual(np.max(np.abs(acceleration)),peak_acc+1e-9)

    def test_retiming_preserves_corners_ratios_and_stops_under_motion_envelope(self):
        path=np.deg2rad([[0,0,0,0,0,0],[10,-5,0,0,0,0],[10,-5,0,4,0,0]])
        for rise,fall in ((5,80),(25,25),(60,20)):
            shape={'kind':'smooth_trapezoid','rise_percent':rise,'fall_percent':fall}
            result=retime_fixed({'converged':True,'target_mm':[100,0,100],
                'frames':[{'q':q.tolist()} for q in np.rad2deg(path)]},80,path=path,velocity_shape=shape)
            curve=result['smoothing'];self.assertEqual(len(curve['segments']),2)
            self.assertLessEqual(scale_needed(curve['motion_bounds']),1.+1e-10)
            for segment in curve['segments']:
                delta=np.asarray(segment['end_deg'])-segment['start_deg']
                for u in np.linspace(0,1,101):
                    q,v,a,_=motion(curve,segment['start_time_s']+segment['duration_s']*u)
                    if u in (0.,1.):
                        np.testing.assert_allclose(v,0.,atol=1e-9);np.testing.assert_allclose(a,0.,atol=1e-9)
                    if 0<u<1:
                        moving=np.abs(delta)>1e-10
                        ratios=v[moving]/delta[moving]
                        np.testing.assert_allclose(ratios,ratios[0],atol=1e-12)
            np.testing.assert_allclose(result['frames'][-1]['q'],np.rad2deg(path[-1]))

    def test_waypoint_schedule_inverts_the_selected_shape(self):
        result=retime_fixed({'converged':True,'target_mm':[100,0,100],
            'frames':[{'q':[0]*6},{'q':[20,0,0,0,0,0]}]},5,
            velocity_shape={'kind':'smooth_trapezoid','rise_percent':20,'fall_percent':50})
        points=[[q,0,0,0,0,0] for q in (1,5,13,20)]
        intervals=waypoint_intervals(result,points)
        for phase,point in zip(np.cumsum(intervals),points):
            np.testing.assert_allclose(motion(result['smoothing'],phase)[0],point,atol=1e-8)
        self.assertAlmostEqual(sum(intervals),result['smoothing']['duration_s'])
