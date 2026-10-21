"""Real scene/IK checks for disk tracking; route rejection is injected only."""
import unittest
from unittest.mock import patch
import numpy as np
from simulation.kinematics import ARM
from simulation.diffusion import classical_tracking, tracking
from simulation.diffusion.env import ArmEnv
from simulation.physical_reach import ENVELOPE


class RegionTrackingTest(unittest.TestCase):
    def test_disjoint_goal_disk_skips_ik_and_motion_search(self):
        body = {'start_deg': [0, -40, -130, 80, 0, 50],
                'cursor_mm': [600., 0., 139.56], 'target_axis': [1., 0., 0.],
                'tracking_region': {'distance_mm': 100., 'half_angle_deg': 30.},
                'obstacle_mm': None, 'continuous_motion': True}
        for planner in (classical_tracking, tracking):
            with self.subTest(planner=planner.__name__), \
                    patch.object(ArmEnv, 'inverse_kinematics', side_effect=AssertionError('Outside workspace')), \
                    patch.object(classical_tracking, 'rrt_plan', side_effect=AssertionError('Outside workspace')), \
                    patch.object(tracking, 'local_pose_plan', side_effect=AssertionError('Outside workspace')), \
                    patch.object(tracking.service, 'load_policy', side_effect=AssertionError('Outside workspace')):
                result = planner.plan(body)
            self.assertEqual(result['status'], 'outside_workspace')
            self.assertFalse(result['physical_reach_precheck']['possible'])
            self.assertFalse(result['playable'])
            self.assertEqual(result['frames'], [])

    def test_object_outside_arm_reach_can_have_a_valid_camera_goal(self):
        # A visible object above the table, beyond the arm's outer envelope.
        q = np.deg2rad([25.13735, 67.378296, 96.757345, 123.963403, -107.835649, 87.377439])
        transform = ARM.transform(q)
        cursor = transform[:3, 3]*1000+350*transform[:3, 2]
        self.assertFalse(ENVELOPE.point(cursor)['possible'])
        body = {'start_deg': np.rad2deg(q).tolist(), 'cursor_mm': cursor.tolist(),
                'tracking_region': {'distance_mm': 350., 'half_angle_deg': 30.},
                'obstacle_mm': None, 'continuous_motion': True}
        for planner in (classical_tracking, tracking):
            with self.subTest(planner=planner.__name__), \
                    patch.object(ArmEnv, 'inverse_kinematics', side_effect=AssertionError('Valid camera pose')):
                result = planner.plan(body)
            self.assertTrue(result['physical_reach_precheck']['possible'])
            self.assertTrue(result['playable'])
            self.assertTrue(result['converged'])

    def test_classical_holds_when_table_occludes_an_in_frame_object(self):
        q = np.deg2rad([0,-40,-130,80,0,50]); pose = ARM.transform(q)
        with patch.object(ArmEnv,'inverse_kinematics',side_effect=AssertionError('Invisible start')):
            result = classical_tracking.plan({'start_deg':np.rad2deg(q).tolist(),
                'cursor_mm':(pose[:3,3]*1000+350*pose[:3,2]).tolist(),
                'tracking_region':{'distance_mm':350.,'half_angle_deg':30.},
                'obstacle_mm':None,'continuous_motion':True})
        self.assertFalse(result['playable'])
        self.assertEqual(result['frames'],[])
        self.assertEqual(result['status'],'object_not_visible')
        self.assertTrue(result['visibility_start']['in_view'])
        self.assertTrue(result['visibility_start']['occluded'])
        self.assertEqual(result['visibility_start']['occluding_object'], 'table')

    def test_webui_visible_demo_has_a_checked_unoccluded_initial_target(self):
        from simulation.service import info
        from simulation.camera_model import configured_camera
        demo = info()['tracking_demo']
        result = classical_tracking.plan({'start_deg':demo['start_deg'], 'cursor_mm':demo['cursor_mm'],
            'tracking_region':{'distance_mm':demo['distance_mm'],'half_angle_deg':demo['half_angle_deg']},
            'obstacle_mm':None, 'continuous_motion':True})
        self.assertTrue(result['playable'])
        self.assertTrue(result['converged'])
        self.assertTrue(result['visibility']['passed'])
        self.assertEqual(result['visibility_model']['camera'],configured_camera().configuration())

    def test_filtered_candidates_do_not_prove_an_overlapping_disk_unreachable(self):
        body = self.body(np.deg2rad([10, -42, -125, 78, 2, 50]))
        for planner in (classical_tracking, tracking):
            with self.subTest(planner=planner.__name__), \
                    patch.object(planner, 'candidate_poses', return_value=[(np.array([.5, 0., .13956]), np.array([1., 0., 0.]))]), \
                    patch.object(ArmEnv, 'inverse_kinematics', side_effect=AssertionError('Filtered point')):
                result = planner.plan(body)
            self.assertTrue(result['physical_reach_precheck']['possible'])
            self.assertEqual(result['status'], 'ik_failed')
            self.assertEqual(result['region_candidates_filtered'], 1)
            self.assertEqual(result['region_candidates_tested'], 0)
            self.assertEqual(result['frames'], [])

    def test_first_unfiltered_classical_candidate_retains_robust_ik(self):
        body = self.body(np.deg2rad([10, -42, -125, 78, 2, 50]))
        candidates = classical_tracking.candidate_poses
        def with_far_point(region, start):
            yield np.array([.5, 0., .13956]), np.array([1., 0., 0.])
            yield from candidates(region, start)
        with patch.object(classical_tracking, 'candidate_poses', with_far_point), \
                patch.object(ArmEnv, 'inverse_kinematics', autospec=True, side_effect=ArmEnv.inverse_kinematics) as solver:
            result = classical_tracking.plan(body)
        self.assertTrue(result['playable'])
        self.assertEqual(result['region_candidates_filtered'], 1)
        self.assertTrue(solver.call_args_list[0].kwargs['robust'])

    def body(self, goal=None):
        start = np.deg2rad([0, -40, -130, 80, 0, 50])
        pose = ARM.transform(start if goal is None else goal)
        # Tilt the frozen cone axis by 8 degrees while the physical tool points
        # exactly at the cursor. This puts the EEF away from the disk center.
        tangent = pose[:3, 0]
        axis = np.cos(np.deg2rad(8))*pose[:3, 2]+np.sin(np.deg2rad(8))*tangent
        cursor = pose[:3, 3]*1000+100/np.cos(np.deg2rad(8))*pose[:3, 2]
        return {'start_deg':np.rad2deg(start).tolist(), 'cursor_mm':cursor.tolist(),
                'target_axis':axis.tolist(), 'tracking_region':{}, 'obstacle_mm':None,
                'continuous_motion':True}

    def test_both_planners_hold_an_already_valid_off_center_pose(self):
        for planner in (classical_tracking, tracking):
            with self.subTest(planner=planner.__name__), \
                    patch.object(ArmEnv, 'inverse_kinematics', side_effect=AssertionError('Valid disk pose requires no IK')):
                result = planner.plan(self.body())
            self.assertTrue(result['playable'], result)
            self.assertTrue(result['converged'], result)
            self.assertTrue(result['cursor_acceptance']['accepted'])
            self.assertEqual(result['tracking_objective'], 'gaze_first')
            self.assertLess(result['gaze_tracking']['final_direction_error_deg'], 1.)
            self.assertTrue(all(frame['cursor_direction_error_deg'] <= 1. for frame in result['frames']))
            self.assertGreater(result['cursor_acceptance']['distance_mm'], 100.)
            self.assertGreater(result['cursor_acceptance']['radial_distance_mm'], 10.)
            np.testing.assert_allclose(result['frames'][-1]['q'], self.body()['start_deg'], atol=1e-8)
            self.assertGreater(np.linalg.norm(np.asarray(result['target_mm'])-result['tracking_region']['center_mm']), 10.)

    def test_gaze_does_not_authorize_an_invalid_current_configuration(self):
        body = {**self.body(), 'obstacle_mm':[0, 0, 15, 250, 250, 50]}
        for planner in (classical_tracking, tracking):
            with self.subTest(planner=planner.__name__):
                result = planner.plan(body)
            self.assertFalse(result['playable'])
            self.assertEqual(result['status'], 'invalid_start')
            self.assertEqual(result['frames'], [])

    def test_classical_tries_another_region_point_after_a_blocked_route(self):
        body = self.body(np.deg2rad([10, -42, -125, 78, 2, 50]))
        actual_route = classical_tracking.rrt_plan
        calls = []
        def route(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                return None, {'status':'iteration_limit'}
            return actual_route(*args, **kwargs)
        with patch.object(classical_tracking, 'rrt_plan', side_effect=route):
            result = classical_tracking.plan(body)
        self.assertTrue(result['playable'], result)
        self.assertGreaterEqual(len(calls), 2)
        self.assertGreaterEqual(result['region_candidates_tested'], 2)
        self.assertLessEqual(result['region_candidates_tested'], 8)
        self.assertTrue(result['goal_cursor_acceptance']['accepted'])
        self.assertEqual(result['region_route_attempts'][0]['status'], 'iteration_limit')

    def test_diffusion_tries_another_disk_point_after_a_failed_local_path(self):
        body = self.body(np.deg2rad([10, -42, -125, 78, 2, 50]))
        actual_local = tracking.local_pose_plan
        calls = []
        def local(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                return {'converged':False, 'local_ik':{'status':'path_invalid'}}
            return actual_local(*args, **kwargs)
        with patch.object(tracking, 'local_pose_plan', side_effect=local), \
                patch.object(tracking.service, 'load_policy', side_effect=AssertionError('Disk route avoids neural fallback')):
            result = tracking.plan(body)
        self.assertTrue(result['playable'], result)
        self.assertTrue(result['converged'], result)
        self.assertGreaterEqual(len(calls), 2)
        self.assertLessEqual(result['region_candidates_tested'], 8)
        self.assertTrue(result['cursor_acceptance']['accepted'])

    def test_controller_candidate_rejection_tries_another_checked_endpoint(self):
        body = self.body(np.deg2rad([10,-42,-125,78,2,50])); calls = []
        def gate(states,times,smoothing,deadline,orientation_mode):
            calls.append(states[-1].copy())
            if len(calls)==1:
                raise ValueError('Injected encoder rounding rejection')
        result = classical_tracking.plan({**body,'_candidate_validator':gate})
        self.assertTrue(result['playable'],result)
        self.assertGreaterEqual(len(calls),2)
        self.assertIn('encoder rounding',result['region_route_attempts'][0]['rejection'])
        self.assertTrue(result['visibility']['passed'])

    def test_visible_region_preserves_measured_moving_boundary(self):
        from simulation.profile_tracking import sample_motion
        start = np.deg2rad([1,-40,-130,-40,1,50]);pose=ARM.transform(start)
        velocity=np.array([.01,0,0,0,0,0]);acceleration=np.zeros(6)
        body={'start_deg':np.rad2deg(start).tolist(),
            'cursor_mm':(pose[:3,3]*1000+350*pose[:3,2]).tolist(),
            'tracking_region':{'distance_mm':330.,'half_angle_deg':30.},'obstacle_mm':None,
            'continuous_motion':True,'speed_rad_s':.35,
            'start_velocity_rad_s':velocity.tolist(),'start_acceleration_rad_s2':acceleration.tolist(),
            'velocity_shape':{'kind':'smooth_trapezoid','rise_percent':5,'fall_percent':10}}
        result=classical_tracking.plan(body)
        self.assertTrue(result['playable'],result)
        for actual,expected in zip(sample_motion(result['smoothing'],0.),(start,velocity,acceleration)):
            np.testing.assert_allclose(actual,expected,atol=1e-10)
        self.assertTrue(result['visibility']['passed'])
        np.testing.assert_allclose(sample_motion(result['smoothing'],result['frames'][-1]['time'])[1:],
                                   np.zeros((2,6)),atol=1e-10)

    def test_short_prefix_is_not_used_as_full_goal_distance_bound(self):
        start=np.deg2rad([1,-40,-130,-40,1,50]);pose=ARM.transform(start)
        result=classical_tracking.plan({'start_deg':np.rad2deg(start).tolist(),
            'cursor_mm':(pose[:3,3]*1000+350*pose[:3,2]).tolist(),
            'tracking_region':{'distance_mm':330.,'half_angle_deg':30.},'obstacle_mm':None,
            'continuous_motion':True,'speed_rad_s':.003,
            'velocity_shape':{'kind':'smooth_trapezoid','rise_percent':5,'fall_percent':10}})
        self.assertTrue(result['playable'],result)
        self.assertTrue(result['replan_required'])
        self.assertLessEqual(result['frames'][-1]['time'],12.)
        distance=np.linalg.norm(np.deg2rad(result['goal_deg'])-start)
        self.assertGreaterEqual(result['selected_route_joint_travel_rad']+1e-9,distance)
        self.assertLess(np.linalg.norm(np.deg2rad(result['frames'][-1]['q'])-start),distance)


if __name__ == '__main__':
    unittest.main()
