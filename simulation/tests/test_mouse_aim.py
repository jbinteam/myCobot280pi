import unittest
from unittest.mock import patch
import numpy as np
from simulation import service, pose5, seed5, tracking
from simulation.kinematics import ARM
from simulation.prm import Roadmap, JointSpace, planning_space


class MouseAimTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.start = [0., -40., -130., 80., 0., 50.]
        cls.q = np.deg2rad(cls.start)
        cls.pose = ARM.transform(cls.q)
        cls.cursor = cls.pose[:3, 3]*1000 + 50*cls.pose[:3, 2]
        cls.road = Roadmap(planning_space('self', 1.))

    def body(self, cursor=None, **extra):
        return {'start_deg': self.start, 'cursor_mm': (self.cursor if cursor is None else cursor).tolist(), **extra}

    def force_fallback(self):
        return patch.object(tracking, 'local_pose_plan', return_value={
            'converged': False, 'local_ik': {'status': 'not_converged', 'reason': 'fixture local failure'}})

    def test_target_geometry_and_pipeline_cannot_be_downgraded(self):
        cursor = self.cursor + [4, 3, 0]
        with self.force_fallback(), patch.object(service, 'plan', return_value={'converged': False}) as plan:
            service.mouse_aim_plan(self.body(cursor, task='position', nn='none', refine=False,
                numerical_fallback=False, collision='limits', gap_mm=1, max_candidates=1,
                pose5_global_recovery=False, recovery_seeds=1, recovery_evaluations=1, recovery_max_candidates=1), self.road)
        request = plan.call_args.args[0]
        axis = pose5.unit_axis(cursor - self.pose[:3, 3]*1000)
        np.testing.assert_allclose(request['target_axis'], axis)
        np.testing.assert_allclose(request['target_mm'], cursor-50*axis)
        self.assertEqual(request['eef_offset_mm'], [0, 0, 0])
        for key, value in {'task':'pose5', 'nn':'seed5', 'refine':True, 'numerical_fallback':True,
                           'collision':'self', 'max_candidates':4, 'roadmap_seeds':4,
                           'random_seeds':4, 'shortcut':True}.items():
            self.assertEqual(request[key], value)
        self.assertTrue(request['pose5_global_recovery'])
        self.assertEqual(request['recovery_seeds'], 32)
        self.assertEqual(request['recovery_evaluations'], 96)
        self.assertEqual(request['recovery_max_candidates'], 8)
        self.assertEqual(request['tolerance_mm'], 1.)
        self.assertEqual(request['orientation_tolerance_deg'], 1.)
        self.assertEqual(request['pose5_method'], 'dls')
        self.assertEqual(request['pose5_recovery_methods'], ('dls',))
        self.assertTrue(callable(plan.call_args.kwargs['goal_constraint']))

    def test_already_at_target_and_local_z_spin(self):
        result = service.mouse_aim_plan(self.body(), self.road)
        self.assertTrue(result['converged'])
        self.assertEqual(result['ik']['goals'][0]['source'], 'current_state')
        self.assertEqual(result['planning_outcome'], 'direct_connection')
        self.assertTrue(result['cursor_acceptance']['accepted'])
        self.assertGreaterEqual(result['server_compute_ms'],result['ik_ms']+result['planning_ms'])
        spun = self.start.copy(); spun[-1] += 40
        other = service.mouse_aim_plan({**self.body(), 'start_deg': spun}, self.road)
        self.assertTrue(other['converged'])
        self.assertEqual(other['ik']['goals'][0]['source'], 'current_state')

    def test_displayed_tolerances_apply_to_the_shortcut_without_hidden_tightening(self):
        # 50.8 mm with exact pointing satisfies the displayed 1 mm / 1 degree.
        cursor = self.cursor + .8*self.pose[:3, 2]
        with patch.object(seed5, 'propose', side_effect=AssertionError('Already at the requested cursor goal')):
            result = service.mouse_aim_plan(self.body(cursor, steps=1), self.road)
        self.assertTrue(result['converged'])
        self.assertAlmostEqual(result['cursor_acceptance']['gap_error_mm'], .8)
        self.assertEqual(result['ik']['goals'][0]['source'], 'current_state')
        self.assertEqual(result['tolerance_mm'], 1.)
        self.assertEqual(result['orientation_tolerance_deg'], 1.)

    def test_reproduced_standalone_dls_solution_is_accepted(self):
        start = [20, -35, -90, 40, 10, 0]
        cursor = [285.69709856079663, 32.10990138603191, -33.48065176338741]
        result = service.mouse_aim_plan({'start_deg':start, 'cursor_mm':cursor}, self.road)
        self.assertEqual(result['ik_outcome'], 'accepted_goals', result.get('reason'))
        self.assertTrue(result['converged'], result.get('reason'))
        self.assertTrue(result['cursor_acceptance']['accepted'])
        self.assertLessEqual(result['final_error_mm'], 1.)
        self.assertLessEqual(result['cursor_acceptance']['gap_error_mm'], 1.)
        self.assertLessEqual(result['cursor_acceptance']['direction_error_deg'], 1.)

    def test_selected_numerical_controller_is_honored(self):
        with patch.object(pose5, 'solve', wraps=pose5.solve) as solve:
            result = service.mouse_aim_plan(self.body(self.cursor+[4, 3, 0], pose5_method='bounded_dls'), self.road)
        self.assertTrue(result['converged'])
        self.assertEqual(result['tracking_source'], 'local_ik')
        self.assertEqual(solve.call_args.kwargs['method'], 'bounded_dls')
        np.testing.assert_allclose(solve.call_args.args[0], self.q)

    def test_local_first_skips_neural_and_roadmap_work_with_speed_checked_frames(self):
        with patch.object(seed5, 'propose', side_effect=AssertionError('Local motion needs no NN')), \
             patch.object(service, 'load_roadmap', side_effect=AssertionError('Local motion needs no roadmap')), \
             patch.object(service, 'plan', side_effect=AssertionError('Local motion needs no global plan')):
            result = service.mouse_aim_plan(self.body(self.cursor + [4, 3, 0]))
        self.assertTrue(result['converged'], result.get('reason'))
        self.assertEqual(result['tracking_strategy'], 'local_first')
        self.assertEqual(result['tracking_source'], 'local_ik')
        self.assertIsNone(result['ik']['nn_checkpoint'])
        self.assertEqual(result['ik']['goals'][0]['source'], 'current_local_ik')
        self.assertGreater(result['local_ik']['raw_path_vertices'], len(result['waypoints_deg']))
        self.assertEqual(len(result['waypoints_deg']), 2)
        self.assertTrue(result['cursor_acceptance']['accepted'])
        self.assertEqual(result['validation'], 'joint_limits_and_self_collision')
        frames = result['frames']
        for a, b in zip(frames, frames[1:]):
            qa, qb = np.deg2rad(a['q']), np.deg2rad(b['q'])
            self.assertTrue(self.road.space.edge_valid(qa, qb, self.road.config.max_step_rad))
            self.assertLessEqual(np.linalg.norm(qb-qa), 1.2*(b['time']-a['time'])+1e-9)

    def test_missing_checkpoint_uses_numerical_fallback(self):
        with self.force_fallback(), patch.object(seed5, 'propose', return_value=([], {'status':'missing', 'reason':'fixture missing checkpoint'}, 0)):
            result = service.mouse_aim_plan(self.body(self.cursor+[4, 3, 0]), self.road)
        self.assertTrue(result['converged'])
        self.assertEqual(result['tracking_source'], 'global_fallback')
        self.assertEqual(result['ik']['nn_checkpoint']['status'], 'missing')
        self.assertTrue(all(not g['source'].startswith('nn_') for g in result['ik']['goals']))
        self.assertEqual(result['ik']['seed_queue'][0]['source'], 'current_dls')

    def test_failed_local_solve_really_falls_back_to_the_trained_pipeline(self):
        actual_solve = pose5.solve
        calls = []
        def solve(*args, **kwargs):
            calls.append(kwargs.get('method', args[3] if len(args) > 3 else 'dls'))
            result = actual_solve(*args, **kwargs)
            if len(calls) == 1:
                result.update(converged=False, stop_reason='stalled')
            return result
        with patch.object(pose5, 'solve', side_effect=solve):
            result = service.mouse_aim_plan(self.body(self.cursor+[4, 3, 0]), self.road)
        self.assertTrue(result['converged'], result.get('reason'))
        self.assertEqual(result['tracking_source'], 'global_fallback')
        self.assertEqual(result['local_ik']['status'], 'not_converged')
        self.assertEqual(result['ik']['nn_checkpoint']['status'], 'ready')
        self.assertTrue(any(g['source'].startswith('nn_') for g in result['ik']['goals']))
        self.assertTrue(result['cursor_acceptance']['accepted'])

    def test_invalid_targets_and_unchecked_space(self):
        for cursor in ([0, 0, float('nan')], [0, 0, float('inf')], [0, 0], ['1', 2, 3], [True, 2, 3], self.pose[:3,3]*1000):
            with self.subTest(cursor=cursor), self.assertRaises(ValueError):
                service.mouse_aim_plan({'start_deg':self.start, 'cursor_mm':list(cursor)}, self.road)
        for settings in ({'tolerance_mm':0}, {'orientation_tolerance_deg':float('nan')}, {'start_deg':[1000]*6}):
            with self.assertRaises(ValueError):service.mouse_aim_plan(self.body(**settings), self.road)
        with self.assertRaises(ValueError):service.mouse_aim_plan(self.body(), Roadmap(JointSpace()))
        for extra in ({'steps':0}, {'pose5_method':'unknown'}, {'speed_rad_s':True},
                      {'guided_route_repair':'true'}, {'route_repair_samples':129}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                service.mouse_aim_plan(self.body(**extra), self.road)

    def test_cursor_geometry_acceptance_is_independent_of_ik_acceptance(self):
        # A nominal IK success pointing the opposite way must never be played.
        other = self.start.copy(); other[3] += 20
        bad = {'converged':True, 'frames':[{'q':other}], 'ik_outcome':'accepted_goals', 'planning_outcome':'direct_connection'}
        with self.force_fallback(), patch.object(service, 'plan', return_value=bad):
            result = service.mouse_aim_plan(self.body(), self.road)
        self.assertFalse(result['converged'])
        self.assertEqual(result['status'], 'mouse_aim_goal_tolerance_not_met')
        self.assertEqual(result['frames'], [])
        self.assertFalse(result['cursor_acceptance']['accepted'])


if __name__ == '__main__': unittest.main()
