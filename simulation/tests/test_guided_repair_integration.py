from dataclasses import asdict, replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from simulation import guided_repair, planning, service
from simulation.kinematics import ARM
from simulation.prm import Roadmap, planning_space


class GuidedRepairIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = json.loads((Path(__file__).parents[1]/'fixtures/self_collision_path.json').read_text())
        cls.start = np.deg2rad(fixture['start_deg'])
        cls.goal = np.deg2rad(fixture['goal_deg'])
        cls.target = ARM.transform(cls.goal)
        cls.road = Roadmap(planning_space('self', 1.))

    def body(self, **changes):
        return {'start_deg': np.rad2deg(self.start).tolist(),
                'target_mm': (self.target[:3, 3]*1000).tolist(), 'target_axis': self.target[:3, 2].tolist(),
                'task': 'pose5', 'nn': 'none', 'shortcut': True,
                'guided_route_repair': True, 'route_repair_samples': 32, **changes}

    def generated(self, config=None, goals=None):
        config = config or planning.IKConfig(task='pose5', nn='none')
        goal = planning.checked_goal(self.goal, self.target[:3, 3], self.road.space, config, self.target[:3, 2])
        return {'goals': [{**goal, 'source': 'validated_fixture'}] if goals is None else goals,
                'attempts': [], 'seed_queue': [], 'raw_accepted_count': 0,
                'best_error': None, 'nn_checkpoint': None, 'ik_ms': 0., 'config': asdict(config),
                'diagnostic': {'best_valid_error': None, 'refinements': 0,
                               'pose_accurate_invalid_candidates': 0, 'goal_constraint_rejections': 0}}

    def test_api_repairs_real_self_collision_detour_and_validates_frames(self):
        self.assertFalse(self.road.space.edge_valid(self.start, self.goal, .04))
        with patch.object(planning, 'generate_goals', return_value=self.generated()):
            result = service.plan(self.body(), self.road)
        self.assertTrue(result['converged'], result.get('reason'))
        self.assertTrue(result['guided_repair']['enabled'])
        self.assertTrue(result['guided_repair']['recovered'])
        self.assertEqual(result['guided_repair']['new_nodes'], 32)
        self.assertEqual(result['validation'], 'joint_limits_and_self_collision')
        self.assertEqual(len(self.road.q), 0)
        np.testing.assert_allclose(result['frames'][0]['q'], np.rad2deg(self.start), atol=1e-12)
        np.testing.assert_allclose(result['frames'][-1]['q'], np.rad2deg(self.goal), atol=1e-12)
        self.assertLessEqual(result['final_error_mm'], 1.)
        self.assertLessEqual(result['final_orientation_error_deg'], 1.)
        for a, b in zip(result['frames'], result['frames'][1:]):
            qa, qb = np.deg2rad(a['q']), np.deg2rad(b['q'])
            self.assertTrue(self.road.space.edge_valid(qa, qb, .04))
            self.assertLessEqual(np.linalg.norm(qb-qa), 1.2*.05+1e-9)

    def test_disabled_repair_returns_original_failure_without_sampling(self):
        with patch.object(planning, 'generate_goals', return_value=self.generated()), \
             patch.object(guided_repair, 'sample_nodes', side_effect=AssertionError('Repair disabled')):
            result = service.plan(self.body(guided_route_repair=False), self.road)
        self.assertFalse(result['converged'])
        self.assertEqual(result['planning_outcome'], 'start_not_connected')
        self.assertFalse(result['guided_repair']['enabled'])
        self.assertEqual(result['frames'], [])

    def test_no_valid_ik_goal_never_attempts_route_repair(self):
        with patch.object(planning, 'generate_goals', return_value=self.generated(goals=[])), \
             patch.object(guided_repair, 'repair_failed_route', side_effect=AssertionError('No valid IK goal')):
            result = service.plan(self.body(), self.road)
        self.assertEqual(result['ik_outcome'], 'ik_solution_not_found')
        self.assertFalse(result['guided_repair']['triggered'])
        self.assertEqual(result['frames'], [])

    def test_final_validation_failure_does_not_expose_playback(self):
        with patch.object(planning, 'generate_goals', return_value=self.generated()), \
             patch.object(planning, 'trajectory', side_effect=ValueError('Rejected final trajectory')):
            result = service.plan(self.body(), self.road)
        self.assertTrue(result['guided_repair']['recovered'])
        self.assertFalse(result['converged'])
        self.assertEqual(result['planning_outcome'], 'trajectory_validation_failed')
        self.assertEqual(result['frames'], [])

    def test_goal_expansion_precedes_one_repair_without_duplicate_baseline_search(self):
        config = planning.IKConfig(task='pose5', nn='none', pose5_global_recovery=True,
                                   guided_route_repair=True)
        first = self.generated(config)
        second_goal = dict(first['goals'][0], q_rad=(self.goal+np.array([0, 0, 0, 0, 0, .1])).tolist())
        expanded = self.generated(config, [*first['goals'], second_goal])
        failure = {'status': 'roadmap_connection_not_found', 'reason': 'start_not_connected', 'planning_ms': 1.}
        events = []
        def normal(*args, **kwargs):
            events.append('route')
            return failure.copy()
        def recover(*args, **kwargs):
            events.append('expand')
            return expanded
        def repair(road, start, goals, baseline, repair_config, **kwargs):
            events.append('repair')
            self.assertEqual(len(goals), 2)
            return {**baseline, 'guided_repair': {'enabled': True, 'triggered': True, 'recovered': False}}
        with patch.object(planning, 'generate_goals', return_value=first), \
             patch.object(self.road, 'plan', side_effect=normal), \
             patch.object(planning, 'recover_goals', side_effect=recover), \
             patch.object(guided_repair, 'repair_failed_route', side_effect=repair):
            result = planning.query(self.start, self.target[:3, 3], self.road, config,
                                    target_axis=self.target[:3, 2])
        self.assertEqual(events, ['route', 'expand', 'route', 'repair'])
        self.assertFalse(result['converged'])

    def test_mouse_aim_forwards_enabled_and_disabled_settings(self):
        from simulation import tracking
        cursor = ARM.transform(self.start)[:3, 3]*1000+50*ARM.transform(self.start)[:3, 2]
        for enabled in (True, False):
            with patch.object(tracking, 'local_pose_plan', return_value={'converged': False, 'local_ik': {'status': 'not_converged'}}), \
                 patch.object(service, 'plan', return_value={'converged': False}) as plan:
                service.mouse_aim_plan({'start_deg': np.rad2deg(self.start).tolist(),
                                        'cursor_mm': cursor.tolist(), 'guided_route_repair': enabled,
                                        'route_repair_samples': 32}, self.road)
            self.assertIs(plan.call_args.args[0]['guided_route_repair'], enabled)
            self.assertEqual(plan.call_args.args[0]['route_repair_samples'], 32)

    def test_configuration_is_bounded_and_existing_api_defaults_stay_disabled(self):
        config = planning.IKConfig()
        self.assertFalse(config.guided_route_repair)
        for changes in ({'guided_route_repair': 1}, {'guided_route_repair': 'true'},
                        {'route_repair_samples': 0}, {'route_repair_samples': 129},
                        {'route_repair_samples': True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(config, **changes)


if __name__ == '__main__':
    unittest.main()
