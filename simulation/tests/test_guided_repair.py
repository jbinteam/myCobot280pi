import json
import unittest
from unittest.mock import patch

import numpy as np

from simulation import guided_repair
from simulation.kinematics import ARM
from simulation.prm import JointSpace, PRMConfig, Roadmap


def angles(x, y=0.):
    q = ARM.center.copy()
    q[:2] = [x, y]
    return q


class GuidedRepairTest(unittest.TestCase):
    def disk(self):
        space = JointSpace(lambda q: np.linalg.norm(q[:2]) >= .35,
                           {'checker': 'disk', 'environment': 'q1_q2_radius_0.35'})
        return Roadmap(space, PRMConfig(samples=0, neighbors=12))

    def test_obstructed_route_recovered_without_mutating_graph(self):
        road = self.disk()
        original = json.dumps(road.edges, sort_keys=True)
        start, goal = angles(-1), angles(1)
        self.assertNotEqual(road.plan(start, [goal])['status'], 'success')
        result = guided_repair.plan(road, start, [goal], guided_repair.RepairConfig(samples=64, seed=280))
        self.assertEqual(result['status'], 'success', result)
        self.assertTrue(result['guided_repair']['recovered'])
        path = np.asarray(result['waypoints_rad'])
        np.testing.assert_array_equal(path[0], start)
        np.testing.assert_array_equal(path[-1], goal)
        self.assertTrue(all(road.space.edge_valid(a, b, .04) for a, b in zip(path, path[1:])))
        self.assertEqual(len(road.q), 0)
        self.assertEqual(json.dumps(road.edges, sort_keys=True), original)
        self.assertLessEqual(result['guided_repair']['connection_checks'], 64*12)

    def test_success_and_invalid_input_never_sample(self):
        road = Roadmap(JointSpace())
        for start, goals in ((angles(-1), [angles(1)]), (ARM.upper+1, [angles(1)]), (angles(-1), [])):
            with patch.object(guided_repair, 'sample_nodes', side_effect=AssertionError('No repair needed')):
                result = guided_repair.plan(road, start, goals)
            self.assertFalse(result['guided_repair']['triggered'])

    def test_disconnected_wall_remains_failed(self):
        space = JointSpace(lambda q: abs(q[0]) > .15,
                           {'checker': 'wall', 'environment': 'q1_wall'})
        road = Roadmap(space, PRMConfig(samples=0))
        result = guided_repair.plan(road, angles(-1), [angles(1)], guided_repair.RepairConfig(samples=32))
        self.assertNotEqual(result['status'], 'success')
        self.assertFalse(result['guided_repair']['recovered'])
        self.assertLessEqual(result['guided_repair']['attempts'], 32*8)

    def test_sampling_is_deterministic_and_preserves_bounded_joint_semantics(self):
        road = Roadmap(JointSpace())
        start, goal = ARM.lower.copy(), ARM.upper.copy()
        config = guided_repair.RepairConfig(samples=64, seed=20)
        a, stats_a = guided_repair.sample_nodes(road, start, [goal], config)
        b, stats_b = guided_repair.sample_nodes(road, start, [goal], config)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(stats_a, stats_b)
        self.assertTrue(np.all(a >= ARM.lower) and np.all(a <= ARM.upper))
        self.assertGreater(stats_a['accepted_guided'], 0)
        self.assertGreater(stats_a['accepted_uniform'], 0)
        self.assertTrue(np.any(np.abs(a[:, 5]) < .5))

    def test_no_valid_sample_has_finite_budget(self):
        road = Roadmap(JointSpace(lambda q: False, {'checker': 'none', 'environment': 'empty'}))
        config = guided_repair.RepairConfig(samples=32, sampling_attempt_factor=2)
        nodes, stats = guided_repair.sample_nodes(road, angles(-1), [angles(1)], config)
        self.assertEqual(nodes.shape, (0, 6))
        self.assertEqual(stats['attempts'], 64)

    def test_config_rejects_unbounded_and_nonfinite_settings(self):
        for settings in ({'samples': 0}, {'samples': 129}, {'samples': True},
                         {'seed': -1}, {'guided_fraction': 2}, {'noise_fraction': float('nan')},
                         {'sampling_attempt_factor': 100}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                guided_repair.RepairConfig(**settings)


if __name__ == '__main__':
    unittest.main()
