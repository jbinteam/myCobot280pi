import unittest
import numpy as np
from simulation.diffusion.endpoint_selection import choose_visible_endpoint, SelectionBudget, SearchBudgetExpired, DeadlineEnv


def goals(*distances):
    return [{'index': i, 'q_rad': [x, 0, 0, 0, 0, 0]} for i,x in enumerate(distances)]


def result(goal, cost=None):
    return {'route': {'goal_index': goal['index'], 'accepted': cost is not None,
                     'joint_travel_rad': cost, 'visibility': {'passed': cost is not None},
                     'planner': {'iterations': 0}}, 'trajectory': goal['index']}


class SelectionTest(unittest.TestCase):
    def test_orders_alternatives_by_bound_and_prunes_without_evaluating(self):
        calls = []
        def evaluate(goal, cap, deadline):
            calls.append(goal['index'])
            return result(goal, {0: 4., 2: 1.}[goal['index']])
        selected, stats = choose_visible_endpoint(np.zeros(6), goals(2., 5., .8, 3.), evaluate)
        self.assertEqual(calls, [0, 2])
        self.assertEqual(selected['trajectory'], 2)
        self.assertEqual(set(stats['lower_bound_pruned_indices']), {1, 3})

    def test_timeout_preserves_validated_first_solution(self):
        def evaluate(goal, cap, deadline):
            if deadline is not None:
                raise SearchBudgetExpired()
            return result(goal, 3.)
        selected, stats = choose_visible_endpoint(np.zeros(6), goals(2., 1.), evaluate)
        self.assertEqual(selected['trajectory'], 0)
        self.assertEqual(stats['stop_reason'], 'improvement_time_limit')
        self.assertLessEqual(stats['iterations_charged'], 800)

    def test_invalid_improvement_and_attempt_limit_preserve_incumbent(self):
        def evaluate(goal, cap, deadline):
            return result(goal, 3. if goal['index'] == 0 else None)
        selected, stats = choose_visible_endpoint(np.zeros(6), goals(2., .8, 1.), evaluate,
                                                   SelectionBudget(improvement_attempts=1))
        self.assertEqual(selected['trajectory'], 0)
        self.assertEqual(stats['improvement_attempts'], 1)
        self.assertEqual(stats['stop_reason'], 'improvement_attempt_limit')

    def test_zero_improvement_budget_and_no_solution(self):
        selected, stats = choose_visible_endpoint(np.zeros(6), goals(2., 1.),
            lambda g,c,d: result(g, 3.), SelectionBudget(improvement_seconds=0))
        self.assertEqual(selected['trajectory'], 0)
        self.assertEqual(stats['improvement_attempts'], 0)
        selected, stats = choose_visible_endpoint(np.zeros(6), goals(2., 1.), lambda g,c,d: result(g))
        self.assertIsNone(selected)
        self.assertEqual(stats['stop_reason'], 'no_visible_route_within_search')

    def test_deadline_interrupts_inside_edge_not_only_at_its_end(self):
        class Space:
            def interpolate(self, a, b, step):
                return np.linspace(a, b, 5)
        class Env:
            space = Space()
            calls = 0
            def valid(self, q):
                self.calls += 1
                return True
        env = Env()
        wrapped = DeadlineEnv(env, 3., clock=lambda: float(env.calls))
        with self.assertRaises(SearchBudgetExpired):
            wrapped.edge_valid(np.zeros(6), np.ones(6))
        self.assertEqual(env.calls, 3)

    def test_shared_rrt_cap_is_preserved_before_first_solution(self):
        caps = []
        def evaluate(goal, cap, deadline):
            caps.append(cap)
            rejected = result(goal)
            rejected['route']['planner']['iterations'] = cap
            return rejected
        selected, stats = choose_visible_endpoint(np.zeros(6), goals(1., 2., 3.), evaluate,
                                                   SelectionBudget(rrt_iterations=250))
        self.assertIsNone(selected)
        self.assertEqual(caps, [200, 50, 0])
        self.assertEqual(stats['iterations_charged'], 250)

    def test_claimed_success_without_visibility_is_rejected(self):
        def evaluate(goal, cap, deadline):
            invalid = result(goal, 2.)
            invalid['route']['visibility']['passed'] = False
            return invalid
        with self.assertRaises(ValueError):
            choose_visible_endpoint(np.zeros(6), goals(1.), evaluate)


if __name__ == '__main__':
    unittest.main()
