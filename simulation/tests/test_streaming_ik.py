import unittest
from unittest.mock import patch
import numpy as np
from simulation.kinematics import ARM
from simulation.pose5 import solve
from simulation.diffusion.env import ArmEnv
from simulation.diffusion.endpoint_selection import SelectionBudget, SearchBudgetExpired
from simulation.diffusion.streaming_ik import choose_streaming_route


class Source:
    def __init__(self, distances):
        self.jobs = [{'index': i, 'q_rad': [d,0,0,0,0,0]} for i,d in enumerate(distances)]
        self.used = 0
    @property
    def has_more(self):
        return self.used < len(self.jobs)
    def next_endpoint(self, deadline):
        goal = self.jobs[self.used]; self.used += 1
        return goal


def checked(goal, cost):
    return {'route': {'goal_index': goal['index'], 'accepted': True,
                     'joint_travel_rad': cost, 'visibility': {'passed': True},
                     'planner': {'iterations': 0}}, 'trajectory': goal['index']}


class StreamingTest(unittest.TestCase):
    def test_pruning_known_endpoint_does_not_certify_uncomputed_candidates(self):
        source = Source([1.,3.,.5]); calls = []
        def evaluate(g, cap, deadline):
            calls.append(g['index'])
            return checked(g, {0:2.,2:.5}[g['index']])
        selected, stats = choose_streaming_route(np.zeros(6), source, evaluate)
        self.assertEqual(calls, [0,2]); self.assertEqual(source.used, 3)
        self.assertEqual(selected['trajectory'], 2)
        self.assertFalse(stats['uncomputed_endpoints_certified'])

    def test_ik_timeout_returns_incumbent_without_eager_generation(self):
        class InterruptingSource(Source):
            def next_endpoint(self, deadline):
                if deadline is not None:
                    raise SearchBudgetExpired()
                return super().next_endpoint(deadline)
        source = InterruptingSource([1.,.5,.2])
        selected, stats = choose_streaming_route(np.zeros(6), source, lambda g,c,d: checked(g, 2.))
        self.assertEqual(selected['trajectory'], 0)
        self.assertEqual(source.used, 1)
        self.assertEqual(stats['stop_reason'], 'improvement_time_limit_in_ik')

    def test_first_solution_only_does_not_generate_next_ik_job(self):
        source = Source([1.,.5])
        selected, stats = choose_streaming_route(np.zeros(6), source, lambda g,c,d: checked(g, 2.),
                                                SelectionBudget(improvement_seconds=0))
        self.assertEqual(source.used, 1); self.assertEqual(selected['trajectory'], 0)

    def test_route_timeout_preserves_incumbent_and_charges_cap(self):
        def evaluate(g, cap, deadline):
            if deadline is not None:
                raise SearchBudgetExpired()
            return checked(g, 2.)
        selected, stats = choose_streaming_route(np.zeros(6), Source([1.,.5]), evaluate)
        self.assertEqual(selected['trajectory'], 0)
        self.assertEqual(stats['iterations_charged'], 80)


class NumericalCancellationTest(unittest.TestCase):
    def test_pose_solver_interrupts_unfinished_iterations(self):
        q = np.deg2rad([0.,-40.,-130.,80.,0.,50.]); t = ARM.transform(q)
        calls = []
        def cancel():
            calls.append(True)
            if len(calls) == 3:
                raise SearchBudgetExpired()
        with self.assertRaises(SearchBudgetExpired):
            solve(q, t[:3,3]+[.03,0.,0.], target_axis=t[:3,2], method='bounded_dls', cancel_check=cancel)
        self.assertEqual(len(calls), 3)

    def test_fallback_checks_deadline_inside_optimizer_and_preserves_diagnostics(self):
        env = ArmEnv.__new__(ArmEnv); env.valid = lambda q: True
        q = np.deg2rad([0.,-40.,-130.,80.,0.,50.]); t = ARM.transform(q)
        failed = {'converged': False, 'frames': [{'q': np.rad2deg(ARM.center).tolist()}]}
        def cancel():
            if env.ik_diagnostics['fallback_attempts']:
                raise SearchBudgetExpired()
        with patch('simulation.diffusion.env.solve', return_value=failed):
            with self.assertRaises(SearchBudgetExpired):
                env.inverse_kinematics(t[:3,3], t[:3,2], ARM.center, attempts=1,
                                       robust=True, learned=False, cancel_check=cancel)
        self.assertEqual(env.ik_diagnostics['fallback_attempts'], 1)
        self.assertGreater(env.ik_diagnostics['elapsed_ms'], 0)

    def test_noop_cancellation_preserves_numerical_result(self):
        q = np.deg2rad([0.,-40.,-130.,80.,0.,50.]); t = ARM.transform(q+.03)
        plain = solve(q, t[:3,3], target_axis=t[:3,2], method='bounded_dls')
        checked = solve(q, t[:3,3], target_axis=t[:3,2], method='bounded_dls', cancel_check=lambda: None)
        self.assertEqual(plain['frames'], checked['frames'])


if __name__ == '__main__':
    unittest.main()
