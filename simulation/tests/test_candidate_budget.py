"""Scheduler behavior with deterministic time, independent of machine speed."""
import unittest
import numpy as np
from simulation.diffusion.endpoint_selection import SelectionBudget, SearchBudgetExpired
from simulation.diffusion.streaming_ik import CandidateBudget, choose_streaming_route
from simulation.tests.test_streaming_ik import Source, checked


class Clock:
    def __init__(self):
        self.now = 0.
    def __call__(self):
        return self.now


class CandidateBudgetTest(unittest.TestCase):
    def test_failed_ik_job_does_not_starve_later_short_route(self):
        clock = Clock()
        class SlowSecond(Source):
            def next_endpoint(self, deadline):
                goal = super().next_endpoint(deadline)
                if goal['index'] == 1:
                    clock.now = deadline
                    raise SearchBudgetExpired()
                return goal
        source = SlowSecond([1.,.8,.2])
        best, stats = choose_streaming_route(np.zeros(6), source,
            lambda g,c,d: checked(g, 2. if g['index']==0 else .2),
            clock=clock, candidate_budget=CandidateBudget(lookahead_seconds=0))
        self.assertEqual(best['trajectory'], 2)
        self.assertEqual(source.used, 3)
        self.assertEqual(stats['ik_timeouts'], [{'scope':'candidate'}])
        self.assertLess(stats['improvement_seconds'], .35)

    def test_local_route_timeout_keeps_searching_and_charges_iterations(self):
        clock = Clock()
        def evaluate(g, cap, deadline):
            if g['index']==1:
                clock.now = deadline
                raise SearchBudgetExpired()
            return checked(g, 2. if g['index']==0 else .2)
        best, stats = choose_streaming_route(np.zeros(6), Source([1.,.8,.2]), evaluate,
            clock=clock, candidate_budget=CandidateBudget(lookahead_seconds=0))
        self.assertEqual(best['trajectory'], 2)
        self.assertEqual(stats['iterations_charged'], 80)
        self.assertEqual(stats['improvement_route_attempts'], 2)
        self.assertEqual(stats['route_attempts'][1]['timeout_scope'], 'candidate')

    def test_overall_deadline_wins_and_preserves_incumbent(self):
        clock = Clock(); source = Source([1.,.8,.2])
        def evaluate(g, cap, deadline):
            if deadline is not None:
                self.assertEqual(deadline, .01)
                clock.now = deadline
                raise SearchBudgetExpired()
            return checked(g, 2.)
        best, stats = choose_streaming_route(np.zeros(6), source, evaluate,
            SelectionBudget(improvement_seconds=.01), clock, CandidateBudget(lookahead_seconds=0))
        self.assertEqual(best['trajectory'], 0)
        self.assertEqual(source.used, 2)
        self.assertEqual(stats['route_attempts'][-1]['timeout_scope'], 'overall')

    def test_late_completed_route_cannot_replace_incumbent(self):
        clock = Clock()
        def evaluate(g, cap, deadline):
            if deadline is not None:
                clock.now = deadline + .001
            return checked(g, 2. if g['index']==0 else .2)
        best, stats = choose_streaming_route(np.zeros(6), Source([1.,.2]), evaluate,
            clock=clock, candidate_budget=CandidateBudget(lookahead_seconds=0))
        self.assertEqual(best['trajectory'], 0)
        self.assertTrue(stats['route_attempts'][-1]['timed_out'])

    def test_local_timeouts_still_obey_shared_effort_cap(self):
        clock = Clock(); source = Source([1.,.8,.6,.4,.2])
        def evaluate(g, cap, deadline):
            if deadline is not None:
                clock.now = deadline
                raise SearchBudgetExpired()
            return checked(g, 2.)
        best, stats = choose_streaming_route(np.zeros(6), source, evaluate,
            SelectionBudget(rrt_iterations=100), clock, CandidateBudget(route_seconds=.05, lookahead_seconds=0))
        self.assertEqual(best['trajectory'], 0)
        self.assertEqual(source.used, 4)
        self.assertEqual(stats['improvement_route_attempts'], 3)
        self.assertEqual(stats['iterations_charged'], 100)
        self.assertEqual(stats['stop_reason'], 'improvement_route_attempt_limit')

    def test_ik_local_slice_clipped_to_overall_deadline(self):
        clock = Clock()
        class SlowSource(Source):
            def next_endpoint(self, deadline):
                goal = super().next_endpoint(deadline)
                if deadline is not None:
                    clock.now = deadline
                    raise SearchBudgetExpired()
                return goal
        source = SlowSource([1.,.8,.2])
        best, stats = choose_streaming_route(np.zeros(6), source,
            lambda g,c,d: checked(g,2.), SelectionBudget(improvement_seconds=.01),
            clock, CandidateBudget(lookahead_seconds=0))
        self.assertEqual(best['trajectory'], 0)
        self.assertEqual(source.used, 2)
        self.assertEqual(stats['ik_timeouts'], [{'scope':'overall'}])

    def test_first_solution_is_not_cut_off_by_improvement_slice(self):
        clock = Clock()
        def evaluate(g, cap, deadline):
            self.assertIsNone(deadline)
            clock.now += 2.
            return checked(g, 2.)
        best, stats = choose_streaming_route(np.zeros(6), Source([1.]), evaluate,
            clock=clock, candidate_budget=CandidateBudget(lookahead_seconds=0))
        self.assertEqual(best['trajectory'], 0)
        self.assertEqual(stats['first_solution_seconds'], 2.)

    def test_slices_require_positive_finite_values(self):
        for v in (0., -.1, float('inf'), float('nan'), True):
            with self.assertRaises(ValueError):
                CandidateBudget(ik_seconds=v)
            with self.assertRaises(ValueError):
                CandidateBudget(route_seconds=v)

    def test_lookahead_checks_shorter_later_endpoint_before_expensive_route(self):
        clock = Clock(); calls = []
        def evaluate(g, cap, deadline):
            calls.append(g['index'])
            if g['index']==1:
                self.fail('Longer endpoint must be pruned after the shorter route')
            return checked(g, 3. if g['index']==0 else .2)
        best, stats = choose_streaming_route(np.zeros(6), Source([1.,.8,.2]), evaluate,
            clock=clock, candidate_budget=CandidateBudget())
        self.assertEqual(calls, [0,2])
        self.assertEqual(best['trajectory'], 2)
        self.assertEqual(stats['pruned_known_endpoint_indices'], [1])

    def test_empty_lookahead_continues_to_later_jobs(self):
        clock = Clock()
        class SlowMiddle(Source):
            def next_endpoint(self, deadline):
                goal = super().next_endpoint(deadline)
                if 1 <= goal['index'] <= 6:
                    clock.now = deadline
                    raise SearchBudgetExpired()
                return goal
        source = SlowMiddle([1.]+[.8]*6+[.2])
        best, stats = choose_streaming_route(np.zeros(6), source,
            lambda g,c,d: checked(g, 2. if g['index']==0 else .2),
            clock=clock, candidate_budget=CandidateBudget())
        self.assertEqual(best['trajectory'], 7)
        self.assertFalse(stats['uncomputed_endpoints_certified'])
        self.assertLessEqual(stats['improvement_seconds'], .35)

    def test_lookahead_global_expiry_retains_incumbent(self):
        clock = Clock()
        class SlowMiddle(Source):
            def next_endpoint(self, deadline):
                goal = super().next_endpoint(deadline)
                if deadline is not None:
                    clock.now = deadline
                    raise SearchBudgetExpired()
                return goal
        source = SlowMiddle([1.,.8,.2])
        best, stats = choose_streaming_route(np.zeros(6), source,
            lambda g,c,d: checked(g,2.), SelectionBudget(improvement_seconds=.01),
            clock, CandidateBudget())
        self.assertEqual(best['trajectory'], 0)
        self.assertEqual(source.used, 2)
        self.assertEqual(stats['ik_timeouts'], [{'scope':'overall'}])


if __name__ == '__main__':
    unittest.main()
