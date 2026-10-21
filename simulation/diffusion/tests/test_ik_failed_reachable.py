"""Characterize a real candidate-limited failure while preserving its solution.

When candidate selection is improved, update the failure expectation to success;
keep the fixture and independent witness checks as a reachable-target regression.
"""
import unittest
from copy import deepcopy
import numpy as np

from scripts.reproduce_ik_failed_reachable import FIXTURE, load_case, reproduce_failure, verify_witness, verify_final_timed_route
from simulation.diffusion import classical_tracking
from simulation.camera_model import CameraModel
from simulation.tracking_geometry import region_for, region_metrics


class ReachableIKFailureTest(unittest.TestCase):
    fixture = FIXTURE

    def test_default_search_exhausts_without_reaching_rrt(self):
        result = reproduce_failure(load_case(self.fixture))
        self.assertEqual(result['status'], 'ik_failed')
        self.assertFalse(result['playable'])
        self.assertEqual(result['region_route_attempts'], [])

    def test_same_request_has_a_visible_collision_free_witness_and_route(self):
        case = load_case(self.fixture)
        proof = verify_witness(case)
        self.assertTrue(proof['acceptance']['accepted'])
        self.assertEqual(proof['candidate_number'], case['witness']['candidate_index'] + 1)
        self.assertTrue(proof['route_audit']['passed'])
        self.assertGreater(proof['route_audit']['samples'], 1000)


class InwardReachableIKFailureTest(ReachableIKFailureTest):
    fixture = FIXTURE.with_name('ik_failed_reachable_inward.json')

    def test_corrected_initial_stage_is_reached_before_the_failed_inward_step(self):
        case = load_case(self.fixture)
        initial = deepcopy(case['approach']['initial_request'])
        initial['_visibility_camera'] = CameraModel(**case['camera'])
        start = np.deg2rad(initial['start_deg'])
        first_region = region_for(initial['cursor_mm'], initial['target_axis'], initial['tracking_region'])
        self.assertTrue(region_metrics(start, first_region, 1., 1.)['accepted'])
        result = classical_tracking.plan(initial)
        self.assertTrue(result['playable'] and result['converged'])
        self.assertTrue(result['visibility']['passed'])
        self.assertFalse(result['replan_required'])
        np.testing.assert_allclose(result['frames'][-1]['q'], case['request']['start_deg'], atol=1e-10, rtol=0)
        self.assertEqual(initial['tracking_region']['distance_mm'], 270.)
        self.assertEqual(case['request']['tracking_region']['distance_mm'], 250.)
        self.assertEqual(initial['cursor_mm'], case['request']['cursor_mm'])


class Final150ReachableIKFailureTest(ReachableIKFailureTest):
    fixture = FIXTURE.with_name('ik_failed_reachable_150.json')

    def test_actual_final_stage_has_a_complete_checked_timed_route(self):
        case = load_case(self.fixture)
        self.assertEqual(case['approach']['measured_initial_distance_mm'], 170.)
        self.assertEqual(case['approach']['final_distance_mm'], 150.)
        self.assertEqual(case['request']['tracking_region']['distance_mm'], 150.)
        proof = verify_final_timed_route(case)
        self.assertTrue(proof['passed'] and proof['complete_route'])
        self.assertTrue(proof['acceptance']['accepted'] and proof['audit']['passed'])
        self.assertEqual(proof['references'], len(case['final_stage_proof']['timed_route']['references']))


if __name__ == '__main__':
    unittest.main()
