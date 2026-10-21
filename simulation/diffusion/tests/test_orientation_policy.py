"""Hybrid search must preserve strict motion and never bypass common failures."""
import unittest
from copy import deepcopy
from unittest.mock import Mock
import numpy as np
from scripts.reproduce_ik_failed_reachable import FIXTURE,load_case
from simulation.camera_model import CameraModel
from simulation.diffusion import classical_tracking
from simulation.diffusion.orientation_policy import plan_with_orientation
from simulation.diffusion.endpoint_selection import SearchBudgetExpired
from simulation.diffusion.visibility import VisibilityEnv
from simulation import profile_tracking


class OrientationPolicyTest(unittest.TestCase):
    def test_checked_strict_prefix_is_kept_without_relaxing(self):
        prefix={'status':'chunk_ready','playable':True,'converged':False,'frames':[{'q':[0]*6}],
                'ik_elapsed_ms':12.,'path_elapsed_ms':20.}
        attempt=Mock(return_value=prefix)
        result=plan_with_orientation(attempt,'strict_then_fov')
        attempt.assert_called_once_with('point_at_object')
        self.assertIs(result['frames'],prefix['frames'])
        self.assertEqual(result['orientation_mode'],'point_at_object')
        self.assertFalse(result['orientation_selection']['fallback_used'])

    def test_fallback_is_limited_to_failed_search_and_records_both_passes(self):
        for status in ('ik_failed','no_path'):
            strict={'status':status,'playable':False,'reason':'exhausted','ik_elapsed_ms':10.,'region_candidate_jobs_tested':16}
            relaxed={'status':'target_reached','playable':True,'converged':True,'reason':'checked',
                     'ik_elapsed_ms':4.,'path_elapsed_ms':7.,'region_candidate_jobs_tested':5}
            attempt=Mock(side_effect=[strict,relaxed])
            result=plan_with_orientation(attempt,'strict_then_fov')
            self.assertEqual([c.args[0] for c in attempt.call_args_list],['point_at_object','camera_fov'])
            self.assertEqual(result['orientation_mode'],'camera_fov')
            self.assertEqual(result['ik_elapsed_ms'],14.)
            self.assertEqual(result['hybrid_ik_jobs'],21)
            self.assertTrue(result['orientation_selection']['fallback_used'])
            self.assertEqual(result['orientation_selection']['strict']['status'],status)

    def test_invalid_or_invisible_start_cannot_trigger_relaxation(self):
        for status in ('invalid_start','outside_workspace','object_not_visible','final_validation_failed'):
            attempt=Mock(return_value={'status':status,'playable':False})
            result=plan_with_orientation(attempt,'strict_then_fov')
            attempt.assert_called_once_with('point_at_object')
            self.assertFalse(result['orientation_selection']['fallback_used'])

    def test_cancellation_propagates_without_starting_fallback(self):
        attempt=Mock(side_effect=SearchBudgetExpired('cancelled'))
        with self.assertRaisesRegex(SearchBudgetExpired,'cancelled'):
            plan_with_orientation(attempt,'strict_then_fov')
        attempt.assert_called_once_with('point_at_object')

    def test_failed_fallback_still_holds(self):
        attempt=Mock(side_effect=[{'status':'ik_failed','playable':False,'reason':'strict exhausted'},
            {'status':'no_path','playable':False,'frames':[],'reason':'no checked relaxed route'}])
        result=plan_with_orientation(attempt,'strict_then_fov')
        self.assertFalse(result['playable']);self.assertEqual(result['frames'],[])
        self.assertEqual(result['status'],'no_path')

    def test_real_hybrid_solves_failure_without_a_witness_and_checks_motion(self):
        case=load_case(FIXTURE.with_name('ik_failed_reachable_150.json'))
        body=deepcopy(case['request']);body['orientation_mode']='strict_then_fov'
        body['_visibility_camera']=CameraModel(**case['camera'])
        result=classical_tracking.plan(body)
        self.assertTrue(result['playable'] and result['converged'])
        self.assertEqual(result['status'],'target_reached')
        self.assertEqual(result['orientation_mode'],'camera_fov')
        self.assertEqual(result['orientation_selection']['strict']['status'],'ik_failed')
        self.assertEqual(result['orientation_selection']['strict']['ik_jobs'],16)
        self.assertTrue(result['orientation_selection']['fallback_used'])
        self.assertIsNone(result['cursor_direction_tolerance_deg'])
        with VisibilityEnv(np.asarray(body['cursor_mm'])/1000,camera=body['_visibility_camera']) as view:
            view.set_obstacle(None)
            states=np.deg2rad([f['q'] for f in result['frames']])
            self.assertTrue(view.path_valid(states,.005))
            self.assertTrue(view.audit(states,.005)['passed'])
            profile_tracking.validate(result['smoothing'],result['frames'],
                (np.deg2rad(body['start_deg']),np.zeros(6),np.zeros(6)),body['speed_rad_s'],view)

    def test_real_initial_hold_stays_strict_and_non_region_hybrid_is_rejected(self):
        case=load_case(FIXTURE.with_name('ik_failed_reachable_150.json'))
        body=deepcopy(case['request']);body['orientation_mode']='strict_then_fov'
        body['tracking_region']['distance_mm']=170.
        result=classical_tracking.plan(body)
        self.assertEqual(result['orientation_mode'],'point_at_object')
        self.assertTrue(result['playable'] and result['converged'])
        self.assertFalse(result['orientation_selection']['fallback_used'])
        self.assertEqual(result['region_candidate_jobs_tested'],0)
        self.assertEqual(result['cursor_direction_tolerance_deg'],1.)
        with self.assertRaisesRegex(ValueError,'cone disk'):
            classical_tracking.plan({**body,'tracking_region':None})
