import unittest
from copy import deepcopy
import numpy as np
from scripts.reproduce_ik_failed_reachable import FIXTURE, load_case
from simulation.camera_model import CameraModel
from simulation.kinematics import ARM
from simulation import profile_tracking
from simulation.tracking_geometry import region_for, region_metrics
from simulation.diffusion import classical_tracking
from simulation.diffusion.visibility import VisibilityEnv
from simulation.diffusion.camera_endpoint_ik import search_camera_endpoint
from simulation.diffusion.endpoint_ik import IKSearchExhausted
from simulation.diffusion.endpoint_selection import SearchBudgetExpired


class CameraEndpointIKTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case=load_case(FIXTURE.with_name('ik_failed_reachable_150.json'))

    def test_real_relaxed_planner_reaches_final_disk_with_checked_full_motion(self):
        body=deepcopy(self.case['request']);body['orientation_mode']='camera_fov'
        body['_visibility_camera']=CameraModel(**self.case['camera'])
        result=classical_tracking.plan(body)
        self.assertTrue(result['playable'] and result['converged'])
        self.assertEqual(result['status'],'target_reached')
        self.assertFalse(result['replan_required'])
        self.assertEqual(result['configuration']['region_candidate_budget'],8)
        self.assertEqual(result['orientation_mode'],'camera_fov')
        states=np.deg2rad([f['q'] for f in result['frames']])
        region=region_for(body['cursor_mm'],body['target_axis'],body['tracking_region'])
        positional=region_metrics(states[-1],region,1.,180.)
        self.assertTrue(positional['accepted'])
        self.assertGreater(positional['direction_error_deg'],3.)
        with VisibilityEnv(np.asarray(body['cursor_mm'])/1000,camera=CameraModel(**self.case['camera'])) as view:
            view.set_obstacle(None)
            self.assertTrue(view.path_valid(states,.005))
            self.assertTrue(view.audit(states,.005)['passed'])
            self.assertTrue(view.inspect(states[-1])['visible'])
            profile_tracking.validate(result['smoothing'],result['frames'],
                (np.deg2rad(body['start_deg']),np.zeros(6),np.zeros(6)),body['speed_rad_s'],view)

    def test_xyz_match_alone_cannot_accept_invisible_or_occluded_endpoint(self):
        start=np.deg2rad(self.case['request']['start_deg']);t=ARM.transform(start)
        camera=CameraModel(**self.case['camera']);target=t[:3,3]
        with VisibilityEnv(target-.17*t[:3,2],camera=camera) as view:
            view.set_obstacle(None)
            with self.assertRaises(IKSearchExhausted):
                search_camera_endpoint(view,target,t[:3,2],start,attempts=1,learned=False)
            view.set_object(target+.17*t[:3,2]);view.set_obstacle(np.r_[target+.085*t[:3,2],[.01,.01,.01]])
            with self.assertRaises(IKSearchExhausted):
                search_camera_endpoint(view,target,t[:3,2],start,attempts=1,learned=False)

    def test_cancellation_and_invalid_mode_are_rejected(self):
        start=np.deg2rad(self.case['request']['start_deg']);t=ARM.transform(start)
        with VisibilityEnv(np.asarray(self.case['request']['cursor_mm'])/1000) as view:
            def cancel():raise SearchBudgetExpired('cancelled test')
            with self.assertRaises(SearchBudgetExpired):
                search_camera_endpoint(view,t[:3,3],t[:3,2],start,cancel_check=cancel)
        with self.assertRaisesRegex(ValueError,'orientation mode'):
            classical_tracking.plan({**self.case['request'],'orientation_mode':'unrestricted'})
        with self.assertRaisesRegex(ValueError,'cone disk'):
            classical_tracking.plan({**self.case['request'],'orientation_mode':'camera_fov','tracking_region':None})
