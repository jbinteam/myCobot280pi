"""Independent visibility gates around server curves and encoder execution."""
import copy
import time
import unittest
from unittest.mock import Mock, patch
import numpy as np
import test_v2_mouse as lifecycle
import test_object_follow_api as fixture
from simulation.kinematics import ARM
from simulation.camera_model import CameraModel
from simulation.diffusion.endpoint_selection import SearchBudgetExpired
from simulation.diffusion.visibility import VisibilityEnv
from webui import v2_mouse, encoder_profile
from webui.tracking_scene import scene_for


class VisibleControllerTest(unittest.TestCase):
    ENGINE=fixture.ObjectFollowAPITest.ENGINE
    PROFILE=fixture.ObjectFollowAPITest.PROFILE
    REGION=fixture.ObjectFollowAPITest.REGION
    setUp=fixture.ObjectFollowAPITest.setUp
    tearDown=fixture.ObjectFollowAPITest.tearDown
    start_ros=fixture.ObjectFollowAPITest.start_ros
    proposal=fixture.ObjectFollowAPITest.proposal
    plan=fixture.ObjectFollowAPITest.plan
    execute=lifecycle.V2MouseTest.execute
    wait=lifecycle.V2MouseTest.wait
    wait_motion=lifecycle.V2MouseTest.wait_motion

    def behind(self):
        t=ARM.transform(np.deg2rad(self.robot.angles))
        return (t[:3,3]-.35*t[:3,2]).tolist()

    def test_camera_is_pinned_for_planner_curve_and_encoder_scenes(self):
        camera = CameraModel(eef_to_optical_translation_m=(.001,0.,0.))
        self.controller.visibility_camera = camera
        state = self.start_ros()
        self.assertEqual(state['ik']['visibility_camera'], camera.configuration())
        self.controller.visibility_camera = CameraModel(eef_to_optical_translation_m=(.01,0.,0.))
        self.plan()
        plan = self.controller.ik.plan
        self.assertEqual(self.planner.call_args.args[0]['_visibility_camera'], camera)
        self.assertEqual(plan['visibility_camera'], camera.configuration())
        self.assertEqual(plan['result']['visibility_camera'], camera.configuration())
        with scene_for(plan) as scene:
            self.assertEqual(scene.camera, camera)
        self.assertTrue(plan['result']['quantized_visibility']['passed'])
        for key in ('visibility_camera', '_visibility_camera'):
            with self.assertRaises(ValueError):
                self.controller.ik.prepare_mouse_aim({'session_id':self.session,
                    'cursor_mm':plan['cursor_mm'], key:CameraModel().configuration()})
        self.assertEqual(self.robot.calls, [])

    def test_retarget_cannot_change_camera_before_switching_object(self):
        from webui.tracking_scene import set_candidate_object
        self.start_ros(); self.plan()
        plan = self.controller.ik.plan
        candidate = copy.deepcopy(plan)
        candidate['visibility_camera'] = CameraModel(eef_to_optical_translation_m=(.01,0.,0.)).configuration()
        candidate['visibility_object_m'] = self.behind()
        with scene_for(plan) as scene:
            original = scene.object_m.copy()
            with self.assertRaisesRegex(ValueError, 'pinned camera'):
                set_candidate_object(scene, candidate)
            np.testing.assert_array_equal(scene.object_m, original)

    def test_curve_visibility_is_recomputed_despite_forged_success_flag(self):
        self.start_ros(); data=self.plan();plan=self.controller.ik.plan
        result=copy.deepcopy(data['result']); result['visibility']={'passed':True}
        v2_mouse.validate_curve(result,plan['expected_motion'],self.controller.ik.v2_mouse.speed_rad_s,
                                 plan['environment'],self.ENGINE,plan['visibility_object_m'])
        with self.assertRaises(ValueError):
            v2_mouse.validate_curve(result,plan['expected_motion'],self.controller.ik.v2_mouse.speed_rad_s,
                                     plan['environment'],self.ENGINE,self.behind())
        self.assertEqual(self.robot.calls,[])

    def test_encoder_quantization_rechecks_object_visibility(self):
        self.start_ros();self.plan();plan=copy.deepcopy(self.controller.ik.plan)
        self.assertTrue(plan['result']['quantized_visibility']['passed'])
        plan['visibility_object_m']=self.behind()
        with self.assertRaisesRegex(ValueError,'visibility'):
            encoder_profile.prepare_mouse(plan,self.controller.ik.v2_mouse.encoder_mapping,30)
        self.assertEqual(self.robot.calls,[])

    def test_expired_encoder_candidate_never_starts_validation(self):
        with patch.object(encoder_profile.encoder_drag, 'direction_evidence') as evidence:
            with self.assertRaises(SearchBudgetExpired):
                encoder_profile.prepare_mouse({}, {}, 30, deadline=time.perf_counter()-1.)
        evidence.assert_not_called()
        self.assertEqual(self.robot.calls, [])

    def test_encoder_candidate_deadline_interrupts_dense_scene_validation(self):
        self.start_ros(); self.plan()
        plan = copy.deepcopy(self.controller.ik.plan)
        plan.pop('encoder_mapping')
        plan['result'].pop('encoder_drag')
        original = VisibilityEnv.valid
        visited = []
        deadline = time.perf_counter()+100.
        clock = Mock(return_value=deadline-1.)
        def expire_after_first_sample(scene, q):
            visited.append(q.copy())
            valid = original(scene, q)
            # Expire between samples without a timing-sensitive sleep. The
            # deadline wrapper must interrupt before checking another sample.
            clock.return_value = deadline+1.
            return valid
        from simulation.diffusion.endpoint_selection import DeadlineEnv
        def checked_scene(scene, deadline):
            return DeadlineEnv(scene, deadline, clock=clock)
        with patch('simulation.diffusion.endpoint_selection.DeadlineEnv', side_effect=checked_scene), \
                patch.object(VisibilityEnv, 'valid', expire_after_first_sample):
            with self.assertRaises(SearchBudgetExpired):
                encoder_profile.prepare_mouse(plan, self.controller.ik.v2_mouse.encoder_mapping,
                                              30, deadline=deadline)
        self.assertEqual(len(visited), 1)
        self.assertNotIn('encoder_mapping', plan)
        self.assertNotIn('encoder_drag', plan['result'])
        self.assertEqual(self.robot.calls, [])

    def test_live_visibility_failure_sends_no_encoder_target(self):
        self.start_ros();data=self.plan()
        # Corrupt a private server fixture after validation. The live executor
        # must still re-evaluate visibility rather than trusting prior flags.
        self.controller.ik.plan['visibility_object_m']=self.behind()
        self.execute(data)
        status=self.wait({'failed'})
        self.assertIn('visibility',status['message'])
        self.assertFalse(any(c[0]=='encoder_drag' for c in self.robot.calls))

    def test_rejected_retarget_restores_previous_visibility_object(self):
        self.start_ros();first=self.plan();active=self.controller.ik.plan
        self.execute(first);self.wait_motion()
        self.goal[0]+=np.deg2rad(.3)
        candidate=self.plan();self.execute(candidate,acknowledge=False)
        with self.controller.lock:
            mouse=self.controller.ik.v2_mouse
            pending=self.controller.ik.mouse_pending_plan
            self.assertIsNotNone(pending)
            pending['confirmed']=True
            pending['plan']['visibility_object_m']=self.behind()
            with scene_for(active) as scene:
                scene.set_obstacle(None);original=scene.object_m.copy()
                self.assertTrue(scene.valid(np.deg2rad(self.robot.angles)))
                kept,_=mouse.advance_pending(active,mouse.clock_origin,time.monotonic(),scene)
                self.assertEqual(kept['id'],active['id'])
                np.testing.assert_array_equal(scene.object_m,original)
                self.assertTrue(scene.valid(np.deg2rad(self.robot.angles)))
            self.assertIsNone(self.controller.ik.mouse_pending_plan)
            self.assertTrue(self.controller.ik.status['feedback_replan_required'])


if __name__=='__main__':unittest.main()
