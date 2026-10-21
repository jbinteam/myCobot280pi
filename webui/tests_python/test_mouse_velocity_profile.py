"""Checked shared velocity profile through both continuous hardware engines."""
from unittest.mock import patch
import test_v2_mouse as lifecycle
from simulation import profile_tracking
from webui import v2_mouse

SHAPE={'kind':'smooth_trapezoid','rise_percent':5,'fall_percent':10}

class ProfileChecks:
    PROFILE = SHAPE
    def wait(self, wanted, seconds=12):
        return super().wait(wanted, seconds)
    def test_same_fixed_target_shape_is_held_by_session_and_returned_curve(self):
        data=self.plan()
        self.assertEqual(data['result']['smoothing']['method'],profile_tracking.METHOD)
        self.assertEqual(data['result']['smoothing']['reference_shape'],SHAPE)
        self.assertEqual(self.controller.snapshot()['ik']['velocity_shape'],SHAPE)
        self.assertEqual(self.controller.snapshot()['ik']['mouse_velocity_profile_protocol'],1)
        self.assertEqual(self.robot.calls,[])
        with self.assertRaisesRegex(ValueError,'profile changed'):
            self.controller.ik.prepare_mouse_aim({'session_id':self.session,'cursor_mm':[100,0,100],
                'velocity_shape':{'kind':'quintic'}})
    def test_planner_cannot_silently_replace_requested_shape(self):
        real=self.proposal
        def changed(body):
            result=real(body);result['smoothing']['reference_shape']={'kind':'quintic'};return result
        self.planner.side_effect=changed
        with self.assertRaisesRegex(ValueError,'selected mouse velocity profile'):
            self.plan()
        self.assertIsNone(self.controller.ik.plan)
        self.assertEqual(self.robot.calls,[])

class ClassicalProfileMouseTest(ProfileChecks,lifecycle.V2MouseTest):
    ENGINE='learned_rrt'

class DiffusionProfileMouseTest(ProfileChecks,lifecycle.V2MouseTest):
    ENGINE='diffusion_v2'
