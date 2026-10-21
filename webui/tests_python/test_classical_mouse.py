"""Run the full continuous hardware lifecycle against the learned IK/RRT engine."""
from unittest.mock import patch
import test_v2_mouse as lifecycle
from webui import v2_mouse


class ClassicalMouseTest(lifecycle.V2MouseTest):
    ENGINE = "learned_rrt"

    def test_selected_engine_checkpoint_and_no_diffusion_inference(self):
        with patch.object(v2_mouse.tracking, 'plan', side_effect=AssertionError('Wrong engine')):
            result = self.plan()
        self.assertEqual(result['result']['method'], 'learned_rrt')
        self.assertEqual(result['plan']['checkpoint_sha256'], v2_mouse.IK_SHA256)
        self.assertTrue(result['plan']['hardware_classical_mouse'])
        self.assertFalse(result['plan']['hardware_diffusion'])
        self.assertEqual(self.controller.snapshot()['ik']['mouse_engine'], self.ENGINE)
        self.assertEqual(self.robot.calls, [])

    def test_rejects_diffusion_identity_in_classical_session(self):
        real = self.proposal
        def changed(body):
            result = real(body)
            result['checkpoint_sha256'] = v2_mouse.V2_SHA256
            return result
        self.planner.side_effect = changed
        with self.assertRaises(ValueError):
            self.plan()
        self.assertIsNone(self.controller.ik.plan)
        self.assertEqual(self.robot.calls, [])
