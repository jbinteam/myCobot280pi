"""Cone-region hardware gates with an in-memory encoder transport."""
import copy
import unittest
import numpy as np

import test_v2_mouse as lifecycle
from simulation.kinematics import ARM
from simulation.tracking_geometry import region_for, region_metrics
from webui import encoder_drag


class ConeHardwareChecks:
    REGION = {'mode': 'cone_disk', 'distance_mm': 100., 'half_angle_deg': 10.}
    PROFILE = {'kind': 'smooth_trapezoid', 'rise_percent': 5., 'fall_percent': 10.}

    def setUp(self):
        lifecycle.V2MouseTest.setUp(self)
        self.goal = self.start.copy(); self.goal[0] += np.deg2rad(4.)
        self.controller.ik.stop_mouse_aim(self.session)
        self.session = self.controller.ik.start_mouse_aim({'armed': True, 'frame_verified': True,
            'tracking_engine': self.ENGINE, 'scene_verified': True, 'speed': 30,
            'velocity_shape': self.PROFILE, 'tracking_region': self.REGION})['session_id']

    def request(self):
        t = ARM.transform(self.goal)
        return {'session_id': self.session, 'cursor_mm': (t[:3, 3]*1000+100*t[:3, 2]).tolist(),
                'target_axis': t[:3, 2].tolist(), 'tracking_region': self.REGION}

    def plan(self):
        return self.controller.ik.prepare_mouse_aim(self.request())

    def proposal(self, body):
        result = lifecycle.V2MouseTest.proposal(self, body)
        if 'tracking_region' in body:
            result['tracking_region'] = region_for(body['cursor_mm'], body['target_axis'], body['tracking_region'])
        return result

    def test_region_settings_and_protocol_are_pinned(self):
        state = self.controller.snapshot()['ik']
        self.assertEqual(state['mouse_tracking_region_protocol'], 1)
        self.assertEqual(state['tracking_region_settings'], self.REGION)
        for setting in ({**self.REGION, 'distance_mm': 150}, {**self.REGION, 'radius_mm': 99}, None):
            request = {**self.request(), 'tracking_region': setting}
            with self.assertRaises(ValueError):
                self.controller.ik.prepare_mouse_aim(request)
        self.assertEqual(self.robot.calls, [])

    def test_axis_is_captured_from_live_pose_and_frozen_for_same_cursor(self):
        body = self.request(); body['target_axis'] = [1., 0., 0.]
        data = self.controller.ik.prepare_mouse_aim(body)
        region = data['result']['tracking_region']
        expected = np.asarray(body['cursor_mm'])-ARM.fk(self.start)*1000
        np.testing.assert_allclose(region['axis'], expected/np.linalg.norm(expected), atol=1e-12)
        self.robot.angles[0] += .5
        body['target_axis'] = [0., 1., 0.]
        replacement = self.controller.ik.prepare_mouse_aim(body)
        np.testing.assert_allclose(replacement['result']['tracking_region']['axis'], region['axis'], atol=1e-12)
        self.assertEqual(data['plan']['tracking_region'], region)

    def test_planner_cannot_change_region_or_claim_an_outside_goal(self):
        data = self.plan(); frozen = copy.deepcopy(data['result']['tracking_region'])
        real = self.proposal
        def altered(body):
            result = real(body); result['tracking_region']['radius_mm'] += 1.
            return result
        self.planner.side_effect = altered
        with self.assertRaisesRegex(ValueError, 'server-owned tracking region'):
            self.plan()
        self.assertEqual(self.controller.ik.v2_mouse.proposed_region, frozen)
        self.planner.side_effect = real
        self.goal[0] += np.deg2rad(25)
        old_request = self.request()
        old_request['cursor_mm'] = frozen['cursor_mm']
        with self.assertRaisesRegex(ValueError, 'outside the tracking region'):
            self.controller.ik.prepare_mouse_aim(old_request)
        self.assertEqual(self.robot.calls, [])

    def test_off_center_goal_completes_with_shared_encoder_executor(self):
        data = self.plan(); region = data['result']['tracking_region']
        self.assertGreater(np.linalg.norm(np.asarray(region['center_mm'])-np.asarray(data['result']['target_mm'])), 5.)
        self.execute(data)
        status = self.wait({'completed', 'failed'})
        self.assertEqual(status['state'], 'completed', status)
        acceptance = status['tracking_region_acceptance']
        self.assertTrue(acceptance['accepted'])
        self.assertAlmostEqual(acceptance['distance_mm'], 100., delta=1.)
        self.assertGreater(acceptance['radial_distance_mm'], 5.)
        self.assertTrue(any(c[0] == 'encoder_drag' for c in self.robot.calls))
        self.assertFalse(any(c[0] == 'move' for c in self.robot.calls))

    def test_executor_independently_rechecks_final_region(self):
        data = self.plan()
        # Deliberately corrupt a server fixture after planning to verify the
        # executor does not trust the planner's accepted/converged flags.
        self.controller.ik.plan['tracking_region']['center_mm'] = (
            np.asarray(data['result']['tracking_region']['center_mm'])+100*np.asarray(data['result']['tracking_region']['axis'])).tolist()
        self.execute(data)
        status = self.wait({'completed', 'failed'})
        self.assertEqual(status['state'], 'completed', status)
        self.assertFalse(status['tracking_region_acceptance']['accepted'])
        self.assertFalse(status['final_goal_accepted'])
        self.assertTrue(status['replan_required'])
        self.assertTrue(status['feedback_replan_required'])
        self.assertIn('correction', status['message'])
        self.assertFalse(any(c[0] == 'stop' for c in self.robot.calls))

    def test_quantized_endpoint_cannot_cross_disk_rim(self):
        body = self.request()
        region = region_for(body['cursor_mm'], np.asarray(body['cursor_mm'])-ARM.fk(self.start)*1000, self.REGION)
        mapping = self.controller.ik.v2_mouse.encoder_mapping
        start = np.asarray(mapping['start_deg'])
        directions = np.asarray(mapping['encoder_directions'])
        displacement = np.rad2deg(self.goal)-start
        quantized = start.copy()
        known = directions != 0
        quantized[known] += np.rint(displacement[known]*encoder_drag.STEPS_PER_DEG*directions[known])/(encoder_drag.STEPS_PER_DEG*directions[known])
        radius = region_metrics(self.goal, region, 1., 1.)['radial_distance_mm']
        quantized_radius = region_metrics(np.deg2rad(quantized), region, 5., 3.)['radial_distance_mm']
        self.assertGreater(quantized_radius, radius)
        self.REGION = {**self.REGION, 'half_angle_deg': float(np.rad2deg(np.arctan((radius+quantized_radius)/200.)))}
        self.controller.ik.stop_mouse_aim(self.session)
        self.session = self.controller.ik.start_mouse_aim({'armed': True, 'frame_verified': True,
            'tracking_engine': self.ENGINE, 'scene_verified': True, 'speed': 30,
            'velocity_shape': self.PROFILE, 'tracking_region': self.REGION})['session_id']
        with self.assertRaisesRegex(ValueError, 'Quantized mouse endpoint'):
            self.plan()
        self.assertEqual(self.robot.calls, [])


class ClassicalConeHardwareTest(ConeHardwareChecks, unittest.TestCase):
    ENGINE = 'learned_rrt'
    tearDown = lifecycle.V2MouseTest.tearDown
    execute = lifecycle.V2MouseTest.execute
    wait = lifecycle.V2MouseTest.wait


class DiffusionConeHardwareTest(ConeHardwareChecks, unittest.TestCase):
    ENGINE = 'diffusion_v2'
    tearDown = lifecycle.V2MouseTest.tearDown
    execute = lifecycle.V2MouseTest.execute
    wait = lifecycle.V2MouseTest.wait


if __name__ == '__main__':
    unittest.main()
