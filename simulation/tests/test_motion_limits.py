import unittest
import numpy as np
from simulation.kinematics import ARM
from simulation.motion_limits import (quintic_path, curve_envelope, scale_needed,
    envelope, JOINT_SPEED, JOINT_ACCELERATION, TCP_SPEED, TCP_ACCELERATION,
    ANGULAR_SPEED, ANGULAR_ACCELERATION)
from simulation.continuous import evaluate, prepare
from simulation.prm import JointSpace
from simulation import service


class MotionLimitsTest(unittest.TestCase):
    def test_fast_multi_joint_path_is_retimed_and_preserves_every_corner(self):
        path = np.array([[0, -.7, -1.5, .8, 0, .3], [.8, -.3, -1., 1.2, .5, -.3],
                         [.8, -.7, -1., 1.2, -.5, .6]])
        states, times, metadata = quintic_path(path, speed=3., acceleration=10.)
        self.assertEqual(metadata['speed_limit_rad_s'], 2.6)
        self.assertTrue(metadata['specification_checked'])
        for segment, a, b in zip(metadata['segments'], path, path[1:]):
            np.testing.assert_array_equal(states[segment['start_frame']], a)
            np.testing.assert_array_equal(states[segment['end_frame']], b)
            controls = np.array([a, a, a, b, b, b]); duration = segment['duration_s']
            self.assertLessEqual(scale_needed(curve_envelope(controls, duration)), 1.+1e-8)
            self.check_derivatives(controls, duration)
        self.assertTrue(np.allclose(np.diff(times), .02))

    def check_derivatives(self, controls, duration):
        vc = 5*np.diff(controls, axis=0)/duration
        ac = 4*np.diff(vc, axis=0)/duration
        for phase in np.linspace(0, 1, 101):
            q = evaluate(controls, phase); v = evaluate(vc, phase); a = evaluate(ac, phase)
            _, jacobian = ARM.spatial_jacobian(q); eps = 1e-5
            jacobian_dot = (ARM.spatial_jacobian(q+eps*v)[1]-ARM.spatial_jacobian(q-eps*v)[1])/(2*eps)
            twist = jacobian@v; twist_acceleration = jacobian@a+jacobian_dot@v
            self.assertLessEqual(np.max(np.abs(v)), JOINT_SPEED+1e-8)
            self.assertLessEqual(np.max(np.abs(a)), JOINT_ACCELERATION+1e-8)
            self.assertLessEqual(np.linalg.norm(twist[:3]), TCP_SPEED+1e-8)
            self.assertLessEqual(np.linalg.norm(twist_acceleration[:3]), TCP_ACCELERATION+1e-8)
            self.assertLessEqual(np.linalg.norm(twist[3:]), ANGULAR_SPEED+1e-8)
            self.assertLessEqual(np.linalg.norm(twist_acceleration[3:]), ANGULAR_ACCELERATION+1e-8)
            bounds = envelope(np.abs(v), np.abs(a))
            self.assertLessEqual(np.linalg.norm(twist[:3]), bounds['tcp_velocity_m_s']+1e-9)
            self.assertLessEqual(np.linalg.norm(twist_acceleration[:3]), bounds['tcp_acceleration_m_s2']+1e-8)
            self.assertLessEqual(np.linalg.norm(twist_acceleration[3:]), bounds['angular_acceleration_rad_s2']+1e-8)

    def test_nonzero_boundary_is_preserved_while_full_curve_passes_specification(self):
        start = np.array([0, -.7, -1.5, .8, 0, .3]); goal = start+[.15, .04, .04, 0, 0, 0]
        v = np.array([.04, .02, 0, 0, 0, 0]); a = np.array([.05, 0, 0, 0, 0, 0])
        space = JointSpace(lambda q: True, {'checker': 'motion test', 'environment': 'bare flange'})
        _, _, metadata = prepare(space, [start, goal], v, a)
        segment = metadata['segments'][0]; c = np.deg2rad(segment['controls_deg']); t = segment['duration_s']
        np.testing.assert_allclose(5*(c[1]-c[0])/t, v, atol=1e-12)
        np.testing.assert_allclose(20*(c[2]-2*c[1]+c[0])/t**2, a, atol=1e-12)
        self.check_derivatives(c, t)

    def test_boundary_exceeding_cartesian_specification_is_rejected(self):
        space = JointSpace(lambda q: True, {'checker': 'motion test', 'environment': 'bare flange'})
        with self.assertRaises(ValueError):
            prepare(space, [np.zeros(6), np.ones(6)*.1], np.ones(6)*.3, np.zeros(6))

    def test_rest_and_invalid_timing(self):
        states, times, metadata = quintic_path([np.zeros(6)])
        self.assertEqual(len(states), 1); self.assertEqual(times[0], 0.)
        self.assertEqual(metadata['segments'], [])
        for speed in (0, -1, float('nan')):
            with self.assertRaises(ValueError): quintic_path([np.zeros(6)], speed)

    def test_ordinary_api_retains_solver_endpoint_with_smooth_specified_timing(self):
        start = np.array([0., -40., -130., 80., 0., 50.]); goal = start+[3, 2, 0, 0, 0, 0]
        result = service.run({'start_deg': start.tolist(), 'target_mm': (ARM.fk(np.deg2rad(goal))*1000).tolist(),
                              'task': 'position', 'methods': ['dls'], 'speed_rad_s': 3.})['results'][0]
        self.assertTrue(result['converged'])
        self.assertEqual(result['smoothing']['method'], 'specification_checked_quintic')
        self.assertEqual(result['smoothing']['speed_limit_rad_s'], 2.6)
        np.testing.assert_allclose(result['frames'][0]['q'], start, atol=1e-12)
        self.assertLessEqual(result['frames'][-1]['error_mm'], 1.)
        self.assertLessEqual(scale_needed(result['smoothing']['motion_bounds']), 1.+1e-8)
