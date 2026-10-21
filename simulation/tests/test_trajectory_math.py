"""Exact trajectory arithmetic parity and conservative scalar fallback tests.

CPU-only: no diffusion inference, CUDA initialization or physical robot IO.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
from simulation import continuous, motion_limits, trajectory_math
from simulation.kinematics import ARM
from simulation.prm import JointSpace


def original_leaves(controls, depth):
    if not depth:
        return [controls]
    rows = [controls]
    while len(rows[-1]) > 1:
        rows.append((rows[-1][:-1]+rows[-1][1:])/2)
    left = np.array([row[0] for row in rows])
    right = np.array([row[-1] for row in rows[::-1]])
    return original_leaves(left, depth-1)+original_leaves(right, depth-1)


def original_bound(controls, depth):
    if not depth:
        return float(np.linalg.norm(controls, axis=1).max(initial=0))
    rows = [controls]
    while len(rows[-1]) > 1:
        rows.append((rows[-1][:-1]+rows[-1][1:])/2)
    return max(original_bound(np.array([row[0] for row in rows]), depth-1),
               original_bound(np.array([row[-1] for row in rows[::-1]]), depth-1))


def original_hull(controls, depth):
    controls = np.asarray(controls, float)
    if not depth:
        return np.max(np.abs(controls), axis=0)
    rows = [controls]
    while len(rows[-1]) > 1:
        rows.append((rows[-1][:-1]+rows[-1][1:])/2)
    return np.maximum(original_hull(np.array([row[0] for row in rows]), depth-1),
                      original_hull(np.array([row[-1] for row in rows[::-1]]), depth-1))


def assert_bytes(test, a, b):
    a, b = np.asarray(a), np.asarray(b)
    test.assertEqual(a.shape, b.shape)
    test.assertEqual(a.dtype, b.dtype)
    test.assertEqual(a.tobytes(), b.tobytes())


class SubdivisionParityTest(unittest.TestCase):
    def test_random_leaves_order_norm_and_hull_are_bitwise_exact(self):
        rng = np.random.default_rng(28020261007)
        with patch.object(trajectory_math, 'BATCH_TRAJECTORY', True):
            for rows in (4, 5):
                for depth in (1, 2, 4, 8):
                    for scale in (1e-100, 1., 1e3):
                        controls = rng.normal(size=(rows, 6))*scale
                        before = controls.tobytes()
                        with self.subTest(rows=rows, depth=depth, scale=scale):
                            leaves = trajectory_math.subdivision_leaves(controls, depth)
                            expected = np.concatenate(original_leaves(controls, depth))
                            self.assertEqual(leaves.shape, (rows*2**depth, 6))
                            assert_bytes(self, leaves, expected)
                            assert_bytes(self, continuous.bound(controls, depth), original_bound(controls, depth))
                            assert_bytes(self, motion_limits.derivative_hull(controls, depth), original_hull(controls, depth))
                            self.assertEqual(controls.tobytes(), before)

    def test_subnormals_signed_zero_magnitude_limit_and_strided_inputs(self):
        tiny = np.nextafter(0., 1.)
        values = np.resize(np.array([0., -0., tiny, -tiny, np.finfo(float).tiny,
                                     -np.finfo(float).tiny]), (5, 6))
        cases = [np.zeros((4, 6)), -np.zeros((5, 6)), values,
                 np.full((4, 6), 1e100), np.full((5, 6), -1e100),
                 np.arange(60., dtype=np.float64).reshape(5, 12)[:, ::2],
                 np.asfortranarray(np.arange(30., dtype=np.float64).reshape(5, 6))]
        cases[-1].flags.writeable = False
        with patch.object(trajectory_math, 'BATCH_TRAJECTORY', True), np.errstate(under='ignore'):
            for controls in cases:
                before = controls.tobytes()
                for depth in (1, 4):
                    with self.subTest(shape=controls.shape, depth=depth, magnitude=controls.flat[0]):
                        leaves = trajectory_math.subdivision_leaves(controls, depth)
                        self.assertIsNotNone(leaves)
                        assert_bytes(self, leaves, np.concatenate(original_leaves(controls, depth)))
                        assert_bytes(self, continuous.bound(controls, depth), original_bound(controls, depth))
                        assert_bytes(self, motion_limits.derivative_hull(controls, depth), original_hull(controls, depth))
                self.assertEqual(controls.tobytes(), before)

    def test_unsupported_inputs_and_depths_dispatch_conservatively(self):
        controls = np.ones((5, 6), dtype=np.float64)
        class ArraySubclass(np.ndarray):
            pass
        unsupported = [controls.tolist(), controls.astype(np.float32), controls.astype(np.int64),
                       controls.astype('>f8'), controls.view(ArraySubclass), np.ones((6, 6)),
                       np.ones((4, 5)), np.ones(6), np.full((5, 6), np.nan),
                       np.full((5, 6), np.inf), np.full((5, 6), np.nextafter(1e100, np.inf))]
        with patch.object(trajectory_math, 'BATCH_TRAJECTORY', True):
            for value in unsupported:
                self.assertIsNone(trajectory_math.subdivision_leaves(value, 4))
            for depth in (0, -1, 9, True, np.int64(4), 2., .5, None):
                self.assertIsNone(trajectory_math.subdivision_leaves(controls, depth))
            # Supported original depths outside the optimized contract preserve outputs.
            for depth in (0, 9, True, np.int64(2), 2.):
                assert_bytes(self, continuous.bound(controls, depth), original_bound(controls, depth))
                assert_bytes(self, motion_limits.derivative_hull(controls, depth), original_hull(controls, depth))
        with patch.object(trajectory_math, 'BATCH_TRAJECTORY', False):
            self.assertIsNone(trajectory_math.subdivision_leaves(controls, 4))

    def test_overflow_and_nan_inputs_retain_original_recursive_behavior(self):
        cases = [np.full((5, 6), 1e308), np.full((4, 6), -1e308),
                 np.full((5, 6), np.inf), np.full((4, 6), np.nan)]
        mixed = np.arange(30., dtype=float).reshape(5, 6); mixed[-1, 0] = np.nan
        cases.append(mixed)
        with patch.object(trajectory_math, 'BATCH_TRAJECTORY', True), np.errstate(all='ignore'):
            for controls in cases:
                self.assertIsNone(trajectory_math.subdivision_leaves(controls, 4))
                with patch.object(continuous, '_bound_recursive', wraps=continuous._bound_recursive) as scalar:
                    result = continuous.bound(controls, 4)
                    self.assertGreater(scalar.call_count, 0)
                assert_bytes(self, result, original_bound(controls, 4))
                with patch.object(motion_limits, '_derivative_hull_recursive',
                                  wraps=motion_limits._derivative_hull_recursive) as scalar:
                    result = motion_limits.derivative_hull(controls, 4)
                    self.assertGreater(scalar.call_count, 0)
                assert_bytes(self, result, original_hull(controls, 4))

    def test_environment_flag_is_read_at_import(self):
        script = 'from simulation.trajectory_math import BATCH_TRAJECTORY; print(int(BATCH_TRAJECTORY))'
        root = Path(__file__).resolve().parents[2]
        for value, expected in (('0', '0'), ('1', '1')):
            process = subprocess.run([sys.executable, '-c', script], cwd=root,
                                     env={**os.environ, 'MYCOBOT_TRAJECTORY_BATCH': value},
                                     capture_output=True, text=True, check=True, timeout=10)
            self.assertEqual(process.stdout.strip(), expected)


class TrajectoryParityTest(unittest.TestCase):
    def test_c2_states_times_metadata_and_collision_sample_order_are_exact(self):
        start = np.array([0., -.7, -1.5, .8, 0., .3])
        goal = start+[.15, .04, .04, 0., 0., 0.]
        fixtures = [([start], np.zeros(6), np.zeros(6)),
                    ([start, goal], np.zeros(6), np.zeros(6)),
                    ([start, goal, goal+[0., .02, 0., .04, 0., 0.]],
                     np.array([.04, .02, 0., 0., 0., 0.]), np.array([.05, 0., 0., 0., 0., 0.]))]
        for path, velocity, acceleration in fixtures:
            outputs = []
            for enabled in (False, True):
                seen = []
                def valid(q):
                    seen.append(np.array(q)); return True
                space = JointSpace(valid, {'checker': 'exact trajectory parity', 'environment': 'test'})
                with patch.object(trajectory_math, 'BATCH_TRAJECTORY', enabled):
                    states, times, metadata = continuous.prepare(space, path, velocity, acceleration)
                outputs.append((states, times, json.dumps(metadata, sort_keys=True, allow_nan=False), np.array(seen)))
            for index in (0, 1, 3):
                assert_bytes(self, outputs[0][index], outputs[1][index])
            self.assertEqual(outputs[0][2], outputs[1][2])

    def test_unsafe_curved_join_and_motion_limit_rejections_match(self):
        fixtures = [([np.zeros(6), np.array([.3, 0., 0., 0., 0., 0.])],
                     np.array([0., -.2, 0., 0., 0., 0.]), np.zeros(6), lambda q: q[1] >= -.00001),
                    ([np.zeros(6), np.ones(6)*.1], np.ones(6)*.3, np.zeros(6), lambda q: True),
                    ([np.zeros(6), np.ones(6)*.1], np.zeros(6), np.zeros(6), lambda q: q[0] < .05)]
        for path, velocity, acceleration, valid in fixtures:
            messages = []
            for enabled in (False, True):
                with patch.object(trajectory_math, 'BATCH_TRAJECTORY', enabled):
                    with self.assertRaises(ValueError) as error:
                        continuous.prepare(JointSpace(valid, {'checker': 'rejection test'}),
                                           path, velocity, acceleration)
                    messages.append(str(error.exception))
            self.assertEqual(messages[0], messages[1])

    def test_frame_fk_bitwise_across_operating_limits_and_block_boundaries(self):
        rng = np.random.default_rng(280513)
        states = np.vstack((ARM.lower, ARM.upper, ARM.center,
                            np.nextafter(ARM.lower, ARM.upper), np.nextafter(ARM.upper, ARM.lower),
                            np.nextafter(ARM.lower, -np.inf), np.nextafter(ARM.upper, np.inf),
                            rng.uniform(ARM.lower, ARM.upper, (506, 6))))
        original = np.array([ARM.transform(q) for q in states])
        for count in (0, 1, 255, 256, 257, 511, 512, 513):
            outputs = []
            for enabled in (False, True):
                with patch.object(trajectory_math, 'BATCH_TRAJECTORY', enabled):
                    result = list(trajectory_math.frame_transforms(states[:count]))
                outputs.append(np.array(result))
            assert_bytes(self, outputs[0], outputs[1])
            if count:
                assert_bytes(self, outputs[1], original[:count])
        with patch.object(trajectory_math, 'BATCH_TRAJECTORY', True), \
                patch.object(ARM, 'link_transforms_batch', wraps=ARM.link_transforms_batch) as batched:
            list(trajectory_math.frame_transforms(states))
        self.assertEqual([len(call.args[0]) for call in batched.call_args_list], [256, 256, 1])


if __name__ == '__main__':
    unittest.main()
