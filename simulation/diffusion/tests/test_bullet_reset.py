"""Exact scalar/batched Bullet reset semantics; CPU-only, no physical IO.

The tests compare the existing geometry and call results rather than asserting
hardware safety or changing collision tolerances.
"""
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pybullet as p
from simulation.diffusion.env import ArmEnv
from simulation.kinematics import ARM

FIXTURE = Path(__file__).with_name('fixtures')/'grazing_collision.json'
START = np.deg2rad([0, -40, -130, 80, 0, 50])


def joint_states(env):
    return p.getJointStates(env.robot, env.joints, physicsClientId=env.client)


def seed_nonzero_velocities(env, q):
    """Both reset paths must erase velocities, not inherit a zero-velocity scene."""
    for index, (joint, value) in enumerate(zip(env.joints, q)):
        p.resetJointState(env.robot, joint, float(value), targetVelocity=.125*(index+1),
                         physicsClientId=env.client)
    assert all(state[1] != 0 for state in joint_states(env))


def snapshot(env):
    """Every loaded link frame and complete relevant Bullet distance tuple."""
    return {'joints': joint_states(env),
            'links': tuple(p.getLinkState(env.robot, index, computeLinkVelocity=True,
                                          computeForwardKinematics=True, physicsClientId=env.client)
                           for index in range(p.getNumJoints(env.robot, physicsClientId=env.client))),
            'closest': tuple((body, distance,
                              p.getClosestPoints(env.robot, body, distance, physicsClientId=env.client))
                             for body in (env.table, env.box) if body is not None
                             for distance in (env.clearance, .02))}


class BulletResetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grazing = json.loads(FIXTURE.read_text())
        random = np.random.default_rng(2802026).uniform(ARM.lower, ARM.upper, (24, 6))
        near = np.array(cls.grazing['invalid_state'])
        perturbations = near+np.array([-1e-8, 0, 1e-8])[:, None]*np.array([1, -1, 1, -1, 1, -1])
        cls.poses = np.vstack((START, random, ARM.lower, ARM.upper,
                               cls.grazing['start'], cls.grazing['goal'], perturbations))

    def test_exact_joint_velocity_link_and_contact_parity(self):
        with ArmEnv() as env:
            for obstacle in (None, self.grazing['obstacle']):
                env.set_obstacle(obstacle)
                for index, q in enumerate(self.poses):
                    values = []
                    for batch in (False, True):
                        env.batch_reset = batch
                        seed_nonzero_velocities(env, START)
                        env._set_joints(q)
                        values.append(snapshot(env))
                        self.assertEqual([state[0] for state in values[-1]['joints']], q.tolist())
                        self.assertEqual([state[1] for state in values[-1]['joints']], [0.]*6)
                    with self.subTest(obstacle=obstacle is not None, pose=index):
                        self.assertEqual(values[0], values[1])

    def test_grazing_boundary_straddles_one_mm_with_exact_contact_parity(self):
        a, b = np.array(self.grazing['start']), np.array(self.grazing['invalid_state'])
        with ArmEnv() as env:
            env.set_obstacle(self.grazing['obstacle']); env.batch_reset = False
            def separation(fraction):
                env._set_joints(a+fraction*(b-a))
                return min(point[8] for point in p.getClosestPoints(env.robot, env.box, .1,
                                                                   physicsClientId=env.client))
            self.assertGreater(separation(0), env.clearance)
            self.assertLess(separation(1), env.clearance)
            lo, hi = 0., 1.
            for _ in range(45):
                middle = (lo+hi)/2
                if separation(middle) > env.clearance:
                    lo = middle
                else:
                    hi = middle
            self.assertLess(abs(separation(lo)-env.clearance), 1e-10)
            self.assertLess(abs(separation(hi)-env.clearance), 1e-10)
            classifications = []
            for fraction in (lo-1e-7, lo, hi, hi+1e-7):
                q = a+fraction*(b-a)
                self.assertTrue(env.space.valid(q))
                comparisons = []
                for batch in (False, True):
                    env.batch_reset = batch
                    seed_nonzero_velocities(env, START)
                    env._set_joints(q)
                    contacts = p.getClosestPoints(env.robot, env.box, env.clearance, physicsClientId=env.client)
                    state = snapshot(env)
                    env._valid_cache.clear()
                    scalar = env.valid(q)
                    env._valid_cache.clear()
                    vector = bool(env.valid_many([q])[0])
                    self.assertEqual(scalar, vector)
                    self.assertEqual(scalar, not bool(contacts))
                    np.testing.assert_array_equal(env.current(), env._display_q)
                    comparisons.append((state, contacts, scalar, vector))
                self.assertEqual(comparisons[0], comparisons[1])
                classifications.append(bool(comparisons[0][1]))
            self.assertFalse(classifications[0])
            self.assertTrue(classifications[-1])

    def test_scalar_and_batched_validity_results_restore_display(self):
        poses = np.vstack((self.poses, self.poses[0], self.poses[0]))
        with ArmEnv() as env:
            for obstacle in (None, self.grazing['obstacle']):
                env.set_obstacle(obstacle)
                comparisons = []
                for batch in (False, True):
                    env.batch_reset = batch; env.reset(START)
                    env._valid_cache.clear()
                    scalar = [env.valid(q) for q in poses]
                    self.assertEqual(joint_states(env), tuple((float(q), 0., (0.,)*6, 0.) for q in START))
                    np.testing.assert_array_equal(env._display_q, START)
                    env._valid_cache.clear()
                    vector = env.valid_many(poses).tolist()
                    np.testing.assert_array_equal(env.current(), START)
                    np.testing.assert_array_equal(env._display_q, START)
                    self.assertEqual([state[1] for state in joint_states(env)], [0.]*6)
                    self.assertEqual(vector, scalar)
                    comparisons.append((scalar, vector))
                self.assertEqual(comparisons[0], comparisons[1])

    def test_query_exceptions_propagate_and_restore_display_in_both_modes(self):
        q = START.copy(); q[0] += .1
        with ArmEnv() as env:
            env.set_obstacle(None)
            for batch in (False, True):
                for method in ('valid', 'valid_many'):
                    with self.subTest(batch=batch, method=method):
                        env.batch_reset = batch; env.reset(START); env._valid_cache.clear()
                        with patch('simulation.diffusion.env.p.getClosestPoints', side_effect=RuntimeError('query failed')):
                            with self.assertRaisesRegex(RuntimeError, 'query failed'):
                                getattr(env, method)(q if method == 'valid' else [q])
                        np.testing.assert_array_equal(env.current(), START)
                        np.testing.assert_array_equal(env._display_q, START)
                        self.assertNotIn(q.tobytes(), env._valid_cache)

    def test_isolated_clients_keep_other_positions_and_nonzero_velocities(self):
        q = START.copy(); q[0] += .2
        with ArmEnv() as first, ArmEnv() as second:
            self.assertNotEqual(first.client, second.client)
            seed_nonzero_velocities(second, q)
            original = snapshot(second)
            for batch in (False, True):
                first.batch_reset = batch
                first.reset(ARM.center)
                first.valid_many([START, q])
                self.assertEqual(snapshot(second), original)

    def test_environment_flag_and_missing_api_select_scalar_fallback(self):
        for flag, api in (('0', p.resetJointStatesMultiDof), ('1', None)):
            with self.subTest(flag=flag, api=api is not None), \
                    patch.dict(os.environ, {'MYCOBOT_BULLET_BATCH_RESET': flag}), \
                    patch('simulation.diffusion.env.p.resetJointStatesMultiDof', api), ArmEnv() as env:
                self.assertIs(env.batch_reset, False)
                seed_nonzero_velocities(env, ARM.center)
                with patch('simulation.diffusion.env.p.resetJointState', wraps=p.resetJointState) as scalar:
                    env._set_joints(START)
                self.assertEqual(scalar.call_count, 6)
                np.testing.assert_array_equal(env.current(), START)
                self.assertEqual([state[1] for state in joint_states(env)], [0.]*6)
        with patch.dict(os.environ, {'MYCOBOT_BULLET_BATCH_RESET': '1'}), ArmEnv() as env:
            self.assertIs(env.batch_reset, True)

    def test_batch_errors_are_not_silently_retried_as_scalar(self):
        with patch.dict(os.environ, {'MYCOBOT_BULLET_BATCH_RESET': '1'}), ArmEnv() as env:
            with patch('simulation.diffusion.env.p.resetJointStatesMultiDof', side_effect=RuntimeError('batch failed')), \
                    patch('simulation.diffusion.env.p.resetJointState') as scalar:
                with self.assertRaisesRegex(RuntimeError, 'batch failed'):
                    env._set_joints(START)
                scalar.assert_not_called()

    def test_batch_partial_failure_restores_display_during_valid_many(self):
        q = START.copy(); q[0] += .1
        with patch.dict(os.environ, {'MYCOBOT_BULLET_BATCH_RESET': '1'}), ArmEnv() as env:
            env.set_obstacle(None); env.reset(START); env._valid_cache.clear()
            original = p.resetJointStatesMultiDof
            calls = []
            def fail_once(*args, **kwargs):
                calls.append(args)
                result = original(*args, **kwargs)
                if len(calls) == 1:
                    raise RuntimeError('post-reset failure')
                return result
            with patch('simulation.diffusion.env.p.resetJointStatesMultiDof', side_effect=fail_once):
                with self.assertRaisesRegex(RuntimeError, 'post-reset failure'):
                    env.valid_many([q])
            self.assertEqual(len(calls), 2)
            np.testing.assert_array_equal(env.current(), START)
            np.testing.assert_array_equal(env._display_q, START)
            self.assertNotIn(q.tobytes(), env._valid_cache)


if __name__ == '__main__':
    unittest.main()
