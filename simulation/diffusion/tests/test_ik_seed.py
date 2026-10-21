"""Learned starts cannot bypass endpoint gates or increase solver budgets."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from simulation.kinematics import ARM
from simulation.diffusion import ik_seed
from simulation.diffusion.env import ArmEnv
from simulation.diffusion.endpoint_ik import IKSearchExhausted


class IKSeedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = {'configuration': ik_seed.configuration(),
                        'training': {'kind': 'smoke', 'completed': True, 'completed_steps': 1},
                        'layers': [{'weight': np.zeros((out, inp)).tolist(),
                                    'bias': np.zeros(out).tolist()}
                                   for inp, out in [(12, 256), (256, 256), (256, 48)]]}

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'model.json'

    def write(self, artifact=None):
        self.path.write_text(json.dumps(self.artifact if artifact is None else artifact))
        return self.path

    def scene(self, valid=lambda q: True):
        env = ArmEnv.__new__(ArmEnv)
        env.valid = valid
        return env

    @staticmethod
    def result(q, converged=False):
        return {'frames': [{'q': np.rad2deg(q).tolist()}], 'converged': converged}

    def test_feature_units_and_full_range_absolute_outputs(self):
        data = copy.deepcopy(self.artifact)
        data['layers'][-1]['bias'] = np.tile([4., -4., 4., -4., 4., -4.], 8).tolist()
        model = ik_seed.load(self.write(data))
        current = ARM.upper.copy()
        target = np.array([.1, .2, .3])
        feature = ik_seed.features(current, target, [0, 0, 5])
        np.testing.assert_allclose(feature[:3], target / ARM.scale)
        np.testing.assert_array_equal(feature[3:6], [0, 0, 1])
        np.testing.assert_allclose(feature[6:], np.ones(6))
        expected = ARM.center + ARM.radius * np.tanh([4., -4., 4., -4., 4., -4.])
        guessed = model.predict(current, target, [0, 0, 1])
        np.testing.assert_allclose(guessed, np.tile(expected, (8, 1)))
        self.assertTrue(np.all(guessed <= ARM.upper) and np.all(guessed >= ARM.lower))
        self.assertLess(guessed[0, 1], ARM.center[1])  # full-range even from upper current

    def test_artifact_metadata_training_shapes_and_nonfinite_rejected(self):
        mutations = [
            lambda d: d['configuration']['fk'].update(urdf_sha256='other robot'),
            lambda d: d['configuration']['limits_rad']['min'].__setitem__(0, -99.),
            lambda d: d['configuration']['outputs'].update(parameterization='current + delta'),
            lambda d: d['training'].update(completed=False),
            lambda d: d['training'].update(completed_steps=True),
            lambda d: d['layers'][0]['weight'].__setitem__(0, [0.]),
            lambda d: d['layers'][2]['bias'].__setitem__(0, float('nan')),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                data = copy.deepcopy(self.artifact)
                mutation(data)
                guesses, status, _ = ik_seed.propose(ARM.center, [0, 0, .2], [0, 0, 1], self.write(data))
                self.assertEqual(status['status'], 'incompatible')
                self.assertEqual(guesses, [])
        for content in ['{', '[]', '{"configuration": null}', '{"layers": 0}']:
            self.path.write_text(content)
            self.assertEqual(ik_seed.checkpoint_status(self.path)['status'], 'incompatible')

    def test_missing_and_environment_path(self):
        guesses, status, _ = ik_seed.propose(ARM.center, [0, 0, .2], [0, 0, 1], self.path)
        self.assertEqual(guesses, [])
        self.assertEqual(status['status'], 'missing')
        self.write()
        with patch.dict('os.environ', {'MYCOBOT_IK_SEED_MODEL': str(self.path)}):
            self.assertEqual(ik_seed.checkpoint_status()['status'], 'ready')

    def test_current_success_does_not_infer_even_when_enabled(self):
        q = ARM.center; target = ARM.transform(q); env = self.scene()
        with patch('simulation.diffusion.env.solve', return_value=self.result(q, True)), \
             patch('simulation.diffusion.endpoint_ik.propose', side_effect=AssertionError('unneeded inference')):
            env.inverse_kinematics(target[:3, 3], target[:3, 2], q, robust=True, learned=True)
        self.assertEqual(env.ik_diagnostics['legacy_attempts'], 1)
        self.assertEqual(env.ik_diagnostics['learned_refine_attempts'], 0)

    def test_learned_start_is_refined_and_same_budget_replaces_random(self):
        q = ARM.center; target = ARM.transform(q); env = self.scene()
        guess = (ARM.center + ARM.upper) / 2
        guesses = np.tile(guess, (8, 1)); starts = []
        def solve(start, *args, **kwargs):
            starts.append(start.copy())
            return self.result(q, len(starts) == 2)
        with patch('simulation.diffusion.env.solve', side_effect=solve), \
             patch('simulation.diffusion.endpoint_ik.propose', return_value=(guesses, {'status': 'ready'}, 2.)):
            actual = env.inverse_kinematics(target[:3, 3], target[:3, 2], q, attempts=3, learned=True)
        np.testing.assert_allclose(actual, q)
        np.testing.assert_array_equal(starts[1], guess)
        self.assertEqual(env.ik_diagnostics['legacy_attempts'], 2)
        self.assertTrue(env.ik_diagnostics['learned_refine_success'])
        self.assertEqual(env.ik_diagnostics['learned_refine_attempts'], 1)

    def test_exhaustion_still_has_exact_legacy_and_fallback_budgets(self):
        q = ARM.center; env = self.scene(); starts = []; fallback = []
        guesses = np.tile((ARM.center + ARM.upper) / 2, (8, 1))
        def failed(start, *args, **kwargs):
            starts.append(start.copy())
            return self.result(q)
        def ls(fun, start, **kwargs):
            fallback.append(start.copy())
            return SimpleNamespace(x=q, nfev=1)
        with patch('simulation.diffusion.env.solve', side_effect=failed), \
             patch('simulation.diffusion.endpoint_ik.least_squares', side_effect=ls), \
             patch('simulation.diffusion.endpoint_ik.propose', return_value=(guesses, {'status': 'ready'}, 1.)):
            with self.assertRaises(IKSearchExhausted):
                env.inverse_kinematics([3, 0, 0], [0, 0, 1], q, attempts=4, robust=True, learned=True)
        self.assertEqual(len(starts), 4)
        self.assertEqual(len(fallback), 32)
        self.assertEqual(env.ik_diagnostics['learned_refine_attempts'], 11)
        np.testing.assert_array_equal(fallback[1], guesses[0])

    def test_missing_model_preserves_numerical_starts_and_default_is_off(self):
        q = ARM.center; all_starts = []
        for learned in (False, True):
            starts = []; env = self.scene()
            def failed(start, *args, **kwargs):
                starts.append(start.copy())
                return self.result(q)
            with patch.dict('os.environ', {'MYCOBOT_IK_SEED': '0', 'MYCOBOT_IK_SEED_MODEL': str(self.path)}), \
                 patch('simulation.diffusion.env.solve', side_effect=failed):
                with self.assertRaises(IKSearchExhausted):
                    env.inverse_kinematics([3, 0, 0], current=q, attempts=4, learned=True if learned else None)
            self.assertEqual(env.ik_diagnostics['learned_status'], 'missing' if learned else 'disabled')
            all_starts.append(starts)
        np.testing.assert_array_equal(all_starts[0], all_starts[1])

    def test_learned_refinement_cannot_bypass_collision_position_or_axis(self):
        q = ARM.center; target = ARM.transform(q)
        for case in ('collision', 'position', 'antipodal', 'limits'):
            env = self.scene(lambda q: case != 'collision')
            position = target[:3, 3] + ([.01, 0, 0] if case == 'position' else np.zeros(3))
            axis = -target[:3, 2] if case == 'antipodal' else target[:3, 2]
            output = ARM.upper + .01 if case == 'limits' else q
            with self.subTest(case=case), \
                 patch('simulation.diffusion.env.solve', return_value=self.result(output, True)), \
                 patch('simulation.diffusion.endpoint_ik.least_squares', return_value=SimpleNamespace(x=output, nfev=1)), \
                 patch('simulation.diffusion.endpoint_ik.propose', return_value=(np.tile(q, (8, 1)), {'status': 'ready'}, 1.)):
                with self.assertRaises(IKSearchExhausted):
                    env.inverse_kinematics(position, axis, q, attempts=2, robust=True, learned=True)
                self.assertFalse(env.ik_diagnostics['learned_refine_success'])

    def test_exact_raw_network_guess_is_never_accepted_directly(self):
        witness = np.deg2rad([23, -42, -78, 45, 36, -22])
        target = ARM.transform(witness); q = ARM.center; env = self.scene()
        with patch('simulation.diffusion.env.solve', return_value=self.result(q, True)), \
             patch('simulation.diffusion.endpoint_ik.least_squares', return_value=SimpleNamespace(x=q, nfev=1)), \
             patch('simulation.diffusion.endpoint_ik.propose', return_value=(np.tile(witness, (8, 1)), {'status': 'ready'}, 1.)):
            with self.assertRaises(IKSearchExhausted):
                env.inverse_kinematics(target[:3, 3], target[:3, 2], q, attempts=2, robust=True, learned=True)
        self.assertEqual(env.ik_diagnostics['learned_refine_attempts'], 9)

    def test_unordered_heads_rank_useful_guess_within_small_budget(self):
        witness=np.deg2rad([23,-42,-78,45,36,-22]);target=ARM.transform(witness)
        current=ARM.center;env=self.scene();starts=[]
        guesses=np.tile(current,(8,1));guesses[-1]=witness
        def refine(start,*args,**kwargs):
            starts.append(start.copy())
            matched=np.allclose(start,witness)
            return self.result(witness if matched else current,matched)
        with patch('simulation.diffusion.env.solve',side_effect=refine), \
             patch('simulation.diffusion.endpoint_ik.propose',return_value=(guesses,{'status':'ready'},1.)):
            goal=env.inverse_kinematics(target[:3,3],target[:3,2],current,attempts=2,learned=True)
        np.testing.assert_allclose(goal,witness);np.testing.assert_array_equal(starts[1],witness)
        self.assertEqual(env.ik_diagnostics['legacy_attempts'],2)

if __name__ == '__main__':
    unittest.main()
