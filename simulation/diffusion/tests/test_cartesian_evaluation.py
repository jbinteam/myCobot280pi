"""Evaluation labels and IK independence must survive accidental regressions."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from simulation.diffusion.cartesian_conditioning import CARTESIAN_CONDITIONING, RELATIVE_CONDITIONING, supports_conditioning
from simulation.diffusion.common import file_sha256, write_json
from simulation.diffusion.compare_cartesian import _cartesian_task, forbid_ik, policy_task, task_index
from simulation.diffusion.deploy_cartesian import prepare_manifest
from simulation.diffusion.service import condition
from simulation.kinematics import ARM


class CartesianEvaluationTest(unittest.TestCase):
    def test_cartesian_task_hides_goal_joint_labels_and_encoder_ignores_goal_argument(self):
        q = ARM.center.copy(); goal = ARM.clip(q + .1)
        task = _cartesian_task({'id': 'test', 'kind': 'static_absent', 'q': np.vstack((q, goal)),
                                'goal': goal, 'box': None})
        transformed = ARM.transform(goal)
        np.testing.assert_allclose(task['targets'][0], transformed[:3, 3])
        np.testing.assert_allclose(task['axes'][0], transformed[:3, 2])
        stripped = policy_task(task, True)
        self.assertNotIn('goal', stripped); self.assertNotIn('goals', stripped)
        for conditioning in (CARTESIAN_CONDITIONING, RELATIVE_CONDITIONING):
            with self.subTest(schema=conditioning['schema']):
                policy = SimpleNamespace(conditioning=conditioning, observation_dim=34)
                expected = condition(policy, q, None, task['targets'][0], task['axes'][0], None)
                poisoned = condition(policy, q, np.full(6, np.nan), task['targets'][0], task['axes'][0], None)
                np.testing.assert_array_equal(expected, poisoned)
                self.assertEqual(expected.shape, (34,))
                self.assertTrue(supports_conditioning(conditioning))
        spoofed = {**RELATIVE_CONDITIONING, 'schema': 3}
        self.assertFalse(supports_conditioning(spoofed))

    def test_ik_disabled_stub_and_time_index_clamping(self):
        with self.assertRaisesRegex(AssertionError, 'never call inverse kinematics'): forbid_ik()
        task = {'times': np.array([0., .1, .4]), 'targets': np.zeros((3, 3))}
        self.assertEqual(task_index(task, -.1), 0)
        self.assertEqual(task_index(task, .3), 1)
        self.assertEqual(task_index(task, 100.), 2)

    def test_candidate_manifest_records_failed_checks_and_preserves_pointer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); folder = root / 'candidate'; (folder / 'model').mkdir(parents=True)
            checkpoint = folder / 'model/diffusion_policy_6dof.pth'; checkpoint.write_bytes(b'fixture')
            baseline = root / 'baseline.json'; active = root / 'diffusion_active.json'
            active.write_text('unchanged pointer')
            write_json(baseline, {'cases': [{'goal_deg': [0.] * 6, 'target_mm': [1., 2., 3.]}],
                                  'checkpoint_sha256': 'old', 'environment': {}, 'schema': 1})
            identity = {'source_sha256': 'source', 'augmentation_sha256': 'augment',
                        'conditioning': CARTESIAN_CONDITIONING}
            training = {'kind': 'full', 'completed_steps': 100, 'goal_joint_input': False,
                        'data_identity': identity, 'dataset_sha256': 'dataset'}
            write_json(folder / 'model/training.json', training)
            evaluation = {'ik_disabled': True, 'goal_joint_labels_removed': True,
                          'summary': {'successes': 0, 'episodes': 2}}
            report = {'kind': 'paired_cartesian_c2_comparison', **{k: v for k, v in identity.items() if k != 'conditioning'},
                      'results': {'candidate': {'checkpoint_sha256': file_sha256(checkpoint),
                                               'fixed': evaluation, 'motion': evaluation},
                                  'baseline': {'checkpoint_sha256': 'old'}},
                      'configuration': {'joint_endpoint_capture': False},
                      'recommended_for_deployment': False, 'deployment_checks': {'usable': False}}
            comparison = root / 'comparison.json'; write_json(comparison, report)
            audit = root / 'audit.json'
            write_json(audit, {'passed': True, **identity, 'dataset_sha256': 'dataset',
                               'episodes': {'train': 2, 'validation': 1, 'test': 1}})
            fake = SimpleNamespace(conditioning=CARTESIAN_CONDITIONING, training=training,
                                   sha256=file_sha256(checkpoint))
            with patch('simulation.diffusion.policy.DiffusionPolicy', return_value=fake):
                result = prepare_manifest(folder, comparison, baseline, audit)
            self.assertFalse(result['activated']); self.assertFalse(result['recommended_for_deployment'])
            self.assertEqual(active.read_text(), 'unchanged pointer')
            manifest = json.loads((folder / 'webui.json').read_text())
            self.assertNotIn('goal_deg', manifest['cases'][0])
            self.assertFalse(manifest['inverse_kinematics_required'])
            self.assertEqual(json.loads(baseline.read_text())['checkpoint_sha256'], 'old')
            # Relative caches nest the preserved v3 source identity and have
            # their own dataset digest despite using the same H5 trajectories.
            training['data_identity'] = {'conditioning': RELATIVE_CONDITIONING,
                                         'source_cache_identity': identity}
            training['dataset_sha256'] = 'relative-dataset'
            write_json(folder / 'model/training.json', training)
            write_json(audit, {'passed': True, **identity, 'conditioning': RELATIVE_CONDITIONING,
                               'dataset_sha256': 'relative-dataset',
                               'episodes': {'train': 2, 'validation': 1, 'test': 1}})
            report['results']['candidate']['conditioning'] = RELATIVE_CONDITIONING
            write_json(comparison, report)
            fake.conditioning = RELATIVE_CONDITIONING
            with patch('simulation.diffusion.policy.DiffusionPolicy', return_value=fake):
                relative = prepare_manifest(folder, comparison, baseline, audit)
            self.assertFalse(relative['activated']); self.assertEqual(active.read_text(), 'unchanged pointer')
            manifest = json.loads((folder / 'webui.json').read_text())
            self.assertEqual(manifest['conditioning'], RELATIVE_CONDITIONING)
            self.assertEqual(manifest['dataset_sha256'], 'relative-dataset')
            report['results']['candidate']['checkpoint_sha256'] = 'changed'
            write_json(comparison, report)
            with self.assertRaisesRegex(ValueError, 'provenance differs'):
                prepare_manifest(folder, comparison, baseline, audit)


if __name__ == '__main__': unittest.main()
