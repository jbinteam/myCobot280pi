"""Cartesian input semantics, isolated cache identity and v2 warm-start contracts."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import h5py
import numpy as np
import torch
from simulation.kinematics import ARM
from simulation.diffusion.common import file_sha256, normalize, robot_schema
from simulation.diffusion.cartesian_conditioning import CONDITIONING, observe, supports
from simulation.diffusion.motion_conditioning import CONDITIONING as MOTION_CONDITIONING
from simulation.diffusion.model import ConditionalUNet1D
from simulation.diffusion.train_cartesian import build_cache, cartesian_labels, warm_start_state


class CartesianTrainingTests(unittest.TestCase):
    def test_cartesian_channels_motion_and_invalid_input(self):
        q = ARM.center.copy(); target = np.array([.12, -.06, .3])
        value = observe(q, target, [0, 0, -4], velocity=[.5, 0, 0, 0, 0, 0],
                        acceleration=[1.5, 0, 0, 0, 0, 0], target_velocity=[0, 2, 0])
        self.assertEqual(value.shape, (34,)); self.assertEqual(value.dtype, np.float32)
        np.testing.assert_array_equal(value[:6], np.zeros(6))
        np.testing.assert_allclose(value[6:9], target / ARM.scale)
        np.testing.assert_array_equal(value[9:12], [0, 0, -1])
        np.testing.assert_array_equal(value[12:19], np.zeros(7))
        self.assertEqual(value[19], .5); self.assertEqual(value[25], .5); self.assertEqual(value[32], 1.)
        self.assertFalse(supports(type('V2', (), {'observation_dim': 34, 'conditioning': MOTION_CONDITIONING})()))
        self.assertTrue(supports(type('V3', (), {'conditioning': CONDITIONING})()))
        for kwargs in ({'tool_axis': [0, 0, 0]}, {'target_position': [np.nan, 0, 0]},
                       {'obstacle': [0, 0, 0, .1, 0, .1]}, {'velocity': [2, 0, 0, 0, 0, 0]}):
            args = {'q': q, 'target_position': target, 'tool_axis': [0, 0, 1]}; args.update(kwargs)
            with self.assertRaises(ValueError): observe(**args)

    def test_fk_batch_deduplicates_goal_labels_and_matches_urdf(self):
        a = ARM.center.copy(); b = ARM.center.copy(); b[0] += .15
        goals = np.array([a, a, b, a, b])
        with patch.object(ARM, 'torch_transform', wraps=ARM.torch_transform) as transform:
            labels = cartesian_labels(goals)
            self.assertEqual(transform.call_args[0][0].shape, (2, 6))
        for label, goal in zip(labels, goals):
            expected = ARM.transform(goal)
            np.testing.assert_allclose(label, np.r_[expected[:3, 3] / ARM.scale, expected[:3, 2]], atol=1e-7)

    def test_warm_start_copies_shared_columns_and_zeros_obsolete_goal_weights(self):
        model = ConditionalUNet1D(8, 34)
        checkpoint = {'configuration': {'robot': robot_schema(), 'conditioning': MOTION_CONDITIONING},
                      'state_dict': model.state_dict()}
        source = checkpoint['state_dict']['observation.0.weight'].clone()
        state = warm_start_state(checkpoint)
        torch.testing.assert_close(state['observation.0.weight'][:, :6], source[:, :6])
        torch.testing.assert_close(state['observation.0.weight'][:, 12:], source[:, 12:])
        self.assertEqual(torch.count_nonzero(state['observation.0.weight'][:, 6:12]).item(), 0)
        torch.testing.assert_close(checkpoint['state_dict']['observation.0.weight'], source)
        checkpoint['configuration']['conditioning'] = CONDITIONING
        with self.assertRaises(ValueError): warm_start_state(checkpoint)

    def make_data(self, folder):
        source = folder / 'original.h5'; augmented = folder / 'augmented.h5'
        q = np.repeat(ARM.center[None], 3, axis=0)
        goal = ARM.center.copy(); goal[0] += .2
        for index, path in enumerate((source, augmented)):
            with h5py.File(path, 'w') as data:
                root = data.create_group('episodes')
                for number, split in enumerate(('train', 'validation', 'test')):
                    group = root.create_group(f'{number:06d}')
                    group.attrs['split'] = split
                    group.create_dataset('q_rad', data=q)
                    group.create_dataset('time_s', data=[0., .1, .2])
                    group.create_dataset('obstacle_m', data=[0., 0., .1, .02, .02, .02])
                    if index:
                        group.attrs['parent_episode'] = f'{number:06d}'; group.attrs['obstacle_present'] = False
                        group.create_dataset('goal_rad', data=np.repeat(goal[None], 3, axis=0))
                        group.create_dataset('velocity_rad_s', data=np.zeros_like(q))
                        group.create_dataset('acceleration_rad_s2', data=np.zeros_like(q))
                        group.create_dataset('target_velocity_m_s', data=np.zeros((3, 3)))
                    else: group.create_dataset('q_goal_rad', data=goal)
                metadata = {'completed_episodes': 3, 'environment': {'robot': robot_schema()}}
                if index: metadata['source_sha256'] = file_sha256(source)
                data.attrs['metadata_json'] = json.dumps(metadata)
        return source, augmented, goal

    def test_cache_cartesian_labels_masks_episode_ends_and_schema_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); source, augmented, goal = self.make_data(folder)
            saved = build_cache(source, augmented, folder / 'cache', chunk_frames=4)
            self.assertEqual(saved['identity']['conditioning'], CONDITIONING)
            obs = np.load(folder / 'cache/train_obs.npy')
            pose = ARM.transform(goal)
            np.testing.assert_allclose(obs[:, 6:12], np.broadcast_to(np.r_[pose[:3, 3] / ARM.scale, pose[:3, 2]], (6, 6)), atol=1e-7)
            np.testing.assert_array_equal(obs[3:, 12:19], np.zeros((3, 7)))
            np.testing.assert_array_equal(np.load(folder / 'cache/train_ends.npy'), [2, 2, 2, 5, 5, 5])
            np.testing.assert_array_equal(np.load(folder / 'cache/train_ranges_0.npy'), [[0, 3]])
            np.testing.assert_array_equal(np.load(folder / 'cache/train_ranges_1.npy'), [[3, 3]])
            self.assertEqual(build_cache(source, augmented, folder / 'cache'), saved)
            with h5py.File(augmented, 'r+') as data: data['episodes/000000'].attrs['split'] = 'test'
            with self.assertRaisesRegex(ValueError, 'sources/schema'): build_cache(source, augmented, folder / 'cache')
            with self.assertRaisesRegex(ValueError, 'split leakage'): build_cache(source, augmented, folder / 'new-cache')


if __name__ == '__main__': unittest.main()
