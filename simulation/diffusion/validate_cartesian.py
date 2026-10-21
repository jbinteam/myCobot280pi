"""Frozen 12-case validation-only Cartesian closed-loop diagnosis.

Original and augmented labels are read from the v3 validation cache. Preserved
source trajectories are replayed completely by the normal comparison gates.
No goal joint labels are supplied to inference and no test data are read.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import h5py
import numpy as np
from .cartesian_conditioning import CONDITIONING, RELATIVE_CONDITIONING, supports, supports_conditioning
from .cartesian_relative_cache import absolute_observations
from .common import file_sha256, write_json
from .compare_cartesian import continuous_suite
from .policy import DiffusionPolicy
from ..kinematics import ARM


def tasks(source, augmentation, cache):
    cache = Path(cache)
    metadata = json.loads((cache/'cache.json').read_text())
    if not supports_conditioning(metadata['identity']['conditioning']):
        raise ValueError('Expected compatible Cartesian cache')
    if (file_sha256(source) != metadata['identity']['source_sha256'] or
            file_sha256(augmentation) != metadata['identity']['augmentation_sha256']):
        raise ValueError('Validation sources do not match cache identity')
    obs = np.load(cache/'validation_obs.npy', mmap_mode='r')
    original_counts = {'blocked_original': 3, 'direct_original': 3}
    augmented_counts = {'static_absent': 2, 'moving_absent': 1, 'moving_block': 1,
                        'retarget_absent': 1, 'retarget_block': 1}
    result = []
    for source_index, path in enumerate((source, augmentation)):
        ranges = np.load(cache/f'validation_ranges_{source_index}.npy', mmap_mode='r')
        desired = original_counts if source_index == 0 else augmented_counts
        counts = Counter(); validation_index = 0
        with h5py.File(path, 'r') as data:
            for name, group in data['episodes'].items():
                if group.attrs['split'] != 'validation': continue
                start, count = map(int, ranges[validation_index]); current_index = validation_index; validation_index += 1
                record = json.loads(group.attrs['record_json'])
                kind = ('blocked_original' if record['direct_blocked'] else 'direct_original') if not source_index else record['kind']
                if counts[kind] >= desired.get(kind, 0): continue
                q = group['q_rad'][:]; observations = np.array(obs[start:start+count])
                if len(q) != count: raise ValueError('Source route and cache episode lengths differ')
                task_observations = (absolute_observations(observations)
                                     if metadata['identity']['conditioning'] == RELATIVE_CONDITIONING else observations)
                box = observations[0, 12:18].astype(float)*ARM.scale if observations[0, 18] == 1 else None
                # Source float64 geometry ensures replay uses the preserved exact scene.
                if box is not None: box = group['obstacle_m'][:]
                static = kind in ('blocked_original', 'direct_original', 'static_absent')
                labels = np.repeat(task_observations[-1][None], 2, axis=0) if static else task_observations
                times = np.array([0., .1]) if static else group['time_s'][:]
                result.append({'id': f'validation:{source_index}:{name}', 'kind': kind,
                               'validation_episode_index': current_index, 'q': q, 'box': box,
                               'blocked': record.get('direct_blocked', False), 'times': times,
                               'targets': labels[:, 6:9].astype(float)*ARM.scale,
                               'axes': labels[:, 9:12].astype(float),
                               'velocities': np.zeros((2, 6)) if static else observations[:, 19:25].astype(float),
                               'accelerations': np.zeros((2, 6)) if static else observations[:, 25:31].astype(float)*3,
                               'target_velocities': np.zeros((2, 3)) if static else observations[:, 31:34].astype(float)*.5})
                counts[kind] += 1
                if all(counts[kind] == target for kind, target in desired.items()): break
        if any(counts[kind] != target for kind, target in desired.items()):
            raise ValueError('Insufficient validation task coverage')
    return result, metadata


def run(checkpoint, source, augmentation, cache, output, max_decisions=80, seed=20261013):
    chosen, metadata = tasks(source, augmentation, cache)
    policy = DiffusionPolicy(checkpoint, 'cuda', seed)
    if not supports(policy): raise ValueError('Expected Cartesian checkpoint')
    if policy.training['dataset_sha256'] != metadata['identity']['dataset_sha256']:
        raise ValueError('Checkpoint/cache identity mismatch')
    result = continuous_suite(policy, chosen, max_decisions=max_decisions)
    report = {'kind': 'validation_cartesian_closed_loop_diagnosis', 'split': 'validation only',
              'checkpoint_sha256': policy.sha256, 'dataset_sha256': metadata['identity']['dataset_sha256'],
              'selected_training_step': policy.training['selected_step'], 'seed': seed,
              'conditioning': policy.conditioning,
              'configuration': {'max_decisions': max_decisions, 'samples': 4, 'inference_steps': 32, 'execute_seconds': .4,
                                'full_expert_route_collision_replay': True, 'IK_disabled': True,
                                'joint_goal_conditioning': False, 'static_neural_prefix_capture': True},
              'task_ids': [task['id'] for task in chosen], 'result': result}
    write_json(output, report); print(json.dumps(result['summary'], indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('simulation/artifacts/diffusion_cartesian_v3/model/diffusion_policy_6dof.pth'))
    parser.add_argument('--source', type=Path, default=Path('simulation/artifacts/diffusion_100k_v1/demonstrations.h5'))
    parser.add_argument('--augmentation', type=Path, default=Path('simulation/artifacts/diffusion_motion_v2/augmentation.h5'))
    parser.add_argument('--cache', type=Path, default=Path('simulation/artifacts/diffusion_cartesian_v3/cache'))
    parser.add_argument('--output', type=Path, default=Path('output/diffusion/cartesian_v3_closed_loop_validation.json'))
    parser.add_argument('--max-decisions', type=int, default=80)
    args = vars(parser.parse_args()); run(**args)
