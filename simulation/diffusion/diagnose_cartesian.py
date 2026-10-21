"""Validation-only terminal corrections, endpoint holds and C2 settling diagnostics.

Uses v3 Cartesian cache observations verbatim; goal joints never enter inference.
This diagnostic does not read test arrays and does not tune model parameters.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from ..kinematics import ARM
from ..continuous import prepare, sample_motion
from .cartesian_conditioning import CONDITIONING, RELATIVE_CONDITIONING, supports, supports_conditioning
from .cartesian_relative_cache import absolute_observations
from .common import denormalize, file_sha256, robot_schema, write_json
from .env import ArmEnv
from .policy import DiffusionPolicy, checked_chunk
from .service import cartesian_score, errors


def validation_cases(cache, episodes=32, seed=20261012):
    if episodes < 2 or episodes % 2:
        raise ValueError('Choose a positive even number of balanced validation episodes')
    cache = Path(cache); metadata = json.loads((cache / 'cache.json').read_text())
    if not supports_conditioning(metadata['identity']['conditioning']) or metadata['identity']['robot'] != robot_schema():
        raise ValueError('Expected compatible Cartesian cache')
    # Deliberately no train or test arrays are opened.
    observations = np.load(cache / 'validation_obs.npy', mmap_mode='r')
    joints = np.load(cache / 'validation_q.npy', mmap_mode='r')
    rng = np.random.default_rng(seed); cases = []
    for source in (0, 1):
        ranges = np.load(cache / f'validation_ranges_{source}.npy', mmap_mode='r')
        if len(ranges) < episodes // 2:
            raise ValueError('Too few validation episodes')
        for chosen in rng.choice(len(ranges), episodes // 2, replace=False):
            start, count = map(int, ranges[chosen])
            for kind, local in (('near_terminal', max(0, count-5)), ('exact_hold', count-1)):
                index = start+local; obs = np.array(observations[index], copy=True)
                labels = (absolute_observations(obs[None])[0]
                          if metadata['identity']['conditioning'] == RELATIVE_CONDITIONING else obs)
                q = np.asarray(denormalize(joints[index]), float)
                q = ARM.clip(q)  # Cache storage float32 can round inward bounds outward by <1e-7.
                box = obs[12:18].astype(float)*ARM.scale if obs[18] == 1 else None
                cases.append({'source': 'original' if source == 0 else 'augmentation',
                              'validation_episode_index': int(chosen), 'kind': kind,
                              'frame_index': local, 'frame_count': count,
                              'q': q, 'obs': obs, 'box': box,
                              'target': labels[6:9].astype(float)*ARM.scale, 'axis': labels[9:12].astype(float),
                              'v': obs[19:25].astype(float), 'a': obs[25:31].astype(float)*3})
    return cases, metadata


def evaluate_case(env, policy, case, samples=4, steps=32):
    q = case['q']; target = case['target']; axis = case['axis']
    env.set_obstacle(case['box']); env.reset(q)
    initial = errors(q, target, axis); rejected = Counter(); candidates = []
    raw = policy.sample(case['obs'][None], samples=samples, inference_steps=steps)[0]
    for index, proposed in enumerate(raw):
        chunk, reason = checked_chunk(env, q, proposed)
        if chunk is None:
            rejected[reason] += 1; continue
        error = [errors(point, target, axis) for point in chunk]
        score = cartesian_score(chunk[-1], target, axis)+.25*cartesian_score(chunk[1], target, axis)
        record = {'candidate': index, 'score': float(score), 'endpoint': error[-1],
                  'best_future': min(error, key=lambda item: item['error_mm']/1000/ARM.scale+.35*np.deg2rad(item['orientation_error_deg'])),
                  'strict_hit_any_future': any(item['error_mm'] <= 1 and item['orientation_error_deg'] <= 1 for item in error),
                  'first_400ms_reference': error[3],
                  'max_reference_step_rad': float(np.linalg.norm(np.diff(np.vstack((q, chunk)), axis=0), axis=1).max()),
                  'max_hold_joint_drift_rad': float(np.linalg.norm(chunk-q, axis=1).max()),
                  'max_hold_TCP_drift_mm': float(max(np.linalg.norm(ARM.fk(point)-ARM.fk(q))*1000 for point in chunk))}
        try:
            _, _, curve = prepare(env, np.vstack((q, chunk)), case['v'], case['a'])
            advance = min(.4, curve['duration_s'])
            pose, velocity, acceleration = sample_motion(curve, advance)
            endpoint, final_velocity, _ = sample_motion(curve, curve['duration_s'])
            grid = np.linspace(0, curve['duration_s'], max(2, int(np.ceil(curve['duration_s']/.02))+1))
            speeds = [np.linalg.norm(sample_motion(curve, time)[1]) for time in grid]
            record['continuous'] = {'duration_s': float(curve['duration_s']), 'advance_s': float(advance),
                                    'after_400ms': errors(pose, target, axis),
                                    'speed_after_400ms_rad_s': float(np.linalg.norm(velocity)),
                                    'acceleration_after_400ms_rad_s2': float(np.linalg.norm(acceleration)),
                                    'endpoint': errors(endpoint, target, axis),
                                    'endpoint_speed_rad_s': float(np.linalg.norm(final_velocity)),
                                    'max_sampled_speed_rad_s': float(max(speeds))}
        except ValueError:
            rejected['continuous_join'] += 1; record['continuous'] = None
        candidates.append(record)
    # Same Cartesian ranking and continuous-join acceptance as runtime.
    available = sorted((item for item in candidates if item['continuous'] is not None), key=lambda item: item['score'])
    selected = available[0] if available else None
    return {'source': case['source'], 'validation_episode_index': case['validation_episode_index'],
            'kind': case['kind'], 'frame_index': case['frame_index'], 'frame_count': case['frame_count'],
            'obstacle_present': case['box'] is not None, 'initial': initial,
            'initial_speed_rad_s': float(np.linalg.norm(case['v'])),
            'target_speed_m_s': float(np.linalg.norm(case['obs'][31:34]*.5)),
            'safe_candidates': len(candidates), 'joinable_candidates': len(available),
            'rejections': dict(rejected), 'selected': selected, 'candidates': candidates}


def summarize(results):
    summary = {}
    for kind in ('near_terminal', 'exact_hold'):
        rows = [item for item in results if item['kind'] == kind]
        chosen = [item for item in rows if item['selected'] is not None]
        def median(values): return None if not values else float(np.median(values))
        summary[kind] = {'cases': len(rows), 'safe_cases': sum(item['safe_candidates'] > 0 for item in rows),
                         'joinable_cases': len(chosen),
                         'strict_hit_any_safe_future': sum(any(c['strict_hit_any_future'] for c in item['candidates']) for item in rows),
                         'selected_strict_hit_any_future': sum(item['selected']['strict_hit_any_future'] for item in chosen),
                         'selected_strict_endpoint': sum(item['selected']['endpoint']['error_mm'] <= 1 and
                                                        item['selected']['endpoint']['orientation_error_deg'] <= 1 for item in chosen),
                         'selected_curves_finish_within400ms': sum(item['selected']['continuous']['duration_s'] <= .4 for item in chosen),
                         'median_initial_mm': median([item['initial']['error_mm'] for item in rows]),
                         'median_best_safe_future_mm': median([min(c['best_future']['error_mm'] for c in item['candidates'])
                                                              for item in rows if item['candidates']]),
                         'median_selected_endpoint_mm': median([item['selected']['endpoint']['error_mm'] for item in chosen]),
                         'median_selected_endpoint_axis_deg': median([item['selected']['endpoint']['orientation_error_deg'] for item in chosen]),
                         'median_selected_after400ms_mm': median([item['selected']['continuous']['after_400ms']['error_mm'] for item in chosen]),
                         'median_selected_after400ms_speed_rad_s': median([item['selected']['continuous']['speed_after_400ms_rad_s'] for item in chosen]),
                         'median_selected_curve_duration_s': median([item['selected']['continuous']['duration_s'] for item in chosen]),
                         'median_selected_hold_max_TCP_drift_mm': median([item['selected']['max_hold_TCP_drift_mm'] for item in chosen])}
    return summary


def diagnose(checkpoint, cache, output, episodes=32, samples=4, steps=32, seed=20261012):
    cases, metadata = validation_cases(cache, episodes, seed)
    policy = DiffusionPolicy(checkpoint, device='cuda', seed=seed)
    if not supports(policy): raise ValueError('Expected Cartesian checkpoint')
    if policy.training['dataset_sha256'] != metadata['identity']['dataset_sha256']:
        raise ValueError('Diagnostic cache does not match checkpoint dataset identity')
    results = []; started = perf_counter()
    with ArmEnv() as env:
        for number, case in enumerate(cases):
            results.append(evaluate_case(env, policy, case, samples, steps))
            if number % 8 == 0: print(f'Cartesian terminal validation {number+1}/{len(cases)}', flush=True)
    report = {'purpose': 'Validation-only terminal correction, exact holds and single C2 braking-curve behavior',
              'checkpoint': str(checkpoint), 'checkpoint_sha256': file_sha256(checkpoint),
              'selected_training_step': policy.training['selected_step'], 'conditioning': policy.conditioning,
              'dataset_sha256': metadata['identity']['dataset_sha256'], 'split': 'validation only',
              'goal_joint_input': False, 'IK_calls': 0, 'seed': seed, 'samples': samples, 'inference_steps': steps,
              'episodes_per_source': episodes//2, 'seconds': perf_counter()-started,
              'limitations': 'Single proposed horizon and C2 join; does not measure repeated-controller convergence. Exact-hold drift is relative to the initial endpoint.',
              'summary': summarize(results), 'results': results}
    write_json(output, report); print(json.dumps(report['summary'], indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('simulation/artifacts/diffusion_cartesian_v3/model/diffusion_policy_6dof.pth'))
    parser.add_argument('--cache', type=Path, default=Path('simulation/artifacts/diffusion_cartesian_v3/cache'))
    parser.add_argument('--output', type=Path, default=Path('output/diffusion/cartesian_v3_terminal_validation.json'))
    parser.add_argument('--episodes', type=int, default=32)
    args = parser.parse_args(); diagnose(args.checkpoint, args.cache, args.output, args.episodes)
