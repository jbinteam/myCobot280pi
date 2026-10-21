#!/usr/bin/env python3
"""Offline timings for the production tracking planner; never opens hardware.

Run with the project venv and OPENBLAS_NUM_THREADS=1. Raw latency measurements
are separate from cProfile runs; profiles explain hotspots, not speedup ratios.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=10)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    began = perf_counter()
    import numpy as np
    from simulation.kinematics import ARM, Arm
    from simulation.diffusion import classical_tracking, ik_seed
    from simulation.diffusion.env import ArmEnv
    from simulation import pose5, profile_tracking
    from simulation.profile_mouse_aim import profile_call
    imports_ms = (perf_counter()-began)*1000

    def summarize(values):
        return {key: float(value) for key, value in zip(
            ('min', 'median', 'p95', 'max'),
            (min(values), np.median(values), np.percentile(values, 95), max(values)))}

    start = np.array([0., -40., -130., 80., 0., 50.])
    cases = []
    for name, goal, distance in (
        ('hold_350', start, 350.),
        ('retarget_small_350', [10., -42., -125., 78., 2., 50.], 350.),
        ('retarget_large_350', [30., -50., -110., 70., 10., 50.], 350.),
        ('approach_step_330', start, 330.),
        ('retarget_small_150', [10., -42., -125., 78., 2., 50.], 150.),
    ):
        pose = ARM.transform(np.deg2rad(goal))
        cursor = pose[:3, 3]*1000 + (350. if name == 'approach_step_330' else distance)*pose[:3, 2]
        cases.append({'id': name, 'body': {
            'start_deg': start.tolist(), 'cursor_mm': cursor.tolist(),
            'tracking_region': {'mode': 'cone_disk', 'distance_mm': distance, 'half_angle_deg': 30.},
            'obstacle_mm': None, 'seed': 42, 'speed_rad_s': float(min(1., np.deg2rad(20.*70/30.))),
            'continuous_motion': True,
            'velocity_shape': {'kind': 'smooth_trapezoid', 'rise_percent': 5, 'fall_percent': 10}}})
    patches = [
        (classical_tracking, 'verified_status', 'model_scene_metadata'),
        (ArmEnv, 'inverse_kinematics', 'inverse_kinematics'),
        (ik_seed, 'propose', 'learned_seed_inference'),
        (pose5, 'solve', 'numerical_ik'),
        (classical_tracking, 'rrt_plan', 'direct_or_rrt_route'),
        (profile_tracking, 'prepare', 'velocity_profile'),
        (profile_tracking, 'prepare_tracking', 'velocity_profile_chunk'),
        (ArmEnv, 'valid', 'collision_single'),
        (ArmEnv, 'valid_many', 'collision_batch'),
        (ArmEnv, 'path_valid', 'path_validation'),
        (Arm, 'transform', 'forward_kinematics'),
        (Arm, 'spatial_jacobian', 'jacobian'),
        (classical_tracking, 'frames_for', 'trajectory_frames'),
        (classical_tracking, 'annotate_gaze', 'gaze_annotation'),
    ]
    report = {'scope': 'Offline production planner; no detector, ROS transport, or physical arm timings',
              'platform': platform.platform(), 'python': platform.python_version(),
              'openblas_num_threads': os.environ.get('OPENBLAS_NUM_THREADS'),
              'imports_ms': imports_ms, 'repeats': args.repeats, 'cases': [],
              'notes': ['Five selected diagnostic cases, not workspace-wide performance.',
                        'Wall times are unprofiled; cProfile stage times include profiling overhead.',
                        'Nested inclusive stage times must not be added.',
                        'No C++ implementation is measured; no C++ speedup is established.',
                        'Reference speed uses the production SDK 70 cap: min(1, radians(20*70/30)); not measured physical speed.']}
    try:
        began = perf_counter()
        cold = classical_tracking.plan(cases[1]['body'])
        report['cold_plan'] = {'wall_ms': (perf_counter()-began)*1000,
                               'status': cold['status'], 'playable': cold['playable']}
        print('Cold plan:', report['cold_plan'], flush=True)
        for case in cases:
            samples = []
            for _ in range(args.repeats):
                began = perf_counter()
                result = classical_tracking.plan(case['body'])
                wall_ms = (perf_counter()-began)*1000
                samples.append({'wall_ms': wall_ms, 'status': result['status'],
                    'playable': result['playable'], 'converged': result['converged'],
                    'route_status': result.get('planning_outcome'),
                    'frames': len(result['frames']),
                    **{k: result.get(k) for k in ('server_compute_ms', 'ik_elapsed_ms', 'path_elapsed_ms')}})
            row = {**case, 'samples': samples,
                   'timing_ms': {k: summarize([s[k] for s in samples if s[k] is not None])
                     for k in ('wall_ms', 'ik_elapsed_ms', 'path_elapsed_ms')
                     if any(s[k] is not None for s in samples)}}
            row['profile'] = profile_call(case['id'], lambda: classical_tracking.plan(case['body']), patches, args.output)
            report['cases'].append(row)
            (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
            print(case['id'], row['timing_ms'], 'statuses:', sorted({s['status'] for s in samples}), flush=True)
    finally:
        if classical_tracking._env is not None:
            classical_tracking._env.close()
    print(args.output/'report.json', flush=True)


if __name__ == '__main__':
    main()
