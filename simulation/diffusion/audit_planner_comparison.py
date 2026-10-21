"""Independent saved-trajectory replay for paired planner benchmarks."""
import argparse
import json
from pathlib import Path
import numpy as np
from ..kinematics import ARM
from ..pose5 import axis_error
from .env import ArmEnv
from .benchmark_ik_refinement import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=128)
    parser.add_argument('--case-count', type=int, help='Audit only the prefix IDs 0..count-1')
    args = parser.parse_args()
    experiment = json.loads((args.output/'experiment.json').read_text())
    inputs = args.output/'inputs.npz'
    if sha256(inputs) != experiment['inputs']['inputs_sha256']:
        raise ValueError('Benchmark inputs changed')
    with np.load(inputs, allow_pickle=False) as archive:
        data = {k: archive[k].copy() for k in archive.files}
    rows = [json.loads(line) for line in (args.output/'rows.jsonl').read_text().splitlines()]
    if args.case_count is not None:
        rows = [r for r in rows if r['case_id'] < args.case_count]
        if set(r['case_id'] for r in rows) != set(range(args.case_count)):
            raise ValueError('Requested case prefix is incomplete')
    if len(rows) != len({r['case_id'] for r in rows}):
        raise ValueError('Duplicate saved case results')
    ordered = sorted(rows, key=lambda r: r['case_id'])
    rng = np.random.default_rng(202610083)
    selected = sorted(rng.choice(len(ordered), min(args.limit, len(ordered)), replace=False).tolist())
    # Include detours and failures in addition to the random audit sample.
    selected = sorted(set(selected) | {i for i, r in enumerate(ordered)
        if r['classical']['planner_status'] != 'direct_optimal'
        or not r['classical']['success'] or not r['diffusion']['success']})
    audits = []
    with ArmEnv() as env:
        env.set_obstacle(None)
        for i in selected:
            row = ordered[i]
            case_id = row['case_id']
            current = data['q_current_rad'][case_id]
            goal = data['q_goal_rad'][case_id]
            target = data['target_xyz_m'][case_id]
            axis = data['target_axis_base'][case_id]
            if not env.valid(current) or not env.valid(goal):
                raise AssertionError('Saved input current/goal fails independent scene')
            item = {'case_id': case_id}
            for mode in ('classical', 'diffusion'):
                result = row[mode]
                path = args.output/result['trajectory_file']
                if sha256(path) != result['trajectory_sha256']:
                    raise AssertionError('Saved trajectory changed')
                with np.load(path, allow_pickle=False) as archive:
                    raw = archive['raw_q_rad']
                    if not np.array_equal(raw[0], current):
                        raise AssertionError('Saved geometric prefix has a different start')
                    states = archive['q_rad'] if result['success'] else raw
                    times = archive['time_s'] if result['success'] else None
                    if (not np.isfinite(states).all() or np.any(states < ARM.lower)
                            or np.any(states > ARM.upper)):
                        raise AssertionError('Saved trajectory violates joint bounds')
                    env._valid_cache.clear()
                    env.reset(current)
                    if not env.path_valid(states, .04):
                        raise AssertionError('Independent collision replay failed')
                    transform = ARM.transform(states[-1])
                    position = float(np.linalg.norm(transform[:3, 3]-target)*1000)
                    direction = float(axis_error(transform[:3, 2], axis)[2])
                    if (abs(position-result['position_error_mm']) > 1e-8
                            or abs(direction-result['direction_error_deg']) > 1e-8):
                        raise AssertionError('Saved endpoint errors do not match recomputed FK')
                    checks = {'hash_verified': True, 'independent_collision_replay': True,
                              'states_replayed': len(states), 'success': result['success']}
                    if result['success']:
                        if position > 1. or direction > 1. or not np.array_equal(states[0], current):
                            raise AssertionError('Saved successful endpoint fails shared pose gates')
                        intervals = np.diff(times)
                        if not np.isfinite(times).all() or np.any(intervals <= 0):
                            raise AssertionError('Invalid saved timing')
                        velocity = np.diff(states, axis=0)/intervals[:, None]
                        padded = np.vstack((np.zeros(6), velocity, np.zeros(6)))
                        acceleration_intervals = np.r_[intervals[0],
                            (intervals[:-1]+intervals[1:])/2, intervals[-1]]
                        acceleration = np.diff(padded, axis=0)/acceleration_intervals[:, None]
                        max_speed = float(np.linalg.norm(velocity, axis=1).max(initial=0))
                        max_acceleration = float(np.linalg.norm(acceleration, axis=1).max(initial=0))
                        if max_speed > 1.+1e-8 or max_acceleration > 3.+1e-8:
                            raise AssertionError('Independent motion-limit replay failed')
                        checks.update(strict_pose_passed=True, motion_bounds_passed=True,
                                      max_velocity_rad_s=max_speed,
                                      max_acceleration_rad_s2=max_acceleration)
                    item[mode] = checks
            audits.append(item)
            if len(audits) % 16 == 0:
                print(f'Independently replayed {len(audits)}/{len(selected)} paired trajectories', flush=True)
    report = {'available_pairs': len(rows), 'independently_replayed_pairs': len(audits),
              'selection': 'Fixed-seed random sample plus every available detour and failure',
              'shared_goal_source_verified': True, 'checks_passed': True, 'rows': audits}
    name = 'audit.json' if args.case_count is None else f'audit_{args.case_count}.json'
    write_json(args.output/name, report)
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
