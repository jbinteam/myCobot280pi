"""Paired offline ablation: raw learned guesses versus the same guesses refined.

Targets are fixed, fresh, scene-valid FK witnesses. No current/random numerical
starts, diffusion paths, training, or hardware IO are part of this experiment.
"""
import argparse
import atexit
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares

from ..kinematics import ARM
from ..pose5 import axis_error, solve
from ..workspace import transforms
from .endpoint_ik import residual_and_jacobian
from .env import ArmEnv
from .ik_seed import DEFAULT_CHECKPOINT, IKSeedModel, configuration
from .ik_seed_data import load_dataset

_ENV = None
_MODEL = None


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def generate(output, count, seed, original_dataset):
    arrays, old_meta = load_dataset(original_dataset)
    old_targets = set()
    old_joints = set()
    for split in ('train', 'validation', 'test'):
        old_targets.update(row.tobytes() for row in np.c_[arrays[split + '_target_xyz_m'],
                                                        arrays[split + '_target_axis_base']])
        for key in ('q_current_rad', 'q_goal_rad'):
            old_joints.update(row.tobytes() for row in arrays[split + '_' + key])
    new_targets = set()
    new_joints = set()
    sampling = {}
    rng = np.random.default_rng(seed)
    tick = perf_counter()
    with ArmEnv() as env:
        env.set_obstacle(None)
        scene = env.metadata()

        def sample(n, label, goals=False):
            accepted = []
            attempts = 0
            while len(accepted) < n:
                if attempts > n * 100 + 1000:
                    raise RuntimeError('Scene-valid sample budget exhausted')
                q = rng.uniform(ARM.lower, ARM.upper)
                attempts += 1
                key = q.tobytes()
                if key in old_joints or key in new_joints or not env.valid(q):
                    continue
                if goals:
                    fk = ARM.transform(q)
                    target_key = np.r_[fk[:3, 3], fk[:3, 2]].tobytes()
                    if target_key in old_targets or target_key in new_targets:
                        continue
                    new_targets.add(target_key)
                new_joints.add(key)
                accepted.append(q)
                if len(accepted) % 1000 == 0:
                    print(f'Generated {label}: {len(accepted)}/{n}', flush=True)
            sampling[label] = {'accepted': n, 'attempts': attempts}
            return np.asarray(accepted)

        goals = sample(count, 'known_reachable_targets', goals=True)
        currents = sample(count, 'independent_current_poses')
        fk = transforms(goals)
        # Batch/scalar FK may round differently: verify exact-label exclusion
        # against both representations before freezing the corpus.
        labels = np.c_[fk[:, :3, 3], fk[:, :3, 2]]
        if len({r.tobytes() for r in labels}) != count or any(r.tobytes() in old_targets for r in labels):
            raise AssertionError('Duplicate or reused target')
    dataset = output / 'cases.npz'
    np.savez_compressed(dataset, q_current_rad=currents, q_witness_rad=goals,
                        target_xyz_m=fk[:, :3, 3], target_axis_base=fk[:, :3, 2],
                        case_id=np.arange(count))
    metadata = {
        'count': count, 'seed': seed, 'configuration': configuration(), 'scene': scene,
        'external_obstacle': None, 'sampling': sampling,
        'generation_seconds': perf_counter() - tick,
        'original_dataset': str(Path(original_dataset).resolve()),
        'original_dataset_sha256': old_meta['dataset_sha256'],
        'dataset_sha256': sha256(dataset),
        'distribution': 'Uniform full-operating-range joint draws, rejected by runtime self/table collision gates; one independent current per unique reachable FK target',
        'leakage': 'Exact target labels and joint witnesses/currents excluded from all original training/validation/test splits; cases are not conditional on current-pose IK failing',
        'units': {'joints': 'radians', 'position': 'meters', 'direction': 'directed tool-Z unit vector in base frame'},
    }
    write_json(output / 'cases.json', metadata)
    return dataset, metadata


def initialize(checkpoint):
    global _ENV, _MODEL
    _MODEL = IKSeedModel(checkpoint)
    _ENV = ArmEnv()
    _ENV.set_obstacle(None)
    atexit.register(_ENV.close)


def assess(q, target, axis):
    fk = ARM.transform(q)
    position = float(np.linalg.norm(fk[:3, 3] - target) * 1000)
    direction = float(axis_error(fk[:3, 2], axis)[2])
    finite_bounds = bool(np.isfinite(q).all() and np.all(q >= ARM.lower) and np.all(q <= ARM.upper))
    safe = bool(finite_bounds and _ENV.valid(q))
    pose_pass = position <= 1. and direction <= 1.
    return {'position_error_mm': position, 'direction_error_deg': direction,
            'within_joint_limits': finite_bounds, 'collision_valid': safe,
            'pose_pass': pose_pass, 'success': bool(pose_pass and safe),
            'joints_rad': np.asarray(q).tolist()}


def best_error(candidates):
    # Both tolerances equal 1 in their reported units. Preserve paired errors
    # from one endpoint, rather than independent position/angle minima.
    return min(candidates, key=lambda r: max(r['position_error_mm'], r['direction_error_deg']))


def evaluate_case(task):
    index, current, witness, target, axis = task
    if not (_ENV.valid(current) and _ENV.valid(witness)):
        raise AssertionError('Evaluation current/witness is not scene-valid')
    wfk = ARM.transform(witness)
    if np.linalg.norm(wfk[:3, 3] - target) > 1e-10 or axis_error(wfk[:3, 2], axis)[2] > 1e-6:
        raise AssertionError('Reachability witness does not match target')
    start = perf_counter()
    guesses = _MODEL.predict(current, target, axis)
    inference_ms = (perf_counter() - start) * 1000
    if guesses.shape != (8, 6) or not np.isfinite(guesses).all():
        raise AssertionError('Invalid learned guesses')
    ranking = sorted(range(8), key=lambda i: float(
        np.sum(residual_and_jacobian(guesses[i], target, axis)[0] ** 2)
        + 1e-5 * np.sum(((guesses[i] - current) / ARM.radius) ** 2)))
    raw_candidates = [dict(assess(guesses[i], target, axis), proposal_index=i) for i in ranking]
    # Give the unrefined mode every opportunity to pass, using all eight guesses.
    # If none passes, report the first guess under the deployment FK ranking.
    raw = next((r.copy() for r in raw_candidates if r['success']), raw_candidates[0].copy())
    raw['elapsed_ms'] = (perf_counter() - start) * 1000
    refined_candidates = []
    dls_attempts = 0
    dls_iterations = 0
    fallback_attempts = 0
    fallback_nfev = 0
    begin_refine = perf_counter()
    selected = None
    for i in ranking:
        dls_attempts += 1
        result = solve(guesses[i], target, target_axis=axis, method='bounded_dls',
                       steps=400, goal_constraint=_ENV.valid)
        dls_iterations += result['iterations']
        q = np.deg2rad(result['frames'][-1]['q'])
        candidate = dict(assess(q, target, axis), proposal_index=i, method='bounded_dls')
        refined_candidates.append(candidate)
        if result['converged'] and candidate['success']:
            selected = candidate
            break
    dls = (selected or best_error(refined_candidates)).copy()
    dls['elapsed_refinement_ms'] = (perf_counter() - begin_refine) * 1000
    if selected is None:
        for i in ranking:
            fallback_attempts += 1
            result = least_squares(
                lambda q: residual_and_jacobian(q, target, axis)[0],
                np.clip(guesses[i], ARM.lower, ARM.upper),
                jac=lambda q: residual_and_jacobian(q, target, axis)[1],
                bounds=(ARM.lower, ARM.upper), max_nfev=160,
                ftol=1e-9, xtol=1e-9, gtol=1e-9)
            fallback_nfev += int(result.nfev)
            candidate = dict(assess(result.x, target, axis), proposal_index=i, method='least_squares')
            refined_candidates.append(candidate)
            if candidate['success']:
                selected = candidate
                break
    refined = (selected or best_error(refined_candidates)).copy()
    refined['elapsed_refinement_ms'] = (perf_counter() - begin_refine) * 1000
    refined['elapsed_ms'] = raw['elapsed_ms'] + refined['elapsed_refinement_ms']
    return {
        'case_id': int(index), 'raw': raw, 'dls_only': dls, 'refined': refined,
        'raw_candidates': raw_candidates, 'refinement_candidates': refined_candidates,
        'inference_ms': inference_ms, 'dls_attempts': dls_attempts,
        'dls_iterations': dls_iterations, 'fallback_attempts': fallback_attempts,
        'fallback_nfev': fallback_nfev,
    }


def stats(values):
    a = np.asarray(values, float)
    if not len(a):
        return None
    return {'mean': float(a.mean()), 'median': float(np.median(a)),
            'p95': float(np.percentile(a, 95)), 'max': float(a.max())}


def wilson(successes, total):
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    half = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [float((center-half)*100), float((center+half)*100)]


def summarize(rows):
    summary = {'cases': len(rows)}
    for mode in ('raw', 'dls_only', 'refined'):
        results = [r[mode] for r in rows]
        successes = [r for r in results if r['success']]
        summary[mode] = {
            'successes': len(successes), 'failures': len(rows)-len(successes),
            'success_rate_percent': len(successes)/len(rows)*100,
            'success_rate_95pct_wilson_percent': wilson(len(successes), len(rows)),
            'pose_only_passes': sum(r['pose_pass'] for r in results),
            'selected_endpoint_collision_invalid': sum(not r['collision_valid'] for r in results),
            'position_error_mm_all_cases': stats([r['position_error_mm'] for r in results]),
            'direction_error_deg_all_cases': stats([r['direction_error_deg'] for r in results]),
            'position_error_mm_successes_only': stats([r['position_error_mm'] for r in successes]),
            'direction_error_deg_successes_only': stats([r['direction_error_deg'] for r in successes]),
        }
    summary['paired_outcomes'] = {
        'both_success': sum(r['raw']['success'] and r['refined']['success'] for r in rows),
        'refinement_recovers': sum(not r['raw']['success'] and r['refined']['success'] for r in rows),
        'raw_only_success': sum(r['raw']['success'] and not r['refined']['success'] for r in rows),
        'both_fail': sum(not r['raw']['success'] and not r['refined']['success'] for r in rows),
    }
    summary['refined_acceptance_method'] = {method: sum(r['refined']['success'] and r['refined']['method'] == method for r in rows)
                                            for method in ('bounded_dls', 'least_squares')}
    summary['dls_attempts'] = stats([r['dls_attempts'] for r in rows])
    summary['fallback_cases'] = sum(r['fallback_attempts'] > 0 for r in rows)
    summary['raw_elapsed_ms_parallel_run'] = stats([r['raw']['elapsed_ms'] for r in rows])
    summary['refined_elapsed_ms_parallel_run'] = stats([r['refined']['elapsed_ms'] for r in rows])
    return summary


def chart(output, rows, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    colors = {'raw': '#c45745', 'refined': '#287a59'}
    labels = {'raw': 'Raw learned guesses', 'refined': 'With numerical refinement'}
    for mode in ('raw', 'refined'):
        for ax, key, xlabel in ((axes[1], 'position_error_mm', 'Position error (mm)'),
                                (axes[2], 'direction_error_deg', 'Direction error (degrees)')):
            values = np.sort([r[mode][key] for r in rows])
            ax.plot(values, np.arange(1, len(values)+1)/len(values)*100,
                    color=colors[mode], label=labels[mode], linewidth=2)
            ax.set_xscale('symlog', linthresh=.01)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Cases at or below this error (%)')
            ax.grid(alpha=.2)
    rates = [summary[m]['success_rate_percent'] for m in ('raw', 'refined')]
    bars = axes[0].bar(['Raw guesses', 'Refined'], rates, color=list(colors.values()), width=.6)
    for b, rate, mode in zip(bars, rates, ('raw', 'refined')):
        axes[0].text(b.get_x()+b.get_width()/2, rate+2,
                     f'{rate:.2f}%\n{summary[mode]["successes"]:,}/{len(rows):,}', ha='center')
    axes[0].set_ylim(0, 114)
    axes[0].set_ylabel('Strict endpoint success (%)')
    axes[0].set_title('Position ≤1 mm, direction ≤1° + collision gates')
    for ax in axes[1:]:
        ax.axvline(1, color='#555', linestyle='--', alpha=.6, label='Acceptance threshold')
    axes[1].legend(loc='lower right', fontsize=8)
    fig.suptitle(f'{len(rows):,} fresh reachable targets • same eight learned guesses • fixed target', fontsize=13)
    fig.tight_layout()
    fig.savefig(output/'comparison.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=202610081)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--original-dataset', type=Path, default=Path('simulation/artifacts/ik_seed_v1/dataset.npz'))
    parser.add_argument('--generate-only', action='store_true')
    parser.add_argument('--skip-chart', action='store_true', help='Generate the optional matplotlib chart separately')
    args = parser.parse_args()
    if args.count < 1 or args.workers < 1:
        raise ValueError('Positive count/workers required')
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    model_hash = sha256(checkpoint)
    IKSeedModel(checkpoint)
    dataset = args.output/'cases.npz'
    if dataset.exists():
        metadata = json.loads((args.output/'cases.json').read_text())
        if metadata['count'] != args.count or metadata['seed'] != args.seed or metadata['dataset_sha256'] != sha256(dataset):
            raise ValueError('Existing corpus differs from requested experiment')
    else:
        dataset, metadata = generate(args.output, args.count, args.seed, args.original_dataset)
    if args.generate_only:
        return
    if (args.output/'rows.jsonl').exists():
        raise ValueError('Choose a fresh output directory; do not overwrite trial results')
    with np.load(dataset, allow_pickle=False) as archive:
        data = {k: archive[k].copy() for k in archive.files}
    tasks = [(i, data['q_current_rad'][i], data['q_witness_rad'][i],
              data['target_xyz_m'][i], data['target_axis_base'][i]) for i in range(args.count)]
    rows = []
    tick = perf_counter()
    with (args.output/'rows.jsonl').open('w') as stream, ProcessPoolExecutor(
            max_workers=args.workers, mp_context=mp.get_context('spawn'),
            initializer=initialize, initargs=(str(checkpoint),)) as executor:
        futures = [executor.submit(evaluate_case, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False)+'\n')
            if len(rows) % 100 == 0 or len(rows) == args.count:
                stream.flush()
                progress = {'completed': len(rows), 'total': args.count,
                            'raw_success': sum(r['raw']['success'] for r in rows),
                            'refined_success': sum(r['refined']['success'] for r in rows),
                            'seconds': perf_counter()-tick}
                write_json(args.output/'progress.json', progress)
                print(json.dumps(progress), flush=True)
    if sha256(checkpoint) != model_hash or sha256(dataset) != metadata['dataset_sha256']:
        raise RuntimeError('Frozen model or corpus changed during evaluation')
    rows.sort(key=lambda r: r['case_id'])
    if [r['case_id'] for r in rows] != list(range(args.count)):
        raise AssertionError('Missing or duplicate case results')
    summary = summarize(rows)
    report = {
        'summary': summary, 'dataset': metadata, 'checkpoint': str(checkpoint),
        'checkpoint_sha256': model_hash, 'benchmark_source_sha256': sha256(__file__),
        'wall_seconds': perf_counter()-tick, 'workers': args.workers,
        'comparison': 'Same eight learned starts per case; raw guess acceptance versus bounded DLS refinement, then least-squares fallback on those same starts only',
        'budgets': {'proposals': 8, 'dls_max_iterations_per_start': 400,
                    'least_squares_max_nfev_per_start': 160},
        'acceptance': {'position_error_mm_max': 1., 'directed_tool_z_error_deg_max': 1.,
                       'joint_limits': True, 'runtime_self_table_collision': True, 'clearance_mm': 1.},
        'selection': {'raw': 'First passing raw guess in deployment FK ranking, otherwise the top-ranked raw guess',
                      'refined': 'First passing DLS endpoint, otherwise first passing least-squares endpoint; if all fail, smallest max(position_mm, direction_deg) among refined endpoints'},
        'limitations': 'Known-reachable endpoint distribution, not arbitrary Cartesian targets or the subset where current-first IK fails. No physical tracking, trajectory planning or external box. Eight learned-only starts isolate refinement and differ from production current+learned+random 32-start search. All-case errors include rejected endpoints; low error alone does not imply collision validity. Parallel timings are descriptive, not latency claims.',
        'source_hashes': {path: sha256(path) for path in ('simulation/diffusion/ik_seed.py',
            'simulation/diffusion/endpoint_ik.py', 'simulation/pose5.py', 'simulation/diffusion/env.py')},
        'per_case_results': str((args.output/'rows.jsonl').resolve()),
    }
    write_json(args.output/'report.json', report)
    if not args.skip_chart:
        chart(args.output, rows, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
