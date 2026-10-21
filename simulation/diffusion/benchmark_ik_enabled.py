"""Paired learned-IK enabled/disabled benchmark on a frozen fresh corpus.

Both modes retain production numerical refinement and equal 32+32 budgets.
No model training, diffusion trajectories or hardware IO occurs.
"""
import argparse
import atexit
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from ..kinematics import ARM
from ..pose5 import axis_error
from .benchmark_ik_refinement import sha256, stats, wilson, write_json
from .endpoint_ik import IKSearchExhausted
from .env import ArmEnv
from .ik_seed import DEFAULT_CHECKPOINT, IKSeedModel, configuration

_ENV = None


def initialize(checkpoint):
    global _ENV
    IKSeedModel(checkpoint)
    os.environ['MYCOBOT_IK_SEED_MODEL'] = checkpoint
    _ENV = ArmEnv()
    _ENV.set_obstacle(None)
    atexit.register(_ENV.close)


def evaluate_case(task):
    index, current, witness, target, axis = task
    if not (_ENV.valid(current) and _ENV.valid(witness)):
        raise AssertionError('Current or reachability witness fails the scene')
    wfk = ARM.transform(witness)
    if np.linalg.norm(wfk[:3, 3] - target) > 1e-10 or axis_error(wfk[:3, 2], axis)[2] > 1e-6:
        raise AssertionError('Reachability witness does not match target')
    row = {'case_id': int(index)}
    for enabled in ([False, True] if index % 2 == 0 else [True, False]):
        _ENV.reset(current)
        tick = perf_counter()
        try:
            q = _ENV.inverse_kinematics(target, axis, current=current, seed=42,
                                        attempts=32, robust=True, learned=enabled)
            fk = ARM.transform(q)
            position = float(np.linalg.norm(fk[:3, 3] - target) * 1000)
            direction = float(axis_error(fk[:3, 2], axis)[2])
            bounds = bool(np.isfinite(q).all() and np.all(q >= ARM.lower) and np.all(q <= ARM.upper))
            if not (bounds and _ENV.valid(q) and position <= 1. and direction <= 1.):
                raise AssertionError('Solver accepted an invalid endpoint')
            result = {'success': True, 'position_error_mm': position,
                      'direction_error_deg': direction, 'joints_rad': q.tolist(),
                      'error_source': 'Independently recomputed FK of accepted endpoint'}
        except IKSearchExhausted:
            diagnostics = _ENV.ik_diagnostics
            result = {'success': False,
                      'position_error_mm': diagnostics['best_position_error_mm'],
                      'direction_error_deg': diagnostics['best_direction_error_deg'],
                      'error_source': 'Paired FK errors from best rejected endpoint under runtime max-error score; not executable'}
        result['elapsed_ms'] = (perf_counter()-tick)*1000
        result['diagnostics'] = dict(_ENV.ik_diagnostics)
        if enabled and result['diagnostics']['legacy_attempts'] > 1 and result['diagnostics']['learned_status'] != 'ready':
            raise AssertionError('Learned model was unavailable')
        row['with_learned' if enabled else 'without_learned'] = result
    return row


def summarize(rows):
    summary = {'cases': len(rows)}
    for mode in ('with_learned', 'without_learned'):
        results = [r[mode] for r in rows]
        success = [r for r in results if r['success']]
        summary[mode] = {
            'successes': len(success), 'failures': len(rows)-len(success),
            'success_rate_percent': len(success)/len(rows)*100,
            'success_rate_95pct_wilson_percent': wilson(len(success), len(rows)),
            'position_error_mm_all_cases': stats([r['position_error_mm'] for r in results]),
            'direction_error_deg_all_cases': stats([r['direction_error_deg'] for r in results]),
            'position_error_mm_successes_only': stats([r['position_error_mm'] for r in success]),
            'direction_error_deg_successes_only': stats([r['direction_error_deg'] for r in success]),
            'elapsed_ms_parallel_run': stats([r['elapsed_ms'] for r in results]),
            'fallback_cases': sum(r['diagnostics']['fallback_attempts'] > 0 for r in results),
            'current_first_successes': sum(r['success'] and r['diagnostics']['legacy_attempts'] == 1 for r in results),
            'learned_start_successes': sum(r['diagnostics']['learned_refine_success'] for r in results),
        }
    summary['paired_outcomes'] = {
        'both_success': sum(r['with_learned']['success'] and r['without_learned']['success'] for r in rows),
        'learned_recovers': sum(r['with_learned']['success'] and not r['without_learned']['success'] for r in rows),
        'learned_regresses': sum(not r['with_learned']['success'] and r['without_learned']['success'] for r in rows),
        'both_fail': sum(not r['with_learned']['success'] and not r['without_learned']['success'] for r in rows),
    }
    summary['recovered_case_ids'] = [r['case_id'] for r in rows if r['with_learned']['success'] and not r['without_learned']['success']]
    summary['regressed_case_ids'] = [r['case_id'] for r in rows if not r['with_learned']['success'] and r['without_learned']['success']]
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    model_hash = sha256(checkpoint)
    IKSeedModel(checkpoint)
    metadata = json.loads(args.corpus.with_suffix('.json').read_text())
    if metadata['dataset_sha256'] != sha256(args.corpus) or metadata['configuration'] != configuration():
        raise ValueError('Corpus content/schema mismatch')
    source_files = ('simulation/diffusion/ik_seed.py', 'simulation/diffusion/endpoint_ik.py',
                    'simulation/diffusion/env.py', 'simulation/pose5.py')
    source_hashes = {p: sha256(p) for p in source_files}
    with np.load(args.corpus, allow_pickle=False) as archive:
        data = {k: archive[k].copy() for k in archive.files}
    count = len(data['case_id'])
    if count != metadata['count'] or list(data['case_id']) != list(range(count)):
        raise AssertionError('Incomplete corpus')
    tasks = [(i, data['q_current_rad'][i], data['q_witness_rad'][i],
              data['target_xyz_m'][i], data['target_axis_base'][i]) for i in range(count)]
    if (args.output/'rows.jsonl').exists():
        raise ValueError('Choose a fresh output directory; do not overwrite an experiment')
    tick = perf_counter()
    rows = []
    with (args.output/'rows.jsonl').open('w') as stream, ProcessPoolExecutor(
            max_workers=args.workers, mp_context=mp.get_context('spawn'),
            initializer=initialize, initargs=(str(checkpoint),)) as executor:
        futures = [executor.submit(evaluate_case, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False)+'\n')
            if len(rows) % 100 == 0 or len(rows) == count:
                stream.flush()
                progress = {'completed': len(rows), 'total': count,
                            'with_learned_success': sum(r['with_learned']['success'] for r in rows),
                            'without_learned_success': sum(r['without_learned']['success'] for r in rows),
                            'seconds': perf_counter()-tick}
                write_json(args.output/'progress.json', progress)
                print(json.dumps(progress), flush=True)
    if sha256(checkpoint) != model_hash or sha256(args.corpus) != metadata['dataset_sha256']:
        raise RuntimeError('Frozen model or corpus changed')
    if source_hashes != {p: sha256(p) for p in source_files}:
        raise RuntimeError('Solver sources changed during experiment')
    rows.sort(key=lambda r: r['case_id'])
    if [r['case_id'] for r in rows] != list(range(count)):
        raise AssertionError('Missing or duplicate results')
    summary = summarize(rows)
    report = {
        'summary': summary, 'dataset': metadata, 'checkpoint': str(checkpoint),
        'checkpoint_sha256': model_hash, 'source_hashes': source_hashes,
        'benchmark_source_sha256': sha256(__file__),
        'wall_seconds': perf_counter()-tick, 'workers': args.workers,
        'comparison': 'Identical numerical IK with learned starting guesses enabled versus disabled; current-first attempt always retained; same 32 bounded-DLS plus 32 robust least-squares start budgets, seed42, alternating method order',
        'acceptance': 'Independent FK position ≤1mm, directed tool-Z ≤1degree, operating joint limits, runtime self/table clearance 1mm',
        'limitations': 'Fresh reachable full-range FK targets, not arbitrary Cartesian requests or the subset failing current-first IK. Endpoint search only; no diffusion paths, external box, or physical tracking. All-case errors include paired best rejected endpoints for failed searches. Parallel timings are descriptive.',
        'per_case_results': str((args.output/'rows.jsonl').resolve()),
    }
    write_json(args.output/'report.json', report)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
