"""Offline scalar versus batched PyBullet joint-reset comparison.

CUDA graph and scheduler cache stay enabled in both modes. A scoped observer
counts _set_joints calls and reads each scene's actual batch_reset flag; it does
not wrap ArmEnv.valid or alter any collision predicate. No physical robot IO.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import functools
import html
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from .metrics import digest, outcome_errors, summarize
from .runner import (ROOT, CHILD_THREAD_ENV, create_output, frozen_inventory,
                     source_inventory, read_events, sha256)
from .scenarios import load_cases, load_manifest
from .telemetry import POLL_INTERVAL_S, MAX_COVERAGE_GAP_S, sample_gpu, attach_windows

MODES = ('scalar_reset', 'batch_reset', 'batch_reset', 'scalar_reset')
REFERENCE_CHANGE_ALLOWLIST = {'simulation/diffusion/env.py',
                              'simulation/diffusion/tests/test_bullet_reset.py'}
DEFAULT_REFERENCE = ROOT/'output/scheduler_lookup_optimization_20261007/quick/report.json'


def select_cases(names):
    catalog = {case['id']: case for case in load_cases('full')}
    if not names or len(names) != len(set(names)):
        raise ValueError('Cases must be nonempty and unique')
    unknown = [name for name in names if name not in catalog]
    if unknown:
        raise ValueError('Unknown cases: '+', '.join(unknown))
    return [catalog[name] for name in names]


def reference_signatures(reference, cases, seeds, sources, manifest):
    """Reference correctness only; production changes restricted to this patch."""
    if reference.get('summary', {}).get('passed') is not True:
        raise ValueError('Historical reference did not pass correctness')
    for field in ('checkpoint_sha256', 'urdf_sha256'):
        if reference.get('inputs', {}).get('expected_'+field) != manifest[field]:
            raise ValueError('Reference '+field+' differs')
    prior = {k: v for k, v in reference.get('source_sha256', {}).items()
             if k.startswith(('simulation/', 'webui/'))}
    current = {k: v for k, v in sources.items() if k.startswith(('simulation/', 'webui/'))}
    if not prior or not current:
        raise ValueError('Reference production source inventory missing')
    changes = {key for key in prior.keys() | current.keys() if prior.get(key) != current.get(key)}
    unexpected = changes - REFERENCE_CHANGE_ALLOWLIST
    if unexpected:
        raise ValueError('Unrelated production changes versus reference: '+', '.join(sorted(unexpected)))
    expected = {}
    for case in cases:
        for seed in seeds:
            rows = [row for row in reference.get('records', []) if row.get('case_id') == case['id']
                    and row.get('seed') == seed and row.get('backend') == 'cache_on']
            if not rows:
                raise ValueError(f"No historical cache_on records for {case['id']} seed {seed}")
            signatures = []
            for row in rows:
                if row.get('request') != {**case['request'], 'seed': seed} or outcome_errors(row, case):
                    raise ValueError('Reference request/outcome mismatch for '+case['id'])
                signature = row.get('signatures')
                if (not isinstance(signature, dict) or not all(signature.get(key) for key in
                        ('full_reference', 'sampled_predictions', 'rng_after')) or 'encoder_profile' not in signature):
                    raise ValueError('Reference signatures incomplete')
                if row.get('device') != 'cuda' or (row.get('metrics', {}).get('sample_calls', 0)
                                                  and not row.get('graph_active')):
                    raise ValueError('Reference neural backend is not active CUDA graph')
                if row.get('neural_backend') != 'cuda_graph' or row.get('scheduler_cache_enabled') is not True:
                    raise ValueError('Reference must have CUDA graph and scheduler cache enabled')
                if (row.get('metrics', {}).get('sample_calls', 0)
                        and (type(row.get('scheduler_cache_hits')) is not int or row['scheduler_cache_hits'] <= 0)):
                    raise ValueError('Reference has no observed scheduler cache activity')
                signatures.append(signature)
            if any(signature != signatures[0] for signature in signatures):
                raise ValueError('Reference signatures inconsistent')
            expected[f"{case['id']}:{seed}"] = signatures[0]
    return expected, sorted(changes)


def cache_observations(policy):
    return {'enabled': getattr(policy, 'scheduler_cache', None),
            'hits': getattr(policy, '_scheduler_cache_hits', None),
            'misses': getattr(policy, '_scheduler_cache_misses', None)}


def worker_main(input_path, output_path, mode, affinity=None):
    """Keep inference fixed and observe every scene's actual reset mode."""
    from . import worker
    if mode not in {'scalar_reset', 'batch_reset'}:
        raise ValueError('Unknown joint reset mode')
    os.environ['MYCOBOT_SCHEDULER_CACHE'] = '1'
    os.environ['MYCOBOT_BULLET_BATCH_RESET'] = '0' if mode == 'scalar_reset' else '1'
    if affinity is not None:
        if not hasattr(os, 'sched_setaffinity'):
            raise RuntimeError('CPU affinity unsupported on this platform')
        os.sched_setaffinity(0, set(affinity))
    original_run = worker.run_case
    @functools.wraps(original_run)
    def observed(case, seed, backend, service, torch):
        from simulation.diffusion.env import ArmEnv
        before = cache_observations(service._policy)
        reset_counts = {'scalar_reset': 0, 'batch_reset': 0, 'unknown': 0}
        original_reset = ArmEnv._set_joints
        @functools.wraps(original_reset)
        def observed_reset(scene, *args, **kwargs):
            enabled = getattr(scene, 'batch_reset', None)
            label = ('batch_reset' if enabled else 'scalar_reset') if type(enabled) is bool else 'unknown'
            reset_counts[label] += 1
            return original_reset(scene, *args, **kwargs)
        ArmEnv._set_joints = observed_reset
        try:
            row = original_run(case, seed, backend, service, torch)
        finally:
            ArmEnv._set_joints = original_reset
        after = cache_observations(service._policy)
        counts = {}
        for key in ('hits', 'misses'):
            a, b = before[key], after[key]
            counts[key] = b-a if type(a) is int and type(b) is int else None
        row.update(neural_backend=backend, scheduler_cache_enabled=after['enabled'],
                   scheduler_cache_hits=counts['hits'], scheduler_cache_misses=counts['misses'],
                   requested_reset_mode=mode, reset_calls=sum(reset_counts.values()),
                   reset_calls_by_mode=reset_counts,
                   observed_reset_modes=sorted(key for key, value in reset_counts.items() if value),
                   child_cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                   requested_cpu_affinity=affinity)
        return row
    previous_argv = sys.argv
    worker.run_case = observed
    sys.argv = ['benchmarks.suite.worker', '--input', str(input_path), '--output', str(output_path),
                '--backend', 'cuda_graph']
    try:
        return worker.main()
    finally:
        worker.run_case = original_run
        sys.argv = previous_argv


def launch_worker(directory, index, mode, timeout, affinity=None):
    output = directory/f'worker-{index:02d}-{mode}.jsonl'
    log = directory/f'worker-{index:02d}-{mode}.log'
    telemetry_path = directory/f'worker-{index:02d}-{mode}-telemetry.jsonl'
    command = [sys.executable, '-m', 'benchmarks.suite.compare_collision', '--worker',
               '--input', str(directory/'inputs.json'), '--worker-output', str(output), '--reset-mode', mode]
    if affinity is not None:
        command += ['--cpu-affinity', *map(str, affinity)]
    began = last_progress = next_poll = time.monotonic()
    observed = 0
    timed_out = False
    samples = []
    with log.open('x') as stream, telemetry_path.open('x') as telemetry_stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True, env={**os.environ, **CHILD_THREAD_ENV,
                                   'MYCOBOT_SCHEDULER_CACHE': '1',
                                   'MYCOBOT_BULLET_BATCH_RESET': '0' if mode == 'scalar_reset' else '1'})
        try:
            while process.poll() is None:
                if time.monotonic() >= next_poll:
                    sample = sample_gpu(); samples.append(sample)
                    telemetry_stream.write(json.dumps(sample, allow_nan=False)+'\n'); telemetry_stream.flush()
                    next_poll = sample['query_started_monotonic_s']+POLL_INTERVAL_S
                events = read_events(output)
                if len(events) > observed:
                    last_progress = time.monotonic(); observed = len(events)
                if time.monotonic()-last_progress > timeout:
                    timed_out = True; process.kill(); break
                time.sleep(.1)
            process.wait()
        finally:
            if process.poll() is None:
                process.kill(); process.wait()
    return {'reset_mode': mode, 'neural_backend': 'cuda_graph', 'execution_position': index,
            'pid': process.pid, 'returncode': process.returncode, 'timed_out': timed_out,
            'process_wall_ms': (time.monotonic()-began)*1000,
            'events': attach_windows(read_events(output), samples, process.pid),
            'gpu_telemetry_file': telemetry_path.name, 'gpu_telemetry_samples': samples,
            'log': log.name, 'data': output.name, 'requested_cpu_affinity': affinity}


def comparison_errors(row, expected=None, affinity=None):
    mode = row.get('backend')
    errors = []
    if mode not in {'scalar_reset', 'batch_reset'}:
        errors.append('unknown collision joint-reset comparison mode')
    if row.get('neural_backend') != 'cuda_graph' or row.get('device') != 'cuda':
        errors.append('both joint-reset modes require the same CUDA graph neural backend')
    if row.get('scheduler_cache_enabled') is not True:
        errors.append('scheduler cache must remain enabled in both joint-reset modes')
    if row.get('requested_reset_mode') != mode or row.get('observed_reset_modes') != [mode]:
        errors.append('observed scene reset modes differ from the requested comparison mode')
    resets, counts = row.get('reset_calls'), row.get('reset_calls_by_mode')
    if (type(resets) is not int or resets <= 0 or not isinstance(counts, dict)
            or set(counts) != {'scalar_reset', 'batch_reset', 'unknown'}
            or any(type(n) is not int or n < 0 for n in counts.values())
            or sum(counts.values()) != resets or counts.get(mode) != resets):
        errors.append('joint-reset call counts are missing, mixed or invalid')
    hits, misses = row.get('scheduler_cache_hits'), row.get('scheduler_cache_misses')
    if type(hits) is not int or hits < 0 or type(misses) is not int or misses < 0:
        errors.append('actual scheduler-cache call counters missing or invalid')
    elif misses:
        errors.append('scheduler cache unexpectedly fell back during a collision comparison')
    if row.get('metrics', {}).get('sample_calls', 0):
        if not row.get('graph_active'):
            errors.append('CUDA graph inactive in a sampled collision comparison trial')
        if type(hits) is not int or hits <= 0:
            errors.append('sampled collision comparison performed no scheduler cache hits')
    if affinity is not None and row.get('child_cpu_affinity') != affinity:
        errors.append('actual child CPU affinity differs from requested CPUs')
    if expected is not None and row.get('signatures') != expected:
        errors.append('signatures differ from historical reference')
    return errors


def _relabel_summary(value):
    """Internal ABBA slots reuse existing gates; exports name the actual modes."""
    mapping = {'eager': 'scalar_reset', 'cuda_graph': 'batch_reset',
               'eager_mean_ms': 'scalar_reset_mean_ms', 'cuda_graph_mean_ms': 'batch_reset_mean_ms'}
    if isinstance(value, dict):
        return {mapping.get(key, key): _relabel_summary(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_relabel_summary(item) for item in value]
    if isinstance(value, str):
        return value.replace('cuda_graph', 'batch_reset').replace('eager', 'scalar_reset')
    return value


def summarize_comparison(records, cases, seeds, repeats, expected=None, affinity=None):
    mapped = []
    reset_counts = {}
    for row in records:
        if type(row.get('reset_calls')) is int:
            reset_counts.setdefault((row.get('case_id'), row.get('seed')), set()).add(row['reset_calls'])
    for row in records:
        converted = deepcopy(row)
        key = f"{row.get('case_id')}:{row.get('seed')}"
        errors = comparison_errors(row, None if expected is None else expected.get(key), affinity)
        if len(reset_counts.get((row.get('case_id'), row.get('seed')), set())) > 1:
            errors.append('joint-reset call counts differ across paired trials for the same case/seed')
        if expected is not None and key not in expected:
            errors.append('historical reference missing case/seed')
        if errors:
            converted['error'] = '; '.join(([str(row['error'])] if row.get('error') else [])+errors)
        converted['backend'] = {'scalar_reset': 'eager', 'batch_reset': 'cuda_graph'}.get(row.get('backend'), row.get('backend'))
        mapped.append(converted)
    result = _relabel_summary(summarize(mapped, cases, seeds, repeats, reliable_timing=True))
    result['comparison'] = 'scalar versus batched collision joint resets; CUDA graph and scheduler cache enabled in both modes'
    result['neural_backend'] = 'cuda_graph'
    result['scheduler_cache'] = 'enabled in both modes'
    return result


def invalidate_summary(summary, errors):
    summary['failures'].extend(errors)
    summary['passed'] = not summary['failures']
    if not summary['passed']:
        for group in summary['groups']:
            group['speedup_ratio'] = None
            if group.get('timing_status') == 'supported_speedup':
                group['timing_status'] = 'inconclusive'
                group.setdefault('timing_reasons', []).append('Global correctness or infrastructure failure')


def write_report(directory, report):
    directory = Path(directory)
    (directory/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    esc = lambda value: html.escape(str(value), quote=True)
    fmt = lambda value: '—' if value is None else f'{value:.3f}'
    rows = []
    for group in report['summary']['groups']:
        timing = group['paired_timing']
        interval = timing['bootstrap_ci95']
        ci = '—' if interval is None else f"[{interval['lower']:.3f}, {interval['upper']:.3f}]"
        latencies = group['warm_success_latency_ms']
        cells = [group['case_id'], group['seed'], 'PASS' if group['correctness_passed'] else 'FAIL',
                 group['timing_status'].replace('_', ' '), timing['complete_block_count'],
                 fmt(latencies['scalar_reset']['median']), fmt(latencies['batch_reset']['median']),
                 fmt(timing['median_ratio']), ci,
                 '; '.join(group['timing_reasons']+group['errors']) or 'None']
        rows.append('<tr>'+''.join('<td>'+esc(value)+'</td>' for value in cells)+'</tr>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Collision joint reset comparison</title><style>body{font:15px system-ui;margin:2rem;color:#18222c}.scroll{overflow:auto}table{border-collapse:collapse;min-width:1200px;width:100%}th,td{border:1px solid #ccd3da;padding:.5rem;text-align:left;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere}p{max-width:100ch}</style>
<h1>Collision joint reset: '''+('PASS' if report['summary']['passed'] else 'FAIL / INCOMPLETE')+'''</h1>
<p>Baseline scalar_reset versus candidate batch_reset, with CUDA graph neural inference and the scheduler timestep cache held on in both. Same current source, model, collision checks and inputs. No robot IO. Historical references check exact outputs only; their absolute times are not used.</p>
<p>One ratio per complete ABBA block: mean scalar_reset slots 0,3 / mean batch_reset slots 1,2. Whole-block bootstrap requires at least 10 blocks. Fewer blocks are diagnostic. Latencies below are instrumented warm planning medians; graph inference and the scheduler timestep cache are held constant. Only supported_speedup with clean observed GPU telemetry, stable repeated timings, exact outputs and a 95% interval above 1 supports a speed claim. The approximate interval assumes exchangeable blocks; autocorrelation and unsampled host variation remain limitations.</p>
<div class="scroll"><table><thead><tr><th>Case</th><th>Seed</th><th>Correctness</th><th>Timing status</th><th>ABBA blocks</th><th>Scalar reset median ms</th><th>Batch reset median ms</th><th>Median block ratio</th><th>95% interval</th><th>Reasons</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div>
<h2>Failures</h2><pre>'''+esc('\n'.join(report['summary']['failures']) or 'None')+'''</pre><details><summary>Complete reproducibility record</summary><pre>'''+esc(json.dumps(report, indent=2, allow_nan=False))+'''</pre></details></html>'''
    (directory/'report.html').write_text(page)


def parser():
    p = argparse.ArgumentParser(description='Offline collision joint-reset ABBA comparison; graph inference in BOTH modes; no robot IO.')
    p.add_argument('--cases', nargs='+', default=['recent_live_start'])
    p.add_argument('--seeds', nargs='+', type=int, default=[42])
    p.add_argument('--repeats', type=int, default=10, help='ABBA blocks; fewer than 10 are diagnostic only')
    p.add_argument('--allow-long-run', action='store_true', help='Required for 10 or more ABBA blocks')
    p.add_argument('--reference', type=Path, default=DEFAULT_REFERENCE, help='Prior passing scheduler comparison; cache_on rows check exact historical outputs')
    p.add_argument('--cpu-affinity', nargs='+', type=int, help='Set only child worker affinity to these allowed CPU IDs')
    p.add_argument('--timeout', type=float, default=240)
    p.add_argument('--output', type=Path)
    p.add_argument('--list-cases', action='store_true')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--input', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--reset-mode', choices=('scalar_reset', 'batch_reset'), help=argparse.SUPPRESS)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.worker:
        return worker_main(args.input, args.worker_output, args.reset_mode, args.cpu_affinity)
    try:
        cases = select_cases(args.cases)
    except ValueError as exc:
        parser().error(str(exc))
    if args.list_cases:
        print(json.dumps(cases, indent=2)); return 0
    if (not 1 <= args.repeats <= 100 or not math.isfinite(args.timeout) or args.timeout <= 0 or
            len(args.seeds) != len(set(args.seeds)) or any(not 0 <= seed < 2**31 for seed in args.seeds)):
        parser().error('repeats must be 1..100, timeout finite positive and seeds unique in 0..2**31-1')
    if args.repeats >= 10 and not args.allow_long_run:
        parser().error('10 or more blocks require --allow-long-run')
    affinity = sorted(args.cpu_affinity) if args.cpu_affinity is not None else None
    if affinity is not None and (len(affinity) != len(set(affinity)) or not hasattr(os, 'sched_getaffinity')
                                 or not set(affinity) <= os.sched_getaffinity(0)):
        parser().error('CPU affinity must contain unique CPU IDs allowed to this process')
    lock_path = ROOT/'tmp/benchmark-suite.lock'; lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Another benchmark owns the project GPU lock.', file=sys.stderr); return 2
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        sources, frozen, manifest = source_inventory(), frozen_inventory(), load_manifest()
        expected, reference_changes = None, []
        reference_sha = None
        if args.reference:
            try:
                expected, reference_changes = reference_signatures(json.loads(args.reference.read_text()), cases,
                                                                    args.seeds, sources, manifest)
                reference_sha = sha256(args.reference)
            except (OSError, ValueError, KeyError) as exc:
                print('Reference rejected: '+str(exc), file=sys.stderr); return 2
        try:
            directory = create_output(args.output or ROOT/'output/collision-comparison'/stamp)
        except (ValueError, FileExistsError) as exc:
            print(str(exc), file=sys.stderr); return 2
        spec = {'cases': cases, 'seeds': args.seeds, 'repeats': args.repeats,
                'expected_checkpoint_sha256': manifest['checkpoint_sha256'],
                'expected_urdf_sha256': manifest['urdf_sha256']}
        (directory/'inputs.json').write_text(json.dumps(spec, indent=2)+'\n')
        report = {'schema_version': 1, 'created_utc': stamp, 'inputs': spec,
                  'comparison': {'baseline': 'scalar_reset', 'candidate': 'batch_reset',
                                 'neural_backend': 'cuda_graph', 'scheduler_cache_enabled': True},
                  'scope': 'Offline same-source collision joint-reset comparison; no physical IO or qualification',
                  'source_sha256': sources, 'source_closure_sha256': digest(sources),
                  'frozen_reference_sha256': frozen, 'input_sha256': digest(spec),
                  'historical_reference': {'path': str(args.reference) if args.reference else None,
                                           'sha256': reference_sha, 'verified': expected is not None,
                                           'allowed_production_changes': reference_changes},
                  'runtime': {'python': sys.version, 'platform': platform.platform(),
                              'parent_cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                              'requested_child_cpu_affinity': affinity, 'child_thread_environment': CHILD_THREAD_ENV},
                  'measurement_policy': {'order': list(MODES)*args.repeats,
                                         'neural_backend': 'cuda_graph in both modes',
                                         'scheduler_cache': 'enabled in both modes with observed hit/miss counters',
                                         'collision_reset_observation': 'Scoped _set_joints call counts by actual ArmEnv.batch_reset flag; ArmEnv.valid unchanged',
                                         'minimum_blocks_for_timing_claim': 10,
                                         'warmup': 'Complete case warmup immediately before measured trial in each fresh process',
                                         'timing': 'Same existing probe/prediction hashing overhead; encoder preparation separately recorded',
                                         'telemetry_poll_interval_s': POLL_INTERVAL_S,
                                         'telemetry_maximum_coverage_gap_s': MAX_COVERAGE_GAP_S},
                  'workers': [], 'records': []}
        errors = []
        try:
            for index, mode in enumerate(report['measurement_policy']['order']):
                print(f'[{index+1}/{4*args.repeats}] {mode} / neural CUDA graph: {len(cases)} cases', flush=True)
                process = launch_worker(directory, index, mode, args.timeout, affinity)
                report['workers'].append(process)
                for event in process['events']:
                    if event.get('type') == 'trial':
                        # Preserve actual neural backend separately, never pretend scalar_reset is eager inference.
                        report['records'].append({**event, 'neural_backend': event.get('backend'), 'backend': mode,
                                                  'execution_position': index, 'repetition': index//4,
                                                  'abba_slot': index%4, 'pid': process['pid']})
                    elif event.get('type') == 'fatal':
                        errors.append(str(event.get('error', 'worker fatal error')))
                if process['returncode'] or process['timed_out']:
                    errors.append(f'worker {index} failed/timed out (exit {process["returncode"]})')
                (directory/'progress.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
                if any(event.get('type') == 'fatal' for event in process['events']):
                    break
        except KeyboardInterrupt:
            errors.append('comparison interrupted')
        except Exception as exc:
            errors.append(f'orchestration error: {type(exc).__name__}: {exc}')
        report['source_unchanged'] = source_inventory() == sources
        report['frozen_unchanged'] = frozen_inventory() == frozen
        if not report['source_unchanged']:
            errors.append('source/config changed during comparison')
        if not report['frozen_unchanged']:
            errors.append('frozen benchmark changed during comparison')
        if args.reference:
            try:
                if sha256(args.reference) != reference_sha:
                    errors.append('historical reference changed during comparison')
            except OSError:
                errors.append('historical reference unavailable after comparison')
        report['summary'] = summarize_comparison(report['records'], cases, args.seeds, args.repeats, expected, affinity)
        invalidate_summary(report['summary'], errors)
        write_report(directory, report)
        print(f'{"PASS" if report["summary"]["passed"] else "FAIL"}: {directory}/report.html', flush=True)
        return 0 if report['summary']['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
