"""Parent-only sampled GPU observation; never queries inside a planner timer.

This observes compute processes at intervals. It cannot prove absence of short
competitors between samples and never terminates another process.
"""
import csv
from concurrent.futures import ThreadPoolExecutor
import math
import subprocess
import time

POLL_INTERVAL_S = 1.
MAX_COVERAGE_GAP_S = 2.5


def _number(value):
    try:
        result = float(value.strip())
    except (ValueError, AttributeError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def _uuid(value):
    value = str(value).strip().lower()
    return value.removeprefix('gpu-')


def parse_gpu_csv(text):
    result = []
    for row in csv.reader(text.splitlines()):
        if not row: continue
        if len(row) != 5 or not row[0].strip().startswith(('GPU-', 'MIG-')):
            raise ValueError('Malformed GPU telemetry row')
        result.append(dict(zip(('gpu_uuid','clock_sm_mhz','temperature_c','power_w','utilization_percent'),
                               [row[0].strip(), *(_number(x) for x in row[1:])])))
    if not result:
        raise ValueError('No GPUs reported')
    return result


def parse_process_csv(text):
    result = []
    for row in csv.reader(text.splitlines()):
        if not row: continue
        if len(row) != 2 or not row[0].strip().startswith(('GPU-', 'MIG-')):
            raise ValueError('Malformed compute process telemetry row')
        pid = int(row[1].strip())
        if pid <= 0: raise ValueError('Invalid compute PID')
        result.append({'gpu_uuid': row[0].strip(), 'pid': pid})
    return result


def sample_gpu(run=subprocess.run, clock=time.monotonic):
    started = clock()
    result = {'query_started_monotonic_s': started, 'available': False, 'gpus': [], 'compute_processes': []}
    try:
        # Query independent GPU/process tables concurrently; serial CLI startup
        # costs can otherwise exceed the declared observation cadence.
        with ThreadPoolExecutor(max_workers=2) as pool:
            gpu_future = pool.submit(run, ['nvidia-smi', '--query-gpu=uuid,clocks.current.sm,temperature.gpu,power.draw,utilization.gpu',
                                          '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=2)
            process_future = pool.submit(run, ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
                                        capture_output=True, text=True, timeout=2)
            gpu, processes = gpu_future.result(), process_future.result()
        if gpu.returncode:
            raise RuntimeError('GPU query failed: '+gpu.stderr.strip())
        result['gpus'] = parse_gpu_csv(gpu.stdout)
        if processes.returncode:
            raise RuntimeError('Compute process query failed: '+processes.stderr.strip())
        result['compute_processes'] = parse_process_csv(processes.stdout)
        result['available'] = all(all(g[key] is not None for key in
                                       ('clock_sm_mhz','temperature_c','power_w','utilization_percent'))
                                  for g in result['gpus'])
        if not result['available']: result['error'] = 'GPU query returned unavailable numeric fields'
    except (OSError, subprocess.TimeoutExpired, ValueError, RuntimeError) as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    result['query_finished_monotonic_s'] = clock()
    result['monotonic_s'] = (started+result['query_finished_monotonic_s'])/2
    return result


def summarize_window(samples, started, finished, worker_pid, gpu_uuid=None):
    """Use every sample in the window; unknown/missing coverage is never clean."""
    # Queries cover intervals, not instantaneous points. Retain overlapping
    # boundary queries for interference/failure evidence even if their midpoint
    # is outside the window; only interior midpoints establish coverage.
    selected = [s for s in samples
                if s.get('query_started_monotonic_s', s['monotonic_s']) <= finished
                and s.get('query_finished_monotonic_s', s['monotonic_s']) >= started]
    interior = [s['monotonic_s'] for s in selected if started <= s['monotonic_s'] <= finished]
    scope = 'worker_gpu_uuid' if gpu_uuid else 'all_nvidia_smi_gpus_conservative'
    observed_gpus, foreign = [], set()
    available = bool(interior) and all(s['available'] for s in selected)
    for sample in selected:
        gpus = [g for g in sample['gpus'] if not gpu_uuid or _uuid(g['gpu_uuid']) == _uuid(gpu_uuid)]
        if not gpus: available = False
        observed_gpus.extend(gpus)
        for process in sample['compute_processes']:
            if (not gpu_uuid or _uuid(process['gpu_uuid']) == _uuid(gpu_uuid)) and process['pid'] != worker_pid:
                foreign.add(process['pid'])
    times = sorted([started, *interior, finished])
    max_gap = max((b-a for a,b in zip(times,times[1:])), default=finished-started)
    if max_gap > MAX_COVERAGE_GAP_S: available = False
    def limits(key):
        values = [g[key] for g in observed_gpus if g.get(key) is not None]
        return (min(values), max(values)) if values else (None, None)
    result = {'available': available, 'foreign_compute_pids': sorted(foreign),
              'gpu_uuids': sorted(set(g['gpu_uuid'] for g in observed_gpus)), 'samples': len(selected),
              'midpoint_samples': len(interior),
              'scope': scope, 'window_start_monotonic_s': started, 'window_end_monotonic_s': finished,
              'sampled_span_s': max(interior)-min(interior) if interior else 0.,
              'maximum_sample_gap_s': max_gap, 'coverage_gap_limit_s': MAX_COVERAGE_GAP_S,
              'window_scope': 'warmup plus measured plan and optional offline encoder preparation',
              'limitation': 'No competing compute process observed does not exclude short-lived activity between samples.'}
    for key in ('clock_sm_mhz','temperature_c','power_w','utilization_percent'):
        result[key+'_min'], result[key+'_max'] = limits(key)
    if not available:
        result['unavailable_reason'] = 'Missing/failed telemetry, UUID mapping absent from samples, or insufficient window coverage'
    return result


def attach_windows(events, samples, worker_pid):
    setup = next((e for e in events if e.get('type') == 'setup'), {})
    starts = {}
    for event in events:
        key = event.get('case_id'), event.get('seed')
        if event.get('type') == 'started': starts[key] = event['monotonic_s']
        elif event.get('type') == 'trial':
            start = starts.get(key, event['monotonic_s'])
            event['gpu_telemetry'] = summarize_window(samples, start, event['monotonic_s'],
                                                      worker_pid, setup.get('gpu_uuid'))
    return events
