"""Pure aggregation and strict correctness gates (no simulation imports)."""
import hashlib
import json
import math
import random
import statistics
from .scenarios import expected_for_seed


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def stable_result(value):
    """Remove only known wall-clock measurements; preserve every route field."""
    if isinstance(value, dict):
        return {k: stable_result(v) for k, v in value.items()
                if k not in ('elapsed_ms', 'v2_prepare_timing_ms')}
    if isinstance(value, (list, tuple)):
        return [stable_result(v) for v in value]
    if isinstance(value, set):
        return sorted(value)
    return value


def linear_quantile(sorted_values, probability):
    """Linear interpolation at (n-1)*p, with already sorted finite values."""
    if not sorted_values or not 0 <= probability <= 1:
        raise ValueError('Quantile needs observations and a probability in [0, 1]')
    index = (len(sorted_values)-1)*probability
    lower = math.floor(index)
    upper = math.ceil(index)
    fraction = index-lower
    return sorted_values[lower]*(1-fraction)+sorted_values[upper]*fraction


def distribution(values):
    values = sorted(float(v) for v in values)
    if not values:
        return {'n': 0, 'median': None, 'min': None, 'max': None, 'spread': None,
                'q1': None, 'q3': None, 'iqr': None, 'mad': None, 'p95': None,
                'p95_note': 'insufficient samples (requires at least 20)'}
    if not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError('Timings must be finite and nonnegative')
    median = statistics.median(values)
    q1, q3 = linear_quantile(values, .25), linear_quantile(values, .75)
    return {'n': len(values), 'median': median, 'q1': q1, 'q3': q3,
            'iqr': q3-q1, 'mad': statistics.median(abs(v-median) for v in values),
            'min': values[0], 'max': values[-1],
            'spread': values[-1]-values[0],
            'p95': values[math.ceil(.95*len(values))-1] if len(values) >= 20 else None,
            'p95_note': 'nearest-rank empirical p95' if len(values) >= 20 else
                        'insufficient samples (requires at least 20)'}


def bootstrap_median_ci(block_ratios):
    """Approximate percentile CI; resample complete blocks, never their trials."""
    if len(block_ratios) < 10:
        return None
    if not all(math.isfinite(v) and v > 0 for v in block_ratios):
        raise ValueError('Block ratios must be finite and positive')
    rng = random.Random(20261007)
    medians = sorted(statistics.median(rng.choices(block_ratios, k=len(block_ratios)))
                     for _ in range(10000))
    return {'lower': linear_quantile(medians, .025),
            'upper': linear_quantile(medians, .975)}


def paired_timing(rows, complete_successful_comparison, timings):
    blocks = []
    repetitions = sorted({r.get('repetition') for r in rows
                          if type(r.get('repetition')) is int})
    for repetition in repetitions:
        block = [r for r in rows if r.get('repetition') == repetition]
        slots = [r.get('abba_slot') for r in block]
        if (len(block) != 4 or any(type(slot) is not int for slot in slots)
                or sorted(slots) != [0, 1, 2, 3]):
            continue
        block.sort(key=lambda r: r['abba_slot'])
        if any(r.get('backend') != ('eager' if i in (0, 3) else 'cuda_graph')
               or r.get('status') != 'success' or r.get('converged') is not True
               or r.get('error') for i, r in enumerate(block)):
            continue
        totals = [r.get('timing_ms', {}).get('total') for r in block]
        if any(isinstance(t, bool) or not isinstance(t, (int, float))
               or not math.isfinite(t) or t <= 0 for t in totals):
            continue
        eager = statistics.mean([totals[0], totals[3]])
        graph = statistics.mean([totals[1], totals[2]])
        blocks.append({'repetition': repetition, 'eager_mean_ms': eager,
                       'cuda_graph_mean_ms': graph, 'ratio': eager/graph})
    ratios = [b['ratio'] for b in blocks]
    warnings = []
    for backend, values in timings.items():
        if values['median'] is not None and values['median'] > 0 and values['iqr']/values['median'] > .10:
            warnings.append(backend+' warm latency IQR/median exceeds 10%')
    drift = None
    if len(ratios) >= 2:
        half = len(ratios)//2
        first, second = statistics.median(ratios[:half]), statistics.median(ratios[-half:])
        drift = abs(second/first-1)
        if drift > .10:
            warnings.append('first-half versus second-half median block ratio drift exceeds 10%')
    return {'complete_block_count': len(blocks), 'blocks': blocks, 'block_ratios': ratios,
            'median_ratio': statistics.median(ratios) if ratios and complete_successful_comparison else None,
            'bootstrap_ci95': bootstrap_median_ci(ratios) if complete_successful_comparison else None,
            'bootstrap_resamples': 10000, 'bootstrap_seed': 20261007,
            'bootstrap_minimum_blocks': 10, 'variability_warnings': warnings,
            'drift_fraction': drift,
            'method_note': 'One ratio per whole ABBA block: mean eager slots 0,3 / mean graph slots 1,2. '
                           'Median ratio with deterministic 10000-resample percentile bootstrap 95% CI '
                           '(linear interpolation), requiring at least 10 blocks. Approximate/exploratory '
                           'interval assumes exchangeable blocks; between-block autocorrelation may '
                           'understate uncertainty. Trials within a block are not independent observations.'}


def telemetry_reasons(rows):
    """Unknown telemetry is not evidence that a GPU measurement was isolated."""
    reasons = set()
    identities = set()
    for row in rows:
        value = row.get('gpu_telemetry')
        if not isinstance(value, dict) or value.get('available') is not True:
            reasons.add('GPU telemetry unavailable or missing on one or more trials')
            continue
        if type(value.get('samples')) is not int or value['samples'] <= 0:
            reasons.add('GPU telemetry contains no valid samples on one or more trials')
        pids = value.get('foreign_compute_pids')
        if not isinstance(pids, list):
            reasons.add('GPU foreign-process telemetry missing')
        elif pids:
            reasons.add('foreign compute processes observed on the measured GPU')
        uuids = value.get('gpu_uuids')
        if not isinstance(uuids, list) or not uuids or any(not isinstance(u, str) or not u for u in uuids):
            reasons.add('GPU identity telemetry missing')
        else:
            identities.add(tuple(sorted(uuids)))
        for field in ('clock_sm_mhz', 'temperature_c'):
            bounds = [value.get(field+'_'+suffix) for suffix in ('min', 'max')]
            if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                    for v in bounds) or bounds[0] > bounds[1]):
                reasons.add(field+' telemetry missing or invalid')
    if len(identities) > 1:
        reasons.add('GPU identity differs across trials')
    return sorted(reasons)


def outcome_errors(record, case):
    errors = []
    if record.get('error'):
        return ['worker error: '+str(record['error'])]
    timing = record.get('timing_ms')
    total = timing.get('total') if isinstance(timing, dict) else None
    if (isinstance(total, bool) or not isinstance(total, (int, float))
            or not math.isfinite(total) or total <= 0):
        errors.append('total timing must be a finite positive number')
    expected = expected_for_seed(case, record['seed'])
    status = record.get('status')
    converged = record.get('converged') is True
    if (status == 'success') != converged:
        errors.append('status and converged flag are inconsistent')
    if status in ('time_limit', 'timeout'):
        errors.append('timeout is never an accepted rejection')
    if expected['kind'] == 'success' and (not converged or status != 'success'):
        errors.append('required successful convergence missing')
    if expected['kind'] == 'reject' and (converged or status == 'success'):
        errors.append('unsafe/unreachable case unexpectedly accepted')
    if expected.get('statuses') and status not in expected['statuses']:
        errors.append('unexpected status '+str(status))
    if converged:
        for key, limit in (('endpoint_error_mm', 1.), ('endpoint_direction_error_deg', 1.)):
            value = record.get('metrics', {}).get(key)
            if value is None or not math.isfinite(value) or value > limit:
                errors.append(key+' fails strict planner tolerance')
        if not record.get('metrics', {}).get('frame_count', 0):
            errors.append('converged plan has no full reference')
    if not record.get('warmup_exact', False):
        errors.append('warmup/measured deterministic signatures differ')
    return errors


def summarize(records, cases, seeds, repeats, *, reliable_timing=False):
    lookup = {c['id']: c for c in cases}
    failures = []
    groups = []
    for case in cases:
        for seed in seeds:
            rows = [r for r in records if r.get('case_id') == case['id'] and r.get('seed') == seed]
            errors = []
            by_backend = {b: [r for r in rows if r.get('backend') == b]
                          for b in ('eager', 'cuda_graph')}
            actual_positions = sorted(((r.get('repetition'), r.get('abba_slot')) for r in rows), key=repr)
            required_positions = sorted(((repeat, slot) for repeat in range(repeats) for slot in range(4)), key=repr)
            if (any(type(r.get('repetition')) is not int or type(r.get('abba_slot')) is not int for r in rows)
                    or actual_positions != required_positions):
                errors.append('missing or duplicate ABBA execution positions')
            if any(r.get('backend') != ('eager' if r.get('abba_slot') in (0, 3) else 'cuda_graph') for r in rows):
                errors.append('backend does not match declared ABBA order')
            for backend, subset in by_backend.items():
                if len(subset) != 2*repeats:
                    errors.append(f'{backend}: incomplete trials {len(subset)}/{2*repeats}')
                for row in subset:
                    errors.extend(backend+': '+e for e in outcome_errors(row, case))
            signatures = [r.get('signatures') for r in rows]
            if not signatures or any(s is None for s in signatures) or any(s != signatures[0] for s in signatures):
                errors.append('exact full-reference / prediction / RNG / encoder signatures differ or missing')
            comparability = bool(rows) and all(r.get('device') == 'cuda' for r in rows)
            # A rejected pre-inference input needs no graph replay; it is excluded from speedup.
            graph_rows = by_backend['cuda_graph']
            sampled_graph = [r for r in graph_rows if r.get('metrics', {}).get('sample_calls', 0)]
            graph_active = bool(sampled_graph) and all(r.get('graph_active') for r in sampled_graph)
            if sampled_graph and not graph_active:
                errors.append('requested CUDA graph backend fell back or was not active')
            timings = {backend: distribution([r['timing_ms']['total'] for r in subset
                       if not r.get('error') and r.get('converged') and r.get('status') == 'success' and not outcome_errors(r, case)])
                       for backend, subset in by_backend.items()}
            speedup = None
            all_successful = bool(rows) and all(r.get('status') == 'success' and r.get('converged') is True for r in rows)
            if not errors and comparability and graph_active and all_successful and all(t['n'] for t in timings.values()):
                a, b = (timings[k]['median'] for k in ('eager', 'cuda_graph'))
                speedup = a/b if b > 0 else None
            paired = paired_timing(rows, not errors and all_successful, timings)
            expected = expected_for_seed(case, seed)
            timing_status = 'descriptive' if speedup is not None else 'inconclusive'
            timing_reasons = []
            if expected['kind'] == 'reject':
                timing_status = 'not_applicable'
                timing_reasons.append('expected rejection is excluded from speedup claims')
                speedup = None
            elif reliable_timing:
                if errors:
                    timing_reasons.append('correctness or complete ABBA pairing failed')
                if not comparability or not graph_active:
                    timing_reasons.append('CUDA backends are not comparable or graph replay was inactive')
                if not all_successful:
                    timing_reasons.append('all scheduled trials must succeed; selected-subset speedups are prohibited')
                if paired['complete_block_count'] < 10:
                    timing_reasons.append('at least 10 complete successful ABBA blocks are required')
                timing_reasons.extend(telemetry_reasons(rows))
                timing_reasons.extend(paired['variability_warnings'])
                ci = paired['bootstrap_ci95']
                if ci is None or ci['lower'] <= 1:
                    timing_reasons.append('bootstrap 95% interval does not establish a ratio above 1')
                timing_status = 'inconclusive' if timing_reasons else 'supported_speedup'
                speedup = paired['median_ratio'] if timing_status == 'supported_speedup' else None
            group = {'case_id': case['id'], 'seed': seed, 'expected': expected_for_seed(case, seed),
                     'correctness_passed': not errors, 'errors': sorted(set(errors)),
                     'comparable': comparability and graph_active,
                     'warm_success_latency_ms': timings, 'speedup_ratio': speedup,
                     'paired_timing': paired, 'timing_status': timing_status,
                     'timing_reasons': sorted(set(timing_reasons)),
                     'success_count': {b: sum(r.get('converged') is True for r in rs) for b,rs in by_backend.items()},
                     'status_counts': {b: {s: sum(r.get('status', 'error') == s for r in rs)
                                          for s in sorted(set(r.get('status', 'error') for r in rs))}
                                       for b,rs in by_backend.items()}}
            groups.append(group)
            failures.extend(f"{case['id']} seed {seed}: {e}" for e in group['errors'])
    unexpected = [r for r in records if r.get('case_id') not in lookup or r.get('seed') not in seeds]
    if unexpected:
        failures.append('unexpected case/seed records')
    return {'passed': not failures, 'failures': failures, 'groups': groups,
            'reliable_timing': reliable_timing,
            'total_records': len(records), 'expected_records': len(cases)*len(seeds)*4*repeats}
