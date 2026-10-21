"""Pure statistical/gating tests: no GPU, planner, or hardware access."""
import copy
import math
import unittest

from benchmarks.suite.metrics import bootstrap_median_ci, distribution, summarize


def case():
    return {'id': 'timing', 'expected': {'kind': 'success', 'statuses': ['success']},
            'qualified_seeds': [42],
            'exploratory_statuses': ['success', 'decision_limit', 'no_safe_chunk']}


def records(ratios=(2.,) * 10, scales=None, seed=42):
    result = []
    for repeat, ratio in enumerate(ratios):
        scale = 10 if scales is None else scales[repeat]
        for slot, backend in enumerate(('eager', 'cuda_graph', 'cuda_graph', 'eager')):
            result.append({'case_id': 'timing', 'seed': seed, 'repetition': repeat,
                           'abba_slot': slot, 'backend': backend, 'device': 'cuda',
                           'status': 'success', 'converged': True, 'warmup_exact': True,
                           'graph_active': backend == 'cuda_graph',
                           'signatures': {'reference': 'unchanged'},
                           'metrics': {'frame_count': 10, 'sample_calls': 1,
                                       'endpoint_error_mm': .1, 'endpoint_direction_error_deg': .1},
                           'timing_ms': {'total': scale * (ratio if backend == 'eager' else 1)},
                           'gpu_telemetry': {'available': True, 'foreign_compute_pids': [],
                                             'gpu_uuids': ['GPU-test'], 'samples': 2,
                                             'clock_sm_mhz_min': 1200, 'clock_sm_mhz_max': 1200,
                                             'temperature_c_min': 45, 'temperature_c_max': 46}})
    return result


def group(rows=None, repeats=10, seed=42, reliable=True, scenario=None):
    return summarize(records() if rows is None else rows, [case() if scenario is None else scenario],
                     [seed], repeats, reliable_timing=reliable)['groups'][0]


class DistributionTests(unittest.TestCase):
    def test_linear_quartiles_and_unscaled_median_absolute_deviation(self):
        d = distribution([1, 2, 3, 4])
        self.assertEqual((d['q1'], d['q3'], d['iqr'], d['median'], d['mad']),
                         (1.75, 3.25, 1.5, 2.5, 1.))
        self.assertEqual(distribution([7])['mad'], 0)
        self.assertEqual(distribution([7])['iqr'], 0)
        for name in ('median', 'q1', 'q3', 'iqr', 'mad', 'p95'):
            self.assertIsNone(distribution([])[name])
        self.assertEqual(distribution(range(1, 21))['p95'], 19)
        for values in ([math.nan], [math.inf], [-1]):
            with self.assertRaises(ValueError):
                distribution(values)

    def test_bootstrap_is_seeded_and_requires_ten_whole_blocks(self):
        self.assertIsNone(bootstrap_median_ci([2] * 9))
        self.assertEqual(bootstrap_median_ci([2] * 10), {'lower': 2., 'upper': 2.})
        values = [1 + i / 100 for i in range(10)]
        a, b = bootstrap_median_ci(values), bootstrap_median_ci(values)
        self.assertEqual(a, b)
        self.assertLessEqual(a['lower'], 1.045)
        self.assertGreaterEqual(a['upper'], 1.045)
        self.assertGreaterEqual(a['lower'], min(values))
        self.assertLessEqual(a['upper'], max(values))

    def test_blocks_are_paired_before_ratios_not_pooled_trials(self):
        rows = records([1, 10], [10, 100])
        value = group(rows, repeats=2, reliable=False)
        paired = value['paired_timing']
        self.assertEqual(paired['block_ratios'], [1, 10])
        self.assertEqual(paired['median_ratio'], 5.5)
        self.assertAlmostEqual(value['speedup_ratio'], 505 / 55)
        self.assertIsNone(paired['bootstrap_ci95'])
        rows = records([2])
        for row, time in zip(rows, [10, 5, 15, 30]):
            row['timing_ms']['total'] = time
        self.assertEqual(group(rows, repeats=1)['paired_timing']['block_ratios'], [2])


class ReliableTimingTests(unittest.TestCase):
    def test_supported_only_with_complete_clean_repeated_success(self):
        value = group()
        self.assertEqual(value['timing_status'], 'supported_speedup')
        self.assertEqual(value['speedup_ratio'], 2)
        self.assertEqual(value['paired_timing']['complete_block_count'], 10)
        self.assertEqual(value['timing_reasons'], [])
        self.assertIn('autocorrelation', value['paired_timing']['method_note'])

    def test_incomplete_duplicate_and_low_block_count_do_not_claim(self):
        rows = records()
        for bad in (rows[:-1], rows + [copy.deepcopy(rows[0])]):
            value = group(bad)
            self.assertFalse(value['correctness_passed'])
            self.assertIsNone(value['speedup_ratio'])
            self.assertIsNone(value['paired_timing']['median_ratio'])
        value = group(records([2] * 9), repeats=9)
        self.assertTrue(value['correctness_passed'])
        self.assertEqual(value['timing_status'], 'inconclusive')
        self.assertIsNone(value['speedup_ratio'])
        self.assertEqual(value['paired_timing']['median_ratio'], 2)
        self.assertIsNone(value['paired_timing']['bootstrap_ci95'])

    def test_malformed_execution_positions_fail_without_sort_exception(self):
        for field in ('abba_slot', 'repetition'):
            for invalid in (None, '0', False):
                rows = records(); rows[0][field] = invalid
                value = group(rows)
                self.assertFalse(value['correctness_passed'])
                self.assertIsNone(value['speedup_ratio'])
                self.assertIsNone(value['paired_timing']['bootstrap_ci95'])

    def test_missing_unknown_interference_and_wrong_gpu_withhold(self):
        for change in (
            lambda r: r.pop('gpu_telemetry'),
            lambda r: r['gpu_telemetry'].update(available=False),
            lambda r: r['gpu_telemetry'].update(foreign_compute_pids=[1234]),
            lambda r: r['gpu_telemetry'].pop('foreign_compute_pids'),
            lambda r: r['gpu_telemetry'].update(samples=0),
            lambda r: r['gpu_telemetry'].update(gpu_uuids=[]),
            lambda r: r['gpu_telemetry'].update(gpu_uuids=['GPU-other']),
            lambda r: r['gpu_telemetry'].update(temperature_c_min=math.nan),
            lambda r: r['gpu_telemetry'].update(clock_sm_mhz_max=None),
            lambda r: r['gpu_telemetry'].update(clock_sm_mhz_min=1500),
        ):
            rows = records(); change(rows[-1])
            value = group(rows)
            self.assertTrue(value['correctness_passed'])
            self.assertEqual(value['timing_status'], 'inconclusive')
            self.assertIsNone(value['speedup_ratio'])
            self.assertEqual(value['paired_timing']['median_ratio'], 2)

    def test_mixed_success_failure_cannot_select_fast_successes(self):
        rows = records(seed=43)
        for row in rows[:4]:
            row.update(status='decision_limit', converged=False)
        value = group(rows, seed=43)
        self.assertTrue(value['correctness_passed'])
        self.assertEqual(value['paired_timing']['complete_block_count'], 9)
        self.assertIsNone(value['speedup_ratio'])
        self.assertIsNone(value['paired_timing']['median_ratio'])
        self.assertIsNone(value['paired_timing']['bootstrap_ci95'])
        self.assertTrue(any('all scheduled trials' in reason for reason in value['timing_reasons']))
        rows = records(); rows[0]['signatures']['reference'] = 'changed'
        value = group(rows)
        self.assertFalse(value['correctness_passed'])
        self.assertIsNone(value['paired_timing']['bootstrap_ci95'])
        self.assertIsNone(value['speedup_ratio'])

    def test_invalid_total_timings_fail_closed_without_summary_exception(self):
        for invalid in (math.nan, math.inf, -1, 0, True, None, '1'):
            rows = records()
            rows[0]['timing_ms']['total'] = invalid
            value = group(rows)
            self.assertFalse(value['correctness_passed'])
            self.assertIsNone(value['speedup_ratio'])
            self.assertIsNone(value['paired_timing']['bootstrap_ci95'])
            self.assertTrue(any('total timing' in error for error in value['errors']))
        rows = records(); rows[0].pop('timing_ms')
        self.assertFalse(group(rows)['correctness_passed'])

    def test_interval_at_or_below_one_is_inconclusive(self):
        for ratio in (1, .8):
            value = group(records([ratio] * 10))
            self.assertEqual(value['timing_status'], 'inconclusive')
            self.assertIsNone(value['speedup_ratio'])
            self.assertEqual(value['paired_timing']['median_ratio'], ratio)

    def test_latency_variation_and_temporal_ratio_drift_withhold(self):
        value = group(records([2] * 10, [10, 20] * 5))
        self.assertEqual(value['timing_status'], 'inconclusive')
        self.assertTrue(any('IQR/median' in w for w in value['paired_timing']['variability_warnings']))
        self.assertEqual(value['paired_timing']['drift_fraction'], 0)
        value = group(records([1.2] * 5 + [1.4] * 5))
        self.assertEqual(value['timing_status'], 'inconclusive')
        self.assertGreater(value['paired_timing']['drift_fraction'], .1)
        self.assertTrue(any('ratio drift' in w for w in value['paired_timing']['variability_warnings']))
        self.assertIsNone(value['speedup_ratio'])

    def test_rejections_are_not_applicable(self):
        c = case(); c['expected'] = {'kind': 'reject', 'statuses': ['invalid_start']}
        rows = records()
        for row in rows:
            row.update(status='invalid_start', converged=False)
            row['metrics']['sample_calls'] = 0
        value = group(rows, scenario=c)
        self.assertTrue(value['correctness_passed'])
        self.assertEqual(value['timing_status'], 'not_applicable')
        self.assertIsNone(value['speedup_ratio'])
        self.assertEqual(value['paired_timing']['block_ratios'], [])


if __name__ == '__main__':
    unittest.main()
