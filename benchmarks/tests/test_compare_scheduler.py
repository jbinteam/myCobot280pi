"""Scheduler comparison gates/orchestration; no GPU execution."""
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from benchmarks.suite import compare_scheduler as compare
from benchmarks.tests.test_timing_metrics import case as timing_case, records as timing_records


def case():
    return {**timing_case(), 'request': {'seed': 42, 'samples': 4}}


def records(repeats=10):
    values = timing_records([2] * repeats)
    for row in values:
        mode = 'cache_off' if row['backend'] == 'eager' else 'cache_on'
        row.update(backend=mode, neural_backend='cuda_graph', graph_active=True,
                   requested_cache_mode=mode, scheduler_cache_enabled=mode == 'cache_on',
                   scheduler_cache_hits=32 if mode == 'cache_on' else 0,
                   scheduler_cache_misses=0, child_cpu_affinity=[4], requested_cpu_affinity=[4],
                   request=case()['request'], signatures={'full_reference': 'reference',
                   'sampled_predictions': 'predictions', 'rng_after': 'rng', 'encoder_profile': None})
    return values


def summary(rows=None, repeats=10, **kwargs):
    return compare.summarize_comparison(records(repeats) if rows is None else rows,
                                         [case()], [42], repeats, **kwargs)


def reference():
    rows = records(1)
    for row in rows:
        row['backend'] = 'cuda_graph'
    return {'summary': {'passed': True}, 'records': rows,
            'inputs': {'expected_checkpoint_sha256': 'model', 'expected_urdf_sha256': 'urdf'},
            'source_sha256': {'simulation/diffusion/policy.py': 'old', 'simulation/kinematics.py': 'same'}}


class ComparisonGateTests(unittest.TestCase):
    def test_graph_is_constant_and_export_labels_are_cache_modes(self):
        value = summary()
        self.assertTrue(value['passed'])
        group = value['groups'][0]
        self.assertEqual(group['timing_status'], 'supported_speedup')
        self.assertEqual(group['speedup_ratio'], 2)
        self.assertEqual(set(group['warm_success_latency_ms']), {'cache_off', 'cache_on'})
        self.assertIn('cache_off_mean_ms', group['paired_timing']['blocks'][0])
        self.assertNotIn('eager', json.dumps(group))
        self.assertNotIn('cuda_graph', json.dumps(group))
        self.assertEqual(value['neural_backend'], 'cuda_graph')

    def test_wrong_neural_backend_flags_counters_or_affinity_fail_closed(self):
        for mutation in (
            {'neural_backend': 'eager'}, {'scheduler_cache_enabled': True},
            {'scheduler_cache_hits': 1}, {'scheduler_cache_hits': None},
            {'scheduler_cache_misses': -1}, {'requested_cache_mode': 'cache_on'},
            {'graph_active': False}, {'child_cpu_affinity': [6]},
        ):
            rows = records(); rows[0].update(mutation)
            with self.subTest(mutation=mutation):
                value = summary(rows, affinity=[4])
                self.assertFalse(value['passed'])
                self.assertIsNone(value['groups'][0]['speedup_ratio'])
        rows = records(); rows[1]['scheduler_cache_hits'] = 0
        self.assertFalse(summary(rows)['passed'])

    def test_incomplete_mismatched_outputs_and_reference_fail(self):
        self.assertFalse(summary(records()[:-1])['passed'])
        rows = records(); rows[1]['signatures']['rng_after'] = 'changed'
        self.assertFalse(summary(rows)['passed'])
        self.assertFalse(summary(expected={'timing:42': {'wrong': 'signature'}})['passed'])
        self.assertFalse(summary(expected={})['passed'])
        value = summary(repeats=1)
        self.assertTrue(value['passed'])
        self.assertEqual(value['groups'][0]['timing_status'], 'inconclusive')
        self.assertIsNone(value['groups'][0]['speedup_ratio'])

    def test_source_failure_removes_supported_claims(self):
        value = summary()
        compare.invalidate_summary(value, ['source/config changed during comparison'])
        self.assertFalse(value['passed'])
        self.assertIsNone(value['groups'][0]['speedup_ratio'])
        self.assertEqual(value['groups'][0]['timing_status'], 'inconclusive')

    def test_historical_reference_allows_only_intended_patch(self):
        sources = {'simulation/diffusion/policy.py': 'new', 'simulation/kinematics.py': 'same',
                   'simulation/diffusion/timestep_cache.py': 'newcache',
                   'simulation/diffusion/tests/test_timestep_cache.py': 'newtests'}
        manifest = {'checkpoint_sha256': 'model', 'urdf_sha256': 'urdf'}
        expected, changes = compare.reference_signatures(reference(), [case()], [42], sources, manifest)
        self.assertEqual(expected['timing:42'], records(1)[0]['signatures'])
        self.assertEqual(set(changes), {'simulation/diffusion/policy.py', 'simulation/diffusion/timestep_cache.py',
                                        'simulation/diffusion/tests/test_timestep_cache.py'})
        for mutate in (
            lambda r: r['inputs'].update(expected_checkpoint_sha256='wrong'),
            lambda r: r['records'][0]['request'].update(samples=8),
            lambda r: r['records'][0]['signatures'].update(rng_after='wrong'),
            lambda r: r['summary'].update(passed=False),
        ):
            prior = reference(); mutate(prior)
            with self.assertRaises(ValueError):
                compare.reference_signatures(prior, [case()], [42], sources, manifest)
        with self.assertRaises(ValueError):
            compare.reference_signatures(reference(), [case()], [42],
                                         {**sources, 'simulation/kinematics.py': 'changed'}, manifest)


class WorkerAndCliTests(unittest.TestCase):
    def test_wrapper_observes_actual_hit_delta_and_restores_binding(self):
        from benchmarks.suite import worker
        policy = types.SimpleNamespace(scheduler_cache=True, _scheduler_cache_hits=100, _scheduler_cache_misses=4)
        service = types.SimpleNamespace(_policy=policy)
        captured = []
        def run(case, seed, backend, service, torch):
            policy._scheduler_cache_hits += 63
            policy._scheduler_cache_misses += 2
            return {'backend': backend}
        def main():
            captured.append(worker.run_case({}, 42, 'cuda_graph', service, None))
            return 0
        with patch.object(worker, 'run_case', run), patch.object(worker, 'main', main), \
                patch.dict(os.environ, {}, clear=False), patch('os.sched_setaffinity') as affinity, \
                patch('os.sched_getaffinity', return_value={4}):
            self.assertEqual(compare.worker_main('input', 'output', 'cache_on', [4]), 0)
            self.assertIs(worker.run_case, run)
            affinity.assert_called_once_with(0, {4})
        self.assertEqual(captured[0]['scheduler_cache_hits'], 63)
        self.assertEqual(captured[0]['scheduler_cache_misses'], 2)
        self.assertEqual(captured[0]['child_cpu_affinity'], [4])
        self.assertIs(captured[0]['scheduler_cache_enabled'], True)

    def test_cli_rejects_invalid_args_and_missing_optin_before_gpu(self):
        with patch.object(compare, 'launch_worker') as launch, patch('sys.stderr', new_callable=io.StringIO):
            for argv in ([], ['--repeats', '0'], ['--repeats', '1', '--timeout', 'nan'],
                         ['--repeats', '1', '--seeds', '42', '42'], ['--cases', 'unknown']):
                with self.assertRaises(SystemExit):
                    compare.main(argv)
            launch.assert_not_called()

    def test_interrupted_incomplete_or_changed_source_writes_failed_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'benchmarks').mkdir()
            def launch(directory, index, mode, timeout, affinity):
                rows = records(1)
                row = copy.deepcopy(rows[index]); row['backend'] = 'cuda_graph'; row['type'] = 'trial'
                return {'events': [row], 'pid': 1, 'returncode': 0, 'timed_out': False}
            with patch.object(compare, 'ROOT', root), patch.object(compare, 'select_cases', return_value=[case()]), \
                    patch.object(compare, 'source_inventory', side_effect=[{'a': 'old'}, {'a': 'changed'}]), \
                    patch.object(compare, 'frozen_inventory', return_value={'frozen': 'same'}), \
                    patch.object(compare, 'launch_worker', side_effect=launch), \
                    patch.object(compare, 'load_manifest', return_value={'checkpoint_sha256': 'model', 'urdf_sha256': 'urdf'}):
                result = compare.main(['--repeats', '1', '--output', str(root/'run')])
            self.assertEqual(result, 1)
            report = json.loads((root/'run/report.json').read_text())
            self.assertFalse(report['summary']['passed'])
            self.assertIn('source/config changed during comparison', report['summary']['failures'])
            self.assertIsNone(report['summary']['groups'][0]['speedup_ratio'])
            self.assertEqual({row['neural_backend'] for row in report['records']}, {'cuda_graph'})
            self.assertEqual({row['backend'] for row in report['records']}, {'cache_off', 'cache_on'})

    def test_html_escapes_data_and_names_constant_neural_backend(self):
        value = summary(repeats=1)
        value['groups'][0]['case_id'] = '<script>alert(1)</script>'
        with tempfile.TemporaryDirectory() as tmp:
            compare.write_report(tmp, {'summary': value})
            page = (Path(tmp)/'report.html').read_text()
        self.assertNotIn('<script>', page)
        self.assertIn('&lt;script&gt;', page)
        self.assertIn('CUDA graph neural inference held constant', page)
        self.assertNotIn('same_source_eager', page)


if __name__ == '__main__':
    unittest.main()
