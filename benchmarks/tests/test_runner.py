"""Fast fail-closed orchestration/gate tests; no CUDA or robot connections."""
import copy
import json
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.suite.metrics import distribution, digest, stable_result, summarize
from benchmarks.suite.report import write_report
from benchmarks.suite.runner import create_output, main


def case():
    return {'id': 'test', 'expected': {'kind': 'success', 'statuses': ['success']},
            'qualified_seeds': [42], 'exploratory_statuses': ['success', 'decision_limit', 'no_safe_chunk']}


def rows(seed=42):
    return [{'case_id': 'test', 'seed': seed, 'backend': backend,
             'repetition': 0, 'abba_slot': i, 'execution_position': i,
             'status': 'success', 'converged': True, 'warmup_exact': True,
             'device': 'cuda', 'graph_active': backend == 'cuda_graph',
             'signatures': {'full_reference': 'same', 'sampled_predictions': 'same',
                            'rng_after': 'same', 'encoder_profile': None},
             'metrics': {'sample_calls': 1, 'frame_count': 10,
                         'endpoint_error_mm': .1, 'endpoint_direction_error_deg': .1},
             'timing_ms': {'total': 20 if backend == 'eager' else 10}}
            for i, backend in enumerate(('eager','cuda_graph','cuda_graph','eager'))]


class GateTests(unittest.TestCase):
    def test_valid_abba(self):
        value = summarize(rows(), [case()], [42], 1)
        self.assertTrue(value['passed'])
        self.assertEqual(value['groups'][0]['speedup_ratio'], 2)

    def test_partial_or_duplicate_positions_fail(self):
        r = rows()
        self.assertFalse(summarize(r[:-1], [case()], [42], 1)['passed'])
        r[3]['abba_slot'] = 0
        self.assertFalse(summarize(r, [case()], [42], 1)['passed'])

    def test_mismatch_is_not_hidden_by_faster_time(self):
        for component in ('full_reference','sampled_predictions','rng_after','encoder_profile'):
            r = rows(); r[1]['signatures'][component] = 'changed'
            result = summarize(r, [case()], [42], 1)
            self.assertFalse(result['passed'])
            self.assertIsNone(result['groups'][0]['speedup_ratio'])

    def test_bad_success_or_worker_error_fails(self):
        for update in ({'error': 'oops'}, {'status': 'no_safe_chunk', 'converged': False},
                       {'warmup_exact': False}, {'graph_active': False}):
            r = rows(); r[1].update(update)
            self.assertFalse(summarize(r, [case()], [42], 1)['passed'])
        r = rows(); r[1]['metrics']['endpoint_error_mm'] = 1.01
        self.assertFalse(summarize(r, [case()], [42], 1)['passed'])

    def test_rejection_must_be_explicit_and_not_timeout(self):
        c = case(); c['expected'] = {'kind': 'reject', 'statuses': ['invalid_start']}
        r = rows()
        self.assertFalse(summarize(r, [c], [42], 1)['passed'])
        for row in r:
            row.update(status='invalid_start', converged=False)
            row['metrics']['sample_calls'] = 0
        result = summarize(r, [c], [42], 1)
        self.assertTrue(result['passed'])
        self.assertIsNone(result['groups'][0]['speedup_ratio'])
        r[0]['status'] = 'time_limit'
        self.assertFalse(summarize(r, [c], [42], 1)['passed'])

    def test_unqualified_seed_is_exploratory(self):
        r = rows(43)
        for row in r: row.update(converged=False, status='decision_limit')
        result = summarize(r, [case()], [43], 1)
        self.assertTrue(result['passed'])
        self.assertEqual(result['groups'][0]['expected']['kind'], 'allowed')
        self.assertIsNone(result['groups'][0]['speedup_ratio'])

    def test_exploratory_status_flag_inconsistency_fails(self):
        for status, converged in (('success', False), ('decision_limit', True)):
            r = rows(43)
            for row in r: row.update(status=status, converged=converged)
            result = summarize(r, [case()], [43], 1)
            self.assertFalse(result['passed'])
            self.assertIsNone(result['groups'][0]['speedup_ratio'])
            self.assertTrue(any('inconsistent' in e for e in result['failures']))

    def test_cpu_never_claims_speedup(self):
        r = rows()
        for row in r: row['device'] = 'cpu'; row['graph_active'] = False
        result = summarize(r, [case()], [42], 1)
        self.assertFalse(result['passed'])
        self.assertIsNone(result['groups'][0]['speedup_ratio'])

    def test_quantiles_and_hashes(self):
        self.assertIsNone(distribution([1,2])['p95'])
        self.assertEqual(distribution(range(1,21))['p95'], 19)
        with self.assertRaises(ValueError): distribution([float('nan')])
        a = {'frames':[{'q':[1,2]}], 'elapsed_ms': 123, 'v2_prepare_timing_ms': {'total': 8}}
        b = copy.deepcopy(a); b['elapsed_ms'] = 9
        self.assertEqual(digest(stable_result(a)), digest(stable_result(b)))
        b['frames'][0]['q'][1] = 3
        self.assertNotEqual(digest(stable_result(a)), digest(stable_result(b)))


class ArtifactTests(unittest.TestCase):
    def test_nonfinite_timeout_rejected_before_worker_start(self):
        with patch('benchmarks.suite.runner.launch_worker') as launch, patch('sys.stderr', new_callable=io.StringIO):
            for value in ('nan', 'inf', '-inf'):
                with self.assertRaises(SystemExit) as exc:
                    main(['--timeout='+value])
                self.assertEqual(exc.exception.code, 2)
            launch.assert_not_called()

    def test_output_protects_frozen_existing_and_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'benchmarks/fixed-snapshot').mkdir(parents=True); (root/'backups').mkdir()
            for path in (root/'backups/new', root/'benchmarks/fixed-snapshot/new'):
                with self.assertRaises(ValueError): create_output(path, root)
            (root/'alias').symlink_to(root/'backups', target_is_directory=True)
            with self.assertRaises(ValueError): create_output(root/'alias/new', root)
            output = create_output(root/'output/run', root)
            with self.assertRaises(FileExistsError): create_output(output, root)

    def test_html_escapes_embedded_data(self):
        report = {'summary': summarize(rows(), [case()], [42], 1),
                  'records': rows(),
                  'untrusted': '<script>alert("x")</script>'}
        with tempfile.TemporaryDirectory() as tmp:
            write_report(tmp, report)
            text = (Path(tmp)/'report.html').read_text()
            self.assertNotIn('<script>', text)
            self.assertIn('&lt;script&gt;', text)
            self.assertIn('success: 2', text)
            self.assertIn('Inference ms', text)
            self.assertIn('Stage timing and reference quality', text)
            self.assertIn('sampled finite-difference speed/acceleration maxima', text)
            self.assertIn('Sampled speed rad/s', text)
            self.assertEqual(json.loads((Path(tmp)/'report.json').read_text()), report)

    def test_runner_incomplete_fails_and_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch('benchmarks.suite.runner.fcntl.flock'), \
             patch('benchmarks.suite.scenarios.load_cases', return_value=[case()]), \
             patch('benchmarks.suite.runner.source_inventory', return_value={}), \
             patch('benchmarks.suite.runner.frozen_inventory', return_value={}), \
             patch('benchmarks.suite.runner.gpu_snapshot', return_value={}), \
             patch('benchmarks.suite.runner.launch_worker', return_value={
                 'returncode':2,'timed_out':False,'pid':1,
                 'events':[{'type':'fatal','error':'CUDA unavailable'}]}):
            output = Path(tmp)/'run'
            self.assertEqual(main(['--output',str(output)]), 1)
            report = json.loads((output/'report.json').read_text())
            self.assertFalse(report['summary']['passed'])
            self.assertTrue(any('CUDA unavailable' in x for x in report['summary']['failures']))

    def test_offline_guard_denies_network_and_serial(self):
        script = '''from benchmarks.suite.worker import offline_guard
import sys
offline_guard()
for event,args in [('socket.connect',(None,('127.0.0.1',9000))),('open',('/dev/ttyUSB0','r',0))]:
    try: sys.audit(event,*args)
    except RuntimeError: pass
    else: raise AssertionError(event)
'''
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__': unittest.main()
