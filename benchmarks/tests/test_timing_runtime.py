"""Mocked timing orchestration and GPU observation tests; no GPU or robot IO."""
import io
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.suite.runner import CHILD_THREAD_ENV, TIMING_CASES, main, selected_cases, launch_worker
from benchmarks.suite.telemetry import sample_gpu, summarize_window, attach_windows


def observation(at, pid=123, gpu='GPU-abc', available=True):
    return {'monotonic_s':at, 'available':available,
            'gpus':[{'gpu_uuid':gpu,'clock_sm_mhz':1000.,'temperature_c':40.,'power_w':55.,'utilization_percent':80.}],
            'compute_processes':[{'gpu_uuid':gpu,'pid':pid}]}


class TelemetryTests(unittest.TestCase):
    def test_parse_queries_and_fail_closed(self):
        def run(cmd, **kwargs):
            return SimpleNamespace(returncode=0, stderr='', stdout=
                'GPU-abc, 1000, 40, 55.0, 80\n' if '--query-gpu=' in cmd[1] else 'GPU-abc, 123\n')
        result = sample_gpu(run=run, clock=lambda:1.)
        self.assertTrue(result['available'])
        self.assertEqual(result['compute_processes'][0]['pid'],123)
        for error in (FileNotFoundError('nvidia-smi absent'), subprocess.TimeoutExpired('nvidia-smi',2)):
            with patch('subprocess.run', side_effect=error) as run:
                missing = sample_gpu(run=run)
            self.assertFalse(missing['available']); self.assertIn('error',missing)
        def missing_field(cmd, **kwargs):
            return SimpleNamespace(returncode=0,stderr='',stdout=
                'GPU-abc, N/A, 40, 55, 80\n' if '--query-gpu=' in cmd[1] else '')
        self.assertFalse(sample_gpu(run=missing_field)['available'])

    def test_every_window_sample_and_device_mapping(self):
        samples=[observation(.1),observation(1.,pid=999),observation(2.)]
        value=summarize_window(samples,0.,2.1,123,'abc')
        self.assertTrue(value['available']); self.assertEqual(value['samples'],3)
        self.assertEqual(value['foreign_compute_pids'],[999])
        self.assertEqual(value['scope'],'worker_gpu_uuid')
        samples[1]['compute_processes']=[{'gpu_uuid':'GPU-other','pid':999}]
        self.assertEqual(summarize_window(samples,0.,2.1,123,'GPU-abc')['foreign_compute_pids'],[])
        conservative=summarize_window(samples,0.,2.1,123)
        self.assertEqual(conservative['foreign_compute_pids'],[999])
        self.assertEqual(conservative['scope'],'all_nvidia_smi_gpus_conservative')

    def test_missing_failed_or_sparse_coverage_never_clean(self):
        for samples in ([],[observation(1.,available=False)],[observation(1.),observation(9.)]):
            self.assertFalse(summarize_window(samples,0.,10.,123,'GPU-abc')['available'])
        self.assertFalse(summarize_window([observation(.1)],0.,.2,123,'GPU-unknown')['available'])

    def test_association_uses_started_through_trial(self):
        events=[{'type':'setup','gpu_uuid':'GPU-abc','monotonic_s':0.},
                {'type':'started','case_id':'x','seed':42,'monotonic_s':1.},
                {'type':'warmup_complete','case_id':'x','seed':42,'monotonic_s':2.},
                {'type':'trial','case_id':'x','seed':42,'monotonic_s':3.}]
        attach_windows(events,[observation(.5,pid=999),observation(1.5),observation(2.5,pid=888)],123)
        telemetry=events[-1]['gpu_telemetry']
        self.assertEqual(telemetry['samples'],2)
        self.assertEqual(telemetry['foreign_compute_pids'],[888])
        self.assertIn('warmup',telemetry['window_scope'])

    def test_boundary_overlap_cannot_hide_interference_or_imply_coverage(self):
        before=observation(.8,pid=999)
        before.update(query_started_monotonic_s=.4,query_finished_monotonic_s=1.2)
        after=observation(3.2,pid=888)
        after.update(query_started_monotonic_s=2.8,query_finished_monotonic_s=3.6)
        value=summarize_window([before,observation(2.),after],1.,3.,123,'GPU-abc')
        self.assertTrue(value['available'])
        self.assertEqual(value['samples'],3)
        self.assertEqual(value['midpoint_samples'],1)
        self.assertEqual(value['foreign_compute_pids'],[888,999])
        boundary_only=summarize_window([before,after],1.,3.,123,'GPU-abc')
        self.assertFalse(boundary_only['available'])
        self.assertEqual(boundary_only['midpoint_samples'],0)
        after['available']=False
        self.assertFalse(summarize_window([before,observation(2.),after],1.,3.,123,'GPU-abc')['available'])


class TimingRunnerTests(unittest.TestCase):
    def test_case_catalog_defaults_selection_and_invalid_requests(self):
        self.assertEqual([c['id'] for c in selected_cases('timing')],TIMING_CASES)
        self.assertEqual([c['id'] for c in selected_cases('full',['recent_live_start'])],['recent_live_start'])
        for names in (['bad'],['recent_live_start','recent_live_start']):
            with self.assertRaises(ValueError): selected_cases('timing',names)
        with patch('benchmarks.suite.runner.launch_worker') as launch, patch('sys.stderr',new_callable=io.StringIO):
            for args in (['--mode','timing'],['--cases','unknown'],['--cases','recent_live_start','recent_live_start']):
                with self.assertRaises(SystemExit): main(args)
            launch.assert_not_called()

    def test_timing_defaults_and_global_failure_downgrades_claim(self):
        fake_summary={'passed':True,'failures':[], 'groups':[{'speedup_ratio':2.,'timing_status':'supported_speedup','timing_reasons':[]}]}
        with tempfile.TemporaryDirectory() as tmp, \
             patch('benchmarks.suite.runner.fcntl.flock'), \
             patch('benchmarks.suite.runner.source_inventory',side_effect=[{}, {'changed':'hash'}]), \
             patch('benchmarks.suite.runner.frozen_inventory',return_value={}), \
             patch('benchmarks.suite.runner.gpu_snapshot',return_value={}), \
             patch('benchmarks.suite.runner.launch_worker',return_value={'returncode':2,'timed_out':False,'pid':1,'events':[{'type':'fatal','error':'mock failure'}]}), \
             patch('benchmarks.suite.runner.summarize',return_value=fake_summary) as summarize, \
             patch('benchmarks.suite.runner.write_report') as write:
            self.assertEqual(main(['--mode','timing','--allow-long-run','--output',str(Path(tmp)/'run')]),1)
            report=write.call_args.args[1]
            self.assertEqual(report['inputs']['seeds'],[42])
            self.assertEqual(report['inputs']['repeats'],10)
            self.assertEqual(len(report['measurement_policy']['order']),40)
            self.assertTrue(summarize.call_args.kwargs['reliable_timing'])
            group=report['summary']['groups'][0]
            self.assertIsNone(group['speedup_ratio']); self.assertEqual(group['timing_status'],'inconclusive')
            self.assertTrue(group['timing_reasons'])
            self.assertIn('source/config changed during benchmark',report['summary']['failures'])

    def test_child_environment_enforced_without_touching_parent(self):
        child=SimpleNamespace(pid=123,returncode=0,poll=lambda:0,wait=lambda:0)
        with tempfile.TemporaryDirectory() as tmp, \
             patch('benchmarks.suite.runner.subprocess.Popen',return_value=child) as popen, \
             patch.dict('os.environ',{'OMP_NUM_THREADS':'99'}):
            result=launch_worker(Path(tmp),0,'eager',10)
            self.assertEqual(result['returncode'],0)
            for key,value in CHILD_THREAD_ENV.items(): self.assertEqual(popen.call_args.kwargs['env'][key],value)
            import os
            self.assertEqual(os.environ['OMP_NUM_THREADS'],'99')
            self.assertEqual(result['gpu_telemetry_samples'],[])


if __name__=='__main__': unittest.main()
