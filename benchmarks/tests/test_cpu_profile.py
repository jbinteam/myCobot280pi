"""Fast profiler scoping, exactness, isolation and failure-path tests; no GPU."""
import copy
import io
import json
from pathlib import Path
import pstats
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from benchmarks.suite.profile_cpu import (run_profiled_case, reference_signatures,
    diagnostic_errors, launch_diagnostic, main)


def fixture():
    case={'id':'test','request':{'seed':42},'expected':{'kind':'success','statuses':['success']},
          'qualified_seeds':[42],'exploratory_statuses':['success','decision_limit','no_safe_chunk']}
    row={'case_id':'test','seed':42,'backend':'cuda_graph','request':{'seed':42},
         'status':'success','converged':True,'warmup_exact':True,'device':'cuda','graph_active':True,
         'timing_ms':{'total':10.},
         'metrics':{'frame_count':10,'sample_calls':1,'endpoint_error_mm':.1,'endpoint_direction_error_deg':.1},
         'signatures':{'full_reference':'x','sampled_predictions':'y','rng_after':'z','encoder_profile':None}}
    sources={'simulation/planner.py':'abc','webui/example.py':'def','benchmarks/suite/new.py':'new'}
    manifest={'checkpoint_sha256':'model','urdf_sha256':'urdf'}
    reference={'summary':{'passed':True},'records':[row],
               'inputs':{'expected_checkpoint_sha256':'model','expected_urdf_sha256':'urdf'},
               'source_sha256':{**sources,'benchmarks/suite/new.py':'old'}}
    return case,row,sources,manifest,reference


class ProfileScopeTests(unittest.TestCase):
    def test_only_service_plan_is_profiled_and_restored(self):
        def planner_helper(): return sum(range(20))
        def encoder_helper(): return sum(range(10))
        original=planner_helper
        service=SimpleNamespace(plan=original)
        def run(case,seed,backend,service,torch):
            result=service.plan()
            encoder_helper()  # Runs after service.plan; must not be in pstats.
            return result
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'test.pstats'
            result=run_profiled_case({},42,'cuda_graph',service,None,path,run=run)
            self.assertEqual(result,sum(range(20)))
            self.assertIs(service.plan,original)
            names=[key[2] for key in pstats.Stats(str(path)).stats]
            self.assertIn('planner_helper',names)
            self.assertNotIn('encoder_helper',names)

    def test_restore_and_keep_raw_profile_when_planning_raises(self):
        def planner(): raise RuntimeError('deliberate failure')
        service=SimpleNamespace(plan=planner)
        def run(*args): return service.plan()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'failed.pstats'
            with self.assertRaisesRegex(RuntimeError,'deliberate'):
                run_profiled_case({},42,'cuda_graph',service,None,path,run=run)
            self.assertIs(service.plan,planner)
            self.assertTrue(path.exists())
            self.assertIn('planner',[key[2] for key in pstats.Stats(str(path)).stats])

    def test_production_reference_exactness_but_benchmark_source_may_grow(self):
        case,row,sources,manifest,reference=fixture()
        self.assertEqual(reference_signatures(reference,[case],42,'cuda_graph',sources,manifest),{'test':row['signatures']})
        for change in ('source','model','request','signatures','outcome'):
            ref=copy.deepcopy(reference)
            if change=='source': ref['source_sha256']['simulation/planner.py']='changed'
            elif change=='model': ref['inputs']['expected_checkpoint_sha256']='changed'
            elif change=='request': ref['records'][0]['request']['seed']=43
            elif change=='signatures': ref['records'][0].pop('signatures')
            elif change=='outcome': ref['records'][0]['converged']=False
            with self.assertRaises(ValueError): reference_signatures(ref,[case],42,'cuda_graph',sources,manifest)

    def test_all_three_phases_and_reference_are_strict(self):
        case,row,_,_,_=fixture()
        record={phase:copy.deepcopy(row) for phase in ('warmup','control','profiled')}
        self.assertEqual(diagnostic_errors(record,case,row['signatures'],'cuda_graph'),[])
        for phase in record:
            bad=copy.deepcopy(record); bad[phase]['signatures']['rng_after']='changed'
            self.assertTrue(diagnostic_errors(bad,case,row['signatures'],'cuda_graph'))
        bad=copy.deepcopy(record); bad['profiled']['graph_active']=False
        self.assertTrue(any('inactive' in e for e in diagnostic_errors(bad,case,row['signatures'],'cuda_graph')))


class DiagnosticRunnerTests(unittest.TestCase):
    def test_cli_rejects_invalid_inputs_before_launch(self):
        with patch('benchmarks.suite.profile_cpu.launch_diagnostic') as launch, patch('sys.stderr',new_callable=io.StringIO):
            for args in (['--repeats','0'],['--repeats','11'],['--timeout','nan'],['--cases','unknown'],
                         ['--cases','recent_live_start','recent_live_start']):
                with self.assertRaises(SystemExit): main(args)
            launch.assert_not_called()

    def test_cpu_windows_separate_phases_and_child_threads_enforced(self):
        base={'case_id':'x','seed':42,'iteration':0}
        events=[{'type':'started','monotonic_s':1.,**base},
                {'type':'phase_started','phase':'control','monotonic_s':2.,**base},
                {'type':'phase_finished','phase':'control','monotonic_s':3.,**base},
                {'type':'phase_started','phase':'profiled','monotonic_s':3.1,**base},
                {'type':'phase_finished','phase':'profiled','monotonic_s':5.,**base},
                {'type':'finished','monotonic_s':5.1,**base}]
        child=SimpleNamespace(pid=123,returncode=0,poll=Mock(side_effect=[None,0,0]),wait=Mock(),kill=Mock())
        with tempfile.TemporaryDirectory() as tmp, \
             patch('benchmarks.suite.profile_cpu.subprocess.Popen',return_value=child) as popen, \
             patch('benchmarks.suite.profile_cpu.read_events',return_value=events), \
             patch('benchmarks.suite.profile_cpu.time.sleep'), \
             patch('benchmarks.suite.cpu_telemetry.sample_cpu',return_value={'monotonic_s':2.5,'available':True}), \
             patch('benchmarks.suite.cpu_telemetry.summarize_cpu',side_effect=lambda samples,start,end,pid: {'start':start,'end':end}) as summarize:
            result=launch_diagnostic(Path(tmp),300)
            final=result['events'][-1]
            self.assertEqual(final['cpu_telemetry_phases']['control'],{'start':2.,'end':3.})
            self.assertEqual(final['cpu_telemetry_phases']['profiled'],{'start':3.1,'end':5.})
            self.assertEqual(final['cpu_telemetry'],{'start':1.,'end':5.1})
            self.assertEqual(summarize.call_count,3)
            self.assertEqual(popen.call_args.kwargs['env']['OPENBLAS_NUM_THREADS'],'1')
            child.kill.assert_not_called()

    def test_timeout_kills_only_own_child(self):
        child=SimpleNamespace(pid=123,returncode=-9,poll=Mock(side_effect=[None,0]),wait=Mock(),kill=Mock())
        clock=iter(range(100))
        with tempfile.TemporaryDirectory() as tmp, \
             patch('benchmarks.suite.profile_cpu.subprocess.Popen',return_value=child), \
             patch('benchmarks.suite.profile_cpu.read_events',return_value=[]), \
             patch('benchmarks.suite.profile_cpu.time.monotonic',side_effect=lambda:float(next(clock))), \
             patch('benchmarks.suite.cpu_telemetry.sample_cpu',return_value={'available':False}):
            result=launch_diagnostic(Path(tmp),.1)
            self.assertTrue(result['timed_out'])
            child.kill.assert_called_once_with()
            child.wait.assert_called_once_with()


if __name__=='__main__': unittest.main()
