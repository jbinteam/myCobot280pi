"""Collision reset comparison gates and observers; no GPU or robot IO."""
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from benchmarks.suite import compare_collision as compare
from benchmarks.tests.test_timing_metrics import case as timing_case, records as timing_records


def case():
    return {**timing_case(), 'request': {'seed':42,'samples':4}}


def records(repeats=10):
    rows=timing_records([2]*repeats)
    for row in rows:
        mode='scalar_reset' if row['backend']=='eager' else 'batch_reset'
        row.update(backend=mode,neural_backend='cuda_graph',graph_active=True,
                   scheduler_cache_enabled=True,scheduler_cache_hits=32,scheduler_cache_misses=0,
                   requested_reset_mode=mode,observed_reset_modes=[mode],reset_calls=2,
                   reset_calls_by_mode={k:2 if k==mode else 0 for k in ('scalar_reset','batch_reset','unknown')},
                   child_cpu_affinity=[4],requested_cpu_affinity=[4],request=case()['request'],
                   signatures={'full_reference':'reference','sampled_predictions':'predictions','rng_after':'rng','encoder_profile':None})
    return rows


def summary(rows=None,repeats=10,**kwargs):
    return compare.summarize_comparison(records(repeats) if rows is None else rows,[case()],[42],repeats,**kwargs)


def reference():
    rows=records(1)
    for row in rows: row['backend']='cache_on'
    return {'summary':{'passed':True},'records':rows,
            'inputs':{'expected_checkpoint_sha256':'model','expected_urdf_sha256':'urdf'},
            'source_sha256':{'simulation/diffusion/env.py':'old','simulation/kinematics.py':'same'}}


class CollisionComparisonTests(unittest.TestCase):
    def test_both_inference_features_fixed_and_export_labels_correct(self):
        value=summary()
        self.assertTrue(value['passed'])
        group=value['groups'][0]
        self.assertEqual(group['timing_status'],'supported_speedup')
        self.assertEqual(group['speedup_ratio'],2)
        self.assertEqual(set(group['warm_success_latency_ms']),{'scalar_reset','batch_reset'})
        self.assertNotIn('eager',json.dumps(group))
        self.assertEqual(value['neural_backend'],'cuda_graph')
        self.assertEqual(value['scheduler_cache'],'enabled in both modes')

    def test_wrong_flags_missing_or_mixed_reset_activity_fail_closed(self):
        for mutation in ({'neural_backend':'eager'},{'scheduler_cache_enabled':False},
                         {'scheduler_cache_hits':0},{'scheduler_cache_misses':1},
                         {'reset_calls':0},{'reset_calls_by_mode':{}},
                         {'observed_reset_modes':['scalar_reset','batch_reset']},
                         {'requested_reset_mode':'batch_reset'},{'graph_active':False},
                         {'child_cpu_affinity':[8]}):
            with self.subTest(mutation=mutation):
                rows=records();rows[0].update(mutation)
                result=summary(rows,affinity=[4])
                self.assertFalse(result['passed'])
                self.assertIsNone(result['groups'][0]['speedup_ratio'])
        rows=records();rows[1]['reset_calls_by_mode']['unknown']=1
        self.assertFalse(summary(rows)['passed'])
        rows=records();rows[1]['reset_calls']=3;rows[1]['reset_calls_by_mode']['batch_reset']=3
        self.assertFalse(summary(rows)['passed'])

    def test_incomplete_wrong_signature_and_source_failure_suppress_claim(self):
        self.assertFalse(summary(records()[:-1])['passed'])
        rows=records();rows[1]['signatures']['rng_after']='changed'
        self.assertFalse(summary(rows)['passed'])
        self.assertFalse(summary(expected={})['passed'])
        result=summary(repeats=1)
        self.assertTrue(result['passed']);self.assertEqual(result['groups'][0]['timing_status'],'inconclusive')
        result=summary();compare.invalidate_summary(result,['source changed'])
        self.assertFalse(result['passed']);self.assertIsNone(result['groups'][0]['speedup_ratio'])

    def test_history_only_cache_on_rows_and_intended_collision_patch(self):
        sources={'simulation/diffusion/env.py':'new','simulation/kinematics.py':'same',
                 'simulation/diffusion/tests/test_bullet_reset.py':'newtest'}
        manifest={'checkpoint_sha256':'model','urdf_sha256':'urdf'}
        prior=reference()
        ignored=copy.deepcopy(prior['records'][0]);ignored.update(backend='cache_off',signatures={})
        prior['records'].append(ignored)
        expected,changes=compare.reference_signatures(prior,[case()],[42],sources,manifest)
        self.assertEqual(expected['timing:42'],records(1)[0]['signatures'])
        self.assertEqual(set(changes),compare.REFERENCE_CHANGE_ALLOWLIST)
        for mutation in (lambda r:r['records'][0].update(scheduler_cache_enabled=False),
                         lambda r:r['records'][0].update(scheduler_cache_hits=0),
                         lambda r:r['records'][0].update(neural_backend='eager'),
                         lambda r:r['records'][0]['signatures'].update(rng_after='changed'),
                         lambda r:r['inputs'].update(expected_checkpoint_sha256='wrong')):
            prior=reference();mutation(prior)
            with self.assertRaises(ValueError): compare.reference_signatures(prior,[case()],[42],sources,manifest)
        with self.assertRaises(ValueError):
            compare.reference_signatures(reference(),[case()],[42],{**sources,'simulation/diffusion/policy.py':'changed'},manifest)
        prior=reference()
        for row in prior['records']: row['backend']='cuda_graph'
        with self.assertRaises(ValueError): compare.reference_signatures(prior,[case()],[42],sources,manifest)

    def test_observer_counts_actual_scene_modes_preserves_valid_and_restores(self):
        from benchmarks.suite import worker
        from simulation.diffusion.env import ArmEnv
        original_valid=ArmEnv.valid
        policy=types.SimpleNamespace(scheduler_cache=True,_scheduler_cache_hits=10,_scheduler_cache_misses=0)
        service=types.SimpleNamespace(_policy=policy)
        captured=[]
        def reset(scene,q): return q
        def run(case,seed,backend,service,torch):
            self.assertIs(ArmEnv.valid,original_valid)
            scene=types.SimpleNamespace(batch_reset=True)
            ArmEnv._set_joints(scene,[1]);ArmEnv._set_joints(scene,[2])
            policy._scheduler_cache_hits+=63
            return {'backend':backend}
        def worker_main():
            self.assertEqual(os.environ['MYCOBOT_SCHEDULER_CACHE'],'1')
            self.assertEqual(os.environ['MYCOBOT_BULLET_BATCH_RESET'],'1')
            captured.append(worker.run_case({},42,'cuda_graph',service,None));return 0
        with patch.object(worker,'run_case',run),patch.object(worker,'main',worker_main), \
             patch.object(ArmEnv,'_set_joints',reset),patch.dict(os.environ,{},clear=False):
            self.assertEqual(compare.worker_main('input','output','batch_reset'),0)
            self.assertIs(worker.run_case,run);self.assertIs(ArmEnv._set_joints,reset)
        self.assertEqual(captured[0]['reset_calls'],2)
        self.assertEqual(captured[0]['observed_reset_modes'],['batch_reset'])
        self.assertEqual(captured[0]['scheduler_cache_hits'],63)
        self.assertIs(ArmEnv.valid,original_valid)
        def fail(*args): raise RuntimeError('deliberate')
        with patch.object(worker,'run_case',fail),patch.object(worker,'main',worker_main), \
             patch.object(ArmEnv,'_set_joints',reset),patch.dict(os.environ,{},clear=False):
            with self.assertRaisesRegex(RuntimeError,'deliberate'): compare.worker_main('input','output','batch_reset')
            self.assertIs(worker.run_case,fail);self.assertIs(ArmEnv._set_joints,reset)

    def test_cli_validation_before_gpu(self):
        with patch.object(compare,'launch_worker') as launch,patch('sys.stderr',new_callable=io.StringIO):
            for args in ([],['--repeats','0'],['--repeats','1','--timeout','nan'],
                         ['--repeats','1','--seeds','42','42'],['--cases','unknown']):
                with self.assertRaises(SystemExit): compare.main(args)
            launch.assert_not_called()

    def test_orchestration_integrity_failure_writes_failed_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'benchmarks').mkdir()
            ref=root/'reference.json';ref.write_text(json.dumps(reference()))
            def launch(directory,index,mode,timeout,affinity):
                row=copy.deepcopy(records(1)[index]);row.update(backend='cuda_graph',type='trial')
                return {'events':[row],'pid':1,'returncode':0,'timed_out':False}
            source={'simulation/diffusion/env.py':'new','simulation/kinematics.py':'same'}
            with patch.object(compare,'ROOT',root),patch.object(compare,'select_cases',return_value=[case()]), \
                 patch.object(compare,'source_inventory',side_effect=[source,{**source,'simulation/kinematics.py':'changed'}]), \
                 patch.object(compare,'frozen_inventory',return_value={'frozen':'same'}), \
                 patch.object(compare,'launch_worker',side_effect=launch), \
                 patch.object(compare,'load_manifest',return_value={'checkpoint_sha256':'model','urdf_sha256':'urdf'}):
                result=compare.main(['--repeats','1','--reference',str(ref),'--output',str(root/'run')])
            self.assertEqual(result,1)
            report=json.loads((root/'run/report.json').read_text())
            self.assertIn('source/config changed during comparison',report['summary']['failures'])
            self.assertEqual({r['backend'] for r in report['records']},{'scalar_reset','batch_reset'})
            self.assertEqual({r['neural_backend'] for r in report['records']},{'cuda_graph'})

    def test_escaped_report_names_both_fixed_inference_features(self):
        result=summary(repeats=1);result['groups'][0]['case_id']='<script>bad</script>'
        with tempfile.TemporaryDirectory() as tmp:
            compare.write_report(tmp,{'summary':result})
            page=(Path(tmp)/'report.html').read_text()
        self.assertNotIn('<script>',page);self.assertIn('&lt;script&gt;',page)
        self.assertIn('scheduler timestep cache held on in both',page)
        self.assertIn('Scalar reset median ms',page)


if __name__=='__main__': unittest.main()
