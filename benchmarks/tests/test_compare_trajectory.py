"""Trajectory arithmetic comparison gates/observers; no GPU or robot IO."""
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from benchmarks.suite import compare_trajectory as compare
from benchmarks.tests.test_timing_metrics import case as timing_case, records as timing_records


def case(): return {**timing_case(), 'request':{'seed':42,'samples':4}}


def records(repeats=10):
    rows=timing_records([2]*repeats)
    for row in rows:
        mode='recursive_math' if row['backend']=='eager' else 'batched_math'
        row.update(backend=mode,neural_backend='cuda_graph',graph_active=True,
                   scheduler_cache_enabled=True,scheduler_cache_hits=32,scheduler_cache_misses=0,
                   observed_reset_modes=['batch_reset'],reset_calls=2,
                   reset_calls_by_mode={'scalar_reset':0,'batch_reset':2,'unknown':0},
                   requested_math_mode=mode,trajectory_batch_before=mode=='batched_math',
                   trajectory_batch_enabled=mode=='batched_math',trajectory_work={
                       'subdivision_calls':3,'subdivision_batch_calls':3 if mode=='batched_math' else 0,
                       'subdivision_fallback_calls':0 if mode=='batched_math' else 3,
                       'frame_calls':1,'frame_requested_poses':10,'frame_yielded_poses':10},
                   child_cpu_affinity=[4],requested_cpu_affinity=[4],request=case()['request'],
                   signatures={'full_reference':'reference','sampled_predictions':'predictions','rng_after':'rng','encoder_profile':None})
    return rows


def summary(rows=None,repeats=10,**kwargs):
    return compare.summarize_comparison(records(repeats) if rows is None else rows,[case()],[42],repeats,**kwargs)


def reference():
    rows=records(1)
    for row in rows: row['backend']='batch_reset'
    return {'summary':{'passed':True},'records':rows,
            'inputs':{'expected_checkpoint_sha256':'model','expected_urdf_sha256':'urdf'},
            'source_sha256':{'simulation/continuous.py':'old','simulation/kinematics.py':'same'}}


class TrajectoryComparisonTests(unittest.TestCase):
    def test_both_prior_optimizations_fixed_and_labels_correct(self):
        result=summary();self.assertTrue(result['passed'])
        group=result['groups'][0]
        self.assertEqual(group['timing_status'],'supported_speedup')
        self.assertEqual(group['speedup_ratio'],2)
        self.assertEqual(set(group['warm_success_latency_ms']),{'recursive_math','batched_math'})
        self.assertNotIn('eager',json.dumps(group))
        self.assertEqual(result['neural_backend'],'cuda_graph')
        self.assertEqual(result['bullet_batch_reset'],'enabled in both modes')

    def test_actual_flags_and_work_are_required(self):
        for change in ({'neural_backend':'eager'},{'scheduler_cache_enabled':False},
                       {'scheduler_cache_hits':0},{'scheduler_cache_misses':1},
                       {'observed_reset_modes':['scalar_reset']},{'reset_calls':0},
                       {'trajectory_batch_before':True},{'requested_math_mode':'batched_math'},
                       {'trajectory_work':{}},{'graph_active':False},{'child_cpu_affinity':[8]}):
            with self.subTest(change=change):
                rows=records();rows[0].update(change)
                result=summary(rows,affinity=[4]);self.assertFalse(result['passed'])
                self.assertIsNone(result['groups'][0]['speedup_ratio'])
        rows=records();rows[1]['trajectory_work'].update(subdivision_batch_calls=0,subdivision_fallback_calls=3)
        self.assertFalse(summary(rows)['passed'])
        rows=records();rows[0]['trajectory_work'].update(subdivision_batch_calls=1,subdivision_fallback_calls=2)
        self.assertFalse(summary(rows)['passed'])

    def test_top_level_work_parity_and_full_generator_consumption(self):
        for field in ('subdivision_calls','frame_calls','frame_requested_poses','frame_yielded_poses'):
            rows=records();rows[1]['trajectory_work'][field]+=1
            self.assertFalse(summary(rows)['passed'])
        rows=records();rows[1]['trajectory_work'].update(frame_requested_poses=11,frame_yielded_poses=11)
        self.assertFalse(summary(rows)['passed'])
        rows=records();rows[1]['reset_calls']=3;rows[1]['reset_calls_by_mode']['batch_reset']=3
        self.assertFalse(summary(rows)['passed'])

    def test_rejection_can_have_zero_trajectory_work(self):
        c=case();c['expected']={'kind':'reject','statuses':['invalid_start']}
        rows=records(1)
        for row in rows:
            row.update(status='invalid_start',converged=False,graph_active=False,scheduler_cache_hits=0)
            row['metrics']['sample_calls']=0
            row['trajectory_work']={k:0 for k in row['trajectory_work']}
        result=compare.summarize_comparison(rows,[c],[42],1)
        self.assertTrue(result['passed']);self.assertIsNone(result['groups'][0]['speedup_ratio'])

    def test_history_only_batch_reset_and_narrow_production_allowlist(self):
        sources={name:'new' for name in compare.REFERENCE_CHANGE_ALLOWLIST}
        sources['simulation/kinematics.py']='same'
        manifest={'checkpoint_sha256':'model','urdf_sha256':'urdf'}
        prior=reference();ignored=copy.deepcopy(prior['records'][0]);ignored.update(backend='scalar_reset',signatures={})
        prior['records'].append(ignored)
        expected,changes=compare.reference_signatures(prior,[case()],[42],sources,manifest)
        self.assertEqual(expected['timing:42'],records(1)[0]['signatures'])
        self.assertEqual(set(changes),compare.REFERENCE_CHANGE_ALLOWLIST)
        for mutate in (lambda r:r['records'][0].update(observed_reset_modes=['scalar_reset']),
                       lambda r:r['records'][0].update(scheduler_cache_enabled=False),
                       lambda r:r['records'][0].update(neural_backend='eager'),
                       lambda r:r['records'][0]['signatures'].update(rng_after='changed')):
            prior=reference();mutate(prior)
            with self.assertRaises(ValueError): compare.reference_signatures(prior,[case()],[42],sources,manifest)
        with self.assertRaises(ValueError):
            compare.reference_signatures(reference(),[case()],[42],{**sources,'simulation/diffusion/env.py':'changed'},manifest)

    def test_scoped_observers_track_actual_work_and_restore_all_bindings(self):
        from benchmarks.suite import worker
        from simulation.diffusion.env import ArmEnv
        from simulation import trajectory_math
        valid=ArmEnv.valid
        policy=types.SimpleNamespace(scheduler_cache=True,_scheduler_cache_hits=0,_scheduler_cache_misses=0)
        service=types.SimpleNamespace(_policy=policy)
        def reset(scene,q): return q
        def leaves(*args): return ['leaf'] if trajectory_math.BATCH_TRAJECTORY else None
        def frames(states): yield from states
        captured=[]
        def run(case,seed,backend,service,torch):
            self.assertIs(ArmEnv.valid,valid)
            ArmEnv._set_joints(types.SimpleNamespace(batch_reset=True),[1])
            trajectory_math.subdivision_leaves(None,4)
            self.assertEqual(list(trajectory_math.frame_transforms([1,2])),[1,2])
            policy._scheduler_cache_hits+=63
            return {'backend':backend}
        def worker_main():
            self.assertEqual(os.environ['MYCOBOT_SCHEDULER_CACHE'],'1')
            self.assertEqual(os.environ['MYCOBOT_BULLET_BATCH_RESET'],'1')
            captured.append(worker.run_case({},42,'cuda_graph',service,None));return 0
        for mode in ('recursive_math','batched_math'):
            with patch.object(worker,'run_case',run),patch.object(worker,'main',worker_main), \
                 patch.object(ArmEnv,'_set_joints',reset),patch.object(trajectory_math,'subdivision_leaves',leaves), \
                 patch.object(trajectory_math,'frame_transforms',frames), \
                 patch.object(trajectory_math,'BATCH_TRAJECTORY',mode=='batched_math'),patch.dict(os.environ,{},clear=False):
                self.assertEqual(compare.worker_main('input','output',mode),0)
                self.assertIs(worker.run_case,run);self.assertIs(ArmEnv._set_joints,reset)
                self.assertIs(trajectory_math.subdivision_leaves,leaves);self.assertIs(trajectory_math.frame_transforms,frames)
        self.assertEqual(captured[0]['trajectory_work']['subdivision_batch_calls'],0)
        self.assertEqual(captured[1]['trajectory_work']['subdivision_batch_calls'],1)
        self.assertEqual(captured[1]['trajectory_work']['frame_yielded_poses'],2)
        self.assertEqual(captured[1]['scheduler_cache_hits'],63)
        def fail(*args): raise RuntimeError('deliberate')
        with patch.object(worker,'run_case',fail),patch.object(worker,'main',worker_main), \
             patch.object(ArmEnv,'_set_joints',reset),patch.object(trajectory_math,'subdivision_leaves',leaves), \
             patch.object(trajectory_math,'frame_transforms',frames),patch.dict(os.environ,{},clear=False):
            with self.assertRaisesRegex(RuntimeError,'deliberate'): compare.worker_main('input','output','batched_math')
            self.assertIs(worker.run_case,fail);self.assertIs(ArmEnv._set_joints,reset)
            self.assertIs(trajectory_math.subdivision_leaves,leaves);self.assertIs(trajectory_math.frame_transforms,frames)

    def test_incomplete_bad_signature_and_global_failure_suppress_claims(self):
        self.assertFalse(summary(records()[:-1])['passed'])
        rows=records();rows[1]['signatures']['rng_after']='changed';self.assertFalse(summary(rows)['passed'])
        self.assertFalse(summary(expected={})['passed'])
        result=summary();compare.invalidate_summary(result,['source changed'])
        self.assertFalse(result['passed']);self.assertIsNone(result['groups'][0]['speedup_ratio'])
        self.assertEqual(summary(repeats=1)['groups'][0]['timing_status'],'inconclusive')

    def test_cli_validation_and_escaped_readable_report(self):
        with patch.object(compare,'launch_worker') as launch,patch('sys.stderr',new_callable=io.StringIO):
            for args in ([],['--repeats','0'],['--repeats','1','--timeout','nan'],['--cases','unknown']):
                with self.assertRaises(SystemExit): compare.main(args)
            launch.assert_not_called()
        result=summary(repeats=1);result['groups'][0]['case_id']='<script>bad</script>'
        with tempfile.TemporaryDirectory() as tmp:
            compare.write_report(tmp,{'summary':result,'records':records(1)})
            page=(Path(tmp)/'report.html').read_text()
        self.assertNotIn('<script>',page);self.assertIn('&lt;script&gt;',page)
        self.assertIn('Trajectory arithmetic:',page)
        self.assertIn('Bullet batch resets remain enabled in both modes',page)
        self.assertIn('Observed work per measured trial',page)
        self.assertIn('Recursive math median ms',page)


if __name__=='__main__': unittest.main()
