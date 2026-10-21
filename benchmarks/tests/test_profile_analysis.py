"""Trusted local cProfile analysis tests; no planner, GPU or hardware imports."""
import cProfile
import json
import marshal
import pstats
from pathlib import Path
import tempfile
import unittest

from benchmarks.suite.profile_analysis import analyze_profile, classify_function, write_report


class ProfileAnalysisTests(unittest.TestCase):
    def test_recursive_caller_counts_use_cprofile_edge_order(self):
        scope = {}
        exec(compile('def bound(n):\n    return 1 if n == 0 else bound(n-1)+1\n',
                     '/project/simulation/continuous.py', 'exec'), scope)
        profile = cProfile.Profile()
        profile.runcall(scope['bound'], 4)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'recursive.pstats'; profile.dump_stats(str(path))
            raw = pstats.Stats(str(path)).stats
            result = analyze_profile(path)
        key = next(key for key in raw if key[2] == 'bound')
        nc, cc, _, _ = raw[key][4][key]
        context = next(row for row in result['caller_context'] if row['name'] == 'bound')
        recursive = next(row for row in context['callers'] if row['name'] == 'bound')
        self.assertGreater(nc, cc)
        self.assertEqual(recursive['total_calls'], nc)
        self.assertEqual(recursive['primitive_calls'], cc)
        self.assertEqual(result['targeted_calls']['continuous.bound']['total_calls'], 5)

    def test_nested_calls_do_not_double_count_exclusive_attribution(self):
        scope = {}
        exec(compile('def link_transforms_batch():\n    return sum(i*i for i in range(100))\n',
                     '/project/simulation/kinematics.py', 'exec'), scope)
        exec(compile('def valid_many():\n    for _ in range(3):\n        link_transforms_batch()\n',
                     '/project/simulation/diffusion/env.py', 'exec'), scope)
        profile = cProfile.Profile()
        profile.enable()
        scope['valid_many'](); scope['valid_many']()
        profile.disable()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'nested.pstats'; profile.dump_stats(str(path))
            result = analyze_profile(path)
        fk = result['targeted_calls']['kinematics.link_transforms_batch']
        self.assertTrue(fk['observed'])
        self.assertEqual(fk['total_calls'], 6)
        self.assertEqual(result['targeted_calls']['env.valid_many']['total_calls'], 2)
        self.assertAlmostEqual(sum(g['exclusive_s'] for g in result['exclusive_groups']),
                               result['total_exclusive_s'])
        self.assertAlmostEqual(sum(g['percent_of_profile'] for g in result['exclusive_groups']), 100)
        self.assertGreater(sum(r['cumulative_s'] for r in result['top_cumulative']),
                           result['total_exclusive_s'])
        context = next(c for c in result['caller_context'] if c['name'] == 'link_transforms_batch')
        self.assertEqual(context['callers'][0]['name'], 'valid_many')
        self.assertEqual(context['callers'][0]['total_calls'], 6)
        self.assertEqual(context['callers'][0]['category'], 'collision_bullet_operations')
        self.assertTrue(any('not total collision cost' in s for s in result['caveats']))
        self.assertFalse(result['targeted_calls']['FCL.visible']['observed'])

    def test_visible_native_counts_and_shared_numpy_stay_distinct(self):
        env = ('/project/simulation/diffusion/env.py', 175, 'valid_many')
        reset = ('~', 0, '<built-in method pybullet.resetJointState>')
        closest = ('~', 0, '<built-in method pybullet.getClosestPoints>')
        numpy = ('~', 0, '<method \'reduce\' of \'numpy.ufunc\' objects>')
        # Synthetic trusted cProfile format: exclusive sum=7, nested cumulative sum=13.
        fixture = {env: (1, 1, 1., 7., {}),
                   reset: (12, 12, 2., 2., {env: (12, 12, 2., 2.)}),
                   closest: (2, 2, 3., 3., {env: (2, 2, 3., 3.)}),
                   numpy: (4, 4, 1., 1., {env: (4, 4, 1., 1.)})}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'native.pstats'
            with path.open('wb') as stream:
                marshal.dump(fixture, stream)
            result = analyze_profile(path)
        self.assertEqual(result['targeted_calls']['pybullet.resetJointState']['total_calls'], 12)
        self.assertEqual(result['targeted_calls']['pybullet.getClosestPoints']['total_calls'], 2)
        self.assertEqual(result['total_exclusive_s'], 7)
        groups = {g['category']: g['exclusive_s'] for g in result['exclusive_groups']}
        self.assertEqual(groups['collision_bullet_operations'], 6)
        self.assertEqual(groups['numpy_shared_ambiguous'], 1)
        self.assertEqual(sum(g['exclusive_s'] for g in result['exclusive_groups']), 7)
        self.assertEqual(sum(r['cumulative_s'] for r in result['top_cumulative']), 13)

    def test_categories_reflect_real_source_and_do_not_claim_shared_builtins(self):
        cases = [
            ('/p/simulation/self_collision.py', 'valid_many', 'self_collision_geometry'),
            ('/p/simulation/self_collision.py', 'collision_mesh', 'scene_setup_provenance'),
            ('/p/simulation/diffusion/service.py', 'info', 'scene_setup_provenance'),
            ('/p/simulation/diffusion/env.py', 'inverse_kinematics', 'inverse_kinematics'),
            ('/p/simulation/continuous.py', 'bound', 'trajectory_work'),
            ('/p/simulation/motion_limits.py', 'derivative_hull', 'trajectory_work'),
            ('/p/diffusers/schedulers/scheduling_ddpm.py', 'step', 'neural_inference_scheduler'),
            ('/p/fcl/fcl.pyx', 'collide', 'collision_fcl_visible'),
            ('/p/numpy/linalg/_linalg.py', 'norm', 'numpy_shared_ambiguous'),
            ('~', '<built-in method builtins.sum>', 'other_shared_ambiguous'),
            ('/p/benchmarks/suite/worker.py', 'run_case', 'benchmark_observer'),
        ]
        for path, function, expected in cases:
            with self.subTest(path=path, function=function):
                self.assertEqual(classify_function(path, function), expected)

    def test_html_escapes_case_functions_telemetry_and_report_payload(self):
        hostile = '<script>alert("unsafe")</script>'
        report = {'summary': {'passed': True, 'failures': []},
                  'records': [{'case_id': hostile, 'seed': 42, 'backend': 'cuda_graph', 'iteration': 0,
                               'control': {'timing_ms': {'total': 10.123456789}},
                               'profiled': {'timing_ms': {'total': 30}},
                               'observer_overhead_ratio': 3, 'signatures_exact': True,
                               'expected_reference_exact': True,
                               'analysis': {'top_exclusive': [{'function': hostile}],
                                            'top_cumulative': [], 'exclusive_groups': []},
                               'cpu_telemetry': {'message': hostile},
                               'cpu_telemetry_phases': {'control': {'message': hostile}}}]}
        with tempfile.TemporaryDirectory() as folder:
            write_report(folder, report)
            page = (Path(folder)/'report.html').read_text()
            saved = json.loads((Path(folder)/'report.json').read_text())
        self.assertEqual(saved, report)
        self.assertNotIn('<script>', page)
        self.assertIn('&lt;script&gt;', page)
        self.assertIn('Control planner ms', page)
        self.assertIn('Cumulative seconds (nonadditive)', page)
        self.assertIn('not an optimization or speedup', page)
        self.assertIn('CPU telemetry', page)
        self.assertIn('<td>10.1235</td>', page)
        self.assertIn('class="table-scroll"', page)


if __name__ == '__main__':
    unittest.main()
