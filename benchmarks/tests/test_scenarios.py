"""CPU-only fixture validation; never calls a planner, transport, or GPU."""
import copy
import json
import unittest

from benchmarks.suite.scenarios import (
    expected_for_seed, load_cases, load_manifest, manifest_sha256, validate_cases,
)


class ScenarioSchemaTest(unittest.TestCase):
    def test_stable_manifest_and_independent_requests(self):
        full, quick = load_cases('full'), load_cases('quick')
        self.assertEqual(len(full), 26)
        self.assertEqual(len(quick), 9)
        self.assertEqual(len({c['id'] for c in full}), len(full))
        self.assertEqual(quick, [c for c in full if c['quick']])
        self.assertEqual(len(manifest_sha256()), 64)
        self.assertEqual(json.dumps(full, sort_keys=True), json.dumps(load_cases('full'), sort_keys=True))
        full[0]['request']['start_deg'][0] = 999
        self.assertNotEqual(full[0], load_cases('full')[0])

    def test_qualification_is_not_invented_for_extra_seeds(self):
        full = load_cases('full')
        self.assertEqual(sum(c['expected']['kind'] == 'success' for c in full), 7)
        self.assertEqual(sum(c['expected']['kind'] == 'allowed' for c in full), 14)
        self.assertEqual(sum(c['expected']['kind'] == 'reject' for c in full), 5)
        for c in full:
            self.assertEqual(expected_for_seed(c, 42), c['expected'])
            other = expected_for_seed(c, 43)
            self.assertEqual(other['kind'], 'reject' if c['expected']['kind'] == 'reject' else 'allowed')
            self.assertNotIn('time_limit', other['statuses'])

    def test_schema_refuses_malformed_or_ambiguous_fixtures(self):
        for change in (
            lambda c: c.update(id='UPPER'),
            lambda c: c.update(quick=1),
            lambda c: c['request'].update(start_deg=[0] * 5),
            lambda c: c['request'].update(target_mm=[float('nan'), 0, 0]),
            lambda c: c['request'].update(target_mm=[True, 0, 0]),
            lambda c: c['request'].update(target_mm=[601, 0, 0]),
            lambda c: c['request'].update(target_axis=[0, 0, 0]),
            lambda c: c['request'].pop('obstacle_mm'),
            lambda c: c['request'].update(obstacle_mm=[0, 0, 0, 10, 10, 10]),
            lambda c: c['request'].update(samples=8),
            lambda c: c['request'].update(seed=True),
            lambda c: c['request'].pop('speed'),
            lambda c: c.update(expected={'kind': 'allowed', 'statuses': ['time_limit']}),
            lambda c: c.update(qualified_seeds=[42, 43]),
            lambda c: c.pop('evidence'),
        ):
            c = load_cases()[0]
            change(c)
            with self.subTest(case=c), self.assertRaises(ValueError):
                validate_cases([c])
        for bad in ([], None, {}, [None]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_cases(bad)
        c = load_cases()[0]
        with self.assertRaises(ValueError):
            validate_cases([c, copy.deepcopy(c)])
        with self.assertRaises(ValueError):
            load_cases('unknown')

    def test_historical_requests_match_immutable_source(self):
        from pathlib import Path
        manifest = json.loads(Path('simulation/artifacts/diffusion_motion_v2/webui.json').read_text())
        sources = {c['id']: c for c in manifest['cases']}
        for c in load_cases('full'):
            if c['category'] == 'heldout':
                source = sources[c['id'].split('_')[1]]
                for field in ('start_deg', 'target_mm', 'target_axis'):
                    self.assertEqual(c['request'][field], source[field])
                self.assertNotIn('goal_deg', c['request'])


class ScenarioGeometryTest(unittest.TestCase):
    def test_endpoints_and_rejection_classes_against_real_scene(self):
        import numpy as np
        from simulation.diffusion.env import ArmEnv
        from simulation.kinematics import ARM
        from simulation.pose5 import axis_error
        from simulation.diffusion.service import request
        self.assertEqual(load_manifest()['urdf_sha256'], ARM.fingerprint)
        with ArmEnv() as env:
            for c in load_cases('full'):
                with self.subTest(case=c['id']):
                    r = c['request']
                    start, target, axis, box, goal, config = request(r)
                    env.set_obstacle(box)
                    status = c['expected']['statuses']
                    self.assertEqual(env.valid(start), status != ['invalid_start'])
                    if goal is not None:
                        self.assertEqual(env.valid(goal), status != ['invalid_goal'])
                        t = ARM.transform(goal)
                        np.testing.assert_allclose(t[:3, 3] * 1000, r['target_mm'], atol=1e-9)
                        np.testing.assert_allclose(t[:3, 2], axis, atol=1e-12)
                    if c['category'] == 'near_limit':
                        margin = np.rad2deg(np.minimum(goal - ARM.lower, ARM.upper - goal))
                        self.assertAlmostEqual(float(margin.min()), 1.0, places=10)
                    if c['category'] == 'direction_change':
                        before = ARM.transform(start)[:3, 2]
                        self.assertGreater(axis_error(before, axis)[2], 15)
                    if c['id'] == 'reject_self_start':
                        self.assertFalse(env.space.valid(start))
                    if c['id'] == 'reject_table_start':
                        self.assertTrue(env.space.valid(start))
                        self.assertIsNone(box)
                    if c['id'] == 'reject_block_start':
                        env.set_obstacle(None)
                        self.assertTrue(env.valid(start))
                    if c['id'] == 'unreachable_corner':
                        # Triangle inequality gives a rigorous conservative reach bound.
                        reach = sum(np.linalg.norm(t[:3, 3]) for t, _ in ARM.chain)
                        self.assertGreater(np.linalg.norm(target), reach + 0.001)


if __name__ == '__main__':
    unittest.main()
