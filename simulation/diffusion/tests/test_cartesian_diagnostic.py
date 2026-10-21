"""Terminal diagnosis reads validation only and samples Cartesian observations."""
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from simulation.kinematics import ARM
from simulation.diffusion.cartesian_conditioning import observe
from simulation.diffusion.diagnose_cartesian import evaluate_case, summarize, validation_cases


class CartesianDiagnosticTests(unittest.TestCase):
    def test_cache_selection_never_reads_test_or_train_arrays(self):
        cache = Path('simulation/artifacts/diffusion_cartesian_v3/cache')
        if not (cache/'cache.json').exists(): self.skipTest('Full v3 cache unavailable')
        load = np.load
        with patch('simulation.diffusion.diagnose_cartesian.np.load', wraps=load) as loader:
            cases, metadata = validation_cases(cache, episodes=2)
        self.assertEqual(len(cases), 4)
        self.assertTrue(all(Path(call.args[0]).name.startswith('validation_') for call in loader.call_args_list))
        self.assertEqual({item['kind'] for item in cases}, {'near_terminal', 'exact_hold'})
        self.assertEqual({item['source'] for item in cases}, {'original', 'augmentation'})
        self.assertFalse(metadata['identity']['conditioning']['goal_joint_input'])

    def test_exact_hold_reports_drift_and_no_join_without_goal_joint_input(self):
        q = ARM.center.copy(); pose = ARM.transform(q)
        obs = observe(q, pose[:3, 3], pose[:3, 2])
        class Env:
            def set_obstacle(self, box): pass
            def reset(self, q): pass
            def path_valid(self, path, step=.04): return True
        class Policy:
            def sample(self, observations, samples, inference_steps):
                np.testing.assert_array_equal(observations, obs[None])
                return np.repeat(q[None, None, None], samples*16, axis=2).reshape(1, samples, 16, 6)
        case = {'q': q, 'target': pose[:3, 3], 'axis': pose[:3, 2], 'obs': obs, 'box': None,
                'v': np.zeros(6), 'a': np.zeros(6), 'source': 'original', 'validation_episode_index': 1,
                'kind': 'exact_hold', 'frame_index': 2, 'frame_count': 3}
        with patch('simulation.diffusion.diagnose_cartesian.prepare', side_effect=ValueError('No join')):
            result = evaluate_case(Env(), Policy(), case)
        self.assertEqual(result['safe_candidates'], 4)
        self.assertEqual(result['joinable_candidates'], 0)
        self.assertIsNone(result['selected'])
        self.assertTrue(all(c['strict_hit_any_future'] for c in result['candidates']))
        self.assertEqual(summarize([result])['exact_hold']['strict_hit_any_safe_future'], 1)
        self.assertEqual(result['candidates'][0]['max_hold_TCP_drift_mm'], 0)


if __name__ == '__main__': unittest.main()
