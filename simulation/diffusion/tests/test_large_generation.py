import json
from pathlib import Path
import tempfile
import unittest
import h5py
from simulation.diffusion.generate_large import generate_large
from simulation.diffusion.planner import RRTConfig
from simulation.diffusion.dataset import read_episodes


class LargeGenerationTest(unittest.TestCase):
    def test_parallel_merge_has_unique_complete_episodes_and_exact_resume_preserves_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'data.h5';config=RRTConfig(iterations=100,post_solution=5)
            metadata=generate_large(path,20,73,2,10,0.,config,audit_episodes=10)
            self.assertEqual(metadata['completed_episodes'],20)
            self.assertEqual(sum(metadata['split_counts'].values()),20)
            _,episodes=read_episodes(path);self.assertEqual(len({e['id'] for e in episodes}),20)
            self.assertEqual(len({s['seed'] for s in metadata['shards']}),2)
            self.assertEqual(metadata['validation']['independent_collision_episodes'],10)
            resumed=generate_large(path,20,73,1,10,0.,config,resume=True,audit_episodes=10)
            self.assertEqual(resumed['sha256'],metadata['sha256'])
            with self.assertRaises(ValueError):generate_large(path,20,73,1,10,0.,config,audit_episodes=10)
            with self.assertRaises(ValueError):generate_large(path,20,74,1,10,0.,config,resume=True,audit_episodes=10)

