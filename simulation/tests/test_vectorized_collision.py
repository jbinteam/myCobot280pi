"""Scalar-parity and fail-closed checks for vectorized collision geometry."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from simulation.kinematics import ARM
from simulation.self_collision import checker
from simulation.diffusion.env import ArmEnv

FIXTURES=Path(__file__).resolve().parents[1]/'fixtures'


class VectorizedCollisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=json.loads((FIXTURES/'self_collision_path.json').read_text())
        cls.known=np.deg2rad([cls.fixture[k] for k in ('start_deg','goal_deg','via_deg')])
        cls.random=np.random.default_rng(7813).uniform(ARM.lower,ARM.upper,(1024,6))

    def test_batch_fk_matches_every_shared_link_frame(self):
        poses=np.vstack((self.known,self.random[:128]))
        batch=ARM.link_transforms_batch(poses)
        for index,q in enumerate(poses):
            for name,t in ARM.link_transforms(q).items():
                np.testing.assert_allclose(batch[name][index],t,rtol=0.,atol=1e-14)
        self.assertEqual(ARM.link_transforms_batch(np.empty((0,6)))[ARM.tcp].shape,(0,4,4))
        for bad in (np.zeros(6),np.zeros((2,5)),np.full((1,6),np.nan)):
            with self.assertRaises(ValueError):ARM.link_transforms_batch(bad)

    def test_self_collision_parity_across_clearances_and_batch_boundaries(self):
        near=json.loads((FIXTURES/'self_collision_clearance.json').read_text())['q_deg']
        poses=np.vstack((self.known,np.deg2rad(near),self.random,ARM.lower,ARM.upper))
        for clearance in (0.,1.,2.):
            with self.subTest(clearance=clearance):
                collision=checker(clearance)
                expected=np.array([collision.valid(q) for q in poses])
                np.testing.assert_array_equal(collision.valid_many(poses),expected)
                # Batch queries must not leave stale native transforms for a
                # subsequent scalar diagnostic or validity query.
                for q in self.known:
                    self.assertEqual(collision.valid(q),collision.check(q)['valid'])

    def test_clearance_boundary_uses_scalar_predicate(self):
        collision=checker();start=self.known[0];end=(self.known[0]+self.known[1])/2
        self.assertTrue(collision.valid(start));self.assertFalse(collision.valid(end))
        lo,hi=0.,1.
        for _ in range(50):
            mid=(lo+hi)/2
            if collision.valid(start+mid*(end-start)):lo=mid
            else:hi=mid
        fractions=np.array([lo-1e-8,lo,hi,hi+1e-8])
        poses=start+fractions[:,None]*(end-start)
        expected=np.array([collision.valid(q) for q in poses])
        with patch.object(collision,'_check',wraps=collision._check) as scalar:
            np.testing.assert_array_equal(collision.valid_many(poses),expected)
            self.assertGreater(scalar.call_count,0)

    def test_scene_parity_with_table_block_and_unchanged_display(self):
        start=np.deg2rad([0,-40,-130,80,0,50])
        poses=np.vstack((start,self.random))
        with ArmEnv() as env:
            env.reset(start)
            for obstacle in (None,np.r_[ARM.fk(start),[.025,.025,.025]]):
                env.set_obstacle(obstacle)
                expected=np.array([env.valid(q) for q in poses])
                env._valid_cache.clear()
                np.testing.assert_array_equal(env.valid_many(poses),expected)
                np.testing.assert_allclose(env.current(),start,rtol=0.,atol=1e-12)
                with patch.object(env.space.collision_checker,'valid_many',side_effect=AssertionError('Cached poses must not be rechecked')):
                    np.testing.assert_array_equal(env.valid_many(poses),expected)

    def test_exact_pose_cache_duplicates_limits_and_nonfinite_states(self):
        q=np.deg2rad([0,-40,-130,80,0,50]);near=q.copy();near[0]=np.nextafter(q[0],np.inf)
        invalid=q.copy();invalid[0]=ARM.upper[0]+.001
        with ArmEnv() as env:
            env.set_obstacle(None)
            collision=env.space.collision_checker
            with patch.object(collision,'valid_many',wraps=collision.valid_many) as batch:
                values=env.valid_many([q,q,near,invalid,np.full(6,np.nan)])
            np.testing.assert_array_equal(values,[True,True,True,False,False])
            self.assertEqual(len(batch.call_args.args[0]),2)
            self.assertIn(q.tobytes(),env._valid_cache);self.assertIn(near.tobytes(),env._valid_cache)
            self.assertFalse(env._valid_cache[invalid.tobytes()])
            self.assertEqual(env.valid_many(np.empty((0,6))).shape,(0,))

    def test_native_failures_propagate_and_bullet_state_is_restored(self):
        q=np.deg2rad(json.loads((FIXTURES/'self_collision_clearance.json').read_text())['q_deg'])
        collision=checker()
        with patch.object(collision.fcl,'distance',return_value=float('nan')):
            with self.assertRaisesRegex(RuntimeError,'Nonfinite FCL distance'):collision.valid_many([q])
        start=np.deg2rad([0,-40,-130,80,0,50]);goal=start.copy();goal[0]+=.1
        with ArmEnv() as env:
            env.set_obstacle(None);env.reset(start)
            with patch('simulation.diffusion.env.p.getClosestPoints',side_effect=RuntimeError('Bullet query failed')):
                with self.assertRaisesRegex(RuntimeError,'Bullet query failed'):env.valid_many([goal])
            np.testing.assert_allclose(env.current(),start,rtol=0.,atol=1e-12)
            self.assertNotIn(goal.tobytes(),env._valid_cache)


if __name__=='__main__':unittest.main()
