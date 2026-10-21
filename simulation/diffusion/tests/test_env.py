import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
from unittest.mock import patch
from simulation.kinematics import ARM
from simulation.prm import JointSpace
from simulation.diffusion.env import ArmEnv


class EnvironmentTest(unittest.TestCase):
    def sampled_scene(self,predicate):
        """Exercise sampling without depending on a physical collision layout."""
        env=ArmEnv.__new__(ArmEnv)
        env.space=JointSpace();env.valid=predicate
        return env

    def test_short_edges_use_exact_endpoints_without_constructing_an_array(self):
        seen=[];env=self.sampled_scene(lambda q:seen.append(q.copy()) or True)
        a=ARM.center.copy();b=a.copy();b[0]+=.04
        with patch.object(env.space,'interpolate',side_effect=AssertionError('Two-point grid needs no interpolation array')):
            self.assertTrue(env.edge_valid(a,b,.04))
        np.testing.assert_array_equal(seen,[a,b])

    def test_path_checks_identical_interior_samples_once_and_rejects_a_barrier(self):
        a=ARM.center.copy();b=a.copy();b[0]+=.12;c=b.copy();c[1]+=.02
        path=np.array([a,b,c]);seen=[]
        env=self.sampled_scene(lambda q:seen.append(q.copy()) or True)
        expected=[*path,*env.space.interpolate(a,b,.04)[1:-1]]
        self.assertTrue(env.path_valid(path,.04))
        np.testing.assert_array_equal(seen,expected)
        blocked=env.space.interpolate(a,b,.04)[1]
        env.valid=lambda q:not np.array_equal(q,blocked)
        self.assertFalse(env.path_valid(path,.04))

    def test_edge_results_match_original_grid_at_sampling_thresholds(self):
        env=self.sampled_scene(lambda q:not (.015<q[0]-ARM.center[0]<.025))
        a=ARM.center.copy()
        for step in [.01,.04]:
            for distance in [0.,np.nextafter(step,0.),step,np.nextafter(step,np.inf),.1]:
                b=a.copy();b[0]+=distance
                expected=all(env.valid(q) for q in env.space.interpolate(a,b,step))
                self.assertEqual(env.edge_valid(a,b,step),expected)
        for step in [0.,-1.,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):env.edge_valid(a,a,step)

    def test_batched_path_preserves_exact_grids_across_flushes_and_thresholds(self):
        rng=np.random.default_rng(280)
        states=rng.uniform(ARM.lower,ARM.upper,(24,6))
        # Include repeated vertices and edges immediately around the threshold.
        a=ARM.center.copy()
        for distance in (0.,np.nextafter(.04,0.),.04,np.nextafter(.04,np.inf)):
            b=a.copy();b[0]+=distance;states=np.vstack((states,a,b))
        seen=[];env=self.sampled_scene(lambda q:seen.append(q.copy()) or True)
        interiors=[env.space.interpolate(a,b,.04)[1:-1] for a,b in zip(states[:-1],states[1:])]
        expected=np.concatenate([states,*interiors])
        self.assertTrue(env.path_valid(states))
        np.testing.assert_array_equal(seen,expected)
        # An interior-only barrier must still reject the route after batching.
        blocked=interiors[5][len(interiors[5])//2]
        env.valid=lambda q:not np.array_equal(q,blocked)
        self.assertFalse(env.path_valid(states))
        for step in (0.,-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):env.path_valid([a],step)

    def test_dense_path_avoids_scalar_edge_dispatch_without_skipping_states(self):
        a=ARM.center.copy();b=a.copy();b[0]+=.2
        states=np.linspace(a,b,1000);seen=[]
        env=self.sampled_scene(lambda q:seen.append(q.copy()) or True)
        with patch.object(env.space,'interpolate',side_effect=AssertionError('Dense edges have no interior samples')):
            self.assertTrue(env.path_valid(states))
        np.testing.assert_array_equal(seen,states)

    def test_cache_keeps_nearby_poses_distinct_and_rejects_reshaped_and_nonfinite_inputs(self):
        env=ArmEnv.__new__(ArmEnv);calls=[];q=ARM.center.copy()
        env.space=JointSpace(lambda state:calls.append(state.copy()) or state[0]<=q[0],
                             {'checker':'cache test','environment':'fixed predicate'})
        env._valid_cache={};env._set_joints=lambda _:None
        env.table=env.box=None;env._display_q=q.copy()
        self.assertTrue(env.valid(q));self.assertTrue(env.valid(q.copy()))
        near=q.copy();near[0]=np.nextafter(near[0],np.inf)
        self.assertFalse(env.valid(near));self.assertFalse(env.valid(near.copy()))
        self.assertEqual(len(calls),2)
        self.assertFalse(env.valid(q.reshape(1,6)))
        bad=q.copy();bad[0]=float('nan');self.assertFalse(env.valid(bad))

    def test_modified_derived_geometry_is_not_loaded(self):
        from simulation.diffusion.env import bullet_urdf
        with tempfile.TemporaryDirectory() as tmp:
            target=bullet_urdf(tmp)
            target.write_text(target.read_text()+'\n')
            with self.assertRaises(ValueError):bullet_urdf(tmp)

    def test_retimed_grazing_collision_is_rejected(self):
        case=json.loads((Path(__file__).parent/'fixtures/grazing_collision.json').read_text())
        with ArmEnv() as env:
            env.set_obstacle(case['obstacle']);env.reset(case['start'])
            self.assertTrue(env.valid(case['start']))
            self.assertTrue(env.valid(case['goal']))
            self.assertFalse(env.valid(case['invalid_state']))
            self.assertFalse(env.path_valid([case['start'],case['invalid_state'],case['goal']]))
            np.testing.assert_allclose(env.current(),case['start'],atol=1e-12)

    def test_fk_and_scene_collision_queries_preserve_execution_state(self):
        with ArmEnv() as env:
            start=np.deg2rad([0,-40,-130,80,0,50])
            env.set_obstacle(None);env.reset(start)
            np.testing.assert_allclose(env.bullet_transform(start),ARM.transform(start),atol=1e-7)
            self.assertTrue(env.valid(start))
            goal=ARM.clip(start+.1)
            env.valid(goal)
            np.testing.assert_allclose(env.current(),start,atol=1e-12)
            point=ARM.fk(goal)
            env.set_obstacle(np.r_[point,[.025,.025,.025]])
            self.assertFalse(env.valid(goal))
            self.assertFalse(env.execute_step(start,goal))
            np.testing.assert_allclose(env.current(),start,atol=1e-12)
            env.set_obstacle(None)
            self.assertTrue(env.valid(goal))
            self.assertFalse(env.execute_step(start,goal,dt=.01))
            np.testing.assert_allclose(env.current(),start,atol=1e-12)
            self.assertTrue(env.execute_step(start,start+.01))
            np.testing.assert_allclose(env.current(),start+.01,atol=1e-12)
            from PIL import Image
            with tempfile.TemporaryDirectory() as tmp:
                image=Path(tmp)/'scene.png';env.render(image,width=320,height=240)
                with Image.open(image) as rendered:self.assertEqual(rendered.size,(320,240))


if __name__=='__main__':unittest.main()
