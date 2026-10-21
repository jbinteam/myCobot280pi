"""Relative Cartesian semantics, FK parity and immutable split/cache conversion."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from simulation.kinematics import ARM
from simulation.diffusion.cartesian_conditioning import CONDITIONING, RELATIVE_CONDITIONING,observe,observe_relative,observe_for_policy,supports
from simulation.diffusion.cartesian_relative_cache import build_cache as relative_cache,relative_observations
from simulation.diffusion.common import file_sha256,robot_schema
from simulation.diffusion.model import ConditionalUNet1D
from simulation.diffusion.train_cartesian import build_cache
from simulation.diffusion.train_cartesian_relative import resume_state
from simulation.diffusion.tests import test_cartesian_training as fixtures


class CartesianRelativeTests(unittest.TestCase):
    def test_explicit_residual_matches_shared_fk_and_preserves_other_channels(self):
        q=ARM.center.copy();q[0]+=.2;current=ARM.transform(q)
        target=current[:3,3]+[.03,-.06,.09];axis=current[:3,2]+[.1,-.1,.05];axis/=np.linalg.norm(axis)
        absolute=observe(q,target,axis,None,[.1,0,0,0,0,0])
        relative=observe_relative(q,target,axis,None,[.1,0,0,0,0,0])
        np.testing.assert_allclose(relative[6:9],[.1,-.2,.3],atol=1e-7)
        np.testing.assert_allclose(relative[9:12],axis-current[:3,2],atol=1e-7)
        np.testing.assert_array_equal(relative[:6],absolute[:6]);np.testing.assert_array_equal(relative[12:],absolute[12:])
        np.testing.assert_allclose(relative_observations(absolute[None])[0],relative,atol=2e-7)
        v3=type('V3',(),{'conditioning':CONDITIONING})();v4=type('V4',(),{'conditioning':RELATIVE_CONDITIONING})()
        self.assertTrue(supports(v3));self.assertTrue(supports(v4))
        np.testing.assert_array_equal(observe_for_policy(v3,q,target,axis),observe(q,target,axis))
        np.testing.assert_array_equal(observe_for_policy(v4,q,target,axis),observe_relative(q,target,axis))
        hold=observe_relative(q,current[:3,3],current[:3,2]);np.testing.assert_allclose(hold[6:12],0,atol=1e-7)

    def test_v3_warm_start_zeroes_only_changed_target_basis(self):
        model=ConditionalUNet1D(8,34);original=model.state_dict()
        state=resume_state({'configuration':{'robot':robot_schema(),'conditioning':CONDITIONING},'state_dict':original})
        torch.testing.assert_close(state['observation.0.weight'][:,:6],original['observation.0.weight'][:,:6])
        torch.testing.assert_close(state['observation.0.weight'][:,12:],original['observation.0.weight'][:,12:])
        self.assertEqual(torch.count_nonzero(state['observation.0.weight'][:,6:12]).item(),0)
        self.assertGreater(torch.count_nonzero(original['observation.0.weight'][:,6:12]).item(),0)

    def test_cache_split_ranges_and_source_arrays_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);source,augmented,_=fixtures.CartesianTrainingTests().make_data(folder)
            absolute=folder/'absolute';relative=folder/'relative'
            original=build_cache(source,augmented,absolute,chunk_frames=3)
            hashes={p.name:file_sha256(p) for p in absolute.iterdir()}
            converted=relative_cache(source,augmented,relative,source_cache=absolute,chunk_frames=2)
            self.assertEqual(converted['identity']['conditioning'],RELATIVE_CONDITIONING)
            self.assertEqual(converted['episodes'],original['episodes']);self.assertEqual(converted['frames'],original['frames'])
            self.assertFalse(converted['identity']['conditioning']['goal_joint_input'])
            self.assertNotEqual(converted['identity']['dataset_sha256'],original['identity']['dataset_sha256'])
            for split in ('train','validation','test'):
                for key in ('q','ends','ranges_0','ranges_1'):
                    self.assertEqual(file_sha256(relative/f'{split}_{key}.npy'),file_sha256(absolute/f'{split}_{key}.npy'))
                expected=relative_observations(np.load(absolute/f'{split}_obs.npy'))
                np.testing.assert_array_equal(np.load(relative/f'{split}_obs.npy'),expected)
            self.assertEqual({p.name:file_sha256(p) for p in absolute.iterdir()},hashes)
            self.assertEqual(relative_cache(source,augmented,relative,source_cache=absolute),converted)
            with self.assertRaisesRegex(ValueError,'Preserve'):relative_cache(source,augmented,absolute,source_cache=absolute)


if __name__=='__main__':unittest.main()
