import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from simulation.diffusion.common import normalize
from simulation.diffusion.trajectory_transformer import TrajectoryTransformer, TransformerPolicy, configuration
from simulation.kinematics import ARM

class TransformerTest(unittest.TestCase):
    def test_posterior_and_decoder_finite_gradients(self):
        torch.set_num_threads(2);m=TrajectoryTransformer();obs=torch.zeros(2,34);action=torch.ones(2,16,6)*.1
        pred,mean,logvar=m(obs,action,torch.zeros(2,16))
        ((pred-action).square().mean()+.001*(mean.square()+logvar.exp()-logvar-1).mean()).backward()
        self.assertEqual(pred.shape,(2,16,6))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_seeded_serving_bounds_shapes_and_checkpoint_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.pth';payload={'configuration':configuration(),'training':{'completed':True,'completed_steps':1},'state_dict':TrajectoryTransformer().state_dict()}
            torch.save(payload,path);policy=TransformerPolicy(path,'cpu',42)
            obs=np.zeros((2,34));obs[:,:6]=normalize(ARM.center)
            a=policy.sample(obs);policy.generator.manual_seed(42);b=policy.sample(obs)
            np.testing.assert_array_equal(a,b);self.assertEqual(a.shape,(2,4,16,6))
            self.assertTrue(np.all(a>=ARM.lower) and np.all(a<=ARM.upper))
            with self.assertRaises(ValueError):policy.sample(np.full((1,34),np.nan))
            with self.assertRaises(ValueError):policy.sample(obs,samples=True)
            payload['training']['completed']=False;torch.save(payload,path)
            with self.assertRaises(ValueError):TransformerPolicy(path,'cpu')

if __name__=='__main__':unittest.main()
