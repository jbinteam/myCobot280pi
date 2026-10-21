import unittest
import numpy as np
import torch
from simulation.diffusion.common import denormalize
from simulation.diffusion.dataset import TrajectoryDataset, read_episodes
from simulation.diffusion.model import ConditionalUNet1D, scheduler
from simulation.diffusion.tests.helpers import demonstrations


class DatasetModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.path=demonstrations()

    def test_future_windows_terminal_padding_and_episode_split(self):
        train=TrajectoryDataset(self.path,'train');validation=TrajectoryDataset(self.path,'validation');test=TrajectoryDataset(self.path,'test')
        self.assertFalse(set(train.episode_ids)&set(validation.episode_ids))
        self.assertFalse(set(train.episode_ids)&set(test.episode_ids))
        self.assertFalse(set(test.episode_ids)&set(validation.episode_ids))
        _,episodes=read_episodes(self.path,'train');first=episodes[0]
        np.testing.assert_allclose(denormalize(train[0]['action'][:,0]),first['q'][1],atol=1e-6)
        terminal=train[len(first['q'])-1]['action'].numpy().T
        np.testing.assert_allclose(denormalize(terminal),np.repeat(first['goal'][None],16,axis=0),atol=1e-6)
        self.assertEqual(train[0]['obs'].shape,(19,));self.assertEqual(train[0]['action'].shape,(6,16))
        self.assertLessEqual(float(train.actions.abs().max()),1.)

    def test_noise_shapes_finite_gradients_and_obstacle_conditioning(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        model=ConditionalUNet1D(16);clean=torch.randn(2,6,16).clamp(-1,1)
        times=torch.tensor([3,80]);noise=torch.randn_like(clean);obs=torch.randn(2,19)
        noisy=scheduler().add_noise(clean,noise,times)
        prediction=model(noisy,times,obs)
        self.assertEqual(prediction.shape,clean.shape)
        torch.nn.functional.mse_loss(prediction,noise).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        changed=obs.clone();changed[:,12:]+=1
        self.assertFalse(torch.allclose(prediction,model(noisy,times,changed)))
        with self.assertRaises(ValueError):model(torch.randn(2,6,15),times,obs)


if __name__=='__main__':unittest.main()
