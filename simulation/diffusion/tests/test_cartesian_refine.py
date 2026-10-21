"""Resume retains learned targets; auxiliary FK gradients respect noise gating."""
import unittest
import numpy as np
import torch
from simulation.diffusion.cartesian_conditioning import CONDITIONING
from simulation.diffusion.common import robot_schema
from simulation.diffusion.model import ConditionalUNet1D,scheduler
from simulation.diffusion.train_cartesian_refine import clean_fk_loss,resume_state,validation_metrics


class CartesianRefineTests(unittest.TestCase):
    def test_resume_retains_cartesian_target_columns_and_source_state(self):
        torch.manual_seed(17); model=ConditionalUNet1D(8,34)
        original=model.state_dict();checkpoint={'configuration':{'robot':robot_schema(),'conditioning':CONDITIONING},'state_dict':original}
        resumed=resume_state(checkpoint)
        for key in original:torch.testing.assert_close(resumed[key],original[key])
        self.assertGreater(torch.count_nonzero(resumed['observation.0.weight'][:,6:12]).item(),0)
        resumed['observation.0.weight'].zero_()
        self.assertGreater(torch.count_nonzero(original['observation.0.weight']).item(),0)

    def test_exact_epsilon_reconstructs_actual_clean_label_not_requested_goal(self):
        torch.manual_seed(18); clean=torch.rand(2,6,16)*.4-.2; noise=torch.randn_like(clean)
        times=torch.tensor([10,40]); schedule=scheduler();noisy=schedule.add_noise(clean,noise,times)
        loss,fraction=clean_fk_loss(noisy,noise,clean,times,schedule.alphas_cumprod)
        self.assertLess(float(loss),1e-9);self.assertEqual(float(fraction),1.)

    def test_low_noise_has_finite_gradients_and_high_noise_is_excluded(self):
        torch.manual_seed(19);clean=torch.zeros(2,6,16); noise=torch.randn_like(clean)
        times=torch.tensor([40,99]); schedule=scheduler();noisy=schedule.add_noise(clean,noise,times)
        prediction=torch.zeros_like(clean,requires_grad=True)
        loss,fraction=clean_fk_loss(noisy,prediction,clean,times,schedule.alphas_cumprod)
        self.assertEqual(float(fraction),.5); self.assertTrue(torch.isfinite(loss))
        loss.backward();self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(torch.linalg.norm(prediction.grad[0]).item(),0)
        self.assertEqual(torch.count_nonzero(prediction.grad[1]).item(),0)
        all_high=torch.tensor([90,99]);prediction=torch.zeros_like(clean,requires_grad=True)
        loss,fraction=clean_fk_loss(noisy,prediction,clean,all_high,schedule.alphas_cumprod)
        self.assertEqual(float(loss.detach()),0.);self.assertEqual(float(fraction),0.)
        loss.backward();self.assertEqual(torch.count_nonzero(prediction.grad).item(),0)

    def test_mask_uses_full_batch_denominator(self):
        torch.manual_seed(20);clean=torch.zeros(2,6,16);noise=torch.randn_like(clean)
        times=torch.tensor([40,99]);schedule=scheduler();noisy=schedule.add_noise(clean,noise,times)
        prediction=torch.zeros_like(clean)
        mixed,_=clean_fk_loss(noisy,prediction,clean,times,schedule.alphas_cumprod)
        low,_=clean_fk_loss(noisy[:1],prediction[:1],clean[:1],times[:1],schedule.alphas_cumprod)
        torch.testing.assert_close(mixed,low/2)

    def test_fixed_validation_objective_is_batch_partition_independent(self):
        class Model:
            def __call__(self,noisy,time,obs):return torch.zeros_like(noisy)
        data={'q':torch.zeros(4,6),'ends':torch.full((4,),3,dtype=torch.long),'obs':torch.zeros(4,34)}
        index=torch.tensor([0,1]);noise=torch.ones(2,6,16);time=torch.tensor([20,99]);future=torch.arange(1,17)[None]
        schedule=scheduler();args=(Model(),data,index,noise,time,future,schedule,schedule.alphas_cumprod,.02)
        whole=validation_metrics(*args,2);partitioned=validation_metrics(*args,1)
        for name in whole:self.assertAlmostEqual(whole[name],partitioned[name],places=6)
        self.assertEqual(whole['validation_noise_mse'],1.)
        self.assertAlmostEqual(whole['validation_composite'],1.+.02*whole['validation_gated_fk_loss'])


if __name__=='__main__':unittest.main()
