"""Exact inference and RNG parity for unchanged eager-kernel CUDA replay."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from simulation.diffusion.cuda_replay import DenoiserReplay
from simulation.diffusion.model import ConditionalUNet1D
from simulation.diffusion import service


@unittest.skipUnless(torch.cuda.is_available(),'CUDA replay requires a GPU')
class CUDAReplayTest(unittest.TestCase):
    @torch.inference_mode()
    def test_changing_inputs_timesteps_and_retained_outputs_match_eager(self):
        torch.manual_seed(281)
        model=ConditionalUNet1D(16,34).cuda().eval();replay=DenoiserReplay(model)
        retained=[]
        for timestep in (99,53,0,71):
            args=(torch.randn(4,6,16,device='cuda'),torch.tensor(timestep,device='cuda'),torch.randn(4,34,device='cuda'))
            expected=model(*args);actual=replay(*args)
            self.assertTrue(torch.equal(actual,expected))
            retained.append((actual,expected.clone()))
        self.assertIsNotNone(replay.graph)
        for actual,expected in retained:self.assertTrue(torch.equal(actual,expected))
        graph=replay.graph
        args=(torch.randn(2,6,16,device='cuda'),torch.tensor(12,device='cuda'),torch.randn(2,34,device='cuda'))
        self.assertTrue(torch.equal(replay(*args),model(*args)))
        self.assertIs(replay.graph,graph)

    @torch.inference_mode()
    def test_initial_mismatch_disables_replay_and_returns_eager_output(self):
        model=ConditionalUNet1D(16,34).cuda().eval();replay=DenoiserReplay(model)
        args=(torch.randn(4,6,16,device='cuda'),torch.tensor(99,device='cuda'),torch.randn(4,34,device='cuda'))
        expected=model(*args)
        with patch('simulation.diffusion.cuda_replay.torch.equal',return_value=False):
            with self.assertWarnsRegex(RuntimeWarning,'using eager inference'):actual=replay(*args)
        self.assertTrue(torch.equal(actual,expected));self.assertTrue(replay.disabled)
        self.assertIsNone(replay.graph)
        self.assertTrue(torch.equal(replay(*args),expected))

    def test_deployed_policy_samples_and_rng_state_are_exact(self):
        status=service.info()
        if status.get('model_name')!='diffusion_motion_v2':self.skipTest('Pinned v2 unavailable')
        from simulation.diffusion.policy import DiffusionPolicy
        policy=DiffusionPolicy(status['checkpoint_path'],device='cuda',cuda_graph=True)
        obs=np.random.default_rng(47).normal(0,.2,(1,34)).astype(np.float32)
        for seed,steps,samples in ((42,32,4),(19,16,4),(0,32,4),(42,8,2)):
            policy.cuda_graph=False;policy.generator.manual_seed(seed)
            expected=policy.sample(obs,samples=samples,inference_steps=steps);state=policy.generator.get_state()
            policy.cuda_graph=True;policy.generator.manual_seed(seed)
            actual=policy.sample(obs,samples=samples,inference_steps=steps)
            np.testing.assert_array_equal(actual,expected)
            self.assertTrue(torch.equal(policy.generator.get_state(),state))
        self.assertIsNotNone(policy._replay.graph)

    @torch.inference_mode()
    def test_unsupported_capture_falls_back_once_but_fatal_faults_propagate(self):
        model=ConditionalUNet1D(16,34).cuda().eval()
        args=(torch.randn(4,6,16,device='cuda'),torch.tensor(99,device='cuda'),torch.randn(4,34,device='cuda'))
        expected=model(*args);replay=DenoiserReplay(model)
        with patch('simulation.diffusion.cuda_replay.torch.cuda.graph',
                   side_effect=RuntimeError('CUDA error: operation not permitted when stream is capturing')) as capture:
            with self.assertWarnsRegex(RuntimeWarning,'capture unsupported'):actual=replay(*args)
            self.assertTrue(torch.equal(actual,expected));self.assertTrue(replay.disabled)
            self.assertTrue(torch.equal(replay(*args),expected));self.assertEqual(capture.call_count,1)
        fatal=DenoiserReplay(model)
        with patch('simulation.diffusion.cuda_replay.torch.cuda.graph',
                   side_effect=RuntimeError('CUDA error: an illegal memory access was encountered')):
            with self.assertRaisesRegex(RuntimeError,'illegal memory access'):fatal(*args)
        self.assertFalse(fatal.disabled);self.assertIsNone(fatal.graph)


if __name__=='__main__':unittest.main()
