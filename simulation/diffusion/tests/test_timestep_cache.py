"""Exact lookup, step output, RNG and restoration tests; CPU only by default."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from diffusers import DDPMScheduler
from simulation.diffusion.model import scheduler, ConditionalUNet1D
from simulation.diffusion.common import robot_schema
from simulation.diffusion.policy import DiffusionPolicy
from simulation.diffusion.motion_conditioning import CONDITIONING
from simulation.diffusion import timestep_cache
from simulation.diffusion.timestep_cache import PreviousTimestepCache


class PreviousTimestepCacheTest(unittest.TestCase):
    @torch.inference_mode()
    def test_every_step_and_rng_bit_exact_varied_schedules_and_shapes(self):
        torch.set_num_threads(1)
        for count in (1,8,32,100):
            for shape in ((1,6,16),(4,6,16),(2,3,7)):
                with self.subTest(steps=count,shape=shape):
                    stock, cached=scheduler(),scheduler()
                    stock.set_timesteps(count,device='cpu'); cached.set_timesteps(count,device='cpu')
                    a=torch.Generator().manual_seed(421); b=torch.Generator().manual_seed(421)
                    start=torch.linspace(-.8,.8,int(np.prod(shape))).reshape(shape)
                    left,right=start.clone(),start.clone()
                    with PreviousTimestepCache(cached) as lookup:
                        self.assertTrue(lookup.active)
                        for index,t in enumerate(lookup):
                            # A deterministic nontrivial prediction, independent of RNG.
                            noise=torch.sin(left*.17+index*.03)
                            original=stock.step(noise,stock.timesteps[index],left,generator=a)
                            result=cached.step(noise,t,right,generator=b)
                            self.assertTrue(torch.equal(result.prev_sample,original.prev_sample))
                            self.assertTrue(torch.equal(result.pred_original_sample,original.pred_original_sample))
                            self.assertTrue(torch.equal(a.get_state(),b.get_state()))
                            left,right=original.prev_sample,result.prev_sample
                        self.assertEqual(lookup.hits,2*count-1)
                        self.assertEqual(lookup.misses,0)
                    self.assertNotIn('previous_timestep',vars(cached))

    def test_same_object_only_ordering_terminal_and_fallbacks(self):
        schedule=scheduler(); schedule.set_timesteps(8)
        original=schedule.previous_timestep
        with PreviousTimestepCache(schedule) as lookup:
            for index,t in enumerate(lookup):
                expected=original(t)
                actual=schedule.previous_timestep(t)
                self.assertTrue(torch.equal(actual,expected))
                self.assertEqual(actual.dtype,expected.dtype)
                self.assertEqual(actual.device,expected.device)
                self.assertEqual(actual.ndim,0)
                # Equal-value copies and Python scalars must use the stock path.
                self.assertTrue(torch.equal(schedule.previous_timestep(t.clone()),expected))
                self.assertTrue(torch.equal(schedule.previous_timestep(int(t)),expected))
                if index:
                    self.assertTrue(torch.equal(schedule.previous_timestep(previous),original(previous)))
                previous=t
            self.assertEqual(lookup.hits,8)
            self.assertEqual(lookup.misses,23)
            self.assertTrue(torch.equal(schedule.previous_timestep(previous),original(previous)))
            self.assertEqual(lookup.misses,24)
        self.assertEqual(schedule.previous_timestep.__func__,original.__func__)

    def test_schedule_changes_and_exception_restore_instance_state(self):
        schedule=scheduler(); schedule.set_timesteps(32)
        original=schedule.previous_timestep
        # Preserve an existing explicit stock bound-method instance attribute.
        schedule.previous_timestep=original
        with self.assertRaisesRegex(RuntimeError,'test exception'):
            with PreviousTimestepCache(schedule) as lookup:
                iterator=iter(lookup)
                t=next(iterator)
                schedule.set_timesteps(8)
                self.assertTrue(torch.equal(schedule.previous_timestep(t),original(t)))
                self.assertEqual(lookup.hits,0);self.assertEqual(lookup.misses,1)
                raise RuntimeError('test exception')
        self.assertIs(vars(schedule)['previous_timestep'],original)
        for count in (8,32,1,100):
            schedule.set_timesteps(count)
            with PreviousTimestepCache(schedule) as lookup:
                for t in lookup:
                    self.assertTrue(torch.equal(schedule.previous_timestep(t),original(t)))
                self.assertEqual(lookup.hits,count)

    def test_unsupported_version_type_methods_and_custom_schedules_bypass(self):
        schedule=scheduler();schedule.set_timesteps(8)
        with patch.object(timestep_cache.diffusers,'__version__','future'):
            with PreviousTimestepCache(schedule) as lookup:
                self.assertFalse(lookup.active)
                self.assertEqual(len(list(lookup)),8)
        class Alternate(DDPMScheduler): pass
        alt=Alternate(num_train_timesteps=100);alt.set_timesteps(8)
        with PreviousTimestepCache(alt) as lookup: self.assertFalse(lookup.active)
        original=schedule.previous_timestep
        schedule.previous_timestep=lambda t:original(t)
        with PreviousTimestepCache(schedule) as lookup: self.assertFalse(lookup.active)
        del schedule.previous_timestep
        for custom in ([99,70,15,0],[3.9,3.1]):
            schedule.set_timesteps(timesteps=custom)
            with PreviousTimestepCache(schedule) as lookup:
                self.assertFalse(lookup.active)
                for t in lookup:
                    self.assertTrue(torch.equal(schedule.previous_timestep(t),original(t)))
        schedule.set_timesteps(8)
        with PreviousTimestepCache(schedule,enabled=False) as lookup:
            self.assertFalse(lookup.active);list(lookup)
        self.assertNotIn('previous_timestep',vars(schedule))

    def test_no_tensor_scalar_conversion_on_cache_hit(self):
        schedule=scheduler();schedule.set_timesteps(8)
        with PreviousTimestepCache(schedule) as lookup:
            for t in lookup:
                with patch.object(torch.Tensor,'item',side_effect=AssertionError('no item on hit')), \
                     patch.object(torch.Tensor,'__int__',side_effect=AssertionError('no int on hit')), \
                     patch.object(torch.Tensor,'nonzero',side_effect=AssertionError('no timetable search on hit')):
                    value=schedule.previous_timestep(t)
                self.assertEqual(value.ndim,0)


class PolicyCacheTest(unittest.TestCase):
    def test_seeded_policy_predictions_rng_and_counters_exact_on_cpu(self):
        for width in (19,34):
            with self.subTest(observation_dim=width):
                self._check_policy(width)

    def _check_policy(self,width):
        torch.manual_seed(6)
        model=ConditionalUNet1D(16,width)
        payload={'configuration':{'robot':robot_schema(),'channels':16,'diffusion_timesteps':100,
                 'scheduler':{'beta_schedule':'squaredcos_cap_v2','prediction_type':'epsilon','clip_sample':True,'timestep_spacing':'linspace'}},
                 'training':{'kind':'smoke','completed_epochs':1},'state_dict':model.state_dict()}
        if width == 34: payload['configuration']['conditioning']=CONDITIONING
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'policy.pth';torch.save(payload,path)
            plain=DiffusionPolicy(path,device='cpu',allow_smoke=True,scheduler_cache=False)
            cached=DiffusionPolicy(path,device='cpu',allow_smoke=True,scheduler_cache=True)
            for steps,samples in ((1,1),(8,2),(32,4)):
                plain.generator.manual_seed(42);cached.generator.manual_seed(42)
                expected=plain.sample(np.zeros((1,width)),samples=samples,inference_steps=steps)
                actual=cached.sample(np.zeros((1,width)),samples=samples,inference_steps=steps)
                np.testing.assert_array_equal(actual,expected)
                self.assertTrue(torch.equal(cached.generator.get_state(),plain.generator.get_state()))
            self.assertEqual(plain._scheduler_cache_hits,0)
            self.assertEqual(cached._scheduler_cache_hits,(2*1-1)+(2*8-1)+(2*32-1))
            self.assertEqual(cached._scheduler_cache_misses,0)
            with patch.object(cached.model,'forward',side_effect=RuntimeError('denoiser failed')):
                with self.assertRaisesRegex(RuntimeError,'denoiser failed'):
                    cached.sample(np.zeros((1,width)),samples=1,inference_steps=8)
            self.assertNotIn('previous_timestep',vars(cached.scheduler))


if __name__=='__main__': unittest.main()
