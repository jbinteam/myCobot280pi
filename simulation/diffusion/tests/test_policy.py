from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from simulation.diffusion.common import robot_schema
from simulation.diffusion.model import ConditionalUNet1D
from simulation.diffusion.policy import DiffusionPolicy, checked_chunk
from simulation.prm import JointSpace


class GateEnv:
    def __init__(self):
        self.space=JointSpace(lambda q:not(abs(q[0])<.15 and abs(q[1])<.3),
            {'checker':'test barrier','environment':'fixture'})
    def path_valid(self,q,step=.04):return all(self.space.valid(s) for s in q) and all(self.space.edge_valid(a,b,step) for a,b in zip(q[:-1],q[1:]))


class PolicyTest(unittest.TestCase):
    def test_collision_gate_checks_whole_chunk_and_speed_projection(self):
        env=GateEnv();start=np.array([-.5,0,0,0,0,0.]);goal=-start
        unsafe=np.linspace(start,goal,16)
        chunk,status=checked_chunk(env,start,unsafe)
        self.assertIsNone(chunk);self.assertEqual(status,'raw_collision')
        start[1]=.5;goal[1]=.5
        chunk,status=checked_chunk(env,start,np.repeat(goal[None],16,axis=0))
        self.assertEqual(status,'accepted')
        self.assertTrue(np.all(np.linalg.norm(np.diff(np.vstack((start,chunk)),axis=0),axis=1)<=.1+1e-10))
        chunk,status=checked_chunk(env,start,np.full((16,6),np.nan))
        self.assertIsNone(chunk);self.assertEqual(status,'malformed')

    def test_safe_load_smoke_guard_and_seeded_batched_denoising(self):
        torch.manual_seed(42)
        model=ConditionalUNet1D(16)
        payload={'configuration':{'robot':robot_schema(),'channels':16,'diffusion_timesteps':100,
                    'scheduler':{'beta_schedule':'squaredcos_cap_v2','prediction_type':'epsilon','clip_sample':True,'timestep_spacing':'linspace'}},
                 'training':{'kind':'smoke','completed_epochs':1},'state_dict':model.state_dict()}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'policy.pth';torch.save(payload,path)
            with self.assertRaises(ValueError):DiffusionPolicy(path,device='cpu')
            one=DiffusionPolicy(path,device='cpu',allow_smoke=True)
            two=DiffusionPolicy(path,device='cpu',allow_smoke=True)
            first=one.sample(np.zeros((2,19)),samples=2,inference_steps=4)
            np.testing.assert_array_equal(first,two.sample(np.zeros((2,19)),samples=2,inference_steps=4))
            self.assertEqual(first.shape,(2,2,16,6));self.assertTrue(np.isfinite(first).all())


if __name__=='__main__':unittest.main()
