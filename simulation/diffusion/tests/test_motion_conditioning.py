"""Motion/schema isolation and causal augmented demonstration contracts."""
import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
import torch
from simulation.diffusion.common import observation
from simulation.diffusion.motion_conditioning import extend,encode
from simulation.diffusion.model import ConditionalUNet1D
from simulation.diffusion.generate_motion import worker,derivatives
from simulation.kinematics import ARM


class MotionConditioningTest(unittest.TestCase):
    def test_motion_channels_are_normalized_without_changing_scene_mask(self):
        base=observation(ARM.center,ARM.center,None)
        result=extend(base,[.5,0,0,0,0,0],[1.5,0,0,0,0,0],[0,.25,0])
        self.assertEqual(result.shape,(34,));np.testing.assert_array_equal(result[:19],base)
        np.testing.assert_array_equal(result[12:19],np.zeros(7))
        self.assertEqual(result[19],.5);self.assertEqual(result[25],.5);self.assertEqual(result[32],.5)
        with self.assertRaises(ValueError):extend(base,[2,0,0,0,0,0])
        with self.assertRaises(ValueError):extend(base,target_velocity=[float('nan'),0,0])
        self.assertAlmostEqual(np.linalg.norm(extend(base,target_velocity=[0,2,0])[-3:]),1.)

    def test_old_policy_gets_exact_old_input_and_new_policy_uses_motion(self):
        base=observation(ARM.center,ARM.center,None)
        old=type('Old',(),{'observation_dim':19})();new=type('New',(),{'observation_dim':34})()
        np.testing.assert_array_equal(encode(old,base,[.1]*6),base)
        self.assertEqual(encode(new,base,[.1]*6).shape,(34,))

    def test_warm_start_preserves_v1_predictions_before_training(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        old=ConditionalUNet1D(8);new=ConditionalUNet1D(8,34);state=old.state_dict()
        expanded=torch.zeros((128,34));expanded[:,:19]=state['observation.0.weight'];state['observation.0.weight']=expanded
        new.load_state_dict(state);q=torch.randn(2,6,16);base=torch.randn(2,19);extra=torch.randn(2,15)
        torch.testing.assert_close(old(q,torch.tensor([1,2]),base),new(q,torch.tensor([1,2]),torch.cat((base,extra),dim=1)))

    def test_augmentation_inherits_parent_split_and_has_actual_moving_goals(self):
        source=Path('simulation/artifacts/diffusion_100k_v1/demonstrations.h5')
        if not source.exists():self.skipTest('100k parent dataset unavailable')
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'shard.h5';worker((str(source),str(output),['000000'],['moving_absent'],123))
            with h5py.File(source,'r') as parents,h5py.File(output,'r') as data:
                group=data['episodes/000000'];self.assertEqual(group.attrs['split'],parents['episodes/000000'].attrs['split'])
                self.assertFalse(group.attrs['obstacle_present']);goals=group['goal_rad'][:]
                self.assertGreater(np.linalg.norm(np.diff(goals,axis=0)),0)
                np.testing.assert_array_equal(goals[-1],group['q_rad'][-1])
                self.assertGreater(np.linalg.norm(group['target_velocity_m_s'][:]),0)
                self.assertLessEqual(np.linalg.norm(group['velocity_rad_s'][:],axis=1).max(),1+1e-6)

    def test_full_preview_carries_reference_velocity_between_decisions(self):
        from time import perf_counter
        from simulation.diffusion import service
        from simulation.diffusion.common import denormalize
        class Scene:
            def __init__(self):self.q=ARM.center.copy()
            def current(self):return self.q.copy()
            def path_valid(self,path,step=.04):return True
            def execute_step(self,current,target,dt,speed):self.q=target.copy();return True
        class Policy:
            observation_dim=34
            def __init__(self):self.inputs=[]
            def sample(self,obs,samples,steps):
                self.inputs.append(obs.copy());current=denormalize(obs[0,:6])
                chunk=np.repeat(current[None],16,axis=0);chunk[:,0]+=np.arange(1,17)*.01
                return chunk[None,None]
        scene=Scene();policy=Policy();goal=ARM.center.copy();goal[0]+=.8;pose=ARM.transform(goal)
        service.rollout(scene,policy,scene.q,goal,pose[:3,3],pose[:3,2],None,
                        {'max_decisions':2,'samples':1,'inference_steps':8},perf_counter()+10)
        self.assertEqual(len(policy.inputs),2);self.assertEqual(policy.inputs[0][0,19],0)
        self.assertGreater(policy.inputs[1][0,19],.09)
        self.assertLessEqual(np.linalg.norm(policy.inputs[1][0,25:31]),1+1e-6)

    def test_full_horizon_controller_advances_checked_curves_from_rest(self):
        import json
        from time import perf_counter
        from simulation.diffusion.continuous_rollout import rollout
        from simulation.diffusion.common import denormalize
        from simulation.diffusion.env import ArmEnv
        case=json.loads(Path('simulation/artifacts/diffusion_v1/webui.json').read_text())['cases'][1]
        start=np.deg2rad(case['start_deg']);goal=np.deg2rad(case['goal_deg']);pose=ARM.transform(goal)
        class Policy:
            observation_dim=34
            def __init__(self):self.inputs=[]
            def sample(self,obs,samples,steps):
                self.inputs.append(obs.copy());q=denormalize(obs[0,:6])
                return np.linspace(q,goal,17)[1:][None,None]
        policy=Policy()
        with ArmEnv() as env:
            env.set_obstacle(None);env.reset(start)
            points,status,*_=rollout(env,policy,start,goal,pose[:3,3],pose[:3,2],None,
                                   {'max_decisions':40,'samples':1,'inference_steps':8},perf_counter()+15)
            self.assertEqual(status,'success');self.assertTrue(env.path_valid(points))
        self.assertGreater(len(policy.inputs),1);self.assertGreater(np.linalg.norm(policy.inputs[1][0,19:25]),.01)
        self.assertLess(np.linalg.norm(points[1]-points[0]),.01)
        self.assertTrue(all(np.linalg.norm(item[0,19:25])<=1+1e-6 and np.linalg.norm(item[0,25:31])<=1+1e-6 for item in policy.inputs))

    def test_fixed_v2_uses_original_continuous_rollout_and_final_shortcuts(self):
        from unittest.mock import patch
        from simulation.diffusion import service
        from simulation.diffusion.motion_conditioning import CONDITIONING
        status=service.info();start=np.deg2rad(status['cases'][1]['start_deg'])
        goal=start.copy();goal[0]+=.04;pose=ARM.transform(goal)
        policy=type('Policy',(),{'observation_dim':34,'conditioning':CONDITIONING,
                               'sha256':status['checkpoint_sha256']})()
        points=np.linspace(start,goal,3)
        body={'start_deg':np.rad2deg(start).tolist(),'goal_deg':np.rad2deg(goal).tolist(),
              'target_mm':(pose[:3,3]*1000).tolist(),'target_axis':pose[:3,2].tolist(),'obstacle_mm':None}
        with patch.object(service,'_policy',policy),patch.object(service,'load_policy',return_value=policy),\
             patch('simulation.diffusion.continuous_rollout.rollout',return_value=(points,'success',1,4,{},1)) as runner:
            result=service.plan(body)
        runner.assert_called_once()
        self.assertTrue(result['converged'],result['reason'])
        self.assertEqual(result['rollout_controller'],'full_horizon_c2')
        self.assertEqual(result['smoothing']['path_vertices'],2)
        self.assertEqual(len(result['smoothing']['segments']),1)
        self.assertTrue(np.isfinite(result['joint_travel_deg']))
        self.assertTrue(all('time' in frame and 'position' in frame for frame in result['frames']))


if __name__=='__main__':unittest.main()
