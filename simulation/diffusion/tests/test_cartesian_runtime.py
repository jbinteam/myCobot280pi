"""Cartesian runtime must work even when every endpoint IK path is disabled."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from simulation.kinematics import ARM
from simulation.diffusion import service,tracking
from simulation.diffusion.cartesian_conditioning import CONDITIONING,RELATIVE_CONDITIONING,supports
from simulation.diffusion.common import denormalize,robot_schema
from simulation.diffusion.env import ArmEnv
from simulation.diffusion.model import ConditionalUNet1D
from simulation.diffusion.motion_conditioning import CONDITIONING as MOTION_CONDITIONING
from simulation.diffusion.motion_conditioning import encode
from simulation.diffusion.policy import DiffusionPolicy


class Proposals:
    """Controlled safe proposals isolate runtime behavior from model accuracy."""
    observation_dim=34
    conditioning=CONDITIONING
    sha256='test-cartesian-checkpoint'

    def __init__(self,goal):self.goal=goal;self.inputs=[]

    def sample(self,obs,samples,steps):
        self.inputs.append(obs.copy())
        current=denormalize(obs[0,:6])
        return np.repeat(np.linspace(current,self.goal,17)[1:][None,None],samples,axis=1)


class CartesianRuntimeTest(unittest.TestCase):
    def setUp(self):
        manifest=json.loads(Path('simulation/artifacts/diffusion_v1/webui.json').read_text())
        self.start=np.deg2rad(manifest['cases'][1]['start_deg'])
        self.goal=self.start.copy();self.goal[0]+=.08
        pose=ARM.transform(self.goal);self.target=pose[:3,3];self.axis=pose[:3,2]
        self.policy=Proposals(self.goal)
        self.expected_position=self.target/ARM.scale;self.expected_axis=self.axis
        self.status={'status':'ready','conditioning':CONDITIONING,'observation_dim':34,
            'environment':manifest['environment'],'checkpoint_sha256':self.policy.sha256,'no_block_note':''}
        self.body={'start_deg':np.rad2deg(self.start).tolist(),'target_mm':(1000*self.target).tolist(),
            'target_axis':self.axis.tolist(),'obstacle_mm':None,'samples':1,'max_decisions':50}

    def disabled_ik(self):
        stack=ExitStack()
        stack.enter_context(patch.object(service,'info',return_value=self.status))
        stack.enter_context(patch.object(service,'load_policy',return_value=self.policy))
        stack.enter_context(patch.object(service,'_policy',self.policy))
        stack.enter_context(patch.object(ArmEnv,'inverse_kinematics',side_effect=AssertionError('Endpoint IK forbidden')))
        stack.enter_context(patch.object(tracking.pose5,'solve',side_effect=RuntimeError('All IK disabled')))
        stack.enter_context(patch.object(tracking,'local_pose_plan',side_effect=RuntimeError('All IK disabled')))
        stack.enter_context(patch.object(tracking,'checked_goal',side_effect=AssertionError('Joint goal forbidden')))
        return stack

    def test_preview_samples_and_succeeds_without_any_ik_or_goal_joint_capture(self):
        with self.disabled_ik():
            result=service.plan({**self.body,'goal_deg':['ignored','even','invalid','goal']})
        self.assertTrue(result['converged'],result)
        self.assertGreater(len(self.policy.inputs),0)
        self.assertEqual(result['endpoint_capture_steps'],0)
        self.assertIsNone(result['goal_deg'])
        self.assertEqual(result['rollout_controller'],'full_horizon_c2')
        self.assertLessEqual(result['final_error_mm'],1.)
        self.assertLessEqual(result['final_orientation_error_deg'],1.)
        np.testing.assert_allclose(self.policy.inputs[0][0,6:9],self.expected_position,atol=1e-7)
        np.testing.assert_allclose(self.policy.inputs[0][0,9:12],self.expected_axis,atol=1e-7)
        with ArmEnv() as env:
            env.set_obstacle(None)
            self.assertTrue(env.path_valid(np.deg2rad([f['q'] for f in result['frames']])))

    def test_tracking_returns_checked_c2_frames_when_all_ik_fails(self):
        body={**self.body,'cursor_mm':(1000*self.target+50*self.axis).tolist(),
              'continuous_motion':True,'goal_hint_deg':['ignored']}
        with self.disabled_ik():result=tracking.plan(body)
        self.assertTrue(result['playable'],result)
        self.assertEqual(result['tracking_source'],'diffusion_fallback')
        self.assertEqual(result['ik_outcome'],'not_required')
        self.assertEqual(result['motion_conditioning']['schema'],self.policy.conditioning['schema'])
        self.assertEqual(result['motion_conditioning']['target_kind'],'cartesian_pose5')
        self.assertIsNone(result['goal_deg']);self.assertIsNone(result['goal_cursor_acceptance'])
        self.assertEqual(result['reference_steps'],16)
        self.assertTrue(result['smoothing']['collision_rechecked'])
        self.assertLessEqual(result['max_velocity_rad_s'],1.+1e-9)
        self.assertLessEqual(result['max_acceleration_rad_s2'],3.+1e-9)
        self.assertEqual(len(self.policy.inputs),1)
        json.dumps(result,allow_nan=False)

    def test_arbitrary_joint_goal_cannot_change_v3_observation_or_ranking(self):
        first=service.condition(self.policy,self.start,None,self.target,self.axis,None)
        second=service.condition(self.policy,self.start,np.full(6,np.nan),self.target,self.axis,None)
        np.testing.assert_array_equal(first,second)
        chunk=np.linspace(self.start,self.goal,16)
        self.assertEqual(service.candidate_score(self.policy,chunk,None,self.target,self.axis),
                         service.candidate_score(self.policy,chunk,np.full(6,np.nan),self.target,self.axis))
        # v2 has the same width but its semantic schema retains joint goals.
        legacy=type('Legacy',(),{'observation_dim':34,'conditioning':MOTION_CONDITIONING})()
        self.assertFalse(supports(legacy))
        old=service.condition(legacy,self.start,self.goal,self.target,self.axis,None)
        self.assertFalse(np.array_equal(first[6:12],old[6:12]))
        with self.assertRaisesRegex(ValueError,'incompatible'):
            encode(self.policy,old[:19])
        foreign=type('Foreign',(),{'observation_dim':34,'conditioning':{'schema':4}})()
        with self.assertRaisesRegex(ValueError,'incompatible'):
            encode(foreign,old[:19])

    def test_equal_width_loader_persists_semantic_schema(self):
        for schema in (MOTION_CONDITIONING,CONDITIONING,RELATIVE_CONDITIONING):
            with self.subTest(schema=schema['schema']),tempfile.TemporaryDirectory() as directory:
                payload={'configuration':{'robot':robot_schema(),'channels':8,'conditioning':schema,
                    'diffusion_timesteps':100,'scheduler':{'beta_schedule':'squaredcos_cap_v2',
                    'prediction_type':'epsilon','clip_sample':True,'timestep_spacing':'linspace'}},
                    'training':{'kind':'full','completed_epochs':1},
                    'state_dict':ConditionalUNet1D(8,34).state_dict()}
                path=Path(directory)/'model.pth';torch.save(payload,path)
                policy=DiffusionPolicy(path,device='cpu')
                self.assertEqual(policy.conditioning,schema)
                self.assertEqual(supports(policy),schema['schema'] in (3,4))

    def test_static_capture_selects_only_existing_neural_point_and_disables_for_motion(self):
        overshoot=self.goal.copy();overshoot[0]+=.08
        chunk=np.vstack((np.repeat(self.start[None],7,axis=0),self.goal,
                         np.repeat(overshoot[None],8,axis=0)))
        prefix,captured=service.static_capture_prefix(chunk,self.target,self.axis)
        self.assertTrue(captured);np.testing.assert_array_equal(prefix,chunk[:8])
        moving,captured=service.static_capture_prefix(chunk,self.target,self.axis,[.01,0,0])
        self.assertFalse(captured);np.testing.assert_array_equal(moving,chunk)
        opposite,captured=service.static_capture_prefix(chunk,self.target,-self.axis)
        self.assertFalse(captured);np.testing.assert_array_equal(opposite,chunk)

    def test_tracking_captures_strict_neural_interior_pose_before_overshoot(self):
        overshoot=self.goal.copy();overshoot[0]+=.08
        chunk=np.vstack((np.repeat(self.start[None],7,axis=0),self.goal,
                         np.repeat(overshoot[None],8,axis=0)))
        body={**self.body,'cursor_mm':(1000*self.target+50*self.axis).tolist(),'continuous_motion':True}
        with self.disabled_ik(),patch.object(self.policy,'sample',return_value=chunk[None,None]):
            result=tracking.plan(body)
        self.assertTrue(result['playable'],result);self.assertTrue(result['converged'],result)
        self.assertTrue(result['static_cartesian_capture']);self.assertEqual(result['reference_steps'],8)
        np.testing.assert_allclose(result['frames'][-1]['q'],np.rad2deg(self.goal),atol=1e-10)
        self.assertTrue(result['smoothing']['collision_rechecked'])
        self.assertLessEqual(result['final_error_mm'],1.);self.assertLessEqual(result['final_orientation_error_deg'],1.)

    def test_static_capture_honors_cursor_constraint_as_well_as_strict_pose(self):
        # A submillimeter sideways position miss can still exceed the cursor's
        # one-degree pointing tolerance over the required 50 mm cursor gap.
        almost=self.goal.copy();almost[1]+=.004
        cursor=1000*self.target+50*self.axis
        error=service.errors(almost,self.target,self.axis)
        self.assertLessEqual(error['error_mm'],1.);self.assertLessEqual(error['orientation_error_deg'],1.)
        accept=lambda q:tracking.cursor_metrics(q,cursor,1.,1.)['accepted']
        self.assertFalse(accept(almost));self.assertTrue(accept(self.goal))
        overshoot=self.goal.copy();overshoot[0]+=.08
        chunk=np.vstack((np.repeat(self.start[None],6,axis=0),almost,self.goal,
                         np.repeat(overshoot[None],8,axis=0)))
        pose_only,captured=service.static_capture_prefix(chunk,self.target,self.axis)
        self.assertTrue(captured);self.assertEqual(len(pose_only),7)
        constrained,captured=service.static_capture_prefix(chunk,self.target,self.axis,accept=accept)
        self.assertTrue(captured);np.testing.assert_array_equal(constrained,chunk[:8])
        body={**self.body,'cursor_mm':cursor.tolist(),'continuous_motion':True}
        with self.disabled_ik(),patch.object(self.policy,'sample',return_value=chunk[None,None]):
            result=tracking.plan(body)
        self.assertTrue(result['converged'],result);self.assertEqual(result['reference_steps'],8)
        self.assertTrue(result['cursor_acceptance']['accepted'])
        np.testing.assert_allclose(result['frames'][-1]['q'],np.rad2deg(self.goal),atol=1e-10)

    def test_strict_neural_prefix_cannot_hide_unsafe_raw_proposal_suffix(self):
        raw=np.vstack((self.goal,np.repeat(ARM.lower[None],15,axis=0)))
        body={**self.body,'cursor_mm':(1000*self.target+50*self.axis).tolist(),'continuous_motion':True}
        with self.disabled_ik(),patch.object(self.policy,'sample',return_value=raw[None,None]):
            result=tracking.plan(body)
        self.assertEqual(result['status'],'no_safe_chunk');self.assertEqual(result['frames'],[])
        self.assertEqual(result['rejections']['raw_collision'],1)

    def test_reached_static_pose_holds_and_brakes_without_ik_or_new_denoising(self):
        body={**self.body,'start_deg':np.rad2deg(self.goal).tolist(),
              'cursor_mm':(1000*self.target+50*self.axis).tolist(),'continuous_motion':True}
        for speed,acceleration in ((0.,0.),(.04,.1)):
            with self.subTest(speed=speed),self.disabled_ik(), \
                    patch.object(self.policy,'sample',side_effect=AssertionError('Reached pose needs no denoising')), \
                    patch.object(tracking.pose5,'solve',side_effect=AssertionError('Reached pose needs no IK')):
                result=tracking.plan({**body,'start_velocity_rad_s':[speed,0,0,0,0,0],
                    'start_acceleration_rad_s2':[acceleration,0,0,0,0,0]})
            self.assertTrue(result['playable'],result);self.assertTrue(result['converged'],result)
            self.assertEqual(result['planning_outcome'],'checked_cartesian_hold')
            self.assertEqual(result['proposed_chunks'],0)
            self.assertTrue(result['smoothing']['ends_at_rest'])
            self.assertEqual(result['smoothing']['starts_and_ends_at_rest'],speed==0)
            np.testing.assert_allclose(result['frames'][-1]['q'],body['start_deg'],atol=1e-10)
        with self.disabled_ik(),patch.object(self.policy,'sample',side_effect=AssertionError('Resting target no denoising')):
            preview=service.plan(body)
        self.assertTrue(preview['converged'],preview);self.assertEqual(len(preview['frames']),1)
        self.assertEqual(preview['proposed_chunks'],0)


class RelativeCartesianRuntimeTest(CartesianRuntimeTest):
    """Run the same no-IK, capture and safety contracts with relative schema4."""
    def setUp(self):
        super().setUp()
        self.policy.conditioning=RELATIVE_CONDITIONING
        self.status['conditioning']=RELATIVE_CONDITIONING
        current=ARM.transform(self.start)
        self.expected_position=(self.target-current[:3,3])/ARM.scale
        self.expected_axis=self.axis-current[:3,2]

    def test_relative_target_features_are_zero_at_current_cartesian_pose(self):
        transform=ARM.transform(self.start)
        condition=service.condition(self.policy,self.start,None,transform[:3,3],transform[:3,2],None)
        np.testing.assert_allclose(condition[6:12],np.zeros(6),atol=1e-7)


if __name__=='__main__':unittest.main()
