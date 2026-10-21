import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from simulation.kinematics import ARM, rotation
from simulation import pose5, seed5, planning, service
from simulation.prm import JointSpace, Roadmap
from simulation.seed5_data import generate, load_dataset


class Seed5Test(unittest.TestCase):
    def setUp(self):
        self.q=np.deg2rad([10,-40,-110,65,20,50])
        self.goal=self.q+np.deg2rad([3,2,-2,3,-2,0])
        self.t=ARM.transform(self.goal);self.xyz=self.t[:3,3];self.axis=self.t[:3,2]
        self.config=planning.IKConfig(task='pose5',nn='seed5')

    def solve_goals(self,proposals,**kwargs):
        status={'status':'ready','reason':'trained fixture'}
        with patch.object(seed5,'propose',return_value=(proposals,status,.1)):
            return planning.generate_goals(self.q,self.xyz,space=kwargs.pop('space',JointSpace()),
                                          config=kwargs.pop('config',self.config),target_axis=self.axis,**kwargs)

    def test_unit_axis_validation_and_extreme_scales(self):
        for value in ([0,0,0],[1,np.nan,0],[1,np.inf,0],[True,0,0],['1',0,0],[1,0],None):
            with self.subTest(value=value),self.assertRaises((ValueError,TypeError)):pose5.unit_axis(value)
        for value in ([1e300,-1e300,0],[1e-300,-1e-300,0]):
            np.testing.assert_allclose(pose5.unit_axis(value),[2**-.5,-2**-.5,0])

    def test_spin_invariance_and_opposite_rejection(self):
        spun=self.t[:3,:3]@rotation([0,0,1],1.7)
        self.assertLess(pose5.axis_error(self.axis,spun[:,2])[2],1e-9)
        self.assertAlmostEqual(pose5.axis_error(self.axis,-self.axis)[2],180)
        self.assertIsNone(planning.checked_goal(self.goal,self.xyz,JointSpace(),self.config,-self.axis))
        self.assertIsNotNone(planning.checked_goal(self.goal,self.xyz,JointSpace(),self.config,self.axis))

    def test_already_at_target_requires_both_constraints(self):
        t=ARM.transform(self.q)
        with patch.object(seed5,'propose',side_effect=AssertionError('shortcut must avoid NN')):
            result=planning.generate_goals(self.q,t[:3,3],space=JointSpace(),config=self.config,target_axis=t[:3,2])
        self.assertEqual(result['goals'][0]['source'],'current_state');self.assertEqual(result['seed_queue'],[])
        with patch.object(seed5,'propose',return_value=([],{'status':'ready'},0)) as proposal:
            result=planning.generate_goals(self.q,t[:3,3],space=JointSpace(),
                config=planning.IKConfig(task='pose5',nn='seed5',numerical_fallback=False),target_axis=-t[:3,2])
        proposal.assert_called_once();self.assertFalse(result['goals'])

    def test_all_four_raw_checked_and_independent_refinement_queue(self):
        guesses=np.tile(self.goal,(4,1));guesses[:,5]+=np.deg2rad([0,10,20,30])
        with patch.object(pose5,'solve',side_effect=AssertionError('four-goal budget already met')):
            result=self.solve_goals(guesses)
        self.assertEqual(result['raw_accepted_count'],4)
        self.assertEqual(len(result['attempts']),4);self.assertEqual(len(result['goals']),4)
        self.assertEqual([s['source'] for s in result['seed_queue']],['nn_seed5_refined']*4)
        self.assertTrue(all(a['added_unique_goal'] for a in result['attempts']))

    def test_colliding_nn_seed_still_refined_and_fallback_independent_of_flag(self):
        colliding=self.q+np.deg2rad([1,0,0,0,0,0])
        space=JointSpace(validity=lambda q:np.linalg.norm(q-colliding)>1e-8,
                         assumptions={'checker':'fixture','environment':'fixture'})
        failed={'frames':[{'q':np.rad2deg(self.q).tolist()}],'iterations':1,'elapsed_ms':.1,'stop_reason':'fixture'}
        road=Roadmap(space);road.q=np.stack([self.q,self.goal]);road.poses=np.stack([ARM.transform(q) for q in road.q])
        with patch.object(pose5,'solve',return_value=failed) as solve:
            result=self.solve_goals([colliding,colliding,self.q,self.q],space=space,roadmap=road)
        self.assertFalse(result['attempts'][0]['accepted'])
        np.testing.assert_allclose(solve.call_args_list[0].args[0],colliding)
        self.assertEqual([s['source'] for s in result['seed_queue']],['nn_seed5_refined']*4+['current_dls']+['roadmap_dls']*2+['random_dls']*4)
        # Duplicate guesses/current/roadmap current are only refined once.
        self.assertEqual(solve.call_count,7)
        config=planning.IKConfig(task='pose5',nn='seed5',refine=False)
        with patch.object(pose5,'solve',return_value=failed):
            result=self.solve_goals([colliding]*4,space=space,roadmap=road,config=config)
        self.assertEqual([s['source'] for s in result['seed_queue']],['current_dls']+['roadmap_dls']*2+['random_dls']*4)
        again=self.solve_goals([],config=planning.IKConfig(task='pose5',nn='seed5',refine=False,steps=1))
        np.testing.assert_allclose(result['seed_queue'][-4:][0]['q_rad'],again['seed_queue'][-4:][0]['q_rad'])

    def test_raw_acceptance_suppresses_fallback_and_unique_goals(self):
        failed={'frames':[{'q':np.rad2deg(self.goal).tolist()}],'iterations':0,'elapsed_ms':0,'stop_reason':'fixture'}
        with patch.object(pose5,'solve',return_value=failed) as solve:
            result=self.solve_goals([self.goal]*4)
        self.assertEqual(len(result['goals']),1);self.assertEqual(result['raw_accepted_count'],1)
        self.assertEqual(solve.call_count,1)
        self.assertEqual([s['source'] for s in result['seed_queue']],['nn_seed5_refined']*4)

    def test_extra_goal_constraint_applies_before_shortcut_raw_budget_and_refinement(self):
        config=planning.IKConfig(task='pose5',nn='seed5',numerical_fallback=False)
        failed={'frames':[{'q':np.rad2deg(self.goal).tolist()}],'iterations':0,'elapsed_ms':0,'stop_reason':'fixture'}
        with patch.object(seed5,'propose',return_value=([self.goal]*4,{'status':'ready'},0)), \
             patch.object(pose5,'solve',return_value=failed) as solve:
            result=planning.generate_goals(self.goal,self.xyz,space=JointSpace(),config=config,
                                           target_axis=self.axis,goal_constraint=lambda q:np.bool_(False))
        self.assertFalse(result['goals']);self.assertEqual(result['raw_accepted_count'],0)
        self.assertEqual(len(result['seed_queue']),4);self.assertEqual(solve.call_count,1)
        self.assertEqual(result['diagnostic']['goal_constraint_rejections'],5)
        self.assertTrue(callable(solve.call_args.kwargs['goal_constraint']))

    def test_dls_recovery_keeps_seed_order_unique_goal_budget_and_valid_endpoints(self):
        guesses=np.tile(self.q,(4,1));guesses[:,0]+=np.deg2rad([1,2,3,4])
        config=planning.IKConfig(task='pose5',nn='seed5',pose5_recovery_methods=('dls',),max_candidates=2)
        calls=[]
        def solved(q, target, orientation, method, **kwargs):
            calls.append((q.copy(),method));end=self.q.copy()
            if method=='dls':end=self.goal.copy();end[5]+=.01*len(calls)
            return {'frames':[{'q':np.rad2deg(end).tolist()}],'iterations':1,'elapsed_ms':.1,'stop_reason':'fixture'}
        with patch.object(pose5,'solve',side_effect=solved):result=self.solve_goals(guesses,config=config)
        self.assertEqual(len(result['goals']),2)
        self.assertEqual([m for q,m in calls],['bounded_dls','dls']*2)
        np.testing.assert_allclose(calls[0][0],guesses[0]);np.testing.assert_allclose(calls[2][0],guesses[1])
        self.assertEqual([s['source'] for s in result['seed_queue']][:5],['nn_seed5_refined']*4+['current_dls'])
        for value in [('unknown',),('dls','dls'),True]:
            with self.assertRaises(ValueError):planning.IKConfig(task='pose5',pose5_recovery_methods=value)
        with self.assertRaises(ValueError):planning.IKConfig(pose5_recovery_methods=('dls',))

    def test_failure_diagnostics_distinguish_accurate_colliding_goals(self):
        space=JointSpace(validity=lambda q:np.linalg.norm(q-self.goal)>1e-8,
                         assumptions={'checker':'fixture','environment':'fixture'})
        config=planning.IKConfig(task='pose5',nn='seed5',refine=False,numerical_fallback=False)
        result=self.solve_goals([self.goal]*4,space=space,config=config)
        self.assertFalse(result['goals'])
        self.assertEqual(result['diagnostic']['pose_accurate_invalid_candidates'],4)
        self.assertTrue(all(a['rejection_reason']=='invalid_configuration' for a in result['attempts']))
        with patch.object(seed5,'propose',return_value=([self.goal,self.q,self.q,self.q],{'status':'ready'},0)):
            failed=planning.query(self.q,self.xyz,Roadmap(space),config,target_axis=self.axis)
        self.assertEqual(failed['ik_outcome'],'ik_solution_not_found')
        self.assertIn('pose-accurate candidates failed collision',failed['reason'])
        self.assertAlmostEqual(failed['final_error_mm'],planning.goal_error(self.q,self.xyz,self.axis)['error_mm'])
        self.assertAlmostEqual(failed['ik']['best_error']['error_mm'],0)

    def test_refinement_stops_at_unique_goal_budget_and_rejects_invalid_seeds(self):
        guesses=np.tile(self.q,(4,1));guesses[:,0]+=np.deg2rad([1,2,3,4])
        calls=[]
        def solved(q,*args,**kwargs):
            goal=self.goal.copy();goal[5]+=len(calls)*.1;calls.append(q.copy())
            return {'frames':[{'q':np.rad2deg(goal).tolist()}],'iterations':1,'elapsed_ms':.1,'stop_reason':'fixture'}
        with patch.object(pose5,'solve',side_effect=solved):result=self.solve_goals(guesses)
        self.assertEqual(result['raw_accepted_count'],0);self.assertEqual(len(calls),4)
        self.assertEqual(len(result['goals']),4)
        self.assertEqual([s['source'] for s in result['seed_queue']][4:],['current_dls']+['random_dls']*4)
        nan=self.q.copy();nan[0]=np.nan;outside=self.q.copy();outside[0]=ARM.upper[0]+.1
        with patch.object(pose5,'solve',side_effect=solved):
            result=self.solve_goals([nan,outside,self.q,self.q],config=planning.IKConfig(task='pose5',nn='seed5',numerical_fallback=False))
        self.assertEqual(len(result['attempts'][:4]),4)
        self.assertEqual([s['source'] for s in result['seed_queue']],['nn_seed5_refined']*2)

    def test_checkpoint_rejects_untrained_and_nonfinite_weights(self):
        smoke=Path('simulation/artifacts/seed5_v1/smoke/models.json')
        model=seed5.Seed5Model(smoke);proposals=model.predict(self.q,self.xyz,self.axis)
        self.assertEqual(proposals.shape,(4,6))
        self.assertTrue(np.all(proposals>=ARM.lower) and np.all(proposals<=ARM.upper))
        self.assertTrue(np.all(np.abs(proposals-self.q)<=ARM.radius+1e-12))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.json'
            invalid=copy.deepcopy(model.data);invalid['training']['completed_steps']=0
            path.write_text(json.dumps(invalid));self.assertEqual(seed5.checkpoint_status(path)['status'],'incompatible')
            invalid=copy.deepcopy(model.data);invalid['layers'][0]['weight'][0][0]=float('nan')
            path.write_text(json.dumps(invalid));self.assertEqual(seed5.checkpoint_status(path)['status'],'incompatible')

    def test_missing_model_uses_supported_fallback_even_without_nn_refinement(self):
        solved={'frames':[{'q':np.rad2deg(self.goal).tolist()}],'iterations':1,'elapsed_ms':.1,'stop_reason':'fixture'}
        status={'status':'missing','reason':'No compatible 5D checkpoint'}
        config=planning.IKConfig(task='pose5',nn='seed5',refine=False,random_seeds=0,roadmap_seeds=0)
        with patch.object(seed5,'propose',return_value=([],status,0)),patch.object(pose5,'solve',return_value=solved):
            result=planning.generate_goals(self.q,self.xyz,space=JointSpace(),config=config,target_axis=self.axis)
        self.assertEqual(result['nn_checkpoint']['status'],'missing')
        self.assertEqual(result['attempts'][0]['stop_reason'],'checkpoint_missing')
        self.assertEqual(result['goals'][0]['source'],'current_dls')
        self.assertEqual([s['source'] for s in result['seed_queue']],['current_dls'])

    def test_checkpoint_status_identifies_collision_training_and_refreshes_active_hash(self):
        legacy=seed5.Seed5Model('simulation/artifacts/seed5_v1/smoke/models.json')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.json'
            path.write_text(json.dumps(legacy.data))
            before=seed5.checkpoint_status(path)
            self.assertFalse(before['collision_training']['enabled'])
            self.assertFalse(before['fine_tuned'])
            trained=copy.deepcopy(legacy.data)
            trained['training']['initialization']={'completed_steps':32000}
            trained['training']['collision_training']={'enabled':True,'scope':'Endpoint self-collision',
                'validity':{'assumptions':{'clearance_mm':1.}}}
            path.write_text(json.dumps(trained))
            guesses,after,_=seed5.propose(self.q,self.xyz,self.axis,path)
            self.assertEqual(len(guesses),4)
            self.assertTrue(after['collision_training']['enabled'])
            self.assertEqual(after['collision_training']['clearance_mm'],1.)
            self.assertTrue(after['fine_tuned'])
            self.assertNotEqual(before['checkpoint_sha256'],after['checkpoint_sha256'])
            self.assertEqual(after['checkpoint_sha256'],hashlib.sha256(path.read_bytes()).hexdigest())

    def test_missing_checkpoint_and_xyz_compatibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'missing.json'
            self.assertEqual(seed5.checkpoint_status(path)['status'],'missing')
            guesses,status,_=seed5.propose(self.q,self.xyz,self.axis,path)
            self.assertEqual(guesses,[]);self.assertEqual(status['status'],'missing')
        self.assertEqual(seed5.checkpoint_status(Path('simulation/artifacts/models.json'))['status'],'incompatible')
        for args in ({'task':'pose5','nn':'seed'},{'task':'position','nn':'seed5'}):
            with self.assertRaises(ValueError):planning.IKConfig(**args)
        self.assertEqual(planning.IKConfig().nn,'seed')
        inputs=seed5.features(self.q,self.xyz,self.axis*100)
        self.assertEqual(inputs.shape,(12,));np.testing.assert_allclose(inputs[3:6],self.axis)

    def test_dataset_fk_labels_and_differentiable_loss(self):
        import torch
        from simulation.train_seed5 import task_costs
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'data.npz';generate(path,16,8,71,72,collision='self')
            data,metadata=load_dataset(path)
            self.assertEqual(metadata['units']['target_xyz_m'],'meters')
            q=torch.tensor(data['q_goal_rad'][:3],dtype=torch.float64,requires_grad=True)
            t=ARM.torch_transform(q)
            np.testing.assert_allclose(t.detach().numpy(),np.stack([ARM.transform(x) for x in q.detach().numpy()]),atol=1e-12)
            xyz=t[:,:3,3].detach();axis=t[:,:3,2].detach()
            aligned=task_costs(t[:,None,:3,3],t[:,None,:3,2],xyz,axis)
            opposed=task_costs(t[:,None,:3,3],t[:,None,:3,2],xyz,-axis)
            np.testing.assert_allclose(aligned.detach(),0,atol=1e-12)
            np.testing.assert_allclose(opposed.detach(),1.44,atol=1e-12)
            opposed.sum().backward();self.assertTrue(torch.isfinite(q.grad).all())

    def test_offset_frame_and_planning_outcomes(self):
        road=Roadmap(JointSpace());offset=np.array([3.,4.,5.]);rotation=ARM.transform(self.q)[:3,:3]
        body={'task':'pose5','nn':'none','start_deg':np.rad2deg(self.q).tolist(),
              'target_mm':(ARM.fk(self.q)*1000-rotation@offset).tolist(),'eef_offset_mm':offset.tolist(),
              'target_axis':ARM.transform(self.q)[:3,2].tolist()}
        result=service.plan(body,road)
        self.assertTrue(result['converged']);self.assertEqual(result['ik_outcome'],'accepted_goals')
        self.assertEqual(result['planning_outcome'],'direct_connection')
        np.testing.assert_allclose(result['target_mm'],ARM.fk(self.q)*1000,atol=1e-9)
        with patch.object(road,'plan',return_value={'status':'roadmap_connection_not_found','reason':'start_not_connected','planning_ms':0}):
            result=service.plan(body,road)
        self.assertEqual(result['ik_outcome'],'accepted_goals');self.assertEqual(result['planning_outcome'],'start_not_connected')


if __name__=='__main__':unittest.main()
