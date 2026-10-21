import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from simulation.kinematics import ARM
from simulation.prm import JointSpace, planning_space
from simulation.seed5 import Seed5Model
from simulation.seed5_data import generate
from simulation.train_seed5 import collision_repair_costs, collision_repair_loss, train


class CollisionTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.space=planning_space('self',1.)
        fixture=json.loads((Path(__file__).resolve().parents[1]/'fixtures/self_collision_path.json').read_text())
        cls.safe=np.deg2rad(fixture['start_deg'])
        cls.bad=(cls.safe+np.deg2rad(fixture['goal_deg']))/2
        assert cls.space.valid(cls.safe) and not cls.space.valid(cls.bad)

    def test_exact_collision_mask_repairs_all_unsafe_heads_and_preserves_safe_heads(self):
        proposals=torch.tensor(np.array([[self.safe,self.bad,self.safe,self.bad]]),dtype=torch.float64,requires_grad=True)
        witness=torch.tensor(self.safe[None],dtype=torch.float64)
        costs,audit=collision_repair_costs(proposals,witness,self.space)
        self.assertEqual(audit['invalid_proposals'],2)
        self.assertEqual(audit['proposals'],4)
        self.assertEqual(audit['invalid_fraction'],.5)
        self.assertEqual(costs[0,0].item(),0.)
        self.assertGreater(costs[0,1].item(),0.)
        costs.mean().backward()
        self.assertTrue(torch.isfinite(proposals.grad).all())
        np.testing.assert_array_equal(proposals.grad[0,0],np.zeros(6))
        self.assertGreater(float(torch.dot(proposals.grad[0,1],proposals[0,1]-witness[0]).detach()),0.)
        self.assertGreater(float(torch.dot(proposals.grad[0,3],proposals[0,3]-witness[0]).detach()),0.)

    def test_safe_alternative_posture_has_no_witness_pull(self):
        alternative=np.zeros(6);self.assertTrue(self.space.valid(alternative))
        proposals=torch.tensor(np.array([[alternative]*4]),dtype=torch.float64,requires_grad=True)
        costs,audit=collision_repair_costs(proposals,torch.tensor(self.safe[None]),self.space)
        self.assertEqual(audit['invalid_proposals'],0)
        costs.sum().backward()
        np.testing.assert_array_equal(proposals.grad,np.zeros((1,4,6)))

    def test_safe_heads_cannot_dilute_rare_collision_repair_gradients(self):
        witness=torch.tensor(self.safe[None],dtype=torch.float64)
        gradients=[];losses=[]
        for heads in (2,4):
            proposals=torch.tensor(np.array([[self.bad]+[self.safe]*(heads-1)]),dtype=torch.float64,requires_grad=True)
            loss,audit=collision_repair_loss(proposals,witness,self.space)
            self.assertEqual(audit['invalid_proposals'],1)
            loss.backward();gradients.append(proposals.grad[0,0]);losses.append(loss.item())
        self.assertEqual(losses[0],losses[1])
        np.testing.assert_array_equal(gradients[0],gradients[1])

    def test_unchecked_space_and_unsafe_witness_are_rejected(self):
        proposals=torch.tensor(np.array([[self.safe]*4]),dtype=torch.float64)
        for space,witness in ((JointSpace(),self.safe),(self.space,self.bad)):
            with self.assertRaises(ValueError):
                collision_repair_costs(proposals,torch.tensor(witness[None]),space)
        with self.assertRaises(ValueError):
            collision_repair_costs(proposals*float('nan'),torch.tensor(self.safe[None]),self.space)

    def test_smoke_export_records_audits_and_exact_collision_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset=Path(tmp)/'dataset.npz';output=Path(tmp)/'smoke'
            generate(dataset,16,8,172,173)
            result=train(dataset,output,steps=2,batch_size=8,smoke=True,hidden=128,
                         collision_every=1,collision_batch_size=8)
            self.assertTrue(result['collision_training']['enabled'])
            self.assertEqual(result['collision_training']['audits']['batches'],2)
            self.assertEqual(result['collision_training']['audits']['proposals'],64)
            metrics=result['validation_metrics']
            self.assertEqual(metrics['raw_proposals'],32)
            self.assertEqual(metrics['raw_invalid_fraction'],metrics['raw_invalid_proposals']/32)
            model=Seed5Model(output/'models.json')
            self.assertEqual(model.data['training']['collision_training'],result['collision_training'])
            second=train(dataset,Path(tmp)/'initialized',steps=1,batch_size=4,smoke=True,hidden=128,
                         init_checkpoint=output/'models.json',learning_rate=.0001)
            self.assertEqual(second['initialization']['completed_steps'],2)

    def test_collision_options_fail_before_loading_a_dataset(self):
        for changes in ({'collision_weight':-1},{'collision_every':0},{'collision_every':True},
                        {'collision_batch_size':0},{'collision_validation_batch_size':0},{'learning_rate':0}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                train('missing.npz','unused',**changes)


if __name__=='__main__':unittest.main()
