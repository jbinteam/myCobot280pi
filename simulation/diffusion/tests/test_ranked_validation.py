"""Lazy validation preserves candidate ordering and full-route gates."""
import unittest
from time import perf_counter
from unittest.mock import patch
import numpy as np
from simulation.kinematics import ARM
from simulation.diffusion.motion_conditioning import CONDITIONING
from simulation.diffusion.policy import checked_chunk,projected_chunk
from simulation.diffusion.continuous_rollout import rollout


class RankedValidationTest(unittest.TestCase):
    def run_candidates(self,unsafe_first=False,failed_first_join=False):
        start=ARM.center.copy();goal=start.copy();goal[0]+=.8;pose=ARM.transform(goal)
        chunks=[]
        for distance in [.45,.3,.2,.1]:
            end=start.copy();end[0]+=distance
            chunks.append(np.linspace(start,end,17)[1:])
        if unsafe_first:chunks[0][8,1]+=1.
        chunks=np.asarray(chunks)
        class Scene:
            def path_valid(self,path,step=.04):return np.all(np.abs(np.asarray(path)[:,1]-start[1])<.5)
            def edge_valid(self,a,b,*args):return True
            def execute_step(self,a,b,dt,speed):return True
        class Policy:
            observation_dim=34;conditioning=CONDITIONING
            def sample(self,obs,samples,steps):return chunks[None]
        joined=[]
        def prepare(env,path,v,a):
            joined.append(path.copy())
            if failed_first_join and len(joined)==1:raise ValueError('Rejected C2 join')
            return None,None,{'duration_s':.4}
        with patch('simulation.diffusion.continuous_rollout.checked_chunk',wraps=checked_chunk) as gate,\
             patch('simulation.diffusion.continuous_rollout.prepare',side_effect=prepare),\
             patch('simulation.diffusion.continuous_rollout.sample_motion',return_value=(start.copy(),np.zeros(6),np.zeros(6))):
            result=rollout(Scene(),Policy(),start,goal,pose[:3,3],pose[:3,2],None,
                           {'samples':4,'inference_steps':32,'max_decisions':1},perf_counter()+10)
        return start,chunks,joined,gate,result

    def test_accepted_winner_does_not_validate_lower_ranked_unused_candidates(self):
        start,chunks,joined,gate,result=self.run_candidates()
        self.assertEqual(gate.call_count,1);self.assertEqual(len(joined),1)
        np.testing.assert_array_equal(joined[0],np.vstack((start,projected_chunk(start,chunks[0]))))
        self.assertEqual(result[3],4)  # All four proposals still come from inference.

    def test_highest_ranked_raw_collision_falls_back_before_constructing_a_curve(self):
        start,chunks,joined,gate,result=self.run_candidates(unsafe_first=True)
        self.assertEqual(gate.call_count,2);self.assertEqual(len(joined),1)
        np.testing.assert_array_equal(joined[0],np.vstack((start,projected_chunk(start,chunks[1]))))
        self.assertEqual(result[4],{'raw_collision':1})

    def test_rejected_c2_join_validates_the_next_candidate_before_using_it(self):
        start,chunks,joined,gate,result=self.run_candidates(failed_first_join=True)
        self.assertEqual(gate.call_count,2);self.assertEqual(len(joined),2)
        np.testing.assert_array_equal(joined[1],np.vstack((start,projected_chunk(start,chunks[1]))))
        self.assertEqual(result[4],{'continuous_join':1})


if __name__=='__main__':unittest.main()
