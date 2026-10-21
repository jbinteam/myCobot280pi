"""Bounded trajectory gradients and complete safety gates for research guidance."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from simulation.kinematics import ARM
from simulation.diffusion.cartesian_guidance import GuidanceConfig,guide_chunk,pose_loss_gradient
from simulation.diffusion.env import ArmEnv
from simulation.diffusion.service import errors


class GuidanceTest(unittest.TestCase):
    def setUp(self):
        case=json.loads(Path('simulation/artifacts/diffusion_v1/webui.json').read_text())['cases'][1]
        self.q=np.deg2rad(case['start_deg'])
        self.goal=self.q.copy();self.goal[0]+=.03
        pose=ARM.transform(self.goal);self.target=pose[:3,3];self.axis=pose[:3,2]

    def test_shared_urdf_gradient_matches_finite_difference_and_directed_axis(self):
        loss,gradient=pose_loss_gradient(self.q,self.target,self.axis)
        epsilon=1e-7;finite=[]
        for joint in range(6):
            offset=np.eye(6)[joint]*epsilon
            finite.append((pose_loss_gradient(self.q+offset,self.target,self.axis)[0]
                -pose_loss_gradient(self.q-offset,self.target,self.axis)[0])/(2*epsilon))
        np.testing.assert_allclose(gradient,finite,atol=1e-9,rtol=1e-6)
        self.assertGreater(pose_loss_gradient(self.q,self.target,-self.axis)[0],loss)

    def test_checked_bounded_refinement_improves_near_pose_without_ik(self):
        end=self.q.copy();end[0]+=.01
        original=np.linspace(self.q,end,17)[1:]
        with ArmEnv() as env, \
                patch.object(ArmEnv,'inverse_kinematics',side_effect=AssertionError('No endpoint IK')), \
                patch('simulation.pose5.solve',side_effect=AssertionError('No local IK')):
            env.set_obstacle(None)
            guided,metadata=guide_chunk(env,self.q,original,self.target,self.axis)
            self.assertTrue(env.path_valid(np.vstack((self.q,guided))))
        self.assertTrue(metadata['applied'],metadata);self.assertTrue(metadata['c2_validated'])
        self.assertLess(metadata['loss_after'],metadata['loss_before'])
        self.assertLess(errors(guided[-1],self.target,self.axis)['error_mm'],
                        errors(original[-1],self.target,self.axis)['error_mm']*.6)
        self.assertLessEqual(metadata['max_correction_rad'],GuidanceConfig().max_correction_rad+1e-9)
        np.testing.assert_array_equal(guided[:metadata['protected_references']],original[:metadata['protected_references']])
        self.assertLessEqual(metadata['max_velocity_rad_s'],1.+1e-9)
        self.assertLessEqual(metadata['max_acceleration_rad_s2'],3.+1e-9)

    def test_full_unsafe_route_rejected_before_any_refinement(self):
        raw=np.vstack((self.goal,np.repeat(ARM.lower[None],15,axis=0)))
        with ArmEnv() as env:
            env.set_obstacle(None)
            with self.assertRaisesRegex(ValueError,'complete neural route'):
                guide_chunk(env,self.q,raw,self.target,self.axis)

    def test_failed_c2_check_returns_original_route_instead_of_corrected_motion(self):
        original=np.repeat(self.q[None],16,axis=0)
        with ArmEnv() as env,patch('simulation.diffusion.cartesian_guidance.prepare',side_effect=ValueError('Join rejected')):
            env.set_obstacle(None)
            guided,metadata=guide_chunk(env,self.q,original,self.target,self.axis)
        np.testing.assert_array_equal(guided,original)
        self.assertFalse(metadata['applied']);self.assertFalse(metadata['c2_validated'])
        self.assertEqual(metadata['reason'],'guided_c2_rejected')


if __name__=='__main__':unittest.main()
