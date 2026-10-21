"""Numerical endpoint contracts, without hardware or learned model IO."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from simulation.kinematics import ARM
from simulation.diffusion.env import ArmEnv
from simulation.diffusion.endpoint_ik import IKSearchExhausted, residual_and_jacobian


class EndpointIKTest(unittest.TestCase):
    def scene(self, valid=lambda q: True):
        env=ArmEnv.__new__(ArmEnv)
        env.valid=valid
        return env

    def legacy(self, q, converged=False):
        return {'converged':converged, 'frames':[{'q':np.rad2deg(q).tolist()}]}

    def test_analytic_jacobian_matches_finite_difference(self):
        q=np.deg2rad([23,-42,-78,45,36,-22]);t=ARM.transform(ARM.center)
        _,jac=residual_and_jacobian(q,t[:3,3],t[:3,2])
        eps=1e-6
        numeric=np.column_stack([(residual_and_jacobian(q+np.eye(6)[i]*eps,t[:3,3],t[:3,2])[0]-residual_and_jacobian(q-np.eye(6)[i]*eps,t[:3,3],t[:3,2])[0])/(2*eps) for i in range(6)])
        np.testing.assert_allclose(jac,numeric,atol=1e-8)

    def test_legacy_success_preserves_exact_goal_and_skips_fallback(self):
        q=np.deg2rad([0,-40,-130,80,0,50]);t=ARM.transform(q)
        result=self.legacy(q,True);env=self.scene()
        expected=np.deg2rad(result['frames'][-1]['q'])
        with patch('simulation.diffusion.env.solve',return_value=result), patch('simulation.diffusion.endpoint_ik.least_squares',side_effect=AssertionError):
            got=env.inverse_kinematics(t[:3,3],t[:3,2],q,robust=True)
        np.testing.assert_array_equal(got,expected)
        self.assertEqual(env.ik_diagnostics['method'],'bounded_dls')

    def test_fallback_recovers_known_reachable_target_after_legacy_miss(self):
        q=np.deg2rad([23,-42,-78,45,36,-22]);t=ARM.transform(q)
        env=self.scene()
        with patch('simulation.diffusion.env.solve',return_value=self.legacy(ARM.center)):
            got=env.inverse_kinematics(t[:3,3],t[:3,2],ARM.center,attempts=1,robust=True)
        fk=ARM.transform(got)
        self.assertLessEqual(np.linalg.norm(fk[:3,3]-t[:3,3]),.001)
        self.assertLessEqual(np.rad2deg(np.arccos(np.clip(fk[:3,2]@t[:3,2],-1,1))),1)
        self.assertEqual(env.ik_diagnostics['method'],'least_squares')

    def test_real_scene_recovers_recorded_legacy_miss(self):
        # FK-generated reachable_04 from deterministic 2026-10-07 validation.
        witness=np.array([-2.63871288791391,.21315299890227513,1.0956456737136926,
                          1.94116365448573,-1.132670670977861,-1.9056764425908146])
        current=np.deg2rad([0,-40,-130,80,0,50]);target=ARM.transform(witness)
        with ArmEnv() as env:
            env.set_obstacle(None)
            self.assertTrue(env.valid(witness))
            goal=env.inverse_kinematics(target[:3,3],target[:3,2],current,robust=True)
            self.assertEqual(env.ik_diagnostics['legacy_attempts'],12)
            self.assertEqual(env.ik_diagnostics['method'],'least_squares')
            self.assertTrue(env.valid(goal))
            actual=ARM.transform(goal)
            self.assertLessEqual(np.linalg.norm(actual[:3,3]-target[:3,3]),.001)
            self.assertLessEqual(np.rad2deg(np.arccos(np.clip(actual[:3,2]@target[:3,2],-1,1))),1)

    def test_antipodal_direction_is_never_accepted(self):
        q=ARM.center;t=ARM.transform(q);env=self.scene()
        residual,_=residual_and_jacobian(q,t[:3,3],-t[:3,2])
        self.assertGreater(np.linalg.norm(residual),1)
        with patch('simulation.diffusion.env.solve',return_value=self.legacy(q,True)), patch('simulation.diffusion.endpoint_ik.least_squares',return_value=SimpleNamespace(x=q,nfev=1)):
            with self.assertRaises(IKSearchExhausted):
                env.inverse_kinematics(t[:3,3],-t[:3,2],q,attempts=1,robust=True)
        self.assertAlmostEqual(env.ik_diagnostics['best_direction_error_deg'],180)

    def test_collision_gate_rejects_accurate_endpoints(self):
        q=ARM.center;t=ARM.transform(q);env=self.scene(lambda q:False)
        with patch('simulation.diffusion.env.solve',return_value=self.legacy(q,True)), patch('simulation.diffusion.endpoint_ik.least_squares',return_value=SimpleNamespace(x=q,nfev=1)):
            with self.assertRaises(IKSearchExhausted) as caught:
                env.inverse_kinematics(t[:3,3],t[:3,2],q,attempts=1,robust=True)
        self.assertEqual(env.ik_diagnostics['collision_rejections'],33)
        self.assertGreater(caught.exception.diagnostics['elapsed_ms'],0)

    def test_failure_is_deterministic_bounded_and_opt_in(self):
        env=self.scene();target=np.array([3.,0.,0.]);q=ARM.center
        seeds=[]
        def failed(fun,start,**kw):
            seeds.append(start.copy())
            self.assertEqual(kw['max_nfev'],160)
            return SimpleNamespace(x=q,nfev=160)
        with patch('simulation.diffusion.env.solve',return_value=self.legacy(q)), patch('simulation.diffusion.endpoint_ik.least_squares',side_effect=failed):
            for _ in range(2):
                with self.assertRaises(IKSearchExhausted):env.inverse_kinematics(target,current=q,attempts=1,robust=True)
                self.assertEqual(env.ik_diagnostics['fallback_attempts'],32)
                self.assertEqual(env.ik_diagnostics['fallback_nfev'],5120)
            with self.assertRaises(IKSearchExhausted):env.inverse_kinematics(target,current=q,attempts=1)
            self.assertEqual(env.ik_diagnostics['fallback_attempts'],0)
        np.testing.assert_array_equal(seeds[:32],seeds[32:])

    def test_unexpected_solver_fault_propagates(self):
        env=self.scene()
        with patch('simulation.diffusion.env.solve',side_effect=RuntimeError('backend fault')):
            with self.assertRaisesRegex(RuntimeError,'backend fault'):
                env.inverse_kinematics([0,0,.2],current=ARM.center,robust=True)

if __name__=='__main__':unittest.main()
