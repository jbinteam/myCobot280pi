import unittest
import numpy as np
from simulation.diffusion.generate_collision_protected import timed_route, final_valid, boundary_pair
from simulation.kinematics import ARM

class CollisionProtectedTest(unittest.TestCase):
    def test_dense_replay_is_same_timed_curve_and_analytic_bounds(self):
        path=np.array([ARM.center, ARM.center+np.array([.2,-.1,.1,0,0,0]), ARM.center])
        sparse,dense,segments=timed_route(path)
        q,v,a,t=sparse; dq,dv,da,dt=dense
        for state,time in zip(q,t):
            index=np.argmin(abs(dt-time));self.assertLess(abs(dt[index]-time),1e-10)
            np.testing.assert_allclose(dq[index],state,atol=1e-12)
        self.assertLessEqual(np.max(np.diff(dt)),.01+1e-10)
        self.assertLessEqual(np.linalg.norm(dv,axis=1).max(),1.)
        self.assertLessEqual(np.linalg.norm(da,axis=1).max(),3.)
        np.testing.assert_array_equal(q[-1],path[-1])

    def test_safe_frames_with_collision_between_are_rejected(self):
        class Env:
            def path_valid(self,states,step):
                # Endpoints alone are valid, but the connecting segment crosses a hazard.
                return not any(a[0]<.05<b[0] for a,b in zip(states[:-1],states[1:]))
        q=np.array([[0.]*6,[.1,0,0,0,0,0]])
        sparse=(q,np.zeros_like(q),np.zeros_like(q),np.array([0.,.1]))
        self.assertFalse(final_valid(Env(),sparse,(q,None,None,np.array([0.,.1])),q[-1]))

    def test_invalid_timing_and_route_refused(self):
        for path in (np.zeros((1,6)),np.full((2,6),np.nan)):
            with self.assertRaises(ValueError):timed_route(path)
        with self.assertRaises(ValueError):timed_route(np.array([ARM.center,ARM.center+.1]),replay_dt=.2)

if __name__=='__main__':unittest.main()
