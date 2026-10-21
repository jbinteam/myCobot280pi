import unittest
import numpy as np
from simulation.kinematics import ARM,rotation
from simulation.pose5 import axis_from_xy,xy_from_axis,axis_error,task,solve,METHODS,bounded_velocity,cases
from simulation import service


class Pose5Test(unittest.TestCase):
    def setUp(self):self.q=np.deg2rad([10,-40,-110,65,20,50])
    def test_direction_convention_and_roundtrip(self):
        np.testing.assert_allclose(axis_from_xy([0,0]),[0,0,1],atol=1e-12)
        np.testing.assert_allclose(axis_from_xy([90,0]),[0,-1,0],atol=1e-12)
        np.testing.assert_allclose(axis_from_xy([0,90]),[1,0,0],atol=1e-12)
        for n in ([0,1,0],[0,-1,0],[0,0,-1],[.3,.8,-.5]):
            n=np.array(n,dtype=float);n/=np.linalg.norm(n)
            np.testing.assert_allclose(axis_from_xy(xy_from_axis(n)),n,atol=1e-12)
    def test_spatial_and_direction_jacobians_match_finite_difference(self):
        t,j=ARM.spatial_jacobian(self.q);step=1e-6;numeric=[];direction=[]
        for i in range(6):
            d=np.eye(6)[i]*step;plus=ARM.transform(self.q+d);minus=ARM.transform(self.q-d)
            omega=((plus[:3,:3]-minus[:3,:3])/(2*step))@t[:3,:3].T
            numeric.append(np.r_[(plus[:3,3]-minus[:3,3])/(2*step),[omega[2,1],omega[0,2],omega[1,0]]])
            direction.append((plus[:3,2]-minus[:3,2])/(2*step))
        np.testing.assert_allclose(j,np.column_stack(numeric),atol=1e-9)
        np.testing.assert_allclose(np.cross(j[3:].T,t[:3,2]).T,np.column_stack(direction),atol=1e-9)
        _,j5,_,_=task(self.q,t[:3,3],t[:3,2])
        self.assertEqual(j5.shape,(5,6));self.assertEqual(np.linalg.matrix_rank(j5),5)
    def test_free_local_z_spin_is_not_an_orientation_error(self):
        t=ARM.transform(self.q);spun=t[:3,:3]@rotation([0,0,1],1.2)
        self.assertLess(axis_error(t[:3,2],spun[:,2])[2],1e-10)
        changed=self.q.copy();changed[5]+=np.deg2rad(45)
        result=solve(changed,t[:3,3],xy_from_axis(t[:3,2]))
        self.assertTrue(result['converged']);self.assertEqual(result['iterations'],0)
        self.assertGreater(np.linalg.norm(ARM.transform(changed)[:3,:3]-t[:3,:3]),.1)
    def test_antipodal_direction_has_180_degree_error_and_finite_command(self):
        e,basis,angle=axis_error(np.array([0.,0.,1.]),np.array([0.,0.,-1.]))
        self.assertAlmostEqual(angle,180);self.assertTrue(np.isfinite(e).all())
        np.testing.assert_allclose(basis@basis.T,np.eye(2),atol=1e-12)
        t=ARM.transform(self.q)
        r=solve(self.q,t[:3,3],xy_from_axis(-t[:3,2]),steps=1)
        self.assertFalse(r['converged']);self.assertTrue(np.isfinite(r['frames'][-1]['q']).all())
    def test_position_alone_never_counts_as_pose_convergence(self):
        t=ARM.transform(self.q);desired=rotation([1,0,0],.3)@t[:3,2]
        r=solve(self.q,t[:3,3],xy_from_axis(desired),steps=1)
        self.assertAlmostEqual(r['frames'][0]['error_mm'],0)
        self.assertGreater(r['frames'][0]['orientation_error_deg'],1)
        self.assertFalse(r['converged'])
    def test_extra_goal_acceptance_continues_refinement_inside_nominal_tolerances(self):
        goal=self.q.copy();goal[0]+=.002;t=ARM.transform(goal)
        ordinary=solve(self.q,t[:3,3],target_axis=t[:3,2])
        self.assertTrue(ordinary['converged']);self.assertEqual(ordinary['iterations'],0)
        constraint=lambda q:np.linalg.norm(ARM.fk(q)-t[:3,3])*1000<=.1
        refined=solve(self.q,t[:3,3],target_axis=t[:3,2],goal_constraint=constraint)
        self.assertTrue(refined['converged']);self.assertGreater(refined['iterations'],0)
        self.assertLessEqual(refined['final_error_mm'],.1)
        rejected=solve(goal,t[:3,3],target_axis=t[:3,2],goal_constraint=lambda q:False)
        self.assertFalse(rejected['converged'])
    def test_all_solvers_reach_both_constraints_and_obey_speed_limits(self):
        goal=self.q+np.deg2rad([3,2,-2,3,-2,0]);t=ARM.transform(goal)
        for method,_,_ in METHODS:
            r=solve(self.q,t[:3,3],xy_from_axis(t[:3,2]),method,speed=.8)
            with self.subTest(method=method):
                self.assertTrue(r['converged']);self.assertLessEqual(r['final_error_mm'],1)
                self.assertLessEqual(r['final_orientation_error_deg'],1)
                qs=np.deg2rad([f['q'] for f in r['frames']])
                self.assertTrue(np.all(qs>=ARM.lower-1e-9) and np.all(qs<=ARM.upper+1e-9))
                self.assertTrue(np.all(np.linalg.norm(np.diff(qs,axis=0),axis=1)<=.8*.05+1e-9))
                for q,f in zip(qs,r['frames']):
                    actual=ARM.transform(q)
                    np.testing.assert_allclose(actual[:3,2],f['tool_axis'],atol=1e-12)
                    self.assertAlmostEqual(axis_error(actual[:3,2],t[:3,2])[2],f['orientation_error_deg'])
                if method=='null_classical':self.assertLess(r['null_residual_max'],1e-9)
                else:self.assertAlmostEqual(r['frames'][-1]['q'][5],np.rad2deg(self.q[5]),places=6)
    def test_service_rejects_position_trained_network_and_bad_orientation(self):
        t=ARM.transform(self.q)
        body={'task':'pose5','start_deg':np.rad2deg(self.q).tolist(),'target_mm':(t[:3,3]*1000).tolist(),'orientation_xy_deg':xy_from_axis(t[:3,2])}
        for extra in ({'methods':['direct']},{'orientation_xy_deg':[0]},{'orientation_xy_deg':[float('nan'),0]},{'orientation_xy_deg':[181,0]},{'orientation_tolerance_deg':0}):
            with self.assertRaises((ValueError,TypeError)):service.run({**body,**extra})
        result=service.run(body)
        self.assertEqual(result['task'],'pose5');self.assertTrue(result['results'][0]['converged'])

    def test_regular_nullspace_only_changes_flange_spin(self):
        t=ARM.transform(self.q);j=task(self.q,t[:3,3],t[:3,2])[1]
        projector=np.eye(6)-np.linalg.pinv(j,rcond=1e-5)@j
        np.testing.assert_allclose(projector,np.diag([0,0,0,0,0,1]),atol=1e-10)

    def test_bounded_velocity_obeys_constraints_and_kkt_conditions(self):
        rng=np.random.default_rng(2026)
        for _ in range(20):
            q=ARM.center+ARM.radius*rng.uniform(-1,1,6)
            q[0]=ARM.upper[0];q[1]=ARM.lower[1]
            j=rng.normal(size=(5,6));e=rng.normal(size=5)
            dt=.05;speed=1.2;damping=.04
            v=bounded_velocity(j,e,q,damping,dt,speed)
            lo=np.maximum((ARM.lower-q)/dt,-speed);hi=np.minimum((ARM.upper-q)/dt,speed)
            self.assertTrue(np.all(v>=lo-1e-9) and np.all(v<=hi+1e-9))
            gradient=(j.T@j+damping*damping*np.eye(6))@v-3*j.T@e
            for k in range(6):
                if v[k]<=lo[k]+1e-8:self.assertGreaterEqual(gradient[k],-1e-8)
                elif v[k]>=hi[k]-1e-8:self.assertLessEqual(gradient[k],1e-8)
                else:self.assertAlmostEqual(gradient[k],0,places=8)

    def test_bounded_dls_recovers_limit_failure_without_increasing_task_error(self):
        # A reachable global target on which the original clipped DLS stalls.
        c=cases()[7];q=np.deg2rad(c['start_deg']);target=np.array(c['target_mm'])/1000
        baseline=solve(q,target,c['orientation_xy_deg'],'dls')
        bounded=solve(q,target,c['orientation_xy_deg'],'bounded_dls')
        self.assertFalse(baseline['converged']);self.assertTrue(bounded['converged'])
        axis=axis_from_xy(c['orientation_xy_deg'])
        errors=[task(np.deg2rad(f['q']),target,axis)[2] for f in bounded['frames']]
        merits=np.array([e@e for e in errors])
        self.assertTrue(np.all(np.diff(merits)<=1e-12))
        service_result=service.run({'task':'pose5','start_deg':c['start_deg'],'target_mm':c['target_mm'],
                                   'orientation_xy_deg':c['orientation_xy_deg'],'methods':['bounded_dls']})
        self.assertTrue(service_result['results'][0]['converged'])
