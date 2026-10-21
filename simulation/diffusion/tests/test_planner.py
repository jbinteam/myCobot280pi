import unittest
import numpy as np
from simulation.prm import JointSpace
from simulation.diffusion.planner import RRTConfig, Tree, plan, smooth, trajectory


class FixtureEnv:
    def __init__(self):
        self.space=JointSpace(lambda q:not (abs(q[0])<.15 and abs(q[1])<.3),
                              {'checker':'test rectangular barrier','environment':'six joint fixture'})
    def valid(self,q):return self.space.valid(q)
    def edge_valid(self,a,b,max_step_rad=.04):return self.space.edge_valid(a,b,max_step_rad)


class PlannerTest(unittest.TestCase):
    def test_blocked_rrt_star_smoothing_and_timing(self):
        env=FixtureEnv();start=np.array([-.5,0,0,0,0,0.]);goal=-start
        self.assertFalse(env.edge_valid(start,goal))
        path,report=plan(env,start,goal,42,RRTConfig(iterations=1200,post_solution=20))
        self.assertIsNotNone(path);self.assertEqual(report['status'],'rrt_star')
        path=smooth(env,path);states,times=trajectory(path,dt=.05,speed=.8,acceleration=2.)
        np.testing.assert_allclose(states[0],start);np.testing.assert_allclose(states[-1],goal)
        self.assertTrue(all(env.edge_valid(a,b) for a,b in zip(states[:-1],states[1:])))
        velocity=np.diff(states,axis=0)/np.diff(times)[:,None]
        self.assertLessEqual(np.linalg.norm(velocity,axis=1).max(),.8+1e-8)
        self.assertLessEqual(np.linalg.norm(np.diff(velocity,axis=0)/.05,axis=1).max(),2.+1e-8)

    def test_rewiring_propagates_costs_and_rejects_cycles(self):
        tree=Tree(np.zeros(6));one=tree.add(np.ones(6),0,3);two=tree.add(np.ones(6)*2,one,5)
        other=tree.add(np.ones(6)*.1,0,.2);tree.reparent(one,other,2)
        self.assertEqual(tree.cost[two],4)
        with self.assertRaises(ValueError):tree.reparent(other,two,1)

    def test_invalid_endpoints_never_return_a_path(self):
        path,report=plan(FixtureEnv(),np.zeros(6),np.ones(6))
        self.assertIsNone(path);self.assertEqual(report['status'],'invalid_endpoint')

    def test_observer_reports_real_tree_and_does_not_change_the_route(self):
        env=FixtureEnv();start=np.array([-.5,0,0,0,0,0.]);goal=-start
        config=RRTConfig(iterations=1200,post_solution=20)
        baseline,baseline_report=plan(env,start,goal,42,config)
        events=[];path,report=plan(env,start,goal,42,config,observer=events.append)
        np.testing.assert_array_equal(path,baseline)
        for key in ('nodes','iterations','rewires','status'):
            self.assertEqual(report[key],baseline_report[key])
        self.assertEqual(events[0]['kind'],'begin');self.assertEqual(events[-1]['kind'],'end')
        self.assertEqual(len([e for e in events if e['kind']=='add']),report['nodes']-1)
        self.assertEqual(len([e for e in events if e['kind']=='rewire']),report['rewires'])
        parents={0:-1};vertices={0:start}
        for event in events:
            if event['kind']=='add':
                self.assertIn(event['parent'],vertices)
                q=np.deg2rad(event['q_deg'])
                self.assertTrue(env.edge_valid(vertices[event['parent']],q,config.max_step_rad))
                vertices[event['index']]=q;parents[event['index']]=event['parent']
            elif event['kind']=='rewire':
                self.assertTrue(env.edge_valid(vertices[event['parent']],vertices[event['index']],config.max_step_rad))
                parents[event['index']]=event['parent']
        np.testing.assert_allclose(np.deg2rad(events[-1]['path_deg']),path)


if __name__=='__main__':unittest.main()
