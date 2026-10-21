import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from simulation.kinematics import ARM
from simulation.prm import JointSpace, PRMConfig, Roadmap, planning_space
from simulation.planning import IKConfig, checked_goal, query, trajectory, local_rollout
from simulation.self_collision import CollisionConfig, checker, collision_mesh
from simulation import service
from simulation.workspace import transforms

FIXTURE = Path(__file__).resolve().parents[1]/'fixtures/self_collision_path.json'


class SelfCollisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.space = planning_space(); cls.collision = cls.space.collision_checker
        cls.fixture = json.loads(FIXTURE.read_text())
        cls.start, cls.goal, cls.via = np.deg2rad([cls.fixture[k] for k in ('start_deg','goal_deg','via_deg')])

    def test_link_frames_match_shared_fk_and_zero_collision_origins(self):
        q = np.deg2rad([20,-35,-90,40,10,50])
        frames = ARM.link_transforms(q)
        self.assertEqual(set(frames), {ARM.base_frame, *ARM.link_names})
        np.testing.assert_allclose(frames[ARM.tcp], ARM.transform(q), atol=1e-12)
        np.testing.assert_allclose(frames[ARM.base_frame], np.eye(4))
        base = next(b for b in self.collision.bodies if b['link']=='g_base')
        np.testing.assert_allclose(base['origin'][:3,3], [0,0,-.03])

    def test_meshes_units_and_no_silent_wrist_geometry_loss(self):
        info = {i['link']:i for i in self.collision.description['mesh_info']}
        self.assertEqual(len(info),8)
        self.assertEqual(info['g_base']['asset_unit_m'], .001)
        self.assertEqual(info['joint2']['asset_unit_m'], 1.)
        self.assertGreater(info['joint5']['source_triangles'], 20000)
        for m in info.values():
            extents = np.diff(np.array(m['local_bounds_m']),axis=0)[0]
            self.assertTrue(np.all(extents>0)); self.assertLess(extents.max(), .2)
            self.assertGreater(m['hull_triangles'],0)

    def test_only_directly_connected_link_pairs_are_excluded(self):
        self.assertEqual(len(self.collision.excluded_pairs),7)
        self.assertEqual(len(self.collision.pairs),21)
        checked = {tuple(sorted((self.collision.bodies[i]['link'],self.collision.bodies[j]['link'])))
                   for i,j in self.collision.pairs}
        self.assertIn(('g_base','joint6'),checked)
        self.assertIn(('joint2','joint4'),checked)  # Two-hop links are checked.
        self.assertNotIn(('joint2','joint3'),checked)

    def test_known_free_and_colliding_configurations(self):
        for q in [np.zeros(6),self.start,self.goal,self.via]: self.assertTrue(self.space.valid(q))
        self.assertFalse(self.space.valid(ARM.lower))
        report = self.collision.check((self.start+self.goal)/2)
        self.assertFalse(report['valid'])
        self.assertIn(['g_base','joint6'], [c['links'] for c in report['conflicts']])
        self.assertTrue(any(c['intersecting'] for c in report['conflicts']))

    def test_clearance_rejects_near_contact_without_intersection(self):
        fixture = json.loads(FIXTURE.with_name('self_collision_clearance.json').read_text())
        q = np.deg2rad(fixture['q_deg'])
        self.assertTrue(planning_space(clearance_mm=0).valid(q))
        self.assertFalse(self.space.valid(q))
        report = self.collision.check(q)
        self.assertTrue(all(not c['intersecting'] and 0<c['distance_mm']<1 for c in report['conflicts']))

    def test_native_convex_collision_detects_containment(self):
        import fcl
        import trimesh
        hull = trimesh.creation.box([1,1,1])
        faces = np.column_stack([np.full(len(hull.faces),3),hull.faces]).ravel().tolist()
        geometry = fcl.Convex(np.array(hull.vertices),len(hull.faces),faces)
        inside = fcl.CollisionObject(fcl.Box(.1,.1,.1))
        self.assertGreater(fcl.collide(fcl.CollisionObject(geometry),inside,
                                      fcl.CollisionRequest(),fcl.CollisionResult()),0)

    def test_goal_accuracy_cannot_admit_self_colliding_goal(self):
        colliding = (self.start+self.goal)/2
        self.assertIsNone(checked_goal(colliding,ARM.fk(colliding),self.space,IKConfig()))
        result = query(colliding,ARM.fk(self.goal),Roadmap(self.space))
        self.assertEqual(result['status'],'invalid_start'); self.assertFalse(result['converged'])
        self.assertIn('Self-collision',result['reason'])
        self.assertTrue(result['self_collision']['conflicts'])

    def test_actual_geometry_blocks_direct_path_and_prm_via_is_valid(self):
        self.assertFalse(self.space.edge_valid(self.start,self.goal,.04))
        road = Roadmap(self.space,PRMConfig(samples=0,max_step_rad=.04))
        road.q = np.array([self.via]); road.poses = transforms(road.q); road.grow(0)
        result = road.plan(self.start,[self.goal],shortcut=True)
        self.assertEqual(result['reason'],'roadmap_path'); self.assertEqual(result['status'],'success')
        self.assertEqual(len(result['waypoints_rad']),3)
        self.assertTrue(all(self.space.edge_valid(a,b,.04) for a,b in
                            zip(result['waypoints_rad'],result['waypoints_rad'][1:])))
        frames = trajectory(result['waypoints_rad'],ARM.fk(self.goal),self.space,max_step_rad=.04)
        self.assertTrue(all(self.space.valid(ARM.clip(np.deg2rad(f['q']))) for f in frames))
        with self.assertRaisesRegex(ValueError,'segment'):
            trajectory([self.start,self.goal],ARM.fk(self.goal),self.space,max_step_rad=.04)

    def test_local_rollout_is_rejected_if_intermediate_self_collision(self):
        frames = [{'q':np.rad2deg(q).tolist(),'error_mm':100.} for q in [self.start,self.goal]]
        fake = {'frames':frames,'elapsed_ms':.1,'stop_reason':'iteration_limit'}
        with patch('simulation.planning.solvers.solve',return_value=fake):
            result = local_rollout(self.start,ARM.fk(self.goal),'conditional',self.space,IKConfig())
        self.assertEqual(result['status'],'local_rollout_constraint_violation')
        self.assertEqual(len(result['frames']),1)

    def test_cache_rejects_limit_only_clearance_and_pair_mask_changes(self):
        road = Roadmap(self.space,PRMConfig(samples=0,max_step_rad=.04))
        road.q = np.array([self.via]); road.poses = transforms(road.q); road.grow(0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'road.json'; road.save(path)
            loaded = Roadmap.load(path,self.space)
            self.assertEqual(loaded.space.validation,'joint_limits_and_self_collision')
            with self.assertRaisesRegex(ValueError,'Stale'): Roadmap.load(path,JointSpace())
            with self.assertRaisesRegex(ValueError,'Stale'): Roadmap.load(path,planning_space(clearance_mm=2))
            data = json.loads(path.read_text()); data['metadata']['assumptions']['excluded_pairs'].append(['g_base','joint6'])
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'Stale'): Roadmap.load(path,self.space)
            Roadmap(JointSpace(),PRMConfig(samples=0)).save(path)
            with self.assertRaisesRegex(ValueError,'Stale'): Roadmap.load(path,self.space)

    def test_roadmap_reuse_invalidates_modified_file_and_validity_profile(self):
        service._PRM_CACHE.clear()
        road = Roadmap(self.space,PRMConfig(samples=0)); road.grow(0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'road.json'; road.save(path)
            with patch('simulation.prm.Roadmap.load',wraps=Roadmap.load) as loader:
                a=service.load_roadmap(path,self.space); b=service.load_roadmap(path,self.space)
                self.assertIs(a,b); self.assertEqual(loader.call_count,1)
                road.save(path)
                service.load_roadmap(path,self.space); self.assertEqual(loader.call_count,2)
                with self.assertRaisesRegex(ValueError,'Stale'):
                    service.load_roadmap(path,JointSpace())
        service._PRM_CACHE.clear()

    def test_invalid_clearance_unknown_modes_and_missing_geometry_fail_closed(self):
        for value in [-1,11,True,float('nan')]:
            with self.assertRaises(ValueError): CollisionConfig(value)
        with self.assertRaises(ValueError): planning_space('unknown')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'empty.dae'
            path.write_text('<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1"/>')
            with self.assertRaises(Exception): collision_mesh(path,[1,1,1])


if __name__=='__main__':unittest.main()
