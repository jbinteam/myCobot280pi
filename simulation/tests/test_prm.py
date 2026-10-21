import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from simulation.kinematics import ARM
from simulation.prm import JointSpace, PRMConfig, Roadmap, fk_samples
from simulation.planning import IKConfig, checked_goal, generate_goals, local_rollout, query, trajectory
from simulation import pose5, service
from simulation.workspace import transforms
from simulation.trajectory import joint_waypoints


def q(x, y=0):
    value = ARM.center.copy(); value[:2] = [x, y]; return value


def obstacle_space():
    return JointSpace(lambda value: np.linalg.norm(value[:2]) >= .35,
                      {'checker': 'synthetic_disk_v1', 'environment': 'q1_q2_disk_radius_0.35'})


def hand_road(space, nodes, neighbors=2):
    road = Roadmap(space, PRMConfig(samples=0, neighbors=neighbors, max_step_rad=.04))
    road.q = np.array(nodes); road.poses = transforms(road.q)
    return road.grow(0)


class JointSpaceTests(unittest.TestCase):
    def test_limits_and_bounded_angular_interpolation(self):
        space = JointSpace(); a = ARM.center.copy(); b = a.copy()
        a[5] = np.deg2rad(177); b[5] = np.deg2rad(-177)
        points = space.interpolate(a, b, .08)
        self.assertGreater(space.distance(a, b), .9)
        self.assertTrue(np.any(np.abs(points[:, 5]) < .05))
        self.assertLessEqual(np.max(np.abs(np.diff(points, axis=0))), .08+1e-12)
        self.assertTrue(all(space.valid(p) for p in points))
        self.assertFalse(space.valid(ARM.upper+.01))
        self.assertFalse(space.valid([float('nan')]*6))
        self.assertTrue(space.valid(ARM.lower)); self.assertTrue(space.valid(ARM.upper))

    def test_invalid_middle_between_valid_endpoints(self):
        space = obstacle_space(); a = q(-1); b = q(1)
        self.assertTrue(space.valid(a)); self.assertTrue(space.valid(b))
        self.assertFalse(space.edge_valid(a, b, .04))
        self.assertTrue(JointSpace().edge_valid(a, b, .04))

    def test_callback_requires_cache_identity(self):
        with self.assertRaises(ValueError): JointSpace(lambda _: True)

    def test_fk_checks_candidate_and_orientation(self):
        space = JointSpace(); config = IKConfig(); angles = q(.6, -.8); target = ARM.fk(angles)
        self.assertIsNotNone(checked_goal(angles, target, space, config))
        self.assertIsNone(checked_goal(angles, target+[.01, 0, 0], space, config))
        self.assertIsNone(checked_goal(ARM.upper+.1, ARM.fk(ARM.upper+.1), space, config))
        axis = ARM.transform(angles)[:3, 2]
        self.assertIsNone(checked_goal(angles, target, space, config, -axis))


class RoadmapTests(unittest.TestCase):
    def test_direct_success_and_via_points(self):
        start = q(-1); goal = q(1)
        direct = Roadmap(JointSpace()).plan(start, [goal])
        self.assertEqual(direct['reason'], 'direct_connection')
        space = obstacle_space()
        road = hand_road(space, [q(-.8, .6), q(0, .8), q(.8, .6)])
        original = json.dumps(road.edges, sort_keys=True)
        result = road.plan(start, [goal], shortcut=True)
        self.assertEqual(result['status'], 'success'); self.assertEqual(result['reason'], 'roadmap_path')
        self.assertGreater(len(result['waypoints_rad']), 2)
        for a, b in zip(result['waypoints_rad'], result['waypoints_rad'][1:]):
            self.assertTrue(space.edge_valid(a, b, .04))
        for angles, transform in zip(result['waypoints_rad'], result['fk_transforms_m']):
            np.testing.assert_allclose(ARM.transform(angles), transform, atol=1e-12)
        self.assertEqual(json.dumps(road.edges, sort_keys=True), original)
        self.assertGreater(result['path_cost'], 0)

    def test_disconnected_and_attachment_failure(self):
        wall = JointSpace(lambda value: abs(value[0]) > .15,
                          {'checker': 'wall_v1', 'environment': 'synthetic_q1_wall'})
        road = hand_road(wall, [q(-.8), q(-.5), q(.5), q(.8)])
        result = road.plan(q(-1), [q(1)])
        self.assertEqual(result['status'], 'path_not_found')
        self.assertEqual(result['reason'], 'disconnected_components')
        self.assertEqual(Roadmap(obstacle_space()).plan(q(-1), [q(1)])['status'], 'roadmap_connection_not_found')

    def test_multiple_goals_try_direct_first(self):
        road = Roadmap(obstacle_space()); start = q(-1)
        self.assertEqual(road.plan(start, [q(1), q(-.5)])['reason'], 'direct_connection')
        self.assertEqual(road.plan([999]*6, [q(1)])['status'], 'invalid_start')
        self.assertEqual(road.plan(start, [])['status'], 'invalid_goal')

    def test_dijkstra_selects_lower_nonnegative_cost(self):
        road = hand_road(obstacle_space(), [q(-.8, .6), q(0, .6), q(.8, .6),
                                           q(-.8, 1.8), q(0, 1.8), q(.8, 1.8)], neighbors=6)
        result = road.plan(q(-1), [q(1)])
        self.assertEqual(result['status'], 'success')
        self.assertLess(max(np.array(result['waypoints_rad'])[:, 1]), 1.)

    def test_seeded_sampling_reuse_and_grow(self):
        samples = fk_samples('simulation/artifacts/dataset.npz')
        config = PRMConfig(samples=20, neighbors=3, seed=8)
        a = Roadmap(JointSpace(), config=config).grow(samples=samples)
        b = Roadmap(JointSpace(), config=config).grow(samples=samples)
        np.testing.assert_array_equal(a.q, b.q)
        self.assertEqual(a.edges, b.edges); self.assertEqual(a.stats['reused_fk_samples'], 10)
        first = a.q.copy(); a.grow(5)
        self.assertEqual(len(a.q), 25); np.testing.assert_array_equal(a.q[:20], first)
        self.assertTrue(all(a.space.valid(angles) for angles in a.q))

    def test_persistence_and_assumption_invalidation(self):
        space = obstacle_space(); road = hand_road(space, [q(-.8, .6), q(0, .8), q(.8, .6)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'road.json'; road.save(path)
            loaded = Roadmap.load(path, space)
            np.testing.assert_array_equal(loaded.q, road.q); self.assertEqual(loaded.edges, road.edges)
            self.assertEqual(loaded.plan(q(-1), [q(1)])['status'], 'success')
            with self.assertRaisesRegex(ValueError, 'Stale'): Roadmap.load(path)
            with self.assertRaisesRegex(ValueError, 'configuration'):
                Roadmap.load(path, space, PRMConfig(samples=1))
            original = json.loads(path.read_text())
            for key in ('geometry_sha256', 'urdf_sha256', 'limits_rad', 'units', 'base_transform', 'assumptions'):
                data = json.loads(json.dumps(original)); data['metadata'][key] = 'changed'
                path.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, 'Stale'): Roadmap.load(path, space)
            data = json.loads(json.dumps(original)); data['edges'][0][2] = -1
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'edge'): Roadmap.load(path, space)
            data = json.loads(json.dumps(original)); data['fk_transforms_m'][0][0][3] += .01
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'FK metadata'): Roadmap.load(path, space)

    def test_fk_sample_provenance_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'dataset.npz'; np.savez(path, q=[q(.5)])
            path.with_name('training.json').write_text(json.dumps({'urdf_sha256': 'old'}))
            with self.assertRaisesRegex(ValueError, 'mismatch'): fk_samples(path)


class QueryTests(unittest.TestCase):
    def test_accurate_endpoint_independent_of_refinement_motion(self):
        start = q(-1); goal = q(1); space = obstacle_space()
        road = hand_road(space, [q(-.8, .6), q(0, .8), q(.8, .6)])
        generated = {'goals': [{'q_rad': goal.tolist()}], 'attempts': [], 'ik_ms': 1.}
        with patch('simulation.planning.generate_goals', return_value=generated) as solver:
            result = query(start, ARM.fk(goal), road, shortcut=True)
        self.assertEqual(solver.call_count, 1)
        self.assertEqual(result['status'], 'success'); self.assertTrue(result['converged'])
        angles = np.deg2rad([f['q'] for f in result['frames']])
        self.assertLessEqual(np.max(np.linalg.norm(np.diff(angles, axis=0), axis=1)), 1.2*.05+1e-12)
        self.assertTrue(all(space.valid(a) for a in angles))
        self.assertLess(result['final_error_mm'], 1e-8)
        commands = result['controller_waypoints_deg']
        self.assertLessEqual(np.max(np.abs(np.diff([np.rad2deg(start).tolist()]+commands, axis=0))), 3.5+1e-10)

    def test_query_direct_numerical_goal_and_diagnostics(self):
        start = np.deg2rad([0, -40, -130, 80, 0, 50]); goal = start.copy(); goal[0] += .1
        config = IKConfig(nn='none', random_seeds=0, roadmap_seeds=0, max_candidates=1)
        result = query(start, ARM.fk(goal), Roadmap(JointSpace()), config)
        self.assertEqual(result['status'], 'success'); self.assertLessEqual(result['final_error_mm'], 1.)
        self.assertEqual(result['validation'], 'joint_limits_only')
        self.assertEqual(query([999]*6, ARM.fk(goal), Roadmap(JointSpace()), config)['status'], 'invalid_start')
        self.assertEqual(query(start, [float('nan'), 0, 0], Roadmap(JointSpace()), config)['status'], 'invalid_target')
        failed = query(start, [.8, 0, 0], Roadmap(JointSpace()), config)
        self.assertEqual(failed['status'], 'ik_solution_not_found'); self.assertFalse(failed['converged'])
        self.assertIn('does not prove', failed['limitations'])

    def test_local_rollout_stall_and_constraint_diagnostics(self):
        start = q(-1); target = ARM.fk(q(1)); config = IKConfig(nn='conditional', refine=False,
                                                           numerical_fallback=False, stall_window=3)
        frames = [{'q': np.rad2deg(start).tolist(), 'error_mm': 100.} for _ in range(8)]
        fake = {'frames': frames, 'elapsed_ms': .1, 'stop_reason': 'iteration_limit'}
        with patch('simulation.planning.solvers.solve', return_value=fake):
            result = query(start, target, Roadmap(JointSpace()), config)
        self.assertEqual(result['status'], 'local_rollout_stalled')
        frames = [{'q': np.rad2deg(p).tolist(), 'error_mm': 100.} for p in [q(-1), q(1)]]
        with patch('simulation.planning.solvers.solve', return_value={**fake, 'frames': frames}):
            rollout = local_rollout(start, target, 'conditional', obstacle_space(), config)
        self.assertEqual(rollout['status'], 'local_rollout_constraint_violation')
        self.assertEqual(len(rollout['frames']), 1)

    def test_raw_prediction_is_not_accepted_without_fk_accuracy(self):
        fake = type('ZeroModel', (), {'predict': lambda *args: np.zeros(6)})()
        config = IKConfig(nn='direct', refine=False, numerical_fallback=False)
        with patch('simulation.planning.solvers.models', return_value=fake):
            goals = generate_goals(q(-1), ARM.fk(q(1)), config=config)
        self.assertEqual(goals['goals'], []); self.assertFalse(goals['attempts'][0]['accepted'])

    def test_colliding_nn_seed_can_refine_to_valid_endpoint(self):
        space = obstacle_space(); start = q(-1); goal = q(1)
        fake = type('ZeroModel', (), {'predict': lambda *args: np.zeros(6)})()
        config = IKConfig(nn='direct', refine=True, numerical_fallback=False)
        refined = {'frames': [{'q': np.rad2deg(goal).tolist()}], 'iterations': 3,
                   'elapsed_ms': .1, 'stop_reason': 'target_reached'}
        with patch('simulation.planning.solvers.models', return_value=fake), \
                patch('simulation.planning.solvers.solve', return_value=refined) as solver:
            generated = generate_goals(start, ARM.fk(goal), space=space, config=config)
        np.testing.assert_array_equal(solver.call_args.args[0], ARM.center)
        self.assertFalse(space.valid(ARM.center))
        self.assertFalse(generated['attempts'][0]['accepted'])
        self.assertEqual(generated['goals'][0]['source'], 'nn_direct_refined')
        self.assertTrue(generated['attempts'][-1]['accepted'])
        # Refinement did not authorize executing the obstructed direct motion.
        self.assertFalse(space.edge_valid(start, generated['goals'][0]['q_rad'], .04))

    def test_refinement_does_not_admit_colliding_endpoint(self):
        space = obstacle_space(); start = q(-1); goal = ARM.center.copy()
        fake = type('ZeroModel', (), {'predict': lambda *args: np.zeros(6)})()
        refined = {'frames': [{'q': np.rad2deg(goal).tolist()}], 'iterations': 3,
                   'elapsed_ms': .1, 'stop_reason': 'target_reached'}
        with patch('simulation.planning.solvers.models', return_value=fake), \
                patch('simulation.planning.solvers.solve', return_value=refined):
            generated = generate_goals(start, ARM.fk(goal), space=space,
                                       config=IKConfig(nn='direct', numerical_fallback=False))
        self.assertEqual(generated['goals'], [])
        self.assertFalse(generated['attempts'][-1]['valid_configuration'])

    def test_pose5_uses_numerical_checkpoints_stay_xyz_only(self):
        with self.assertRaisesRegex(ValueError, 'XYZ only'): IKConfig(task='pose5')
        start = q(.3, -.8); t = ARM.transform(start)
        result = query(start, t[:3, 3], Roadmap(JointSpace()), IKConfig(task='pose5', nn='none'), pose5.xy_from_axis(t[:3, 2]))
        self.assertEqual(result['status'], 'success'); self.assertLess(result['final_orientation_error_deg'], 1e-8)
        start = np.deg2rad([0, -40, -130, 80, 0, 50]); goal = start.copy(); goal[0] += .08
        t = ARM.transform(goal)
        result = query(start, t[:3, 3], Roadmap(JointSpace()),
                       IKConfig(task='pose5', nn='none', random_seeds=0, roadmap_seeds=0, max_candidates=1),
                       pose5.xy_from_axis(t[:3, 2]))
        self.assertEqual(result['status'], 'success')
        self.assertLessEqual(result['final_error_mm'], 1.)
        self.assertLessEqual(result['final_orientation_error_deg'], 1.)

    def test_pose5_reuses_selected_numerical_solver_endpoint(self):
        start = q(.3, -.8); goal = q(.5, -.6); t = ARM.transform(goal)
        config = IKConfig(task='pose5', nn='none', pose5_method='null_classical',
                          random_seeds=0, roadmap_seeds=0, max_candidates=1)
        solved = {'frames': [{'q': np.rad2deg(goal).tolist()}], 'iterations': 3,
                  'elapsed_ms': .1, 'stop_reason': 'target_reached'}
        with patch('simulation.planning.pose5.solve', return_value=solved) as solver:
            result = generate_goals(start, t[:3, 3], space=JointSpace(), config=config,
                                    orientation_xy_deg=pose5.xy_from_axis(t[:3, 2]))
        self.assertEqual(solver.call_count, 1)
        self.assertEqual(solver.call_args.args[3], 'null_classical')
        self.assertEqual(result['attempts'][0]['refinement_method'], 'null_classical')
        self.assertLess(result['goals'][0]['orientation_error_deg'], 1e-8)

    def test_pose5_selected_solver_failure_tries_bounded_dls(self):
        start = q(.3, -.8); goal = q(.5, -.6); t = ARM.transform(goal)
        config = IKConfig(task='pose5', nn='none', pose5_method='null_classical',
                          random_seeds=0, roadmap_seeds=0, max_candidates=1)
        def solved(q):
            return {'frames': [{'q': np.rad2deg(q).tolist()}], 'iterations': 3,
                    'elapsed_ms': .1, 'stop_reason': 'stalled'}
        with patch('simulation.planning.pose5.solve', side_effect=[solved(start), solved(goal)]) as solver:
            result = generate_goals(start, t[:3, 3], space=JointSpace(), config=config,
                                    orientation_xy_deg=pose5.xy_from_axis(t[:3, 2]))
        self.assertEqual([call.args[3] for call in solver.call_args_list], ['null_classical', 'bounded_dls'])
        self.assertFalse(result['attempts'][0]['accepted'])
        self.assertTrue(result['attempts'][1]['accepted'])

    def test_pose5_selected_solver_cannot_admit_colliding_goal(self):
        start = q(-1); goal = ARM.center.copy(); t = ARM.transform(goal)
        config = IKConfig(task='pose5', nn='none', pose5_method='null_classical',
                          random_seeds=0, roadmap_seeds=0)
        solved = {'frames': [{'q': np.rad2deg(goal).tolist()}], 'iterations': 3,
                  'elapsed_ms': .1, 'stop_reason': 'target_reached'}
        with patch('simulation.planning.pose5.solve', return_value=solved):
            result = generate_goals(start, t[:3, 3], space=obstacle_space(), config=config,
                                    orientation_xy_deg=pose5.xy_from_axis(t[:3, 2]))
        self.assertEqual(result['goals'], [])
        self.assertEqual(len(result['attempts']), 2)
        self.assertTrue(all(not a['valid_configuration'] for a in result['attempts']))

    def test_service_offset_and_shared_controller_waypoint_contract(self):
        start = q(.6, -.8); t = ARM.transform(start); offset = np.array([10., 3., -5.])
        body = {'start_deg': np.rad2deg(start).tolist(), 'target_mm': (t[:3, 3]*1000-t[:3, :3]@offset).tolist(),
                'eef_offset_mm': offset.tolist(), 'nn': 'none'}
        result = service.plan(body, Roadmap(JointSpace()))
        self.assertEqual(result['status'], 'success'); self.assertTrue(result['simulation_only'])
        np.testing.assert_allclose(result['target_mm'], t[:3, 3]*1000, atol=1e-10)
        points = joint_waypoints([0]*6, [[1]*6, [2.5]*6, [10]*6], compress=True)
        self.assertEqual(len(points), 4)
        self.assertLessEqual(np.max(np.abs(np.diff([[0]*6]+points, axis=0))), 3.5)

    def test_config_errors_and_invalid_trajectory(self):
        for change in ({'neighbors': 0}, {'max_step_rad': 0}, {'samples': True}):
            with self.assertRaises(ValueError): PRMConfig(**change)
        for change in ({'random_seeds': -1}, {'steps': 0}, {'nn': 'unknown'}, {'refine': 1}, {'pose5_method': 'unknown'}):
            with self.assertRaises(ValueError): IKConfig(**change)
        with self.assertRaisesRegex(ValueError, 'segment'):
            trajectory([q(-1), q(1)], ARM.fk(q(1)), obstacle_space(), .04)

    def test_trajectory_preserves_exact_operating_bound_endpoints(self):
        rng = np.random.default_rng(7); space = JointSpace()
        for i in range(12):
            start = rng.uniform(ARM.lower, ARM.upper); end = ARM.lower if i%2 == 0 else ARM.upper
            frames = trajectory([start, end], ARM.fk(end), space)
            np.testing.assert_array_equal(frames[-1]['q'], np.rad2deg(end))
            self.assertTrue(all(space.valid(np.deg2rad(f['q'])) for f in frames[:-1]))
            commands = joint_waypoints(np.rad2deg(start), [np.rad2deg(end)])
            np.testing.assert_array_equal(commands[-1], np.rad2deg(end))


if __name__ == '__main__': unittest.main()
