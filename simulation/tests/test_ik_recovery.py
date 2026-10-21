from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np
from simulation import ik_recovery, planning, pose5, seed5
from simulation.kinematics import ARM
from simulation.prm import JointSpace, Roadmap


class RecoveryTest(unittest.TestCase):
    def setUp(self):
        self.q = np.deg2rad([10, -40, -110, 65, 20, 0])
        self.goal = self.q + np.deg2rad([3, 2, -2, 3, -2, 0])
        self.pose = ARM.transform(self.goal)
        self.target, self.axis = self.pose[:3, 3], self.pose[:3, 2]
        self.road = Roadmap(JointSpace())
        self.config = planning.IKConfig(task='pose5', nn='seed5', pose5_method='dls', pose5_global_recovery=True)

    def generated(self, goals=None):
        return {'goals': goals or [], 'attempts': [], 'seed_queue': [], 'raw_accepted_count': 0,
                'best_error': None, 'nn_checkpoint': None, 'ik_ms': 0.,
                'diagnostic': {'best_valid_error': None, 'refinements': 0,
                               'pose_accurate_invalid_candidates': 0, 'goal_constraint_rejections': 0}}

    def accepted(self, q, space=None, constraint=None):
        return planning.recovery_goal(q, self.target, space or self.road.space,
                                      self.config, self.axis, constraint) is not None

    def test_directed_residual_jacobian_and_local_spin_invariance(self):
        rng = np.random.default_rng(5102)
        for q in [self.q, *rng.uniform(ARM.lower, ARM.upper, (10, 6))]:
            _, r, j = ik_recovery.residual_jacobian(q, self.target, self.axis)
            h = 1e-6
            numeric = np.column_stack([(ik_recovery.residual_jacobian(q+h*np.eye(6)[k], self.target, self.axis)[1]
                                      - ik_recovery.residual_jacobian(q-h*np.eye(6)[k], self.target, self.axis)[1])/(2*h)
                                      for k in range(6)])
            np.testing.assert_allclose(j, numeric, atol=1e-9)
            spun = q.copy(); spun[5] += .7
            np.testing.assert_allclose(ik_recovery.residual_jacobian(spun, self.target, self.axis)[1], r, atol=1e-12)
        _, aligned, j = ik_recovery.residual_jacobian(self.goal, self.target, self.axis)
        _, opposite, _ = ik_recovery.residual_jacobian(self.goal, self.target, -self.axis)
        self.assertEqual(np.linalg.matrix_rank(j), 5)
        self.assertLess(aligned@aligned, 1e-20)
        self.assertAlmostEqual(opposite@opposite, 4*pose5.ORIENTATION_WEIGHT**2)

    def test_optimizer_recovers_a_clipped_dls_failure(self):
        case = pose5.cases()[7]
        q = np.deg2rad(case['start_deg']); target = np.array(case['target_mm'])/1000
        axis = pose5.axis_from_xy(case['orientation_xy_deg'])
        self.assertFalse(pose5.solve(q, target, target_axis=axis)['converged'])
        accept = lambda x: planning.recovery_goal(x, target, self.road.space, self.config, axis) is not None
        result = ik_recovery.optimize(q, target, axis, accept)
        self.assertTrue(result['converged'])
        self.assertTrue(accept(result['q_rad']))
        self.assertLessEqual(result['evaluations'], 96)
        self.assertTrue(np.all(result['q_rad'] >= ARM.lower) and np.all(result['q_rad'] <= ARM.upper))

    def test_optimizer_stationary_and_antipodal_states_are_not_accepted(self):
        result = ik_recovery.optimize(self.goal, self.target, self.axis, lambda q: False)
        self.assertFalse(result['converged'])
        self.assertEqual(result['stop_reason'], 'stationary_without_valid_goal')
        result = ik_recovery.optimize(self.goal, self.target, -self.axis, lambda q: False, evaluations=1)
        self.assertFalse(result['converged']); self.assertTrue(result['antipodal_escape'])
        self.assertTrue(np.isfinite(result['q_rad']).all())
        self.assertFalse(self.accepted(self.goal, constraint=lambda q: False))
        self.assertIsNone(planning.recovery_goal(self.goal, self.target, self.road.space,
                                                self.config, -self.axis))

    def test_acceptance_rejects_collision_nonfinite_and_out_of_limit_results(self):
        space = JointSpace(lambda q: False, {'checker': 'fixture', 'environment': 'none_valid'})
        self.assertFalse(self.accepted(self.goal, space))
        for value in ([np.nan]*6, ARM.upper+.1, [0]*5):
            self.assertFalse(self.accepted(value))

    def test_stratified_coverage_is_deterministic_and_visits_all_joint_bins(self):
        seeds = ik_recovery.stratified_seeds(12, 19)
        np.testing.assert_array_equal(seeds, ik_recovery.stratified_seeds(12, 19))
        self.assertFalse(np.array_equal(seeds, ik_recovery.stratified_seeds(12, 20)))
        bins = np.floor(12*(seeds-ARM.lower)/(ARM.upper-ARM.lower)).astype(int)
        for joint in range(6):
            np.testing.assert_array_equal(np.sort(bins[:, joint]), np.arange(12))
        self.assertTrue(np.isfinite(seeds).all())
        self.assertEqual(ik_recovery.stratified_seeds(0, 19).shape, (0, 6))

    def test_roadmap_seed_selection_covers_postures_not_only_free_spins(self):
        spins = [self.goal + np.array([0, 0, 0, 0, 0, value]) for value in (0, .3, .6, .9)]
        different = self.q.copy(); different[0] += .9*ARM.radius[0]
        self.road.q = np.stack([*spins, different])
        self.road.poses = np.stack([ARM.transform(q) for q in self.road.q])
        with patch.object(seed5, 'propose', side_effect=AssertionError('No trained checkpoint')):
            seeds, _ = ik_recovery.coverage_seeds(self.q, self.target, self.axis, self.road,
                                                 self.generated(), 32, 9001)
        selected = [s for s in seeds if s['source'] == 'recovery_roadmap']
        self.assertEqual(len(selected), 2)
        np.testing.assert_allclose(selected[0]['q_rad'][:5], self.goal[:5])
        np.testing.assert_allclose(selected[1]['q_rad'], different)
        self.assertTrue(any(s['source'] == 'recovery_stratified' for s in seeds))
        self.assertEqual(len({tuple(s['q_rad']) for s in seeds}), len(seeds))

    def test_reconditioned_nn_keeps_its_recorded_anchor_bounds(self):
        generated = self.generated(); generated['nn_checkpoint'] = {'status': 'ready'}
        generated['seed_queue'] = [{'source': 'nn_seed5_refined', 'q_rad': self.q.tolist()}]*4
        def proposal(q, target, axis):
            return np.stack([ARM.clip(q+value*ARM.radius) for value in (.1, .2, .3, .4)]), {'status':'ready'}, 0.
        with patch.object(seed5, 'propose', side_effect=proposal):
            seeds, duplicates = ik_recovery.coverage_seeds(self.q, self.target, self.axis, self.road,
                                                          generated, 32, 9001)
        self.assertGreaterEqual(duplicates, 4)
        anchors = [s for s in seeds if s['source'] == 'recovery_nn_anchor']
        self.assertEqual(len(anchors), 8)
        for s in anchors:
            self.assertTrue(np.all(np.abs(np.array(s['q_rad'])-s['anchor_q_rad']) <= ARM.radius+1e-12))

    def test_selected_solvers_finish_before_the_separate_trust_region_pass(self):
        failed = {'frames':[{'q':np.rad2deg(self.q).tolist()}], 'iterations':1, 'elapsed_ms':.1, 'stop_reason':'stalled'}
        def optimize(q, target, axis, accept, evaluations):
            self.assertTrue(accept(self.goal))
            return {'q_rad':self.goal, 'iterations':2, 'evaluations':3, 'elapsed_ms':.1,
                    'optimizer_status':-2, 'antipodal_escape':False, 'stop_reason':'target_reached'}
        cfg = replace(self.config, max_candidates=1, random_seeds=0, roadmap_seeds=0)
        with patch.object(seed5, 'propose', return_value=([self.q]*4, {'status':'ready'}, 0)), \
             patch.object(pose5, 'solve', return_value=failed) as solve, \
             patch.object(ik_recovery, 'optimize', side_effect=optimize) as rescue:
            result = planning.query(self.q, self.target, self.road, config=cfg, target_axis=self.axis)
        self.assertTrue(result['converged'])
        result=result['ik']
        self.assertEqual(solve.call_count, 2)
        self.assertEqual([call.args[3] for call in solve.call_args_list], ['dls','bounded_dls'])
        self.assertEqual(rescue.call_count, 1)
        self.assertEqual(result['attempts'][-1]['refinement_method'], 'trust_region')
        np.testing.assert_allclose(rescue.call_args.args[0], self.q)
        self.assertEqual(result['raw_accepted_count'], 0)
        self.assertEqual([s['source'] for s in result['seed_queue']], ['nn_seed5_refined']*4+['current_dls'])

    def test_second_pass_stops_at_goal_budget_and_filters_near_duplicate_goals(self):
        seeds = [{'source':'recovery_stratified', 'q_rad':(self.q+np.array([value,0,0,0,0,0])).tolist()}
                 for value in (.01, .02, .03, .04)]
        endings = [self.goal, self.goal+np.array([0,0,0,0,0,1e-4]), self.goal+np.array([0,0,0,0,0,.5])]
        def optimize(*args):
            return {'q_rad':endings.pop(0), 'iterations':2, 'evaluations':3, 'elapsed_ms':.1,
                    'optimizer_status':-2, 'antipodal_escape':False, 'stop_reason':'target_reached'}
        with patch.object(ik_recovery, 'coverage_seeds', return_value=(seeds, 0)), \
             patch.object(ik_recovery, 'optimize', side_effect=optimize) as rescue:
            result = planning.recover_goals(self.generated(), self.q, self.target, self.road,
                                            self.config, self.axis, goal_budget=2)
        self.assertEqual(rescue.call_count, 3); self.assertEqual(len(result['goals']), 2)
        self.assertEqual(result['recovery']['phases'][0]['used_seeds'], 3)
        self.assertEqual(result['recovery']['added_goals'], 2)

    def test_route_failure_expands_goals_and_retries_a_valid_direct_connection(self):
        space = JointSpace(lambda q: abs(q[5])>.15, {'checker':'wrist_wall', 'environment':'fixture'})
        road = Roadmap(space); start=self.q.copy(); start[5]=-1.
        goals=[]
        for spin in (.5, .8, 1.1, 1.4):
            q=self.goal.copy(); q[5]=spin
            goals.append({**planning.checked_goal(q,self.target,space,self.config,self.axis),'source':'fixture'})
        generated=self.generated(goals); candidate=self.goal.copy(); candidate[5]=-1.
        with patch.object(planning,'generate_goals',return_value=generated), \
             patch.object(ik_recovery,'coverage_seeds',return_value=([{'source':'recovery_roadmap','q_rad':candidate.tolist()}],0)):
            result=planning.query(start,self.target,road,self.config,target_axis=self.axis,shortcut=True)
        self.assertTrue(result['converged'],result.get('reason'))
        self.assertEqual(result['planning_outcome'],'direct_connection')
        self.assertEqual(len(result['ik']['goals']),5)
        self.assertEqual(result['ik']['recovery']['phases'][0]['trigger'],'route_failure')
        self.assertEqual(result['ik']['recovery']['phases'][0]['goal_budget'],8)
        self.assertTrue(all(space.valid(np.deg2rad(f['q'])) for f in result['frames']))

    def test_disabled_fallback_and_successful_routes_do_not_expand_search(self):
        with patch.object(planning,'generate_goals',return_value=self.generated()), \
             patch.object(planning,'recover_goals',side_effect=AssertionError('Fallback disabled')):
            result=planning.query(self.q,self.target,self.road,replace(self.config,numerical_fallback=False),target_axis=self.axis)
        self.assertEqual(result['ik_outcome'],'ik_solution_not_found')
        goal={**planning.checked_goal(self.goal,self.target,self.road.space,self.config,self.axis),'source':'fixture'}
        with patch.object(planning,'generate_goals',return_value=self.generated([goal])), \
             patch.object(planning,'recover_goals',side_effect=AssertionError('Already has a route')):
            result=planning.query(self.q,self.target,self.road,self.config,target_axis=self.axis)
        self.assertTrue(result['converged']);self.assertNotIn('recovery',result['ik'])

    def test_recovery_configuration_preserves_xyz_and_rejects_bad_budgets(self):
        with self.assertRaises(ValueError):planning.IKConfig(pose5_global_recovery=True)
        for settings in ({'pose5_global_recovery':1},{'recovery_seeds':0},
                         {'recovery_evaluations':601},{'recovery_max_candidates':True}):
            with self.subTest(settings=settings),self.assertRaises(ValueError):
                replace(self.config,**settings)
        self.assertFalse(planning.IKConfig().pose5_global_recovery)
        self.assertEqual(planning.IKConfig().max_candidates,4)

    def test_trajectory_failure_does_not_start_another_ik_search(self):
        goal={**planning.checked_goal(self.goal,self.target,self.road.space,self.config,self.axis),'source':'fixture'}
        with patch.object(planning,'generate_goals',return_value=self.generated([goal])), \
             patch.object(planning,'trajectory',side_effect=ValueError('fixture validation failed')), \
             patch.object(planning,'recover_goals',side_effect=AssertionError('Do not hide invalid trajectories')):
            result=planning.query(self.q,self.target,self.road,self.config,target_axis=self.axis)
        self.assertFalse(result['converged']);self.assertEqual(result['frames'],[])
        self.assertEqual(result['ik_outcome'],'accepted_goals')
        self.assertEqual(result['planning_outcome'],'trajectory_validation_failed')

    def test_route_recovery_skips_seeds_used_by_an_earlier_coverage_pass(self):
        first=self.generated()
        seed={'source':'recovery_current','q_rad':self.q.tolist()}
        first['recovery']={'phases':[{'seed_queue':[seed],'used_seeds':1}], 'elapsed_ms':1., 'added_goals':0}
        next_seed={'source':'recovery_stratified','q_rad':self.goal.tolist()}
        with patch.object(ik_recovery,'coverage_seeds',return_value=([seed,next_seed],0)):
            result=planning.recover_goals(first,self.q,self.target,self.road,self.config,self.axis,goal_budget=1,trigger='route_failure')
        phase=result['recovery']['phases'][-1]
        self.assertEqual(phase['used_seeds'],1)
        self.assertEqual(phase['seed_queue'],[next_seed])
        self.assertEqual(phase['previous_seeds_skipped'],1)


if __name__ == '__main__':unittest.main()
