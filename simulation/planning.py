"""Goal IK and joint-path planning are separate, simulation-only stages."""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from .kinematics import ARM
from . import pose5, solvers, seed5, guided_repair
from .prm import JointSpace, Roadmap, planning_space
from .trajectory import joint_waypoints


@dataclass(frozen=True)
class IKConfig:
    task: str = 'position'
    nn: str = 'seed'
    refine: bool = True
    numerical_fallback: bool = True
    random_seeds: int = 4
    roadmap_seeds: int = 4
    max_candidates: int = 4
    seed: int = 9001
    steps: int = 240
    speed_rad_s: float = 1.2
    tolerance_mm: float = 1.
    damping: float = .04
    orientation_tolerance_deg: float = 1.
    stall_window: int = 30
    min_progress_mm: float = .01
    pose5_method: str = 'bounded_dls'
    pose5_recovery_methods: tuple = ()
    pose5_global_recovery: bool = False
    recovery_seeds: int = 32
    recovery_evaluations: int = 96
    recovery_max_candidates: int = 8
    guided_route_repair: bool = False
    route_repair_samples: int = 32

    def __post_init__(self):
        if self.task not in ('position', 'pose5') or self.nn not in ('none', 'seed', 'seed5', 'direct', 'conditional', 'differential'):
            raise ValueError('Unknown IK task or neural proposal method')
        if self.task == 'pose5' and self.nn not in ('none','seed5'):
            raise ValueError('Existing NN checkpoints support XYZ only; use nn=seed5 or none for pose5')
        if self.task == 'position' and self.nn == 'seed5':
            raise ValueError('Seed5 checkpoints require the pose5 task; use the existing seed for XYZ')
        if self.pose5_method not in [method[0] for method in pose5.METHODS]:
            raise ValueError('Unknown 5D numerical IK method')
        if (not isinstance(self.pose5_recovery_methods, (tuple, list)) or
                any(method not in [m[0] for m in pose5.METHODS] for method in self.pose5_recovery_methods) or
                len(set(self.pose5_recovery_methods)) != len(self.pose5_recovery_methods)):
            raise ValueError('5D recovery methods must be distinct supported solver names')
        if self.task != 'pose5' and self.pose5_recovery_methods:
            raise ValueError('5D recovery methods require the pose5 task')
        for key, low, high in [('random_seeds', 0, 100), ('roadmap_seeds', 0, 100), ('max_candidates', 1, 100),
                               ('seed', 0, 2**32-1), ('stall_window', 2, 600),
                               ('recovery_seeds', 1, 100), ('recovery_evaluations', 1, 600),
                               ('recovery_max_candidates', 1, 100), ('route_repair_samples', 1, 128)]:
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f'{key} must be an integer between {low} and {high}')
        if not isinstance(self.refine, bool) or not isinstance(self.numerical_fallback, bool):
            raise ValueError('refine and numerical_fallback must be booleans')
        if not isinstance(self.pose5_global_recovery, bool):
            raise ValueError('pose5_global_recovery must be a boolean')
        if not isinstance(self.guided_route_repair, bool):
            raise ValueError('guided_route_repair must be a boolean')
        if self.pose5_global_recovery and self.task != 'pose5':
            raise ValueError('Global 5D recovery requires the pose5 task')
        solvers.validate(ARM.center, np.zeros(3), 'dls', self.steps, self.speed_rad_s, self.tolerance_mm, self.damping)
        if not np.isfinite(self.orientation_tolerance_deg) or not .1 <= self.orientation_tolerance_deg <= 10:
            raise ValueError('Orientation tolerance must be between 0.1 and 10 degrees')
        if not np.isfinite(self.min_progress_mm) or self.min_progress_mm <= 0:
            raise ValueError('min_progress_mm must be positive')


def goal_error(q, target, target_axis=None):
    t = ARM.transform(q)
    return {'error_mm': float(np.linalg.norm(t[:3, 3]-target)*1000),
            'orientation_error_deg': pose5.axis_error(t[:3, 2], target_axis)[2] if target_axis is not None else None}


def checked_goal(q, target, space, config, target_axis=None):
    if config.task=='pose5':
        if target_axis is None:raise ValueError('A 5D goal requires a directed tool-Z target')
        target_axis=pose5.unit_axis(target_axis)
    if not space.valid(q): return None
    error = goal_error(q, target, target_axis)
    if error['error_mm'] > config.tolerance_mm: return None
    if target_axis is not None and error['orientation_error_deg'] > config.orientation_tolerance_deg: return None
    return {'q_rad': np.asarray(q).tolist(), 'q_deg': np.rad2deg(q).tolist(),
            'fk_transform_m': ARM.transform(q).tolist(), **error}


def recovery_goal(q, target, space, config, target_axis, goal_constraint=None):
    """Same acceptance, with cheap FK gates before optimizer collision checks."""
    q = np.asarray(q, float)
    if q.shape != (6,) or not np.isfinite(q).all() or np.any(q < ARM.lower) or np.any(q > ARM.upper):
        return None
    error = goal_error(q, target, target_axis)
    if error['error_mm'] > config.tolerance_mm or error['orientation_error_deg'] > config.orientation_tolerance_deg:
        return None
    if goal_constraint is not None and not goal_constraint(q):
        return None
    return checked_goal(q, target, space, config, target_axis)


def local_rollout(q, target, method, space, config, max_step_rad=.08):
    """Evaluate the existing increment/velocity controller, never a single step.

    Retain only the prefix before a detected stall or invalid transition. This
    adapter diagnoses progress without changing old checkpoints or solve().
    """
    result = solvers.solve(q, target, method, steps=config.steps, speed=config.speed_rad_s,
                           tolerance=config.tolerance_mm, damping=config.damping)
    frames = result['frames']; kept = [frames[0]]; reason = None
    for frame in frames[1:]:
        a = np.deg2rad(kept[-1]['q']); b = np.deg2rad(frame['q'])
        if not space.edge_valid(a, b, max_step_rad):
            reason = 'local_rollout_constraint_violation'; break
        kept.append(frame)
        if frame['error_mm'] <= config.tolerance_mm: break
        if len(kept) > config.stall_window:
            recent = kept[-config.stall_window:]
            previous_best = min(f['error_mm'] for f in kept[:-config.stall_window+1])
            if previous_best-min(f['error_mm'] for f in recent) < config.min_progress_mm:
                reason = 'local_rollout_stalled'; break
    if reason is None and kept[-1]['error_mm'] > config.tolerance_mm:
        reason = 'local_rollout_stalled' if 'stalled' in result['stop_reason'] else 'local_rollout_budget_exhausted'
    return {'q_rad': np.deg2rad(kept[-1]['q']), 'frames': kept,
            'status': reason or 'success', 'elapsed_ms': result['elapsed_ms']}


def generate_goals(q_current, target, roadmap=None, space=None, config=None, orientation_xy_deg=None, target_axis=None,goal_constraint=None):
    """Propose/refine endpoints; IK refinement frames are not planned motion.

    Numerical refinement may traverse invalid motion: only its final endpoint
    is used here. The independent planner must validate the eventual path.
    """
    space = space or (roadmap.space if roadmap is not None else JointSpace())
    config = config or IKConfig(); began = perf_counter(); attempts = []; goals = []; seeds = []; best = None; best_valid = None
    if goal_constraint is not None and (config.task != 'pose5' or not callable(goal_constraint)):
        raise ValueError('An additional goal constraint requires a callable and pose5 task')
    target_axis = pose5.resolve_axis(orientation_xy_deg,target_axis) if config.task == 'pose5' else None
    if target_axis is not None:orientation_xy_deg=pose5.xy_from_axis(target_axis)
    model_status=None;inference_ms=0.
    q_current = np.asarray(q_current, float); target = np.asarray(target, float)
    solvers.validate(q_current, target, 'dls', config.steps, config.speed_rad_s, config.tolerance_mm, config.damping)

    def admit(q, source):
        nonlocal best, best_valid
        q = np.asarray(q, float)
        error = goal_error(q, target, target_axis) if q.shape == (6,) and np.isfinite(q).all() else None
        def score(value):
            return value['error_mm']/config.tolerance_mm+(value['orientation_error_deg']/config.orientation_tolerance_deg if target_axis is not None else 0)
        if error is not None and (best is None or score(error)<score(best)): best = error
        valid = space.valid(q)
        if valid and error is not None and (best_valid is None or score(error)<score(best_valid)):best_valid = error
        candidate = checked_goal(q, target, space, config, target_axis)
        constraint_passed = bool(goal_constraint(q)) if candidate is not None and goal_constraint is not None else None
        if constraint_passed is False:candidate = None
        added=candidate is not None and not any(np.linalg.norm(q-np.array(g['q_rad'])) < 1e-6 for g in goals)
        if added:
            goals.append({**candidate, 'source': source})
        rejection = ('invalid_configuration' if not valid else
                     'position_tolerance' if error['error_mm'] > config.tolerance_mm else
                     'direction_tolerance' if target_axis is not None and error['orientation_error_deg'] > config.orientation_tolerance_deg else
                     'goal_constraint' if constraint_passed is False else None)
        attempts.append({'source': source, 'valid_configuration': valid, 'accepted': candidate is not None, 'added_unique_goal':added,
                         'rejection_reason':rejection,
                         **(error or {})})
        if not attempts[-1]['valid_configuration']:
            attempts[-1]['configuration_diagnostic'] = space.diagnostic(q)

    # Already at target is a valid query even when every neural proposal fails.
    if (checked_goal(q_current, target, space, config, target_axis) is not None
            and (goal_constraint is None or goal_constraint(q_current))):
        admit(q_current, 'current_state')
    elif config.nn != 'none':
        normalized = (q_current-ARM.center)/ARM.radius
        if config.nn == 'seed5':
            proposed,model_status,inference_ms=seed5.propose(q_current,target,target_axis)
            if model_status['status']!='ready':
                attempts.append({'source':'nn_seed5','accepted':False,'model_status':model_status['status'],
                                 'stop_reason':'checkpoint_'+model_status['status'],'model_diagnostic':model_status['reason']})
        elif config.nn == 'seed':
            raw = solvers.models().predict('seed', np.r_[target/ARM.scale, normalized]).reshape(4, 6)
            proposed = ARM.clip(q_current+ARM.radius*np.tanh(raw))
        elif config.nn == 'direct':
            proposed = [ARM.center+ARM.radius*np.tanh(solvers.models().predict('direct', target/ARM.scale))]
        else:
            rollout = local_rollout(q_current, target, config.nn, space, config,
                                    roadmap.config.max_step_rad if roadmap else .08)
            attempts.append({'source': config.nn, 'rollout_status': rollout['status'],
                             'rollout_steps': len(rollout['frames'])-1})
            proposed = [rollout['q_rad']] if rollout['status'] != 'local_rollout_constraint_violation' else []
        for q in proposed:
            admit(q, 'nn_'+config.nn)
            # A seed is an optimizer starting point, not executed motion.
            # Like random numerical seeds, it may be self-colliding; the
            # refined endpoint and independently planned path must be valid.
            if (config.refine and np.asarray(q).shape == (6,) and np.isfinite(q).all()
                    and np.all(q >= ARM.lower) and np.all(q <= ARM.upper)):
                seeds.append((q, 'nn_'+config.nn+'_refined'))
    raw_accepted_count=len(goals)
    if not goals and config.numerical_fallback:
        # Add conventional seeds independently of any NN success or rollout.
        seeds.append((q_current, 'current_dls'))
        if roadmap is not None and len(roadmap.q):
            distances = np.linalg.norm(roadmap.poses[:, :3, 3]-target, axis=1)/ARM.scale
            if target_axis is not None:
                angles=np.arctan2(np.linalg.norm(np.cross(roadmap.poses[:,:3,2],target_axis),axis=1),
                                  np.clip(roadmap.poses[:,:3,2]@target_axis,-1,1))
                distances += .6*angles
            for i in np.argsort(distances, kind='stable')[:config.roadmap_seeds]:
                seeds.append((roadmap.q[i], 'roadmap_dls'))
        rng = np.random.default_rng(config.seed)
        seeds.extend((q, 'random_dls') for q in rng.uniform(ARM.lower, ARM.upper, (config.random_seeds, 6)))
    seed_queue=[{'source':source,'q_rad':np.asarray(q).tolist()} for q,source in seeds]
    seen = set()
    for q, source in seeds:
        if len(goals) >= config.max_candidates: break
        key = tuple(np.asarray(q).round(10))
        if key in seen: continue
        seen.add(key)
        kwargs = {'steps': config.steps, 'speed': config.speed_rad_s,
                  'tolerance': config.tolerance_mm, 'damping': config.damping}
        methods = (list(dict.fromkeys([config.pose5_method, *config.pose5_recovery_methods] + (['bounded_dls'] if config.pose5_method != 'bounded_dls' else [])))
                   if config.task == 'pose5' else ['dls'])
        if goal_constraint is not None:kwargs['goal_constraint']=goal_constraint
        for method in methods:
            solved = (pose5.solve(q, target, orientation_xy_deg, method,
                                 orientation_tolerance_deg=config.orientation_tolerance_deg,target_axis=target_axis, **kwargs)
                      if config.task == 'pose5' else solvers.solve(q, target, method, **kwargs))
            # Optimizer traces are not motion; retain only a validated endpoint.
            admit(ARM.clip(np.deg2rad(solved['frames'][-1]['q'])), source)
            attempts[-1].update(iterations=solved['iterations'], refinement_ms=solved['elapsed_ms'],
                                stop_reason=solved['stop_reason'], refinement_method=method)
            if config.pose5_global_recovery:
                attempts[-1]['end_q_rad'] = ARM.clip(np.deg2rad(solved['frames'][-1]['q'])).tolist()
            if attempts[-1]['accepted']: break
    accurate_invalid = sum(not a.get('valid_configuration',True) and a.get('error_mm',np.inf)<=config.tolerance_mm
                           and (target_axis is None or a.get('orientation_error_deg',np.inf)<=config.orientation_tolerance_deg) for a in attempts)
    diagnostic={'pose_accurate_invalid_candidates':accurate_invalid,
                'goal_constraint_rejections':sum(a.get('rejection_reason')=='goal_constraint' for a in attempts),
                'refinements':sum('refinement_method' in a for a in attempts),
                'best_valid_error':best_valid}
    return {'goals': goals[:config.max_candidates], 'attempts': attempts, 'best_error': best, 'diagnostic':diagnostic,
            'target_axis':target_axis.tolist() if target_axis is not None else None,
            'seed_queue':seed_queue,'raw_accepted_count':raw_accepted_count,'nn_checkpoint':model_status,'inference_ms':inference_ms,
            'ik_ms': (perf_counter()-began)*1000, 'config': asdict(config)}


def recover_goals(generated, q_current, target, roadmap, config, target_axis,
                  goal_constraint=None, trigger='ik_failure', goal_budget=None):
    """Extend a completed normal search without changing its raw/fallback policy."""
    from . import ik_recovery
    began = perf_counter(); budget = goal_budget or config.max_candidates
    result = {**generated, 'goals': list(generated['goals']), 'attempts': list(generated['attempts']),
              'diagnostic': dict(generated['diagnostic'])}
    recovery = generated.get('recovery', {'phases': [], 'elapsed_ms': 0., 'added_goals': 0})
    previous = {tuple(np.asarray(seed['q_rad']).round(10))
                for phase in recovery['phases'] for seed in phase['seed_queue'][:phase['used_seeds']]}
    queue, duplicates = ik_recovery.coverage_seeds(q_current, target, target_axis, roadmap, generated,
                config.recovery_seeds, config.seed, use_nn=config.nn == 'seed5')
    queue = [seed for seed in queue if tuple(np.asarray(seed['q_rad']).round(10)) not in previous]
    before = len(result['goals']); used = 0; evaluations = 0

    def candidate(q):
        return recovery_goal(q, target, roadmap.space, config, target_axis, goal_constraint)

    def record(q, source, optimized=None):
        q = np.asarray(q, float); error = goal_error(q, target, target_axis)
        valid = roadmap.space.valid(q); goal = candidate(q)
        # Expanded routing choices should cover different joint postures;
        # nearby duplicate endpoints are not eight useful IK alternatives.
        added = goal is not None and not any(roadmap.space.distance(q, g['q_rad']) < .025 for g in result['goals'])
        if added:
            result['goals'].append({**goal, 'source': source})
        score = lambda e: e['error_mm']/config.tolerance_mm + e['orientation_error_deg']/config.orientation_tolerance_deg
        if result['best_error'] is None or score(error) < score(result['best_error']):
            result['best_error'] = error
        best_valid = result['diagnostic']['best_valid_error']
        if valid and (best_valid is None or score(error) < score(best_valid)):
            result['diagnostic']['best_valid_error'] = error
        rejection = ('invalid_configuration' if not valid else
                     'position_tolerance' if error['error_mm'] > config.tolerance_mm else
                     'direction_tolerance' if error['orientation_error_deg'] > config.orientation_tolerance_deg else
                     'goal_constraint' if goal is None else None)
        attempt = {'source': source, 'valid_configuration': valid, 'accepted': goal is not None,
                   'added_unique_goal': added, 'rejection_reason': rejection, **error}
        if not valid:
            attempt['configuration_diagnostic'] = roadmap.space.diagnostic(q)
        if optimized is not None:
            attempt.update(refinement_method='trust_region', refinement_ms=optimized['elapsed_ms'],
                           iterations=optimized['iterations'], evaluations=optimized['evaluations'],
                           stop_reason=optimized['stop_reason'], optimizer_status=optimized['optimizer_status'],
                           antipodal_escape=optimized['antipodal_escape'], escape_probes=optimized.get('escape_probes', 0))
        result['attempts'].append(attempt)

    for seed in queue:
        if len(result['goals']) >= budget:
            break
        used += 1
        q = np.asarray(seed['q_rad'])
        record(q, seed['source']+'_raw')
        if len(result['goals']) >= budget:
            break
        solved = ik_recovery.optimize(q, target, target_axis, lambda x: candidate(x) is not None,
                                      config.recovery_evaluations)
        evaluations += solved['evaluations']
        record(solved['q_rad'], seed['source'], solved)
    elapsed = (perf_counter()-began)*1000
    phase = {'trigger': trigger, 'goal_budget': budget, 'seed_budget': config.recovery_seeds,
             'evaluations_per_seed': config.recovery_evaluations, 'seed_queue': queue,
             'used_seeds': used, 'duplicate_seeds_skipped': duplicates,
             'previous_seeds_skipped': len(previous), 'evaluations': evaluations,
             'goals_before': before, 'added_goals': len(result['goals'])-before, 'elapsed_ms': elapsed}
    result['recovery'] = {'phases': [*recovery['phases'], phase],
                          'elapsed_ms': recovery['elapsed_ms']+elapsed,
                          'added_goals': recovery['added_goals']+phase['added_goals']}
    attempts = result['attempts']
    result['diagnostic'].update(refinements=sum('refinement_method' in a for a in attempts),
        pose_accurate_invalid_candidates=sum(not a.get('valid_configuration', True)
            and a.get('error_mm', np.inf) <= config.tolerance_mm
            and a.get('orientation_error_deg', np.inf) <= config.orientation_tolerance_deg for a in attempts),
        goal_constraint_rejections=sum(a.get('rejection_reason') == 'goal_constraint' for a in attempts))
    result['ik_ms'] += elapsed
    return result


def trajectory(path, target, space, max_step_rad=.08, speed_rad_s=1.2, dt=.05, target_axis=None):
    """Existing simulation frame contract: degrees/mm, 0.05 s speed cap.

    Known joint waypoints are interpolated directly; no repeated waypoint IK.
    Every new segment is validated at the roadmap resolution. These frames can
    be inspected by the existing viewer/controller interfaces, but this module
    performs no execution and does not create an executable hardware plan.
    """
    if not np.isfinite(speed_rad_s) or speed_rad_s <= 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError('Trajectory speed and dt must be positive')
    path = np.asarray(path, float).reshape(-1, 6)
    if not len(path) or not all(space.valid(q) for q in path): raise ValueError('Invalid trajectory waypoints')
    points = [path[0]]
    for a, b in zip(path, path[1:]):
        if not space.edge_valid(a, b, max_step_rad): raise ValueError('Invalid trajectory segment')
        delta = space.delta(a, b)
        if np.array_equal(a, b): continue
        pieces = max(1, int(np.ceil(max(np.linalg.norm(delta)/(speed_rad_s*dt),
                                       np.max(np.abs(delta))/max_step_rad))))
        segment = a+np.arange(1, pieces+1)[:, None]/pieces*delta
        segment[-1] = b  # Retain exact endpoints at operating joint bounds.
        points.extend(segment)
    # Checking the actual frame intervals also covers the denser speed-derived
    # interpolation; interpolation is a discretized validity test, not CCD.
    if not all(space.edge_valid(a, b, max_step_rad) for a, b in zip(points, points[1:])):
        raise ValueError('Invalid interpolated trajectory')
    frames = []
    for i, q in enumerate(points):
        t = ARM.transform(q); error = goal_error(q, target, target_axis)
        frames.append({'q': np.rad2deg(q).tolist(), 'position': (t[:3, 3]*1000).tolist(),
                       'tool_axis': t[:3, 2].tolist(), 'time': i*dt, **error})
    return frames


def query(q_current, target, roadmap, config=None, orientation_xy_deg=None, shortcut=False, use_roadmap=True,target_axis=None,goal_constraint=None):
    config = config or IKConfig(); space = roadmap.space
    result = {'simulation_only': True, 'method': 'nn_prm', 'task': config.task,
              'converged': False, 'validation': space.validation, 'frames': [], 'ik_ms': 0., 'planning_ms': 0.,
              'waypoints_rad': [], 'waypoints_deg': [], 'fk_transforms_m': [],
              'ik_outcome':'not_started','planning_outcome':'not_started',
              'guided_repair': {'enabled': config.guided_route_repair, 'triggered': False, 'recovered': False},
              'limitations': 'Finite IK/search budget and discretized motion validation; failure does not prove physical unreachability. '
                             + ('Conservative convex URDF self-collision checking; no environment, tool or cable collision checking.'
                                if space.validation == 'joint_limits_and_self_collision' else
                                ('No collision checking.' if space.validity is None else 'Validity is supplied by the configured callback.'))}
    if space.validation == 'joint_limits_and_self_collision':
        result['self_collision_assumptions'] = space.metadata()['assumptions']
    try:
        q_current = np.asarray(q_current, float)
        if not space.valid(q_current): return {**result, 'status': 'invalid_start', **space.diagnostic(q_current)}
    except (ValueError, TypeError):
        return {**result, 'status': 'invalid_start', 'reason': 'Expected six finite joint angles'}
    try:
        target = np.asarray(target, float)
        solvers.validate(q_current, target, 'dls', config.steps, config.speed_rad_s, config.tolerance_mm, config.damping)
        target_axis = pose5.resolve_axis(orientation_xy_deg,target_axis) if config.task == 'pose5' else None
    except (ValueError, TypeError) as exc:
        return {**result, 'status': 'invalid_target', 'reason': str(exc)}
    if target_axis is not None:orientation_xy_deg=pose5.xy_from_axis(target_axis)
    generated = generate_goals(q_current, target, roadmap, config=config, orientation_xy_deg=orientation_xy_deg,target_axis=target_axis,goal_constraint=goal_constraint)
    recovery_enabled = config.task == 'pose5' and config.pose5_global_recovery and config.numerical_fallback
    if recovery_enabled and not generated['goals']:
        generated = recover_goals(generated, q_current, target, roadmap, config, target_axis, goal_constraint)
    result.update(ik_ms=generated['ik_ms'], ik=generated, target_mm=(target*1000).tolist(),
                  start_deg=np.rad2deg(q_current).tolist(), target_axis=target_axis.tolist() if target_axis is not None else None,
                  orientation_xy_deg=orientation_xy_deg)
    if not generated['goals']:
        local = [a['rollout_status'] for a in generated['attempts'] if 'rollout_status' in a]
        status = local[-1] if local and not config.refine and not config.numerical_fallback else 'ik_solution_not_found'
        diagnostic=generated['diagnostic'];reason='No FK-accurate valid goal configuration found within the search budget.'
        if diagnostic['pose_accurate_invalid_candidates']:
            reason+=f" {diagnostic['pose_accurate_invalid_candidates']} pose-accurate candidates failed collision/joint-limit checks."
        if diagnostic['goal_constraint_rejections']:
            reason+=f" {diagnostic['goal_constraint_rejections']} pose candidates missed the cursor gap/angle tolerances."
        if target_axis is not None and diagnostic['best_valid_error']:
            best=diagnostic['best_valid_error']
            reason+=f" Best collision-valid candidate: {best['error_mm']:.2f} mm / {best['orientation_error_deg']:.2f}°; requested {config.tolerance_mm:g} mm / {config.orientation_tolerance_deg:g}°."
        failure_error=diagnostic['best_valid_error'] or generated['best_error'] or {}
        return {**result, 'status': status, 'reason': reason,
                'ik_outcome':'ik_solution_not_found','planning_outcome':'not_started',
                'final_error_mm': failure_error.get('error_mm'),
                'final_orientation_error_deg':failure_error.get('orientation_error_deg')}
    plan_began = perf_counter()
    search_roadmap = roadmap if use_roadmap else Roadmap(space, roadmap.config)
    planned = search_roadmap.plan(q_current, [g['q_rad'] for g in generated['goals']], shortcut=shortcut)
    recovery_ms = 0.; search_ms = planned['planning_ms']
    if (recovery_enabled and planned['reason'] in ('start_not_connected', 'goals_not_connected', 'disconnected_components')
            and len(generated['goals']) < config.recovery_max_candidates):
        old_count = len(generated['goals']); before_recovery = perf_counter()
        generated = recover_goals(generated, q_current, target, roadmap, config, target_axis,
                                  goal_constraint, trigger='route_failure', goal_budget=config.recovery_max_candidates)
        recovery_ms = (perf_counter()-before_recovery)*1000
        result.update(ik=generated, ik_ms=generated['ik_ms'])
        if len(generated['goals']) > old_count:
            planned = search_roadmap.plan(q_current, [g['q_rad'] for g in generated['goals']], shortcut=shortcut)
            search_ms += planned['planning_ms']
    if config.guided_route_repair:
        baseline_ms = planned['planning_ms']
        planned = guided_repair.repair_failed_route(
            search_roadmap, q_current, [g['q_rad'] for g in generated['goals']], planned,
            guided_repair.RepairConfig(samples=config.route_repair_samples, seed=config.seed), shortcut=shortcut)
        search_ms += planned['planning_ms']-baseline_ms
    result.update(planned)
    result.update(ik_outcome='accepted_goals',planning_outcome=planned['reason'])
    result.update(final_error_mm=generated['goals'][0].get('error_mm'),
                  final_orientation_error_deg=generated['goals'][0].get('orientation_error_deg'))
    if planned['status'] != 'success':
        result.update(reason=planned['reason'], planning_ms=(perf_counter()-plan_began)*1000-recovery_ms,
                      search_ms=search_ms)
        return result
    try:
        frames = trajectory(planned['waypoints_rad'], target, space, roadmap.config.max_step_rad,
                            config.speed_rad_s, target_axis=target_axis)
    except ValueError as exc:
        return {**result, 'status': 'path_not_found', 'planning_outcome':'trajectory_validation_failed',
                'planning_ms': (perf_counter()-plan_began)*1000-recovery_ms, 'search_ms':search_ms,
                'reason': 'trajectory_validation_failed: '+str(exc)}
    commands = joint_waypoints(np.rad2deg(q_current), [f['q'] for f in frames[1:]], compress=False)
    result.update(frames=frames, converged=True, final_error_mm=frames[-1]['error_mm'],
                  final_orientation_error_deg=frames[-1]['orientation_error_deg'],
                  joint_travel_deg=float(np.rad2deg(planned['path_length_rad'])),
                  tolerance_mm=config.tolerance_mm, orientation_tolerance_deg=config.orientation_tolerance_deg,
                  achieved_axis=frames[-1]['tool_axis'],stop_reason='target_reached', dt=.05)
    result['controller_waypoints_deg'] = commands
    result['search_ms'] = search_ms
    result['planning_ms'] = (perf_counter()-plan_began)*1000-recovery_ms
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roadmap', type=Path, default=Path('simulation/artifacts/roadmap.json'))
    parser.add_argument('--collision', choices=['self', 'limits'], default='self')
    parser.add_argument('--self-clearance-mm', type=float, default=1.)
    parser.add_argument('--start-deg', type=float, nargs=6, required=True)
    parser.add_argument('--target-mm', type=float, nargs=3, required=True)
    parser.add_argument('--task', choices=['position', 'pose5'], default='position')
    parser.add_argument('--pose5-method', choices=[method[0] for method in pose5.METHODS], default='bounded_dls')
    parser.add_argument('--pose5-global-recovery', action='store_true', help='Bounded extra seed/solver search after 5D IK or routing failure')
    parser.add_argument('--recovery-seeds', type=int, default=32)
    parser.add_argument('--recovery-evaluations', type=int, default=96)
    parser.add_argument('--recovery-max-candidates', type=int, default=8)
    parser.add_argument('--orientation', type=float, nargs=2)
    parser.add_argument('--nn', choices=['none', 'seed', 'seed5', 'direct', 'conditional', 'differential'])
    parser.add_argument('--target-axis',type=float,nargs=3,help='Directed tool-Z vector in g_base; normalized, local-Z spin free')
    parser.add_argument('--no-refine', action='store_true')
    parser.add_argument('--no-fallback', action='store_true')
    parser.add_argument('--steps', type=int, default=240)
    parser.add_argument('--speed-rad-s', type=float, default=1.2)
    parser.add_argument('--damping', type=float, default=.04)
    parser.add_argument('--max-candidates', type=int, default=4)
    parser.add_argument('--random-seeds', type=int, default=4)
    parser.add_argument('--roadmap-seeds', type=int, default=4)
    parser.add_argument('--seed', type=int, default=9001)
    parser.add_argument('--tolerance-mm', type=float, default=1.)
    parser.add_argument('--direction-tolerance-deg', type=float, default=1.)
    parser.add_argument('--shortcut', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('simulation/artifacts/prm_query.json'))
    args = parser.parse_args()
    config = IKConfig(task=args.task, nn=args.nn or ('seed5' if args.task=='pose5' else 'seed'), pose5_method=args.pose5_method,
                      refine=not args.no_refine, numerical_fallback=not args.no_fallback,
                      pose5_global_recovery=args.pose5_global_recovery, recovery_seeds=args.recovery_seeds,
                      recovery_evaluations=args.recovery_evaluations, recovery_max_candidates=args.recovery_max_candidates,
                      steps=args.steps, random_seeds=args.random_seeds, roadmap_seeds=args.roadmap_seeds,
                      seed=args.seed, speed_rad_s=args.speed_rad_s, damping=args.damping, max_candidates=args.max_candidates,
                      tolerance_mm=args.tolerance_mm, orientation_tolerance_deg=args.direction_tolerance_deg)
    result = query(np.deg2rad(args.start_deg), np.array(args.target_mm)/1000,
                   Roadmap.load(args.roadmap, planning_space(args.collision, args.self_clearance_mm)),
                   config, args.orientation, args.shortcut,target_axis=args.target_axis)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: result.get(k) for k in ('status', 'reason', 'validation', 'final_error_mm', 'ik_ms', 'planning_ms')}))


if __name__ == '__main__': main()
