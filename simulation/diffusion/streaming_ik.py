"""Experimental on-demand IK and anytime visible-route selection; offline only."""
from dataclasses import asdict, dataclass
from time import perf_counter
import numpy as np
from ..kinematics import ARM
from ..physical_reach import ENVELOPE, target_precheck
from ..tracking_geometry import candidate_poses, region_metrics
from .endpoint_ik import IKSearchExhausted
from .endpoint_selection import SelectionBudget, SearchBudgetExpired


@dataclass(frozen=True)
class CandidateBudget:
    """Cooperative per-job limits after the first verified route exists.

    An interrupted solve is discarded, not resumed. The first-solution search
    keeps its original numerical effort limits so difficult branches remain
    available before an incumbent exists.
    """
    ik_seconds: float = .025
    route_seconds: float = .200
    lookahead_seconds: float = .140

    def __post_init__(self):
        for key in ('ik_seconds', 'route_seconds'):
            value = getattr(self, key)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f'Positive finite {key} required')
        if (isinstance(self.lookahead_seconds, bool) or not np.isfinite(self.lookahead_seconds)
                or self.lookahead_seconds < 0):
            raise ValueError('Nonnegative finite lookahead_seconds required')


class EndpointSource:
    """Produce the same ordered candidates as the eager experimental IK bank.

    Preparing jobs performs no IK. Each next_endpoint call solves at most one
    point/seed job, and returns None for a failed/duplicate/filtered candidate.
    """
    def __init__(self, env, start, region, seed, count=8, clock=perf_counter, *,
                 poses=None, position_tolerance_mm=1., orientation_tolerance_deg=1., refine=None,
                 solver=None, acceptance=None, cancel_check=None):
        self.env = env; self.start = np.asarray(start, float); self.region = region; self.clock = clock
        self.position_tolerance_mm = position_tolerance_mm
        self.orientation_tolerance_deg = orientation_tolerance_deg
        self.refine = refine
        self.solver = env.inverse_kinematics if solver is None else solver
        self.acceptance = acceptance
        self.cancel_check = cancel_check
        self.precheck = target_precheck(np.array(region['center_mm'])/1000, region, position_tolerance_mm)
        self.filtered = 0
        self.jobs = []; self.position = 0; self.goals = []; self.attempts = []; self.ik_seconds = 0.
        if not self.precheck['possible']:
            return
        rng = np.random.default_rng(seed+10000)
        poses = list(candidate_poses(region, self.start) if poses is None else poses)[:count]
        usable = [(i, pose) for i, pose in enumerate(poses)
                  if ENVELOPE.point(pose[0]*1000, min(1., position_tolerance_mm))['possible']]
        self.filtered = len(poses)-len(usable)
        for branch in range(2):
            for ordinal, (index, (target, axis)) in enumerate(usable):
                current = self.start if branch == 0 else rng.uniform(ARM.lower, ARM.upper)
                self.jobs.append({'point_index': index, 'branch_pass': branch,
                    'target': target, 'axis': axis, 'current': current,
                    'settings': {'attempts': 4, 'robust': branch == 0 and ordinal == 0,
                                 'learned': branch == 0, 'seed': seed+100*branch+index}})

    @property
    def has_more(self):
        return self.position < len(self.jobs)

    def next_endpoint(self, deadline):
        def check():
            if self.cancel_check is not None:
                self.cancel_check()
            if deadline is not None and self.clock() >= deadline:
                raise SearchBudgetExpired('IK improvement deadline exhausted')
        check()
        job = self.jobs[self.position]; self.position += 1
        record = {k: job[k] for k in ('point_index', 'branch_pass', 'settings')}
        record['phase'] = 'find_first' if deadline is None else 'improve'
        began = self.clock()
        try:
            q = self.solver(job['target'], job['axis'], current=job['current'],
                **job['settings'], **({} if deadline is None and self.cancel_check is None else {'cancel_check': check}))
            check()
            if self.refine is not None:
                q = self.refine(q, job, check)
                check()
                if q is None:
                    record['status'] = 'refinement_rejected'
                    return None
            accepted = (self.acceptance(q) if self.acceptance is not None else
                region_metrics(q,self.region,self.position_tolerance_mm,self.orientation_tolerance_deg)['accepted'])
            duplicate = any(np.linalg.norm(q-np.array(g['q_rad'])) < .05 for g in self.goals)
            record['status'] = 'accepted' if accepted and not duplicate else ('duplicate' if duplicate else 'region_rejected')
            if accepted and not duplicate:
                assert self.env.valid(q)
                check()
                goal = {'index': len(self.goals), 'q_rad': q.tolist(),
                        'point_index': job['point_index'], 'branch_pass': job['branch_pass']}
                self.goals.append(goal)
                return goal
        except IKSearchExhausted:
            record['status'] = 'search_exhausted'
        except SearchBudgetExpired:
            record['status'] = 'timed_out'
            raise
        finally:
            record['seconds'] = self.clock()-began
            record['diagnostics'] = dict(self.env.ik_diagnostics)
            self.attempts.append(record); self.ik_seconds += record['seconds']
        return None

    def report(self):
        return {'jobs_available': len(self.jobs), 'jobs_started': self.position,
                'points_filtered': self.filtered,
                'jobs_unstarted': len(self.jobs)-self.position, 'ik_seconds': self.ik_seconds,
                'goals': self.goals, 'attempts': self.attempts, 'precheck': self.precheck,
                'legacy_attempts_used': sum(a['diagnostics']['legacy_attempts'] for a in self.attempts),
                'fallback_attempts_used': sum(a['diagnostics']['fallback_attempts'] for a in self.attempts)}


def choose_streaming_route(start, source, evaluate, budget=SelectionBudget(), clock=perf_counter,
                           candidate_budget=None):
    """Generate a candidate, check its route, and repeat within bounded effort.

    The first solution starts one deadline shared by IK and path improvement.
    Unlike a fully generated bank, unseen IK endpoints cannot be pruned using
    joint-distance bounds. Pruning one known endpoint never stops production.
    With candidate_budget, a local timeout skips only that job; expiry of the
    overall improvement deadline still ends the search. No partial result may
    replace the incumbent. None preserves the original experimental policy.
    """
    start = np.asarray(start, float)
    if start.shape != (6,) or not np.isfinite(start).all():
        raise ValueError('Expected finite six-joint start')
    began = clock(); remaining = budget.rrt_iterations
    best = None; first = None; first_seconds = None; deadline = None
    extra = 0; attempts = []; pruned = []; ik_timeouts = []; pending = []
    stop = 'endpoint_jobs_exhausted'
    while source.has_more or pending:
        if best is not None:
            if extra >= budget.improvement_attempts:
                stop = 'improvement_route_attempt_limit'; break
            if clock() >= deadline:
                stop = 'improvement_time_limit'; break
        if best is not None and candidate_budget is not None and candidate_budget.lookahead_seconds:
            # Collect a bounded group before spending route checks on it. This
            # lets later, shorter endpoints compete with earlier long routes.
            # Only known joint endpoints are ordered/pruned; no claim is made
            # about endpoints whose IK has not completed.
            if not pending:
                collect_until = min(deadline, clock()+candidate_budget.lookahead_seconds)
                while source.has_more and clock() < collect_until:
                    ik_deadline = min(collect_until, clock()+candidate_budget.ik_seconds)
                    try:
                        goal = source.next_endpoint(ik_deadline)
                    except SearchBudgetExpired:
                        ik_timeouts.append({'scope': 'candidate' if clock() < deadline else 'overall'})
                        continue
                    if goal is not None:
                        pending.append(goal)
                pending.sort(key=lambda g: np.linalg.norm(np.asarray(g['q_rad'])-start))
            if clock() >= deadline:
                stop = 'improvement_time_limit_in_ik'; break
            if not pending:
                continue
            goal = pending.pop(0)
        else:
            ik_deadline = (min(deadline, clock()+candidate_budget.ik_seconds)
                           if best is not None and candidate_budget is not None else deadline)
            try:
                goal = source.next_endpoint(ik_deadline)
            except SearchBudgetExpired:
                local = candidate_budget is not None and deadline is not None and clock() < deadline
                ik_timeouts.append({'scope': 'candidate' if local else 'overall'})
                if local:
                    continue
                stop = 'improvement_time_limit_in_ik'; break
        if goal is None:
            continue
        if best is not None and np.linalg.norm(np.array(goal['q_rad'])-start) >= best['route']['joint_travel_rad']-1e-9:
            pruned.append(goal['index'])
            continue
        improving = best is not None
        cap = min(remaining, budget.improvement_goal_iterations if improving else budget.first_goal_iterations)
        if improving:
            extra += 1
        tick = clock()
        route_deadline = (min(deadline, tick+candidate_budget.route_seconds)
                          if improving and candidate_budget is not None else deadline)
        try:
            result = evaluate(goal, cap, route_deadline)
            # Final native work/audits may return just after a checkpoint.
            # Discard late results as well as explicitly interrupted results.
            if route_deadline is not None and clock() >= route_deadline:
                raise SearchBudgetExpired('Completed route exceeded its deadline')
        except SearchBudgetExpired:
            remaining -= cap
            local = candidate_budget is not None and deadline is not None and clock() < deadline
            attempts.append({'goal_index': goal['index'], 'timed_out': True, 'iterations_charged': cap,
                             'seconds': clock()-tick, 'phase': 'improve',
                             'timeout_scope': 'candidate' if local else 'overall'})
            if local:
                continue
            stop = 'improvement_time_limit_in_route'; break
        row = result['route']; used = row['planner']['iterations']
        if not 0 <= used <= cap:
            raise ValueError('Evaluator exceeded the RRT cap')
        remaining -= used
        if row['accepted']:
            cost = row['joint_travel_rad']
            if not np.isfinite(cost) or cost < 0 or not row['visibility']['passed']:
                raise ValueError('Only finite, checked visible routes may be accepted')
            if best is None or cost < best['route']['joint_travel_rad']-1e-9:
                best = result
            if first is None:
                first = result; first_seconds = clock()-began
                deadline = clock()+budget.improvement_seconds
        attempts.append({'goal_index': goal['index'], 'timed_out': False, 'accepted': bool(row['accepted']),
                         'iterations_charged': used, 'seconds': clock()-tick,
                         'phase': 'improve' if improving else 'find_first'})
    elapsed = clock()-began
    if first is None:
        stop = 'no_visible_route_within_search'
    return best, {'configuration': asdict(budget), 'seconds': elapsed,
        'candidate_configuration': None if candidate_budget is None else asdict(candidate_budget),
        'ik_timeouts': ik_timeouts,
        'untried_generated_endpoint_indices': [g['index'] for g in pending],
        'first_solution_seconds': first_seconds,
        'first_goal_index': None if first is None else first['route']['goal_index'],
        'first_joint_travel_rad': None if first is None else first['route']['joint_travel_rad'],
        'first_trajectory': None if first is None else first['trajectory'],
        'improvement_seconds': 0. if first is None else elapsed-first_seconds,
        'deadline_overrun_seconds': 0. if deadline is None else max(0., clock()-deadline),
        'improvement_route_attempts': extra, 'iterations_charged': budget.rrt_iterations-remaining,
        'stop_reason': stop, 'pruned_known_endpoint_indices': pruned, 'route_attempts': attempts,
        'uncomputed_endpoints_certified': False}
