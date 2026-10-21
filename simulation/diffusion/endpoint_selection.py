"""Offline anytime endpoint selection with a fully validated incumbent.

The budget begins after the first visible solution. Deadlines are cooperative
at collision/visibility samples, not a real-time scheduling guarantee. Only a
completed evaluator result can replace the incumbent.
"""
from dataclasses import dataclass, asdict
from time import perf_counter
import numpy as np


class SearchBudgetExpired(RuntimeError):
    pass


class DeadlineEnv:
    """Apply a cooperative deadline to the existing environment predicates."""
    def __init__(self, env, deadline, clock=perf_counter):
        self.env = env; self.space = env.space
        self.deadline = deadline; self.clock = clock

    def check(self):
        if self.clock() >= self.deadline:
            raise SearchBudgetExpired('Endpoint improvement time budget exhausted')

    def valid(self, q):
        self.check()
        return self.env.valid(q)

    def edge_valid(self, a, b, max_step_rad=.04):
        self.check()
        return all(self.valid(q) for q in self.space.interpolate(a, b, max_step_rad))

    def path_valid(self, states, max_step_rad=.04):
        self.check()
        states = np.asarray(states, float)
        if states.ndim != 2 or states.shape[1] != 6 or not len(states) or not np.isfinite(states).all():
            return False
        return (all(self.valid(q) for q in states)
                and all(self.edge_valid(a, b, max_step_rad) for a,b in zip(states[:-1], states[1:])))


@dataclass(frozen=True)
class SelectionBudget:
    improvement_seconds: float = .35
    improvement_attempts: int = 3
    rrt_iterations: int = 800
    first_goal_iterations: int = 200
    improvement_goal_iterations: int = 80

    def __post_init__(self):
        if (isinstance(self.improvement_seconds, bool) or not np.isfinite(self.improvement_seconds)
                or self.improvement_seconds < 0):
            raise ValueError('Nonnegative improvement time required')
        for key in ('improvement_attempts', 'rrt_iterations', 'first_goal_iterations', 'improvement_goal_iterations'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == 'improvement_attempts' else 1):
                raise ValueError(f'Invalid {key}')


def choose_visible_endpoint(start, goals, evaluate, budget=SelectionBudget(), clock=perf_counter):
    """Evaluate original ordering until success, then lower-bound ordered goals.

    evaluate(goal, iteration_cap, deadline) returns {'route': checked row,
    'trajectory': ...}. A row is accepted only after all endpoint, smoothing,
    collision, and visibility gates. A deadline exception discards that entire
    attempt. The callback's completed iteration count must not exceed its cap.
    """
    start = np.asarray(start, float)
    if start.shape != (6,) or not np.isfinite(start).all():
        raise ValueError('Expected finite six-joint start')
    pending = list(goals)
    for goal in pending:
        q = np.asarray(goal['q_rad'], float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError('Expected finite endpoint joints')
    began = clock(); remaining = budget.rrt_iterations
    best = None; first = None; deadline = None; first_seconds = None
    attempts = []; pruned = []; extra = 0; stop = 'candidate_pool_exhausted'
    while pending:
        if best is not None:
            # Euclidean joint distance bounds every polyline's total joint travel.
            pending.sort(key=lambda g: np.linalg.norm(np.asarray(g['q_rad'])-start))
            while pending and np.linalg.norm(np.asarray(pending[0]['q_rad'])-start) >= best['route']['joint_travel_rad']-1e-9:
                pruned.extend(g['index'] for g in pending); pending.clear()
            if not pending:
                stop = 'remaining_endpoint_lower_bounds'; break
            if extra >= budget.improvement_attempts:
                stop = 'improvement_attempt_limit'; break
            if clock() >= deadline:
                stop = 'improvement_time_limit'; break
        goal = pending.pop(0)
        improving = best is not None
        cap = min(remaining, budget.improvement_goal_iterations if improving else budget.first_goal_iterations)
        if improving:
            extra += 1
        tick = clock()
        try:
            result = evaluate(goal, cap, deadline)
        except SearchBudgetExpired:
            # Exact iteration progress is unavailable on interruption. Charging
            # the entire cap conservatively preserves the shared global budget.
            remaining -= cap
            attempts.append({'goal_index': goal['index'], 'phase': 'improve',
                             'timed_out': True, 'iterations_charged': cap,
                             'seconds': clock()-tick})
            stop = 'improvement_time_limit'; break
        row = result['route']
        used = row['planner']['iterations']
        if not 0 <= used <= cap:
            raise ValueError('Evaluator exceeded the RRT iteration cap')
        remaining -= used
        accepted = bool(row['accepted'])
        if accepted:
            cost = row['joint_travel_rad']
            if not np.isfinite(cost) or cost < 0 or not row['visibility']['passed']:
                raise ValueError('Accepted route must have finite travel and verified visibility')
            if best is None or cost < best['route']['joint_travel_rad']-1e-9:
                best = result
            if first is None:
                first = result; first_seconds = clock()-began
                deadline = clock()+budget.improvement_seconds
        attempts.append({'goal_index': goal['index'], 'phase': 'improve' if improving else 'find_first',
                         'accepted': accepted, 'timed_out': False, 'iterations_charged': used,
                         'seconds': clock()-tick,
                         'joint_travel_rad': row.get('joint_travel_rad')})
    elapsed = clock()-began
    if first is None:
        stop = 'no_visible_route_within_search'
    return best, {'configuration': asdict(budget), 'stop_reason': stop, 'seconds': elapsed,
        'first_solution_seconds': first_seconds, 'improvement_seconds': 0. if first is None else elapsed-first_seconds,
        'deadline_overrun_seconds': 0. if deadline is None else max(0., clock()-deadline),
        'first_goal_index': None if first is None else first['route']['goal_index'],
        'first_joint_travel_rad': None if first is None else first['route']['joint_travel_rad'],
        'selected_goal_index': None if best is None else best['route']['goal_index'],
        'improvement_attempts': extra, 'iterations_charged': budget.rrt_iterations-remaining,
        'lower_bound_pruned_indices': pruned, 'untried_indices': [g['index'] for g in pending],
        'attempts': attempts}
