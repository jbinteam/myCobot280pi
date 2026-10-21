"""Opt-in bounded joint-space roadmap repair; no IK or hardware operations.

Adaptation of MS-Bi-PRM's guided/uniform sampling mixture. The reusable graph
is never modified. Existing successes return their original path. New nodes
and edges use the existing validity profile and interpolation resolution.
"""
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from .prm import Roadmap
from .workspace import transforms


@dataclass(frozen=True)
class RepairConfig:
    samples: int = 32
    guided_fraction: float = .75
    noise_fraction: float = .08
    sampling_attempt_factor: int = 8
    seed: int = 9001

    def __post_init__(self):
        for name, low, high in (('samples', 1, 128), ('sampling_attempt_factor', 1, 20),
                                ('seed', 0, 2**32-1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f'{name} must be an integer in [{low}, {high}]')
        for name, low, high in (('guided_fraction', 0., 1.), ('noise_fraction', .001, .5)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f'{name} must be finite and in [{low}, {high}]')


def sample_nodes(roadmap, start, goals, config):
    """Sample all six bounded joints; Gaussian noise uses operating ranges."""
    space = roadmap.space
    start = np.asarray(start, float)
    goals = np.asarray(goals, float).reshape(-1, 6)
    rng = np.random.default_rng(config.seed)
    seen = {tuple(q) for q in np.vstack([roadmap.q, start, goals])}
    nodes = []
    stats = {'attempts': 0, 'guided_attempts': 0, 'uniform_attempts': 0,
             'accepted_guided': 0, 'accepted_uniform': 0, 'invalid_samples': 0, 'duplicate_samples': 0}
    for _ in range(config.samples*config.sampling_attempt_factor):
        if len(nodes) >= config.samples:
            break
        guided = bool(rng.random() < config.guided_fraction)
        if guided:
            goal = goals[int(rng.integers(len(goals)))]
            candidate = start+rng.random()*(goal-start)+rng.normal(size=6)*config.noise_fraction*space.scale
            candidate = np.clip(candidate, space.lower, space.upper)
        else:
            candidate = rng.uniform(space.lower, space.upper)
        stats['attempts'] += 1
        stats['guided_attempts' if guided else 'uniform_attempts'] += 1
        key = tuple(candidate)
        if key in seen:
            stats['duplicate_samples'] += 1
            continue
        if not space.valid(candidate):
            stats['invalid_samples'] += 1
            continue
        seen.add(key)
        nodes.append(candidate)
        stats['accepted_guided' if guided else 'accepted_uniform'] += 1
    return np.asarray(nodes).reshape(-1, 6), stats


def temporary_roadmap(roadmap, nodes):
    """Preserve original edges and validate at most k connections per new node."""
    temporary = Roadmap(roadmap.space, roadmap.config)
    temporary.q = np.vstack([roadmap.q, nodes])
    temporary.poses = np.concatenate([roadmap.poses, transforms(nodes)])
    temporary.edges = {i: dict(neighbors) for i, neighbors in roadmap.edges.items()}
    checks = 0
    added_edges = 0
    for index in range(len(roadmap.q), len(temporary.q)):
        temporary.edges.setdefault(index, {})
        order = np.argsort(roadmap.space.distance(temporary.q[index], temporary.q), kind='stable')
        for other in order[order != index][:roadmap.config.neighbors]:
            other = int(other)
            if other in temporary.edges[index]:
                continue
            checks += 1
            if roadmap.space.edge_valid(temporary.q[index], temporary.q[other], roadmap.config.max_step_rad):
                cost = float(roadmap.space.distance(temporary.q[index], temporary.q[other]))
                temporary.edges[index][other] = cost
                temporary.edges.setdefault(other, {})[index] = cost
                added_edges += 1
    return temporary, {'connection_checks': checks, 'added_edges': added_edges}


def plan(roadmap, start, goals, config=None, shortcut=True):
    """Run the normal planner first; permit one repair only on routing failure."""
    began = perf_counter()
    baseline = roadmap.plan(start, goals, shortcut=shortcut)
    result = repair_failed_route(roadmap, start, goals, baseline, config, shortcut)
    return {**result, 'planning_ms': (perf_counter()-began)*1000}


def repair_failed_route(roadmap, start, goals, baseline, config=None, shortcut=True):
    """Repair an existing route result without repeating the baseline search.

    Callers pass the normal planner's result for the same graph/start/goals.
    Successful routes and validity failures never trigger sampling.
    """
    config = config or RepairConfig()
    began = perf_counter()
    diagnostic = {'enabled': True, 'config': asdict(config), 'triggered': False, 'recovered': False,
                  'baseline_status': baseline['status'], 'baseline_reason': baseline['reason'],
                  'baseline_ms': baseline['planning_ms'], 'repair_ms': 0.,
                  'sampling_ms': 0., 'connections_ms': 0., 'retry_ms': 0.,
                  'new_nodes': 0, 'connection_checks': 0, 'added_edges': 0}
    if baseline['reason'] not in ('start_not_connected', 'goals_not_connected', 'disconnected_components'):
        return {**baseline, 'guided_repair': diagnostic}
    diagnostic['triggered'] = True
    sample_began = perf_counter()
    nodes, sampling = sample_nodes(roadmap, start, goals, config)
    diagnostic.update(sampling, new_nodes=len(nodes), sampling_ms=(perf_counter()-sample_began)*1000)
    if not len(nodes):
        diagnostic['repair_ms'] = (perf_counter()-began)*1000
        return {**baseline, 'planning_ms': baseline['planning_ms']+diagnostic['repair_ms'], 'guided_repair': diagnostic}
    connections_began = perf_counter()
    temporary, connections = temporary_roadmap(roadmap, nodes)
    diagnostic.update(connections, connections_ms=(perf_counter()-connections_began)*1000)
    retry_began = perf_counter()
    repaired = temporary.plan(start, goals, shortcut=shortcut)
    diagnostic.update(retry_ms=(perf_counter()-retry_began)*1000, recovered=repaired['status'] == 'success',
                      retry_status=repaired['status'], retry_reason=repaired['reason'])
    diagnostic['repair_ms'] = (perf_counter()-began)*1000
    return {**repaired, 'planning_ms': baseline['planning_ms']+diagnostic['repair_ms'], 'guided_repair': diagnostic}
