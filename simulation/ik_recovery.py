"""Bounded, deterministic second-pass 5D IK search; optimizer states are not motion.

The residual has six entries but rank five: XYZ plus a directed unit tool-Z
vector. Its orientation cost is weight**2 * (1 - dot(actual, target)), with
no rotation-matrix, Euler yaw or local-spin residual.
"""
from time import perf_counter
import numpy as np
from .kinematics import ARM
from . import pose5, seed5


def residual_jacobian(q, target, axis, orientation_weight=pose5.ORIENTATION_WEIGHT):
    t, j = ARM.spatial_jacobian(q)
    residual = np.r_[(t[:3, 3]-target)/ARM.scale,
                     orientation_weight*(t[:3, 2]-axis)]
    jacobian = np.vstack([j[:3]/ARM.scale,
                         orientation_weight*np.cross(j[3:].T, t[:3, 2]).T])
    return t, residual, jacobian


def optimize(q, target, axis, accept, evaluations=96):
    """Trust-region reflective least squares, with analytic FK derivatives.

    Acceptance is supplied by planning and includes actual goal tolerances,
    finite/limit/collision checks and, for mouse aim, cursor gap/angle checks.
    SciPy's optimizer convergence flag alone never establishes IK success.
    """
    from scipy.optimize import least_squares
    began = perf_counter(); cache = {}; iterations = 0

    def data(x):
        if 'q' not in cache or not np.array_equal(x, cache['q']):
            t, r, j = residual_jacobian(x, target, axis)
            cache.update(q=x.copy(), t=t, residual=r, jacobian=j)
        return cache

    def callback(x):
        nonlocal iterations
        iterations += 1
        if accept(x):
            raise StopIteration

    initial = np.asarray(q, float)
    # An exact antipodal vector is a stationary maximum of the smooth axis
    # cost. Escape deterministically rather than reporting it as alignment.
    escape = False; escape_probes = 0
    if pose5.axis_error(data(initial)['t'][:3, 2], axis)[2] > 179.999:
        candidates = [initial]
        for k in range(6):
            for sign in (-1, 1):
                candidates.append(ARM.clip(initial+sign*.15*np.eye(6)[k]))
        initial = min(candidates, key=lambda x: float(data(x)['residual']@data(x)['residual']))
        escape = not np.array_equal(initial, q)
        escape_probes = len(candidates)
    result = least_squares(lambda x: data(x)['residual'], initial,
                           jac=lambda x: data(x)['jacobian'],
                           bounds=(ARM.lower, ARM.upper), method='trf',
                           max_nfev=evaluations, callback=callback,
                           ftol=1e-8, xtol=1e-8, gtol=1e-8)
    reached = bool(accept(result.x))
    return {'q_rad': result.x, 'converged': reached, 'iterations': iterations,
            'evaluations': int(result.nfev), 'elapsed_ms': (perf_counter()-began)*1000,
            'escape_probes': escape_probes,
            'antipodal_escape': escape, 'optimizer_status': int(result.status),
            'stop_reason': 'target_reached' if reached else
                ('evaluation_limit' if result.status == 0 else 'stationary_without_valid_goal')}


def stratified_seeds(count, seed):
    """Latin-hypercube coverage: each joint visits every range bin exactly once."""
    if not count:
        return np.empty((0, 6))
    rng = np.random.default_rng(seed)
    unit = np.column_stack([(rng.permutation(count)+rng.random(count))/count for _ in range(6)])
    return ARM.lower + unit*(ARM.upper-ARM.lower)


def coverage_seeds(q_current, target, axis, roadmap, generated, count, seed, use_nn=True):
    """Continuation -> original NN/current -> diverse roadmap -> NN anchors -> strata.

    Roadmap proximity retains normalized XYZ + 0.6 * directed-axis radians.
    Greedy posture diversity ignores joint 6, whose task effect is only local
    tool-Z spin in this URDF. This does not constrain joint 6 in the optimizer.
    The original NN bounds still apply around each explicitly recorded anchor.
    """
    seeds = []; seen = set(); duplicates = 0

    def add(q, source, anchor=None):
        nonlocal duplicates
        q = np.asarray(q, float)
        if q.shape != (6,) or not np.isfinite(q).all() or np.any(q < ARM.lower) or np.any(q > ARM.upper):
            return
        key = tuple(q.round(10))
        if key in seen:
            duplicates += 1; return
        seen.add(key)
        entry = {'q_rad': q.tolist(), 'source': source}
        if anchor is not None:
            entry['anchor_q_rad'] = np.asarray(anchor).tolist()
        seeds.append(entry)

    endpoints = [a for a in generated['attempts'] if a.get('end_q_rad') is not None]
    endpoints.sort(key=lambda a: a.get('error_mm', np.inf)/1000/ARM.scale
                   + .6*np.deg2rad(a.get('orientation_error_deg', 180)))
    for a in endpoints[:4]:
        add(a['end_q_rad'], 'recovery_continuation')
    for entry in generated['seed_queue']:
        if entry['source'].startswith('nn_'):
            add(entry['q_rad'], 'recovery_nn_seed')
    add(q_current, 'recovery_current')
    selected = []
    if roadmap is not None and len(roadmap.q):
        transforms = roadmap.poses
        angles = np.arctan2(np.linalg.norm(np.cross(transforms[:, :3, 2], axis), axis=1),
                            np.clip(transforms[:, :3, 2]@axis, -1, 1))
        distances = np.linalg.norm(transforms[:, :3, 3]-target, axis=1)/ARM.scale + .6*angles
        # Search the full validated graph, not just the four nearest task poses.
        for i in np.argsort(distances, kind='stable'):
            q = roadmap.q[i]
            if any(np.linalg.norm(((q-other)/ARM.radius)[:5]) < .55 for other in selected):
                continue
            selected.append(q)
            add(q, 'recovery_roadmap')
            if len(selected) == 8:
                break
    anchors = selected[:2] or [ARM.center, ARM.clip(2*ARM.center-q_current)]
    if use_nn and (generated.get('nn_checkpoint') or {}).get('status') == 'ready':
        for anchor in anchors:
            guesses, _, _ = seed5.propose(anchor, target, axis)
            for q in guesses:
                add(q, 'recovery_nn_anchor', anchor)
    for q in stratified_seeds(12, seed ^ 0x5D280):
        add(q, 'recovery_stratified')
    return seeds[:count], duplicates
