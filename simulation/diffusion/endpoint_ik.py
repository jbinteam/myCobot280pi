"""Bounded offline endpoint search; exhaustion is not proof of unreachability."""
from time import perf_counter
import os
import numpy as np
from scipy.optimize import least_squares
from ..kinematics import ARM
from ..pose5 import axis_error, ORIENTATION_WEIGHT
from .ik_seed import propose

FALLBACK_ATTEMPTS = 32
FALLBACK_MAX_NFEV = 160


class IKSearchExhausted(RuntimeError):
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__('No collision-free IK endpoint found within the search budget; '
                         'this does not prove the target is unreachable')


def residual_and_jacobian(q, target, axis):
    """Directed chord residual avoids the false zero of antipodal cross products."""
    transform, spatial = ARM.spatial_jacobian(q)
    direction = transform[:3, 2]
    residual = np.r_[(transform[:3, 3] - target) / ARM.scale,
                     ORIENTATION_WEIGHT * (direction - axis)]
    jacobian = np.vstack((spatial[:3] / ARM.scale,
                          ORIENTATION_WEIGHT * np.cross(spatial[3:].T, direction).T))
    return residual, jacobian


def search_endpoint(env, target, axis, current, seed, attempts, legacy_solve, robust=False, learned=None, cancel_check=None):
    started = perf_counter()
    stats = env.ik_diagnostics
    best_score = float('inf')
    enabled = os.environ.get('MYCOBOT_IK_SEED', '0') == '1' if learned is None else bool(learned)
    stats.update(learned_status='pending' if enabled else 'disabled',
                 learned_proposal_count=0, learned_refine_attempts=0, learned_inference_ms=0.,
                 learned_refine_success=False, learned_success_stage=None)
    learned_starts = []
    if cancel_check is not None and not callable(cancel_check):
        raise ValueError('Cancellation check must be callable')

    def checkpoint():
        if cancel_check is not None:cancel_check()

    def assess(q):
        nonlocal best_score
        checkpoint()
        q = np.asarray(q, float)
        if q.shape != (6,) or not np.isfinite(q).all():
            return False
        transform = ARM.transform(q)
        position = float(np.linalg.norm(transform[:3, 3] - target) * 1000)
        direction = axis_error(transform[:3, 2], axis)[2]
        score = max(position, direction)  # both acceptance thresholds are 1
        if score < best_score:
            best_score = score
            stats['best_position_error_mm'] = position
            stats['best_direction_error_deg'] = direction
        if np.any(q < ARM.lower) or np.any(q > ARM.upper):
            return False
        if position > 1. or direction > 1.:
            return False
        if not env.valid(q):
            stats['collision_rejections'] += 1
            return False
        return True

    # Count rejected collision-check calls, including repeated candidates.
    def legacy_valid(q):
        checkpoint()
        valid = env.valid(q)
        if not valid:
            stats['collision_rejections'] += 1
        return valid

    def prepare_learned():
        nonlocal learned_starts
        checkpoint()
        if not enabled:
            return
        guesses, status, inference_ms = propose(current, target, axis)
        stats['learned_status'] = status['status']
        stats['learned_checkpoint'] = status
        stats['learned_inference_ms'] = inference_ms
        learned_starts = [np.asarray(q, float) for q in guesses
                          if np.asarray(q).shape == (6,) and np.isfinite(q).all()
                          and np.all(q >= ARM.lower) and np.all(q <= ARM.upper)][:8]
        # Best-of-K training gives no head-order meaning. Rank virtual guesses
        # with the trained directed FK objective before spending solver budget.
        # The tiny travel term breaks near-pose ties without adding a spin gate.
        learned_starts.sort(key=lambda q: float(np.sum(residual_and_jacobian(q, target, axis)[0]**2)
                            + 1e-5*np.sum(((q-current)/ARM.radius)**2)))
        stats['learned_proposal_count'] = len(learned_starts)

    try:
        checkpoint()
        rng = np.random.default_rng(seed)
        starts = [current] + [rng.uniform(ARM.lower, ARM.upper) for _ in range(attempts - 1)]
        legacy_endpoints = []
        for index in range(attempts):
            checkpoint()
            # Preserve the successful current-pose attempt before any inference.
            if index == 1:
                prepare_learned()
            is_learned = index > 0 and index <= len(learned_starts)
            start = learned_starts[index - 1] if is_learned else starts[index]
            if is_learned:
                stats['learned_refine_attempts'] += 1
            stats['legacy_attempts'] += 1
            result = legacy_solve(start, target, method='bounded_dls', steps=400,
                                  target_axis=axis, goal_constraint=legacy_valid,
                                  **({} if cancel_check is None else {'cancel_check': cancel_check}))
            q = np.deg2rad(result['frames'][-1]['q'])
            legacy_endpoints.append(q)
            accepted = assess(q)
            if result['converged'] and accepted:
                if enabled and index == 0:
                    stats['learned_status'] = 'skipped_current_success'
                if is_learned:
                    stats['learned_refine_success'] = True
                    stats['learned_success_stage'] = 'bounded_dls'
                stats['method'] = 'bounded_dls'
                return q

        if not robust:
            raise IKSearchExhausted(stats)

        if attempts == 1:
            prepare_learned()

        # An independent RNG makes the fallback independent of how many
        # legacy restarts were configured. Fixed seeds reproduce the search.
        fallback_rng = np.random.default_rng(seed)
        ranked = sorted(legacy_endpoints, key=lambda q: np.linalg.norm(residual_and_jacobian(q, target, axis)[0]))
        fallback_starts = [current, *learned_starts, ARM.center, *ranked[:2]]
        fallback_starts += list(fallback_rng.uniform(ARM.lower, ARM.upper, (FALLBACK_ATTEMPTS, 6)))
        for index, start in enumerate(fallback_starts[:FALLBACK_ATTEMPTS]):
            checkpoint()
            if 0 < index <= len(learned_starts):
                stats['learned_refine_attempts'] += 1
            stats['fallback_attempts'] += 1
            def checked_residual(q):
                checkpoint()
                return residual_and_jacobian(q, target, axis)[0]
            def checked_jacobian(q):
                checkpoint()
                return residual_and_jacobian(q, target, axis)[1]
            result = least_squares(
                checked_residual,
                np.clip(start, ARM.lower, ARM.upper),
                jac=checked_jacobian,
                bounds=(ARM.lower, ARM.upper), max_nfev=FALLBACK_MAX_NFEV,
                ftol=1e-9, xtol=1e-9, gtol=1e-9)
            stats['fallback_nfev'] += int(result.nfev)
            if assess(result.x):
                if 0 < index <= len(learned_starts):
                    stats['learned_refine_success'] = True
                    stats['learned_success_stage'] = 'least_squares'
                stats['method'] = 'least_squares'
                return result.x.copy()
        raise IKSearchExhausted(stats)
    finally:
        stats['elapsed_ms'] = (perf_counter() - started) * 1000
