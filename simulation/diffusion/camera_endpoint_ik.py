"""Bounded XYZ IK with camera visibility as a hard endpoint constraint.

Numerical iterates are virtual. Endpoints and the complete timed route still
pass the normal visibility/collision checks before any preview motion.
"""
from time import perf_counter
import numpy as np
from scipy.optimize import least_squares
from ..kinematics import ARM
from .. import pose5
from .ik_seed import propose
from .endpoint_ik import IKSearchExhausted


def search_camera_endpoint(view, target, axis, current, *, seed=42, attempts=4,
                           learned=True, robust=False, cancel_check=None):
    began = perf_counter()
    stats = {'solver':'xyz_with_camera_fov', 'legacy_attempts':0, 'fallback_attempts':0,
             'collision_rejections':0, 'visibility_rejections':0, 'nfev':0,
             'learned_status':'not_needed', 'learned_proposal_count':0}
    view.ik_diagnostics = stats
    rng = np.random.default_rng(seed)
    guesses = [np.asarray(current,float)]
    def check():
        if cancel_check is not None: cancel_check()
    def residual(q):
        check()
        return (ARM.fk(q)-target)/ARM.scale
    def jacobian(q):
        check()
        return ARM.spatial_jacobian(q)[1][:3]/ARM.scale
    try:
        for index in range(attempts):
            check()
            if index == 1 and learned:
                proposals, status, inference_ms = propose(current,target,axis)
                stats.update(learned_status=status['status'],learned_proposal_count=len(proposals),
                             learned_inference_ms=inference_ms)
                proposals = [np.asarray(q,float) for q in proposals if np.asarray(q).shape==(6,)
                    and np.isfinite(q).all() and np.all(q>=ARM.lower) and np.all(q<=ARM.upper)]
                guesses.extend(sorted(proposals,key=lambda q:np.linalg.norm(ARM.fk(q)-target)))
            guess = guesses[index] if index < len(guesses) else rng.uniform(ARM.lower,ARM.upper)
            stats['fallback_attempts'] += 1
            solved = least_squares(residual,guess,jac=jacobian,bounds=(ARM.lower,ARM.upper),
                max_nfev=160,ftol=1e-9,gtol=1e-9,xtol=1e-9)
            stats['nfev'] += solved.nfev
            check()
            q = solved.x
            error = float(np.linalg.norm(ARM.fk(q)-target)*1000)
            stats['best_position_error_mm'] = min(stats.get('best_position_error_mm',float('inf')),error)
            if error > 1.: continue
            projection = view.inspect(q)
            if not projection['visible']:
                stats['visibility_rejections'] += 1
                continue
            if not view.valid(q):
                stats['collision_rejections'] += 1
                continue
            stats.update(status='found',seconds=perf_counter()-began,
                         direction_error_deg=pose5.axis_error(ARM.transform(q)[:3,2],axis)[2])
            return q
        stats.update(status='search_exhausted',seconds=perf_counter()-began)
        raise IKSearchExhausted(stats)
    finally:
        stats['seconds'] = perf_counter()-began
