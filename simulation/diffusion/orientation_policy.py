"""Strict-first orientation policy for checked cone-region planning."""
from time import perf_counter

HYBRID_MODE = 'strict_then_fov'
FALLBACK_STATUSES = frozenset(('ik_failed', 'no_path'))


def effective_orientation_mode(result, requested_mode='point_at_object'):
    """Verify the strict-first report before accepting a playable proposal."""
    mode = result.get('orientation_mode', 'point_at_object')
    if requested_mode != HYBRID_MODE:
        if mode != requested_mode:
            raise ValueError('Planner changed the requested orientation mode')
        return mode
    selection = result.get('orientation_selection') or {}
    strict = selection.get('strict') or {}
    if (result.get('requested_orientation_mode') != requested_mode
            or selection.get('policy') != requested_mode or selection.get('selected_mode') != mode
            or strict.get('mode') != 'point_at_object'):
        raise ValueError('Planner omitted the checked strict-first orientation selection')
    if (mode == 'point_at_object' and selection.get('fallback_used') is False
            and selection.get('relaxed') is None):
        return mode
    relaxed = selection.get('relaxed') or {}
    if (mode == 'camera_fov' and selection.get('fallback_used') is True
            and strict.get('status') in FALLBACK_STATUSES and strict.get('playable') is False
            and relaxed.get('mode') == 'camera_fov'):
        return mode
    raise ValueError('Camera-FOV fallback requires a failed strict search')


def plan_with_orientation(attempt, requested_mode):
    """Retain any playable strict continuation; relax only after search failure.

    The caller keeps its planner lock, frozen region, camera and motion boundary
    across both passes. Exceptions, including cancellation, propagate unchanged.
    """
    if requested_mode != HYBRID_MODE:
        return attempt(requested_mode)
    reports = []
    results = []

    def run(mode):
        began = perf_counter()
        result = attempt(mode)
        results.append(result)
        reports.append(dict(mode=mode,status=result['status'],
            playable=result.get('playable',False),converged=result.get('converged',False),
            compute_ms=(perf_counter()-began)*1000,ik_ms=result.get('ik_elapsed_ms',0.),
            path_ms=result.get('path_elapsed_ms'),ik_jobs=result.get('region_candidate_jobs_tested',0)))
        return result

    strict = run('point_at_object')
    fallback = not strict.get('playable',False) and strict['status'] in FALLBACK_STATUSES
    selected = run('camera_fov') if fallback else strict
    result = dict(selected)
    effective_mode = 'camera_fov' if fallback else 'point_at_object'
    result.update(orientation_mode=effective_mode,requested_orientation_mode=HYBRID_MODE,
        orientation_selection=dict(policy=HYBRID_MODE,selected_mode=effective_mode,
            fallback_used=fallback,strict=reports[0],relaxed=reports[1] if fallback else None),
        ik_elapsed_ms=sum(r.get('ik_elapsed_ms',0.) for r in results),
        hybrid_ik_jobs=sum(r.get('region_candidate_jobs_tested',0) for r in results))
    if any('path_elapsed_ms' in r for r in results):
        result['path_elapsed_ms'] = sum(r.get('path_elapsed_ms',0.) for r in results)
    if fallback:
        result['reason'] = f"Strict planning returned {strict['status']}; camera-FOV fallback: {selected['reason']}"
    return result
