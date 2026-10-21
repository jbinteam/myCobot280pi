"""Batch independent trajectory calculations without changing their arithmetic."""
import os
import numpy as np
from .kinematics import ARM

# Fixed at process import, allowing identical-source offline comparisons.
BATCH_TRAJECTORY = os.environ.get('MYCOBOT_TRAJECTORY_BATCH', '1') != '0'


def subdivision_leaves(controls, depth):
    """Return original de Casteljau leaves, or None for scalar dispatch.

    Restrict batching to the finite six-joint derivative arrays used by the
    planner. The magnitude guard also excludes midpoint/norm overflow; unusual
    inputs retain the original recursive behavior, including NaN ordering.
    """
    if (not BATCH_TRAJECTORY or type(controls) is not np.ndarray
            or controls.dtype != np.dtype('float64') or controls.shape not in ((4,6),(5,6))
            or type(depth) is not int or not 1 <= depth <= 8
            or not np.isfinite(controls).all() or np.max(np.abs(controls)) > 1e100):
        return None
    leaves = controls[None]
    for _ in range(depth):
        rows = [leaves]
        while rows[-1].shape[1] > 1:
            rows.append((rows[-1][:,:-1]+rows[-1][:,1:])/2)
        left = np.stack([row[:,0] for row in rows], axis=1)
        right = np.stack([row[:,-1] for row in rows[::-1]], axis=1)
        # Interleave each parent's left/right children in recursive order.
        leaves = np.stack((left,right), axis=1).reshape(-1,*controls.shape)
    return leaves.reshape(-1,6)


def frame_transforms(states):
    """Yield the same flange FK for output frames in bounded 256-pose blocks."""
    if not BATCH_TRAJECTORY:
        for q in states:
            yield ARM.transform(q)
        return
    for start in range(0,len(states),256):
        yield from ARM.link_transforms_batch(states[start:start+256])[ARM.tcp]
