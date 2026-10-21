"""Estimate targets solvable from a fixed start using the production IK solver.

FK candidate generation uses local direction projection. Each displayed target
must then converge with the selected solver from the captured start. Sampling
is incomplete and failures do not prove unreachability. No robot transport is used.
"""
import argparse
import itertools
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from .kinematics import ARM
from .pose5 import resolve_axis


def transforms(q, jacobian=False):
    """Batched FK from the same URDF chain used by the solver."""
    t = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
    axes = []; index = 0
    for base, axis in ARM.chain:
        t = t @ base
        if axis is not None:
            axes.append(t[:, :3, :3] @ axis)
            x, y, z = axis / np.linalg.norm(axis)
            skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            angle = q[:, index, None, None]
            r = np.broadcast_to(np.eye(4), t.shape).copy()
            r[:, :3, :3] = np.eye(3) + np.sin(angle)*skew + (1-np.cos(angle))*(skew@skew)
            t = t @ r; index += 1
    return (t, np.stack(axes, axis=2)) if jacobian else t


def finite_number(body, key, default, low, high, integer=False):
    value = body.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value) or not low <= value <= high or (integer and int(value) != value):
        raise ValueError(f'{key} must be between {low} and {high}')
    return int(value) if integer else float(value)


def geometric_estimate(body):
    start = perf_counter()
    mode = body.get('task', 'pose5')
    if mode not in ('position', 'pose5'): raise ValueError('Unknown workspace task')
    count = finite_number(body, 'samples', 12000, 100, 50000, True)
    seed = finite_number(body, 'seed', 9001, 0, 2**32-1, True)
    tolerance = finite_number(body, 'tolerance_mm', 1., .1, 10.)
    angle_tol = finite_number(body, 'orientation_tolerance_deg', 1., .1, 10.)
    voxel = finite_number(body, 'voxel_mm', 8., 4., 25.)
    target_axis = resolve_axis(body.get('orientation_xy_deg', [0, 0]),body.get('target_axis')) if mode == 'pose5' else None
    q = np.random.default_rng(seed).uniform(ARM.lower, ARM.upper, (count, 6))
    if target_axis is not None:
        # Move random starts toward the allowed cone, retaining broad position
        # coverage. Exact final FK checks, not projection success, admit samples.
        for _ in range(40):
            t, jw = transforms(q, True)
            n = t[:, :3, 2]
            error = target_axis - n
            j = np.cross(jw.transpose(0, 2, 1), n[:, None, :]).transpose(0, 2, 1)
            velocity = (j.transpose(0, 2, 1) @ np.linalg.solve(j@j.transpose(0, 2, 1) + .0025*np.eye(3), error[..., None]))[..., 0]
            velocity *= np.minimum(1., .35/np.maximum(np.linalg.norm(velocity, axis=1), 1e-12))[:, None]
            done = n@target_axis >= np.cos(np.deg2rad(angle_tol))
            velocity[done] = 0
            q = ARM.clip(q + velocity)
            if done.all(): break
    t = transforms(q)
    angles = np.rad2deg(np.arccos(np.clip(t[:, :3, 2]@target_axis, -1, 1))) if target_axis is not None else np.zeros(count)
    keep = angles <= angle_tol if target_axis is not None else np.ones(count, bool)
    points = t[keep, :3, 3]*1000
    cells = envelope(points, tolerance, voxel)
    return {'task':mode, 'points_mm':points.tolist(), 'voxels':cells, 'voxel_mm':voxel,
            'accepted':len(points), 'samples':count, 'seed':seed, 'tolerance_mm':tolerance,
            'orientation_tolerance_deg':angle_tol, 'target_axis':target_axis.tolist() if target_axis is not None else None,
            'max_direction_error_deg':float(angles[keep].max()) if keep.any() and target_axis is not None else None,
            'bounds_mm':{'min':points.min(axis=0).tolist(), 'max':points.max(axis=0).tolist()} if len(points) else None,
            'elapsed_ms':(perf_counter()-start)*1000, 'limits_deg':{'min':np.rad2deg(ARM.lower).tolist(), 'max':np.rad2deg(ARM.upper).tolist()},
            'urdf_sha256':ARM.fingerprint, 'simulation_only':True,
            'envelope_extra_mm_max':float(np.sqrt(3)*voxel),
            'limitations':'Sampled geometric estimate, biased direction projection; gaps are inconclusive. Voxel shell is an outer approximation of sampled tolerance balls, not certified reachability. No collisions, path, speed or start-pose feasibility.'}


def envelope(points, tolerance, voxel):
    # Outer voxel approximation to the union of position-tolerance balls.
    # A cell is retained if its center is within radius + half diagonal of a
    # witness. This preserves holes; it does not fill a convex hull.
    cells = set(); radius = tolerance + np.sqrt(3)*voxel/2
    reach = int(np.ceil(radius/voxel))
    offsets = np.array(list(itertools.product(range(-reach, reach+1), repeat=3)))
    for chunk in np.array_split(points, max(1, int(np.ceil(len(points)/1000)))):
        if not len(chunk): continue
        indices = np.floor(chunk/voxel).astype(int)[:, None, :] + offsets
        centers = (indices+.5)*voxel
        valid = np.linalg.norm(centers-chunk[:, None, :], axis=2) <= radius
        cells.update(map(tuple, indices[valid].tolist()))
    return sorted(cells)


def estimate(body):
    """Test candidate target poses using the production solver from one start."""
    from .service import run
    from .solvers import validate
    from .pose5 import METHODS as POSE_METHODS
    began = perf_counter()
    mode = body.get('task', 'pose5')
    method = body.get('method', 'dls')
    start_deg = np.asarray(body.get('start_deg'), dtype=float)
    steps = finite_number(body, 'steps', 240, 1, 600, True)
    speed = finite_number(body, 'speed_rad_s', 1.2, .1, 3.)
    damping = finite_number(body, 'damping', .04, .001, .3)
    tolerance = finite_number(body, 'tolerance_mm', 1., .1, 10.)
    targets = finite_number(body, 'targets', 128, 1, 512, True)
    if mode == 'pose5' and method not in [m[0] for m in POSE_METHODS]:
        raise ValueError('Selected method does not support 5D poses')
    validate(np.deg2rad(start_deg), np.zeros(3), 'dls' if method == 'bounded_dls' else method, steps, speed, tolerance, damping)
    pool = geometric_estimate(body)
    candidates = np.array(pool['points_mm']).reshape(-1, 3)
    rng = np.random.default_rng(pool['seed'])
    if len(candidates):
        candidates = candidates[rng.choice(len(candidates), min(targets-1, len(candidates)), replace=False)]
    # Include the current position, but still test the requested direction.
    candidates = np.vstack([ARM.fk(np.deg2rad(start_deg))*1000, candidates])
    successes = []; outcomes = []; final_angles = []
    for target in candidates:
        result = run({**body, 'task':mode, 'start_deg':start_deg.tolist(), 'target_mm':target.tolist(),
                      'eef_offset_mm':[0, 0, 0], 'methods':[method], 'steps':steps,
                      'speed_rad_s':speed, 'damping':damping, 'tolerance_mm':tolerance})['results'][0]
        outcomes.append({'target_mm':target.tolist(), 'converged':result['converged'],
                         'final_error_mm':result['final_error_mm'], 'final_orientation_error_deg':result.get('final_orientation_error_deg'),
                         'iterations':result['iterations'], 'stop_reason':result['stop_reason']})
        if result['converged']:
            successes.append(target.tolist())
            if mode == 'pose5': final_angles.append(result['final_orientation_error_deg'])
    points = np.array(successes).reshape(-1, 3)
    pool.update({'basis':'solver', 'fk_points_mm':pool['points_mm'], 'fk_voxels':pool['voxels'],
                 'fk_bounds_mm':pool['bounds_mm'],
                 'points_mm':successes, 'voxels':envelope(points, tolerance, pool['voxel_mm']),
                 'accepted':len(points), 'tested':len(candidates), 'requested_targets':targets,
                 'geometric_candidates':pool['accepted'], 'start_deg':start_deg.tolist(),
                 'method':method, 'steps':steps, 'speed_rad_s':speed, 'damping':damping,
                 'outcomes':outcomes, 'max_direction_error_deg':max(final_angles, default=None),
                 'bounds_mm':{'min':points.min(axis=0).tolist(), 'max':points.max(axis=0).tolist()} if len(points) else None,
                 'elapsed_ms':(perf_counter()-began)*1000,
                 'limitations':'Only converged tested target poses are shown, using one captured start and solver budget. Biased finite sampling; failures are not proof of unreachability. The voxel shell and neighboring tolerance regions are not solver-tested. No collision or physical-path validation.'})
    return pool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-deg', type=float, nargs=6, required=True)
    parser.add_argument('--method', default='dls')
    parser.add_argument('--targets', type=int, default=128)
    parser.add_argument('--steps', type=int, default=240)
    parser.add_argument('--speed-rad-s', type=float, default=1.2)
    parser.add_argument('--damping', type=float, default=.04)
    parser.add_argument('--task', choices=['pose5', 'position'], default='pose5')
    parser.add_argument('--orientation', type=float, nargs=2, default=[0, 0], metavar=('ROLL', 'PITCH'))
    parser.add_argument('--direction-tolerance', type=float, default=1.)
    parser.add_argument('--tolerance-mm', type=float, default=1.)
    parser.add_argument('--samples', type=int, default=12000)
    parser.add_argument('--voxel-mm', type=float, default=8.)
    parser.add_argument('--seed', type=int, default=9001)
    parser.add_argument('--output', type=Path, default=Path('simulation/artifacts/workspace.json'))
    args = parser.parse_args()
    result = estimate({'start_deg':args.start_deg, 'method':args.method, 'targets':args.targets, 'steps':args.steps, 'speed_rad_s':args.speed_rad_s, 'damping':args.damping, 'task':args.task, 'orientation_xy_deg':args.orientation, 'orientation_tolerance_deg':args.direction_tolerance,
                       'tolerance_mm':args.tolerance_mm, 'samples':args.samples, 'voxel_mm':args.voxel_mm, 'seed':args.seed})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result))
    print(f'{result["accepted"]}/{result["tested"]} targets solved, {len(result["voxels"])} voxels, {result["elapsed_ms"]:.0f} ms → {args.output}')


if __name__ == '__main__': main()
