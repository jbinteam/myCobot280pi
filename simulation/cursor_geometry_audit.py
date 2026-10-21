"""Conservative reach/limit certificates for this URDF, used only by audits.

The arm's near-right-angle origins admit a planar two-link decomposition.
Uniform operator-norm bounds account for the URDF's 1.5708 rotations rather
than treating them as exact pi/2. Both yaw and wrist branches are considered.
An unresolved certificate does not establish reachability or collision freedom.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from .kinematics import ARM
from .pose5 import unit_axis


def model_bounds():
    """Bound exact/ideal position and directed-axis differences for all joints."""
    expected_xyz = np.array([[0, 0, 0], [0, 0, .13956], [0, 0, -.001],
                             [-.1104, 0, 0], [-.096, 0, .06462],
                             [0, -.07318, -.001], [0, .0456, 0]])
    expected_rotations = [np.eye(3), np.eye(3),
        np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]), np.eye(3),
        np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
        np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]),
        np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])]
    if len(ARM.chain) != 7:
        raise ValueError('Unsupported chain for geometric audit')
    rotation_bound = 0.
    position_bound = 0.
    for index, (base, axis) in enumerate(ARM.chain):
        ideal_rotation = expected_rotations[index]
        if (not np.allclose(base[:3, 3], expected_xyz[index], rtol=0, atol=1e-12)
                or not np.allclose(np.round(base[:3, :3]), ideal_rotation, rtol=0, atol=1e-12)
                or (index == 0 and axis is not None)
                or (index != 0 and (axis is None or not np.array_equal(axis, [0, 0, 1])))):
            raise ValueError('Unsupported URDF geometry for geometric audit')
        position_bound += np.linalg.norm(base[:3, 3])*rotation_bound
        rotation_bound += np.linalg.norm(base[:3, :3]-ideal_rotation, ord=2)
    # Small padding for floating-point arithmetic; not a task tolerance change.
    return {'position_m': float(position_bound+1e-12),
            'axis_chord': float(rotation_bound+1e-12)}


def _intersects_periodic(low, high, operating_low, operating_high):
    return any(low+shift <= operating_high and high+shift >= operating_low
               for shift in (-2*np.pi, 0., 2*np.pi))


def geometry_certificate(target, axis, tolerance_mm=1., orientation_tolerance_deg=1.):
    """Exclude all ideal yaw/wrist branches only using necessary constraints.

    Let w = position - 45.6 mm*tool_Z - shoulder. Its lateral component in
    the shoulder plane is fixed at 63.62 mm. This gives two yaw branches.
    Tool-Z fixes the planar wrist angle up to its two wrist branches. After
    subtracting the 73.18 mm wrist segment, the two arm links can span only
    14.4--206.4 mm. Position/axis tolerances and exact/ideal model error enlarge
    every interval before comparing it with operating limits and link reach.
    """
    target = np.asarray(target, float)
    axis = unit_axis(axis)
    if (target.shape != (3,) or not np.isfinite(target).all()
            or not np.isfinite(tolerance_mm) or tolerance_mm < 0
            or not np.isfinite(orientation_tolerance_deg) or not 0 <= orientation_tolerance_deg <= 180):
        raise ValueError('Expected finite geometry and nonnegative tolerances')
    bounds = model_bounds()
    axis_error = 2*np.sin(np.deg2rad(orientation_tolerance_deg)/2)+bounds['axis_chord']
    wrist_error = tolerance_mm/1000+bounds['position_m']+.0456*axis_error
    wrist = target-.0456*axis-np.array([0., 0., .13856])
    rho = float(np.linalg.norm(wrist[:2]))
    lateral = .06362
    result = {'model_error_bound': bounds, 'wrist_position_error_bound_mm': float(wrist_error*1000),
              'geometric_infeasible_with_tolerances': False, 'branches': []}
    if rho+wrist_error < lateral:
        result.update(geometric_infeasible_with_tolerances=True, exclusion='lateral_offset')
        return result
    if rho-wrist_error <= lateral:
        result['unresolved_reason'] = 'Yaw branches merge within the error interval'
        return result
    phi_error = np.arcsin(min(1., wrist_error/rho))
    offset = np.arcsin(lateral/rho)
    offset_error = max(np.arcsin(lateral/(rho-wrist_error))-offset,
                       offset-np.arcsin(lateral/(rho+wrist_error)))
    yaw_error = float(phi_error+offset_error)
    axis_frame_error = float(axis_error+2*np.sin(yaw_error/2))
    planar_wrist_error = float(wrist_error+np.linalg.norm(wrist)*2*np.sin(yaw_error/2))
    phi = np.arctan2(wrist[1], wrist[0])
    shoulder_rotation = np.array([[0., 1., 0.], [0., 0., -1.], [-1., 0., 0.]])
    for yaw in (phi+offset, phi+np.pi-offset):
        yaw = float((yaw+np.pi) % (2*np.pi)-np.pi)
        c, s = np.cos(yaw), np.sin(yaw)
        inverse = shoulder_rotation.T @ np.array([[c, s, 0.], [-s, c, 0.], [0., 0., 1.]])
        v, u = inverse@wrist, inverse@axis
        sine = float(-u[2])
        angle_low, angle_high = np.arcsin(np.clip([sine-axis_frame_error, sine+axis_frame_error], -1, 1))
        projected_length = float(np.linalg.norm(u[:2]))
        for sign, (low, high) in ((1., (angle_low, angle_high)),
                                 (-1., (np.pi-angle_high, np.pi-angle_low))):
            nominal_wrist = float(np.arcsin(np.clip(sine, -1, 1)))
            if sign < 0:
                nominal_wrist = np.pi-nominal_wrist
            nominal_wrist = float((nominal_wrist+np.pi) % (2*np.pi)-np.pi)
            theta = float(np.arctan2(-sign*u[0], sign*u[1]))
            required = float(np.linalg.norm(v[:2]+.07318*np.array([np.cos(theta), np.sin(theta)])))
            branch = {'yaw_deg': float(np.rad2deg(yaw)), 'yaw_error_bound_deg': float(np.rad2deg(yaw_error)),
                      'wrist_deg': float(np.rad2deg(nominal_wrist)),
                      'required_planar_reach_mm': required*1000, 'exclusion': None}
            if not _intersects_periodic(yaw-yaw_error, yaw+yaw_error, ARM.lower[0], ARM.upper[0]):
                branch['exclusion'] = 'joint_1_operating_limit'
            elif not _intersects_periodic(low, high, ARM.lower[4], ARM.upper[4]):
                branch['exclusion'] = 'joint_5_operating_limit'
            elif projected_length <= axis_frame_error:
                branch['unresolved_reason'] = 'Planar axis projection can vanish within the error interval'
            else:
                theta_error = np.arcsin(axis_frame_error/projected_length)
                reach_error = planar_wrist_error+.07318*2*np.sin(theta_error/2)
                branch['planar_reach_interval_mm'] = [max(0., required-reach_error)*1000,
                                                       (required+reach_error)*1000]
                if required-reach_error > .1104+.096+1e-12:
                    branch['exclusion'] = 'outer_planar_reach'
                elif required+reach_error < abs(.1104-.096)-1e-12:
                    branch['exclusion'] = 'inner_planar_reach'
            result['branches'].append(branch)
    result['geometric_infeasible_with_tolerances'] = all(branch['exclusion'] is not None for branch in result['branches'])
    return result


def audit_geometry(benchmark_path, output):
    """Check saved successes as controls and audit failed target geometry."""
    benchmark_path, output = map(Path, (benchmark_path, output))
    benchmark = json.loads(benchmark_path.read_text())
    if benchmark['tolerance_mm'] != 1. or benchmark['orientation_tolerance_deg'] != 1.:
        raise ValueError('This cursor audit expects the saved 1 mm / 1 degree benchmark')
    cases = {case['id']: case for case in benchmark['cursor']['tasks']}
    rows = []
    for saved in benchmark['cursor']['rows']['upgraded']:
        case = cases[saved['task_id']]
        cursor = np.asarray(case['cursor_mm'])/1000
        axis = unit_axis(cursor-ARM.fk(np.deg2rad(case['start_deg'])))
        target = cursor-.05*axis
        certificate = geometry_certificate(target, axis)
        if saved['planned'] and certificate['geometric_infeasible_with_tolerances']:
            raise ValueError('Geometry certificate contradicts a saved success: '+case['id'])
        rows.append({'id': case['id'], 'saved_planned': saved['planned'],
                     'target_mm': (target*1000).tolist(), 'target_axis': axis.tolist(), **certificate})
    failures = [row for row in rows if not row['saved_planned']]
    report = {'simulation_only': True, 'benchmark': str(benchmark_path),
              'benchmark_sha256': hashlib.sha256(benchmark_path.read_bytes()).hexdigest(),
              'urdf_sha256': ARM.fingerprint,
              'tolerance_mm': 1., 'orientation_tolerance_deg': 1.,
              'method': 'Necessary planar reach and joint-1/joint-5 bounds for all four '
                        'yaw/wrist branches, enlarged by task tolerances and uniform exact/ideal URDF error bounds.',
              'limitations': 'An exclusion applies to this URDF and its operating limits only. '
                             'An unresolved result does not establish feasibility. Collision, joint-2/3/4 '
                             'limits and cursor geometry can impose additional constraints. '
                             'The certificates are audit evidence and do not bypass production IK.',
              'summary': {'tasks': len(rows), 'saved_success_controls': len(rows)-len(failures),
                          'excluded_saved_successes': 0, 'failed_tasks': len(failures),
                          'geometrically_excluded_failures': sum(row['geometric_infeasible_with_tolerances'] for row in failures),
                          'unresolved_failures': [row['id'] for row in failures if not row['geometric_infeasible_with_tolerances']],
                          'excluded_branch_reasons': dict(Counter(branch['exclusion'] for row in failures
                              for branch in row['branches'] if branch['exclusion']))},
              'rows': rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report['summary'], indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, default=Path('output/ik_recovery/benchmark.json'))
    parser.add_argument('--output', type=Path, default=Path('output/ik_recovery/cursor_geometry_audit.json'))
    args = parser.parse_args()
    audit_geometry(args.benchmark, args.output)
