"""Replay reachable targets with a blocked direct edge or a bounded IK miss.

Offline only; imports no ROS controller, camera driver, or robot transport.
Run: OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/generate_reachable_failure_scenarios.py
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulation.kinematics import ARM
from simulation.physical_reach import target_precheck
from simulation.pose5 import axis_error, solve
from simulation.diffusion.env import ArmEnv
from simulation.diffusion.endpoint_ik import IKSearchExhausted
from simulation.diffusion.planner import plan
from simulation.diffusion.smoothing import prepare

FIXTURES = ROOT / 'simulation/scenarios/reachable_failures.json'
STEP_RAD = .005


def dense_path(env, path):
    return np.concatenate([env.space.interpolate(a, b, STEP_RAD)
                           for a, b in zip(path[:-1], path[1:])])


def visibility(env, states, object_m):
    """Synthetic 80x80 degree pinhole and a center-point URDF ray test.

    Assumes the camera is at the flange with aligned optical Z. The ray starts
    5 mm ahead to avoid its own surface. This is not measured camera visibility,
    an object silhouette test, or a certificate between sampled configurations.
    """
    gaze, horizontal, vertical, ranges = [], [], [], []
    outside, occluded = 0, []
    try:
        for index, q in enumerate(states):
            t = ARM.transform(q)
            vector = object_m - t[:3, 3]
            distance = float(np.linalg.norm(vector))
            camera = t[:3, :3].T @ vector
            angles = np.abs(np.rad2deg(np.arctan2(camera[:2], camera[2])))
            horizontal.append(float(angles[0])); vertical.append(float(angles[1]))
            gaze.append(axis_error(t[:3, 2], vector)[2]); ranges.append(distance * 1000)
            outside += int(camera[2] <= 0 or np.any(angles > 40))
            env._set_joints(q)
            hit = p.rayTest(t[:3, 3] + .005 * vector / distance, object_m,
                            physicsClientId=env.client)[0]
            if hit[0] >= 0:
                occluded.append({'sample': index, 'body_id': hit[0], 'link_index': hit[1]})
    finally:
        env._set_joints(env._display_q)
    return {'model': 'synthetic_aligned_flange_pinhole_80x80_deg_center_point',
            'calibrated_hardware_visibility': False, 'camera_ray_start_offset_mm': 5.,
            'samples': len(states), 'outside_view_samples': outside,
            'occluded_samples': len(occluded), 'occlusion_examples': occluded[:3],
            'passed': outside == 0 and not occluded,
            'start_gaze_deg': gaze[0], 'end_gaze_deg': gaze[-1],
            'maximum_gaze_deg': max(gaze), 'maximum_horizontal_deg': max(horizontal),
            'maximum_vertical_deg': max(vertical), 'minimum_object_range_mm': min(ranges)}


def table_conflicts(env, states):
    rejected = []
    try:
        for index, q in enumerate(states):
            env._set_joints(q)
            if p.getClosestPoints(env.robot, env.table, env.clearance,
                                  physicsClientId=env.client):
                rejected.append(index)
    finally:
        env._set_joints(env._display_q)
    return rejected


def pose_error(q, target):
    t = ARM.transform(q)
    return {'position_error_mm': float(np.linalg.norm(t[:3, 3] - target[:3, 3]) * 1000),
            'direction_error_deg': axis_error(t[:3, 2], target[:3, 2])[2]}


def ik_trial(env, start, target, *, attempts, robust, learned, seed):
    settings = dict(attempts=attempts, robust=robust, learned=learned, seed=seed)
    try:
        q = env.inverse_kinematics(target[:3, 3], target[:3, 2], start, **settings)
        assert env.valid(q)
        errors = pose_error(q, target)
        assert errors['position_error_mm'] <= 1 and errors['direction_error_deg'] <= 1
        return {**settings, 'status': 'found', 'goal_rad': q.tolist(),
                **errors, 'diagnostics': dict(env.ik_diagnostics)}
    except IKSearchExhausted as exc:
        return {**settings, 'status': 'search_exhausted', 'diagnostics': dict(exc.diagnostics)}


def evaluate(env, case):
    name = case['name']; start = np.array(case['start_rad'])
    witness = np.array(case['goal_witness_rad']); obj = np.array(case['object_m'])
    object_witness = np.array(case['object_position_witness_rad'])
    target = ARM.transform(witness)
    assert env.valid(start) and env.valid(witness) and env.valid(object_witness)
    object_error = float(np.linalg.norm(ARM.transform(object_witness)[:3, 3] - obj) * 1000)
    assert object_error < 1
    assert np.allclose(obj, target[:3, 3] + .15 * target[:3, 2], atol=1e-12)
    reach = target_precheck(target[:3, 3]); assert reach['possible']
    direct = env.space.interpolate(start, witness, STEP_RAD)
    self_valid = env.space.collision_checker.valid_many(direct)
    bad = np.flatnonzero(~self_valid)
    table_bad = table_conflicts(env, direct)
    report = {**case, 'start_deg': np.rad2deg(start).tolist(),
              'goal_witness_deg': np.rad2deg(witness).tolist(),
              'object_mm': (obj * 1000).tolist(),
              'approach_target_mm': (target[:3, 3] * 1000).tolist(),
              'approach_target_axis': target[:3, 2].tolist(), 'standoff_mm': 150.,
              'goal_witness_valid': True, 'object_position_witness_valid': True,
              'object_position_witness_error_mm': object_error, 'physical_precheck': reach,
              'direct_edge': {'valid': bool(env.path_valid(direct, STEP_RAD)),
                              'samples': len(direct), 'self_rejected_samples': len(bad),
                              'table_rejected_samples': len(table_bad)}}
    panels = [('Start', start)]
    if name == 'direct_self_collision':
        intersects = [(int(i), env.space.collision_checker.check(direct[i])['conflicts'])
                      for i in bad]
        intersects = [(i, conflicts) for i, conflicts in intersects
                      if any(c['intersecting'] for c in conflicts)]
        assert intersects and not table_bad
        index, conflicts = intersects[len(intersects) // 2]
        report['direct_edge'].update(blocking_sample=index,
            interpolation_fraction=index / (len(direct) - 1),
            blocking_q_rad=direct[index].tolist(), conflicts=conflicts)
        panels.append(('Direct path: self-collision', direct[index]))
        report['diagnosis'] = 'Valid reachable endpoint; this straight joint-space edge self-collides.'
    else:
        # This case isolates an IK search miss: a valid visible direct route
        # already exists to the witness, even though the bounded search misses it.
        assert report['direct_edge']['valid']
        report['direct_edge']['visibility'] = visibility(env, direct, obj)
        assert report['direct_edge']['visibility']['passed']
        trials = [ik_trial(env, start, target, attempts=a, robust=r, learned=l, seed=case['seed'])
                  for a, r, l in [(12, False, False), (12, False, True),
                                  (32, True, False), (4, True, True)]]
        assert trials[0]['status'] == 'search_exhausted'
        assert trials[2]['status'] == 'found'
        report['ik_trials'] = trials
        report['diagnosis'] = ('A 12-attempt collision-aware IK search misses a valid branch; '
                               'a direct route to the known witness is valid. Larger search recovers.')
        # Record the actual final candidates of the same 12 seeded restarts,
        # distinguishing a stalled pose from a pose-correct colliding branch.
        rng = np.random.default_rng(case['seed'])
        starts = [start, *rng.uniform(ARM.lower, ARM.upper, (11, 6))]
        candidates = []
        for guess in starts:
            failed = solve(guess, target[:3, 3], method='bounded_dls', steps=400,
                           target_axis=target[:3, 2], goal_constraint=env.valid)
            q = np.deg2rad(failed['frames'][-1]['q'])
            candidates.append({'q_rad': q.tolist(), **pose_error(q, target),
                'valid': bool(env.valid(q)),
                'self_collision_conflicts': env.space.collision_checker.check(q)['conflicts']})
        report['bounded_ik_endpoint_candidates'] = candidates
        rejected = next(c for c in candidates if c['position_error_mm'] <= 1
                        and c['direction_error_deg'] <= 1 and not c['valid']
                        and c['self_collision_conflicts'])
        report['pose_correct_colliding_ik_endpoint'] = rejected
        panels += [('IK branch: self-collision', np.array(rejected['q_rad'])),
                   ('Recovered IK endpoint', np.array(trials[2]['goal_rad']))]
    path, diagnostics = plan(env, start, witness, seed=case['seed'])
    assert path is not None and env.path_valid(path, STEP_RAD)
    if name == 'direct_self_collision':
        assert diagnostics['status'] == 'rrt_star'
        panels.append(('RRT* detour', path[1]))
    dense = dense_path(env, path)
    view = visibility(env, dense, obj); assert view['passed']
    states, times, smoothing = prepare(env, path)
    assert env.path_valid(states, STEP_RAD)
    timed_view = visibility(env, dense_path(env, states), obj)
    assert timed_view['passed']
    report['witness_route'] = {'planner': diagnostics, 'vertices_rad': path.tolist(),
        'collision_replay_step_rad': STEP_RAD, 'collision_replay_passed': True,
        'visibility': view, 'timed_visibility': timed_view, 'smoothing': smoothing,
        'endpoint_error': pose_error(states[-1], target)}
    panels.append(('Known reachable goal', witness))
    return report, panels, (states, times)


def preview(env, reports, panels, output):
    from PIL import Image, ImageDraw, ImageFont
    width, height = 420, 330
    canvas = Image.new('RGB', (width * 4, (height + 65) * 2), 'white')
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 16)
    for row, (report, entries) in enumerate(zip(reports, panels)):
        marker = p.createVisualShape(p.GEOM_SPHERE, radius=.007,
                                    rgbaColor=[1, .45, .05, 1], physicsClientId=env.client)
        body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=marker,
                                 basePosition=report['object_m'], physicsClientId=env.client)
        for col, (label, q) in enumerate(entries):
            env.reset(q)
            conflicts = env.space.collision_checker.check(q)['conflicts']
            affected = {link for c in conflicts for link in c['links']}
            for link in range(-1, p.getNumJoints(env.robot, physicsClientId=env.client)):
                name = (ARM.base_frame if link < 0 else
                        p.getJointInfo(env.robot, link, physicsClientId=env.client)[12].decode())
                color = [1, .15, .15, 1] if name in affected else [.35, .65, .88, 1]
                p.changeVisualShape(env.robot, link, rgbaColor=color, physicsClientId=env.client)
            frame = output / f'{report["name"]}_{col}.png'
            view = p.computeViewMatrixFromYawPitchRoll([0, -.025, .10], .65, 40, -20, 0, 2)
            projection = p.computeProjectionMatrixFOV(50, width / height, .01, 3.)
            pixels = p.getCameraImage(width, height, view, projection,
                renderer=p.ER_TINY_RENDERER, physicsClientId=env.client)[2]
            Image.fromarray(np.asarray(pixels, dtype=np.uint8).reshape(height, width, 4)).save(frame)
            canvas.paste(Image.open(frame).convert('RGB'), (col * width, row * (height + 65) + 65))
            draw = ImageDraw.Draw(canvas)
            draw.text((col * width + 10, row * (height + 65) + 8), label, font=font, fill='black')
            draw.text((col * width + 10, row * (height + 65) + 33),
                      'Collision case' if row == 0 else 'Bounded IK case', font=font, fill='#555555')
        p.removeBody(body, physicsClientId=env.client)
    canvas.save(output / 'preview.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'output/reachable_failure_scenarios_20261012')
    parser.add_argument('--no-preview', action='store_true')
    args = parser.parse_args(); output = args.output; output.mkdir(parents=True, exist_ok=True)
    fixtures = json.loads(FIXTURES.read_text())
    with ArmEnv() as env:
        env.set_obstacle(None)
        reports, panels = [], []
        for case in fixtures['cases']:
            report, frames, (states, times) = evaluate(env, case)
            reports.append(report); panels.append(frames)
            np.savez_compressed(output / f'{case["name"]}_trajectory.npz',
                                joints_rad=states, times_s=times)
        sources = [Path(__file__), FIXTURES, ROOT / 'simulation/kinematics.py',
                   ROOT / 'simulation/diffusion/env.py', ROOT / 'simulation/diffusion/endpoint_ik.py',
                   ROOT / 'simulation/diffusion/planner.py', ROOT / 'simulation/diffusion/smoothing.py',
                   ROOT / 'simulation/self_collision.py', ROOT / 'simulation/pose5.py']
        result = {'schema': 'reachable_failure_replay_v1', 'hardware_motion': False,
                  'scene': env.metadata(), 'external_obstacle': None,
                  'pose_definition': '5D: position + directed tool Z; axial roll free',
                  'verification': 'Assertions passed for both fixtures; sampled collision and modeled visibility checks.',
                  'limitations': ['Conservative convex URDF hulls, not calibrated hardware geometry.',
                                 'Visibility uses a synthetic camera and point object, not detector frames.',
                                 'Dense sampled checks do not prove continuous safety.',
                                 'Recovered IK endpoints are endpoint results; their routes require separate checks.'],
                  'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in sources}, 'cases': reports}
        (output / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        if not args.no_preview:
            preview(env, reports, panels, output)
    print(json.dumps({'output': str(output), 'cases': [
        {'name': r['name'], 'direct_valid': r['direct_edge']['valid'],
         'planner_status': r['witness_route']['planner']['status'],
         'modeled_visibility_passed': r['witness_route']['timed_visibility']['passed']}
        for r in reports]}, indent=2))


if __name__ == '__main__':
    main()
