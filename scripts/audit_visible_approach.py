"""Replay saved fake-motor approach plans in fresh collision/visibility scenes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulation.camera_model import CameraModel
from simulation.diffusion.visibility import VisibilityEnv
from simulation.tracking_geometry import region_metrics


def audit(trace):
    if trace.get('physical_motion') is not False:
        raise ValueError('Expected a fake-motor approach trace')
    camera = CameraModel(**trace['camera'])
    rows = []
    for index, item in enumerate(trace['plans']):
        body, result = item['request'], item['result']
        if not result.get('playable'):
            continue
        if result['visibility_model']['camera'] != camera.configuration():
            raise ValueError('Planner camera differs from pinned trace camera')
        frames = np.deg2rad([frame['q'] for frame in result['frames']])
        route = np.asarray(result['planned_route_rad'])
        start = np.deg2rad(body['start_deg'])
        if not (np.allclose(route[0], start, atol=1e-10, rtol=0)
                and np.allclose(frames[0], start, atol=1e-10, rtol=0)
                and np.allclose(route[-1], np.deg2rad(result['goal_deg']), atol=1e-10, rtol=0)):
            raise ValueError('Trace route/start/goal mismatch')
        goal = region_metrics(route[-1], result['tracking_region'], 1., 1.)
        if not goal['accepted']:
            raise ValueError('Geometric route ends outside its accepted region')
        with VisibilityEnv(np.asarray(body['cursor_mm'])/1000, camera=camera) as env:
            env.set_obstacle(None)
            if not env.path_valid(route, .005) or not env.path_valid(frames, .005):
                raise ValueError(f'Plan {index} failed fresh dense scene replay')
            geometric = env.audit(route, .005)
            timed = env.audit(frames, .005)
            if not geometric['passed'] or not timed['passed']:
                raise ValueError(f'Plan {index} failed fresh visibility replay')
        rows.append({'plan_index':index, 'distance_mm':body['tracking_region']['distance_mm'],
            'status':result['status'], 'duration_s':result['frames'][-1]['time'],
            'joint_travel_rad':float(np.linalg.norm(np.diff(route,axis=0),axis=1).sum()),
            'geometric':geometric, 'timed':timed, 'goal':goal})
    final = trace['final_status']
    q = np.deg2rad(final['samples'][-1]['q'])
    acceptance = region_metrics(q, final['tracking_region'], 10., 3.)
    if not acceptance['accepted'] or final['state'] != 'completed':
        raise ValueError('Final measured pose did not complete the approach')
    with VisibilityEnv(np.asarray(final['tracking_region']['cursor_mm'])/1000, camera=camera) as env:
        env.set_obstacle(None)
        if not env.valid(q):
            raise ValueError('Final measured pose fails fresh scene/visibility replay')
    encoder_audit = None
    if 'encoder_path_deg' in trace:
        encoder_path = np.deg2rad(trace['encoder_path_deg'])
        if encoder_path.ndim != 2 or encoder_path.shape[1] != 6 or not np.isfinite(encoder_path).all():
            raise ValueError('Invalid recorded encoder path')
        with VisibilityEnv(np.asarray(trace['object_mm'])/1000, camera=camera) as env:
            env.set_obstacle(None)
            if not env.path_valid(encoder_path, .005):
                raise ValueError('Recorded rounded encoder path failed fresh scene replay')
            encoder_audit = env.audit(encoder_path, .005)
            if not encoder_audit['passed']:
                raise ValueError('Recorded rounded encoder path failed fresh visibility replay')
            encoder_audit['commanded_poses'] = len(encoder_path)
    return {'passed':True, 'camera':camera.configuration(), 'physical_motion':False,
        'plan_count':len(rows), 'stages_mm':sorted({r['distance_mm'] for r in rows}, reverse=True),
        'sample_count':sum(r[k]['samples'] for r in rows for k in ('geometric','timed')),
        'minimum_view_margin_deg':min(r[k]['minimum_view_margin_deg'] for r in rows for k in ('geometric','timed')),
        'final_measured_acceptance':acceptance, 'plans':rows,
        'encoder_path_audit':encoder_audit,
        'limitations':'Point-object and assumed FOV; sampled geometric checks, fake motors. No physical camera or motor qualification.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; preserve the earlier audit')
    report = audit(json.loads(args.trace.read_text()))
    report['trace'] = str(args.trace)
    report['trace_sha256'] = hashlib.sha256(args.trace.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('passed','plan_count','stages_mm','sample_count','minimum_view_margin_deg')}, indent=2))


if __name__ == '__main__':
    main()
