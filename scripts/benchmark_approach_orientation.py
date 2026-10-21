"""Compare strict/FOV approach on a frozen, independently reachable cohort.

Simulation only. Each planner receives starting joints and an object, never the
certificate endpoint/route. Generation does not consult either planner's result.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter, process_time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.optimize import least_squares
from simulation import continuous, profile_tracking
from simulation.camera_model import CameraModel
from simulation.diffusion import classical_tracking
from simulation.diffusion.endpoint_ik import residual_and_jacobian
from simulation.diffusion.visibility import VisibilityEnv
from simulation.kinematics import ARM
from simulation.tracking_geometry import region_for, region_metrics
from scripts.reproduce_ik_failed_reachable import verify_final_timed_route

FIXTURE = ROOT/'simulation/diffusion/tests/fixtures/ik_failed_reachable_150.json'
MODES = ('point_at_object', 'camera_fov')


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def generate(count, seed, output, fixture):
    """One difficult regression plus random final poses with direct timed proofs.

    Construct a start 20 mm behind each sampled final pose along its tool axis.
    Filter only feasibility/visibility/distinctness, never comparison outcomes.
    This samples locally approachable tasks, not a uniform workspace distribution.
    """
    camera = CameraModel(**fixture['camera'])
    known_proof = verify_final_timed_route(fixture)
    assert known_proof['passed']
    cases = [dict(id='case_01', origin='existing difficult regression',
        request=deepcopy(fixture['request']), initial_distance_mm=170.,
        feasibility_proof=dict(kind='saved full timed strict route',
            fixture=str(FIXTURE.relative_to(ROOT)), validation=known_proof))]
    rng = np.random.default_rng(seed)
    reasons = {}
    def reject(reason):
        reasons[reason] = reasons.get(reason, 0)+1
    with VisibilityEnv(camera=camera) as view:
        view.set_obstacle(None)
        for draw in range(2000):
            if len(cases) == count:
                break
            goal = rng.uniform(ARM.lower, ARM.upper)
            if not view.space.valid(goal):
                reject('final joints/self collision'); continue
            transform = ARM.transform(goal)
            xyz, axis = transform[:3, 3], transform[:3, 2]
            obj = xyz+.15*axis
            if obj[2] < .005:
                reject('object below working surface'); continue
            view.set_object(obj)
            if not view.valid(goal):
                reject('final collision/visibility'); continue
            target = xyz-.02*axis
            solved = least_squares(lambda q: residual_and_jacobian(q,target,axis)[0],goal,
                jac=lambda q: residual_and_jacobian(q,target,axis)[1],
                bounds=(ARM.lower,ARM.upper),max_nfev=160,ftol=1e-11,xtol=1e-11,gtol=1e-11)
            start = solved.x
            if np.linalg.norm(ARM.fk(start)-target)*1000 > .001:
                reject('start construction position error'); continue
            if not view.valid(start):
                reject('start collision/visibility'); continue
            delta = obj-ARM.fk(start)
            distance = float(np.linalg.norm(delta)*1000)
            body = deepcopy(fixture['request'])
            body.update(start_deg=np.rad2deg(start).tolist(),cursor_mm=(obj*1000).tolist(),
                        target_axis=(delta/np.linalg.norm(delta)).tolist())
            region = region_for(body['cursor_mm'],body['target_axis'],body['tracking_region'])
            if not region_metrics(goal,region,1.,1.)['accepted']:
                reject('strict witness misses final disk'); continue
            initial = region_for(body['cursor_mm'],body['target_axis'],
                {**body['tracking_region'],'distance_mm':round(distance,6)})
            if not region_metrics(start,initial,1.,1.)['accepted']:
                reject('initial pointing error'); continue
            if any(np.linalg.norm(obj*1000-np.asarray(c['request']['cursor_mm']))<5. or
                   np.linalg.norm(start-np.deg2rad(c['request']['start_deg']))<.05 for c in cases):
                reject('insufficiently distinct'); continue
            path = np.stack([start,goal])
            if not view.path_valid(path,.005):
                reject('direct witness route collision/visibility'); continue
            try:
                states,times,curve = profile_tracking.prepare(view,path,np.zeros(6),np.zeros(6),
                    body['speed_rad_s'],velocity_shape=body['velocity_shape'])
                frames = continuous.frames_for(states,times,xyz,axis)
                profile_tracking.validate(curve,frames,(start,np.zeros(6),np.zeros(6)),
                                          body['speed_rad_s'],view)
                assert view.path_valid(states,.005)
            except ValueError:
                reject('direct witness timing'); continue
            audit = view.audit(states,.005)
            assert audit['passed']
            case = dict(id=f'case_{len(cases)+1:02d}',origin='random reachable final pose',
                draw=draw,request=body,initial_distance_mm=round(distance,6),
                feasibility_proof=dict(kind='independently checked complete strict direct motion',
                    goal_deg=np.rad2deg(goal).tolist(),route_rad=path.tolist(),
                    curve=curve,frames=frames,visibility=audit,
                    endpoint=region_metrics(goal,region,1.,1.),duration_s=float(times[-1])))
            cases.append(case)
            write(output/'generation_progress.json',dict(accepted=len(cases),draws=draw+1,rejections=reasons))
            print(json.dumps(dict(event='generated',case=case['id'],draw=draw,
                object_mm=body['cursor_mm'],witness_duration_s=float(times[-1]))),flush=True)
    assert len(cases)==count, (len(cases),reasons)
    result = dict(seed=seed,count=count,camera=fixture['camera'],cases=cases,
        protocol='one known difficult regression and random final poses, starts constructed 20 mm behind; '
                 'feasibility-filtered before comparison; no selection by planner outcome',
        rejections=reasons)
    write(output/'scenarios.json',result)
    return result


def checked_motion(result, body, camera):
    states = np.deg2rad([f['q'] for f in result['frames']])
    with VisibilityEnv(np.asarray(body['cursor_mm'])/1000,camera=camera) as view:
        view.set_obstacle(None)
        assert view.path_valid(states,.005)
        audit = view.audit(states,.005)
        assert audit['passed']
        profile_tracking.validate(result['smoothing'],result['frames'],
            (np.deg2rad(body['start_deg']),np.zeros(6),np.zeros(6)),body['speed_rad_s'],view)
        region = region_for(body['cursor_mm'],body['target_axis'],body['tracking_region'])
        effective=result['orientation_mode']
        if body['orientation_mode']=='strict_then_fov':
            selection=result['orientation_selection']
            assert selection['selected_mode']==effective
            if effective=='camera_fov':
                assert selection['fallback_used'] and selection['strict']['playable'] is False
                assert selection['strict']['status'] in ('ik_failed','no_path')
        else:
            assert effective==body['orientation_mode']
        endpoint = region_metrics(states[-1],region,1.,1. if effective=='point_at_object' else 180.)
        if result['converged']:
            assert endpoint['accepted'] and view.valid(states[-1])
        return dict(passed=True,visibility=audit,endpoint=endpoint)


def run(case, mode, camera, repeat, output=None):
    q = np.asarray(case['request']['start_deg'])
    calls = []
    durations = 0.
    completed = []
    for stage in (case['initial_distance_mm'],150.):
        # Capture the fresh axis once per stage; prefixes keep that disk frozen.
        vector = np.asarray(case['request']['cursor_mm'])-ARM.fk(np.deg2rad(q))*1000
        axis = vector/np.linalg.norm(vector)
        for chunk in range(12):
            body = deepcopy(case['request'])
            body.update(start_deg=q.tolist(),target_axis=axis.tolist(),orientation_mode=mode,
                        _visibility_camera=camera)
            body['tracking_region']['distance_mm'] = stage
            tick=perf_counter(); cpu=process_time()
            result=classical_tracking.plan(body)
            cpu_ms=(process_time()-cpu)*1000;wall_ms=(perf_counter()-tick)*1000
            assert result['configuration']['region_candidate_budget']==8
            selection = result.get('endpoint_selection',{})
            call = dict(stage_mm=stage,chunk=chunk,status=result['status'],wall_ms=wall_ms,cpu_ms=cpu_ms,
                server_compute_ms=result['server_compute_ms'],ik_ms=result.get('ik_elapsed_ms',0),
                first_checked_route_ms=None if selection.get('first_solution_seconds') is None else
                    selection['first_solution_seconds']*1000,
                improvement_ms=selection.get('improvement_seconds',0)*1000,
                ik_jobs=result.get('region_candidate_jobs_tested',0),
                route_attempts=len(result.get('region_route_attempts',[])),
                motion_duration_s=result['frames'][-1]['time'] if result['frames'] else 0.,
                effective_orientation_mode=result['orientation_mode'],
                orientation_selection=result.get('orientation_selection'))
            if output is not None:
                dest=output/'runs'/case['id']/mode/f'repeat_{repeat}'
                dest.mkdir(parents=True,exist_ok=True)
                serial = {k:v for k,v in body.items() if not k.startswith('_')}
                write(dest/f'stage_{stage:.6f}_chunk_{chunk}.json',dict(request=serial,result=result))
            if result['playable']:
                assert result['visibility']['passed']
                call['independent_validation']=checked_motion(result,body,camera)
                durations+=call['motion_duration_s']
                q=np.asarray(result['frames'][-1]['q'])
            calls.append(call)
            if not result['playable']:
                return finish(case,mode,repeat,calls,completed,durations,result['status'])
            if result['converged'] and not result['replan_required']:
                completed.append(stage); break
        else:
            return finish(case,mode,repeat,calls,completed,durations,'chunk_limit')
    return finish(case,mode,repeat,calls,completed,durations,'completed')


def finish(case,mode,repeat,calls,completed,duration,status):
    final = [c for c in calls if c['stage_mm']==150.]
    return dict(case=case['id'],mode=mode,repeat=repeat,status=status,completed=status=='completed',
        completed_stages_mm=completed,wall_ms=sum(c['wall_ms'] for c in calls),
        final_stage_ms=sum(c['wall_ms'] for c in final),
        first_final_request_ms=final[0]['wall_ms'] if final else None,
        ik_ms=sum(c['ik_ms'] for c in calls),motion_duration_s=duration,
        planner_calls=len(calls),calls=calls)


def stats(values):
    a=np.asarray(values,dtype=float)
    if not len(a):return None
    return dict(n=len(a),median=float(np.median(a)),mean=float(np.mean(a)),
        p95=float(np.percentile(a,95)),minimum=float(a.min()),maximum=float(a.max()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'output/orientation_scenarios_20_20261015')
    parser.add_argument('--count',type=int,default=20)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--generation-seed',type=int,default=20261016)
    args=parser.parse_args();output=args.output;output.mkdir(parents=True,exist_ok=True)
    assert args.count>=2 and args.repeats>0
    if (output/'results.json').exists():raise ValueError('Choose a new output directory to retain prior results')
    fixture=json.loads(FIXTURE.read_text())
    cohort=json.loads((output/'scenarios.json').read_text()) if (output/'scenarios.json').exists() else \
        generate(args.count,args.generation_seed,output,fixture)
    assert len(cohort['cases'])==args.count
    camera=CameraModel(**cohort['camera'])
    print(json.dumps(dict(event='cohort_frozen',count=args.count,seed=cohort['seed'])),flush=True)
    for mode in MODES:
        run(cohort['cases'][0],mode,camera,-1)
    rows=[]
    for index,case in enumerate(cohort['cases']):
        for repeat in range(args.repeats):
            order=MODES if (index+repeat)%2==0 else tuple(reversed(MODES))
            for mode in order:
                row=run(case,mode,camera,repeat,output)
                rows.append(row)
                write(output/'partial_results.json',rows)
                print(json.dumps({k:row[k] for k in ('case','mode','repeat','status','wall_ms',
                    'final_stage_ms','motion_duration_s','planner_calls')}),flush=True)
    per_case=[]
    for case in cohort['cases']:
        record=dict(case=case['id'],origin=case['origin'],initial_distance_mm=case['initial_distance_mm'],
            start_deg=case['request']['start_deg'],object_mm=case['request']['cursor_mm'])
        for mode in MODES:
            group=[r for r in rows if r['case']==case['id'] and r['mode']==mode]
            record[mode]=dict(completed_runs=sum(r['completed'] for r in group),runs=len(group),
                outcomes=sorted({r['status'] for r in group}),
                total_processing_ms=stats([r['wall_ms'] for r in group]),
                final_stage_ms=stats([r['final_stage_ms'] for r in group]),
                first_final_request_ms=stats([r['first_final_request_ms'] for r in group]),
                motion_duration_s=stats([r['motion_duration_s'] for r in group if r['completed']]))
        per_case.append(record)
    summary={}
    for mode in MODES:
        group=[r for r in rows if r['mode']==mode]
        summary[mode]=dict(completed_runs=sum(r['completed'] for r in group),total_runs=len(group),
            scenarios_completed_all_repeats=sum(c[mode]['completed_runs']==args.repeats for c in per_case),
            scenarios_completed_any_repeat=sum(c[mode]['completed_runs']>0 for c in per_case),
            outcomes={s:sum(r['status']==s for r in group) for s in sorted({r['status'] for r in group})},
            across_scenario_medians=dict(total_processing_ms=stats([c[mode]['total_processing_ms']['median'] for c in per_case]),
                final_stage_ms=stats([c[mode]['final_stage_ms']['median'] for c in per_case]),
                first_final_request_ms=stats([c[mode]['first_final_request_ms']['median'] for c in per_case])),
            successful_run_processing_ms=stats([r['wall_ms'] for r in group if r['completed']]))
    both=[c for c in per_case if all(c[m]['completed_runs']==args.repeats for m in MODES)]
    comparison=dict(both_complete=len(both),
        relaxed_only=sum(c['camera_fov']['completed_runs']==args.repeats and c['point_at_object']['completed_runs']==0 for c in per_case),
        strict_only=sum(c['point_at_object']['completed_runs']==args.repeats and c['camera_fov']['completed_runs']==0 for c in per_case),
        neither_complete=sum(all(c[m]['completed_runs']==0 for m in MODES) for c in per_case),
        paired_both_complete_speed_ratio=stats([c['point_at_object']['total_processing_ms']['median']/
            c['camera_fov']['total_processing_ms']['median'] for c in both]))
    import csv
    with (output/'per_case.csv').open('w',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(['case','origin','strict_completed','relaxed_completed','runs_per_mode',
            'strict_total_median_ms','relaxed_total_median_ms','strict_final_median_ms','relaxed_final_median_ms',
            'strict_motion_median_s','relaxed_motion_median_s'])
        for c in per_case:
            a,b=c[MODES[0]],c[MODES[1]]
            writer.writerow([c['case'],c['origin'],a['completed_runs'],b['completed_runs'],args.repeats,
                a['total_processing_ms']['median'],b['total_processing_ms']['median'],
                a['final_stage_ms']['median'],b['final_stage_ms']['median'],
                a['motion_duration_s']['median'] if a['motion_duration_s'] else '',
                b['motion_duration_s']['median'] if b['motion_duration_s'] else ''])
    sources=['scripts/benchmark_approach_orientation.py','simulation/diffusion/classical_tracking.py',
        'simulation/diffusion/region_tracking.py','simulation/diffusion/streaming_ik.py',
        'simulation/diffusion/camera_endpoint_ik.py','simulation/diffusion/endpoint_ik.py',
        'simulation/diffusion/endpoint_selection.py',str(FIXTURE.relative_to(ROOT))]
    report=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        protocol=dict(scenarios=args.count,repeats_per_mode=args.repeats,generation_seed=cohort['seed'],
            planner_seed=42,alternating_order=True,candidate_points=8,
            retries_on_failure=False,max_playable_prefixes_per_stage=12,witness_supplied=False,
            selection=cohort['protocol'],camera=cohort['camera'],
            timing='warm local planner only; excludes generation, independent replay, HTTP, UI and playback',
            caveat='19 generated scenarios are locally approachable and feasibility-filtered; '
                   'one prior difficult case is intentionally included; not a uniform/random task population'),
        machine=dict(platform=platform.platform(),python=platform.python_version(),
            cpu=next(s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines()
                if s.startswith('model name')),openblas_threads=os.environ.get('OPENBLAS_NUM_THREADS')),
        source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sources},
        cohort_sha256=hashlib.sha256((output/'scenarios.json').read_bytes()).hexdigest(),
        summary=summary,comparison=comparison,per_case=per_case,independent_replay_passed=True)
    write(output/'results.json',rows);write(output/'summary.json',report)
    print(json.dumps(dict(summary=summary,comparison=comparison),indent=2),flush=True)


if __name__=='__main__':main()
