"""Normal fixed-pose Learned IK + RRT* versus hybrid staged approaching.

Use the frozen approach cohort. The normal production API is unmodified: one
strict 150 mm standoff point at the initial disk center, 32-attempt learned IK,
default RRT*, and no staged disks/FOV fallback. Separately qualify its returned
route under the approach camera and shared motion profile. No witness hints or
hardware IO. Native time and common qualification time are reported separately.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter,process_time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from simulation import continuous,profile_tracking
from simulation.camera_model import CameraModel
from simulation.diffusion import classical_service,service
from simulation.diffusion.visibility import VisibilityEnv
from simulation.kinematics import ARM
from scripts.benchmark_approach_orientation import run as approach_run,stats,write


def normal_request(case):
    start=case['request']['start_deg']
    delta=np.asarray(case['request']['cursor_mm'])-ARM.fk(np.deg2rad(start))*1000
    axis=delta/np.linalg.norm(delta)
    return dict(start_deg=list(start),target_mm=(np.asarray(case['request']['cursor_mm'])-150*axis).tolist(),
        target_axis=axis.tolist(),obstacle_mm=case['request']['obstacle_mm'],seed=case['request']['seed'])


def qualify(native,request,case,camera):
    """Audit actual native geometry and retime it without another IK/RRT search."""
    states=np.deg2rad([f['q'] for f in native['frames']])
    with VisibilityEnv(np.asarray(case['request']['cursor_mm'])/1000,camera=camera) as view:
        view.set_obstacle(None)
        assert view.metadata()==native['environment']
        visibility=view.audit(states,.005)
        scene_passed=bool(view.path_valid(states,.005))
        if not visibility['passed'] or not scene_passed:
            return dict(passed=False,status='camera_or_dense_scene_rejected',native_visibility=visibility,
                        dense_scene_passed=scene_passed,references=[])
        # These corners are the actually returned smoothed normal route, not
        # the saved scenario proof or an independently selected endpoint.
        path=np.array([states[0],*[np.deg2rad(s['end_deg']) for s in native['smoothing']['segments']]])
        assert np.allclose(path[-1],states[-1],atol=1e-12,rtol=0)
        references=[]
        speed=case['request']['speed_rad_s'];shape=case['request']['velocity_shape'];zero=np.zeros(6)
        def reference(points):
            q,times,curve=profile_tracking.prepare(view,points,zero,zero,speed,velocity_shape=shape)
            frames=continuous.frames_for(q,times,np.asarray(request['target_mm'])/1000,np.asarray(request['target_axis']))
            assert view.path_valid(q,.005)
            audit=view.audit(q,.005);assert audit['passed']
            profile_tracking.validate(curve,frames,(np.asarray(points[0]),zero,zero),speed,view)
            references.append(dict(curve=curve,frames=frames,visibility=audit))
        split=False
        try:
            try:reference(path)
            except ValueError:
                # A long complete path can require several <=12 s references.
                # No standoff stages, endpoint retries or new goals are added.
                split=True
                for a,b in zip(path[:-1],path[1:]):
                    pieces=max(1,int(np.ceil(np.linalg.norm(b-a)/.4)))
                    for i in range(pieces):reference(np.stack([a+(b-a)*i/pieces,a+(b-a)*(i+1)/pieces]))
            final=np.deg2rad(references[-1]['frames'][-1]['q'])
            assert np.allclose(final,states[-1],atol=1e-8,rtol=0)
            endpoint=service.errors(final,np.asarray(request['target_mm'])/1000,np.asarray(request['target_axis']))
            assert endpoint['error_mm']<=1. and endpoint['orientation_error_deg']<=1.
        except ValueError as exc:
            return dict(passed=False,status='shared_motion_profile_failed',reason=str(exc),
                native_visibility=visibility,dense_scene_passed=scene_passed,references=references)
        return dict(passed=True,status='completed',native_visibility=visibility,dense_scene_passed=True,
            shared_profile_split=split,references=references,endpoint=endpoint,
            common_motion_duration_s=sum(x['curve']['duration_s'] for x in references),
            joint_travel_rad=float(np.linalg.norm(np.diff(states,axis=0),axis=1).sum()))


def run_normal(case,camera,repeat,output=None):
    request=normal_request(case)
    assert not {'goal_deg','tracking_region','orientation_mode','cursor_mm'} & request.keys()
    began=perf_counter();cpu=process_time()
    native=classical_service.plan(request)
    api_ms=(perf_counter()-began)*1000
    assert native['configuration']['ik_attempts']==32 and native['configuration']['robust_ik']
    assert native['configuration']['learned_ik']
    qualification=None;post_ms=0.
    if native['status']=='success':
        tick=perf_counter();qualification=qualify(native,request,case,camera);post_ms=(perf_counter()-tick)*1000
    total=(perf_counter()-began)*1000;cpu_ms=(process_time()-cpu)*1000
    complete=bool(qualification and qualification['passed'])
    row=dict(case=case['id'],mode='normal_fixed_pose',repeat=repeat,
        status='completed' if complete else qualification['status'] if qualification else native['status'],
        completed=complete,native_status=native['status'],native_success=native['status']=='success',
        native_api_ms=api_ms,common_qualification_ms=post_ms,total_checked_ms=total,cpu_ms=cpu_ms,
        native_ik_ms=native.get('ik_elapsed_ms',native.get('ik_diagnostics',{}).get('elapsed_ms')),
        native_motion_duration_s=native['frames'][-1]['time'] if native['frames'] else None,
        common_motion_duration_s=qualification.get('common_motion_duration_s') if qualification else None,
        native_visibility=qualification.get('native_visibility') if qualification else None,
        common_reference_count=len(qualification['references']) if qualification else 0)
    if output is not None:
        dest=output/'runs'/case['id']/'normal_fixed_pose'/f'repeat_{repeat}'
        dest.mkdir(parents=True,exist_ok=True)
        write(dest/'normal.json',dict(request=request,native_result=native,qualification=qualification))
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenarios',type=Path,default=ROOT/'output/orientation_scenarios_20_20261015/scenarios.json')
    parser.add_argument('--output',type=Path,default=ROOT/'output/normal_vs_approach_20_20261015')
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists():raise ValueError('Choose a new output directory to preserve previous results')
    cohort=json.loads(args.scenarios.read_text());camera=CameraModel(**cohort['camera'])
    assert len(cohort['cases'])==20 and args.repeats>0
    run_normal(cohort['cases'][1],camera,-1)
    approach_run(cohort['cases'][1],'strict_then_fov',camera,-1)
    rows=[]
    for index,case in enumerate(cohort['cases']):
        for repeat in range(args.repeats):
            modes=('normal_fixed_pose','strict_then_fov')
            for mode in modes if (index+repeat)%2==0 else tuple(reversed(modes)):
                if mode=='normal_fixed_pose':row=run_normal(case,camera,repeat,out)
                else:
                    raw=approach_run(case,mode,camera,repeat,out)
                    row=dict(case=case['id'],mode=mode,repeat=repeat,status=raw['status'],
                        completed=raw['completed'],total_checked_ms=raw['wall_ms'],
                        common_motion_duration_s=raw['motion_duration_s'] if raw['completed'] else None,
                        calls=raw['calls'],fallback_used=any(c.get('orientation_selection',{}).get('fallback_used',False)
                            for c in raw['calls'] if c.get('orientation_selection')))
                rows.append(row);write(out/'partial_results.json',rows)
                print(json.dumps({k:row.get(k) for k in ('case','mode','repeat','status','native_status',
                    'native_api_ms','common_qualification_ms','total_checked_ms','common_motion_duration_s')}),flush=True)
    per_case=[]
    for case in cohort['cases']:
        item=dict(case=case['id'],origin=case['origin'],normal_request=normal_request(case))
        for mode in ('normal_fixed_pose','strict_then_fov'):
            group=[r for r in rows if r['case']==case['id'] and r['mode']==mode]
            item[mode]=dict(completed_runs=sum(r['completed'] for r in group),total_runs=len(group),
                outcomes=sorted({r['status'] for r in group}),
                total_checked_ms=stats([r['total_checked_ms'] for r in group]),
                common_motion_duration_s=stats([r['common_motion_duration_s'] for r in group if r['completed']]))
            if mode=='normal_fixed_pose':
                item[mode].update(native_successes=sum(r['native_success'] for r in group),
                    native_api_ms=stats([r['native_api_ms'] for r in group]),
                    common_qualification_ms=stats([r['common_qualification_ms'] for r in group]))
        per_case.append(item)
    summary={}
    for mode in ('normal_fixed_pose','strict_then_fov'):
        group=[r for r in rows if r['mode']==mode]
        summary[mode]=dict(completed_runs=sum(r['completed'] for r in group),runs=len(group),
            scenarios_completed_all_repeats=sum(c[mode]['completed_runs']==args.repeats for c in per_case),
            outcomes={s:sum(r['status']==s for r in group) for s in sorted({r['status'] for r in group})},
            across_scenario_medians=dict(total_checked_ms=stats([c[mode]['total_checked_ms']['median'] for c in per_case]),
                common_motion_duration_s=stats([c[mode]['common_motion_duration_s']['median'] for c in per_case
                    if c[mode]['common_motion_duration_s']])) )
        if mode=='normal_fixed_pose':
            summary[mode].update(native_successful_runs=sum(r['native_success'] for r in group),
                native_api_ms=stats([c[mode]['native_api_ms']['median'] for c in per_case]))
    sources=['scripts/benchmark_normal_vs_approach.py','scripts/benchmark_approach_orientation.py',
        'simulation/diffusion/classical_service.py','simulation/diffusion/classical_tracking.py',
        'simulation/diffusion/orientation_policy.py','simulation/profile_tracking.py']
    report=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        protocol=dict(scenarios=20,repeats_per_mode=args.repeats,seed=42,alternating_order=True,
            scenario_file=str(args.scenarios),scenario_sha256=hashlib.sha256(args.scenarios.read_bytes()).hexdigest(),
            camera=cohort['camera'],normal='unmodified classical_service.plan, one exact 150 mm center pose',
            approach='strict_then_fov with frozen stage disks and actual initial distance',
            goal_difference='normal must reach center; approach can reach any checked point on the 150 mm disk',
            budget_difference='normal: 32 IK restarts + robust fallback for one pose; approach: 8 points, 2 branch passes, 4 restarts per job, plus relaxed pass only after strict failure',
            native_motion='normal native API uses its default 1 rad/s quintic timing; native metrics are reported separately',
            common_motion='retime the returned normal route to the approach speed/profile and validate camera visibility, scene and each <=12 s reference; no new IK goals/standoff stages',
            timing='normal total includes native API plus qualification/retiming; approach total is production planning (already checks camera/profile); IO/playback excluded',
            witness_supplied=False,hardware_used=False),
        machine=dict(cpu=next(s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines()
            if s.startswith('model name')),python=platform.python_version(),openblas_threads=os.environ.get('OPENBLAS_NUM_THREADS')),
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
        summary=summary,per_case=per_case)
    import csv
    with (out/'per_case.csv').open('w',newline='') as handle:
        w=csv.writer(handle);w.writerow(['case','normal_native_successes','normal_checked_completed','hybrid_completed',
            'runs_per_mode','normal_native_median_ms','normal_checked_median_ms','hybrid_median_ms',
            'normal_common_motion_s','hybrid_motion_s'])
        for c in per_case:
            a,b=c['normal_fixed_pose'],c['strict_then_fov']
            w.writerow([c['case'],a['native_successes'],a['completed_runs'],b['completed_runs'],args.repeats,
                a['native_api_ms']['median'],a['total_checked_ms']['median'],b['total_checked_ms']['median'],
                a['common_motion_duration_s']['median'] if a['common_motion_duration_s'] else '',
                b['common_motion_duration_s']['median'] if b['common_motion_duration_s'] else ''])
    write(out/'results.json',rows);write(out/'summary.json',report)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
