"""Replay moving cursor targets with independent scene audits and timing."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from ..kinematics import ARM
from ..service import mouse_aim_plan
from .service import info
from .env import ArmEnv
from .common import write_json


def benchmark(output, decisions=60):
    metadata=info();rows=[]
    for case in metadata['cases']:
        for kind in ('moving_nearby','distant_cursor'):
            start=np.deg2rad(case['start_deg']);pose=ARM.transform(start)
            anchor=1000*pose[:3,3]+50*pose[:3,2]
            goal=ARM.transform(np.deg2rad(case['goal_deg']))
            distant=1000*goal[:3,3]+50*goal[:3,2]
            q=start.copy();sources=Counter();rejects=Counter();timings=[];trace=[];failures=0;axis=None;hint=None
            with ArmEnv() as env:
                env.set_obstacle(np.asarray(case['obstacle_mm'])/1000)
                for step in range(decisions):
                    # Eight changing targets, then hold the last pointer still.
                    offset=np.array([4+2*np.sin(min(step,7)*.4),2+np.sin(min(step,7)*.3),1.])
                    cursor=anchor+offset if kind=='moving_nearby' else distant
                    if axis is None or (kind=='moving_nearby' and step<=7):
                        direction=cursor-ARM.fk(q)*1000;axis=(direction/np.linalg.norm(direction)).tolist()
                    body={'tracking_engine':'local_diffusion','start_deg':np.rad2deg(q).tolist(),
                        'cursor_mm':cursor.tolist(),'obstacle_mm':case['obstacle_mm'],
                        'pose5_method':'bounded_dls','seed':42+step,'target_axis':axis}
                    if hint is not None:body['goal_hint_deg']=hint
                    result=mouse_aim_plan(body);timings.append(result['server_compute_ms'])
                    sources[result.get('tracking_source','none')]+=1;rejects.update(result['rejections'])
                    trace.append({'cursor_mm':cursor.tolist(),'status':result['status'],
                                  'source':result.get('tracking_source'),'compute_ms':timings[-1],
                                  'frames':result['frames']})
                    if not result['playable']:
                        failures+=1
                        if result['status']!='no_safe_chunk' or failures>=5:break
                        continue
                    failures=0;states=np.deg2rad([f['q'] for f in result['frames']])
                    assert np.allclose(states[0],q,atol=1e-10) and env.path_valid(states)
                    times=np.array([f['time'] for f in result['frames']])
                    if len(states)>1:assert np.all(np.linalg.norm(np.diff(states,axis=0)/np.diff(times)[:,None],axis=1)<=1+1e-9)
                    from .smoothing import motion_bounds
                    assert motion_bounds(states,times)[1]<=3+1e-8
                    q=states[-1];hint=result['goal_deg']
                    if result['converged'] and (kind!='moving_nearby' or step>=7):break
            row={'case':case['id'],'kind':kind,'success':result['converged'],
                 'status':result['status'],'decisions':len(trace),'sources':dict(sources),'rejections':dict(rejects),
                 'latency_ms':{'first':timings[0],'median':float(np.median(timings)),'p95':float(np.percentile(timings,95))},
                 'final_cursor_acceptance':result.get('cursor_acceptance'),'trace':trace}
            rows.append(row);print({k:v for k,v in row.items() if k!='trace'},flush=True)
    report={'kind':'experimental_hybrid_mouse_tracking','checkpoint_sha256':metadata['checkpoint_sha256'],
        'configuration':{'decisions':decisions,'horizon':16,'execute_steps':2,'max_smoothed_chunk_s':1.3,'acceleration_rad_s2':3.,'samples':4,'inference_steps':32},
        'rows':rows,'independent_scene_and_speed_audit_passed':True,
        'limitations':'Six deterministic cursor scenarios, not a general tracking benchmark. Fixed-goal checkpoint. Holds between queries; acceleration bounded on smoothed references; interruptions can stop early; no hardware.'}
    write_json(output,report);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('output/diffusion/mouse_tracking_benchmark.json'))
    parser.add_argument('--decisions',type=int,default=60)
    args=parser.parse_args();benchmark(args.output,args.decisions)
