"""Offline paired transformer vs pinned motion-v2 comparison; no hardware IO.

Frozen-model held-out suites share the same C2 controller and acceptance gates.
Extra full service checks include learned IK, whole-path checks and smoothing.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
import h5py
import numpy as np
import torch
from ..kinematics import ARM
from .common import file_sha256, write_json, observation
from .compare_motion import tasks, motion_suite
from .compare_continuous import fixed_motion_tasks, fixed_duration_metrics
from .env import ArmEnv
from .motion_conditioning import encode
from .policy import DiffusionPolicy
from .trajectory_transformer import TransformerPolicy

BASELINE=Path('simulation/artifacts/diffusion_motion_v2/model/diffusion_policy_6dof.pth')
PIN='6582cbf4e1d2575007cfead8f99ed5f15a7b7021083d0eb200c61b224401bddc'
SOURCE=Path('simulation/artifacts/diffusion_100k_v1/demonstrations.h5')
AUGMENT=Path('simulation/artifacts/diffusion_motion_v2/augmentation.h5')


def protected_tasks(path,count_per_kind=5):
    rng=np.random.default_rng(202610084);result=[]
    with h5py.File(path,'r') as data:
        kinds={}
        for name,g in data['episodes'].items():
            if g.attrs['split']=='test':kinds.setdefault(json.loads(g.attrs['record_json'])['kind'],[]).append(name)
        for kind,names in sorted(kinds.items()):
            for name in rng.choice(names,min(count_per_kind,len(names)),replace=False):
                g=data['episodes'][name]
                result.append({'id':'protected_'+name,'kind':kind,'q':g['q_rad'][:],
                               'goal':g['goal_rad'][-1], 'box':g['obstacle_m'][:] if g.attrs['obstacle_present'] else None})
    return result


def latency(policies, task):
    obs=encode(policies['baseline'],observation(task['q'][0],task['goal'],task['box']))[None]
    for policy in policies.values():
        for _ in range(5):policy.sample(obs,4,32)
    timings={k:[] for k in policies};blocks=[]
    for block in range(3):
        local={k:[] for k in policies}
        for i in range(20):
            order=list(policies) if (block+i)%2==0 else list(reversed(policies))
            for key in order:
                torch.cuda.synchronize();began=perf_counter();policies[key].sample(obs,4,32);torch.cuda.synchronize()
                value=(perf_counter()-began)*1000;timings[key].append(value);local[key].append(value)
        blocks.append({k:float(np.median(v)) for k,v in local.items()})
    return {'device':torch.cuda.get_device_name(),'pairs':60,'alternating_order':True,'blocks':blocks,
            'summary':{k:{'median_ms':float(np.median(v)),'p95_ms':float(np.percentile(v,95))} for k,v in timings.items()},
            'measurements_ms':timings,'scope':'synchronized GPU; one observation; four proposals; baseline 32 DDPM steps and enabled production CUDA graph; candidate one decoder pass; inference and NumPy conversion only'}


def full_pipeline(policies, selected):
    from . import service
    rows=[];status=service.info()
    for task in selected:
        transform=ARM.transform(task['goal']);body={'start_deg':np.rad2deg(task['q'][0]).tolist(),
            'target_mm':(transform[:3,3]*1000).tolist(),'target_axis':transform[:3,2].tolist(),
            'obstacle_mm':None if task['box'] is None else (task['box']*1000).tolist(),
            'seed':42,'samples':4,'inference_steps':32,'max_decisions':200}
        row={'id':task['id'],'kind':task['kind']}
        for name,policy in policies.items():
            effective={**status,'checkpoint_sha256':policy.sha256,'model_name':name}
            def load(_status,seed):policy.generator.manual_seed(seed);return policy
            began=perf_counter()
            with patch.object(service,'info',return_value=effective),patch.object(service,'load_policy',side_effect=load),patch.object(service,'_policy',policy):
                result=service.plan(body)
            row[name]={k:result.get(k) for k in ('status','converged','final_error_mm','final_orientation_error_deg','ik_diagnostics','max_velocity_rad_s','max_acceleration_rad_s2')}
            row[name]['elapsed_ms']=(perf_counter()-began)*1000
            if result.get('converged'):
                q=np.deg2rad([f['q'] for f in result['frames']])
                with ArmEnv() as env:
                    env.set_obstacle(task['box'])
                    if not env.path_valid(q,.01):raise RuntimeError('Independent complete-plan replay failed')
                row[name]['independent_path_replay']=True
            else:row[name]['independent_path_replay']=None
        rows.append(row);print(json.dumps({'pipeline':row['id'],'baseline':row['baseline']['status'],'candidate':row['candidate']['status']}),flush=True)
    return rows


def moving_from_rest(candidate, output):
    """Diagnostic suite: same held-out moving targets, zero incoming motion.

    Keeps controller gates intact when historical labels cannot satisfy the
    current envelope. Models remain frozen; this is not a tuning dataset.
    """
    output = Path(output)
    report = json.loads(output.read_text())
    if 'moving_from_rest' in report['results']['candidate']:
        raise ValueError('Moving-from-rest result already recorded')
    if (file_sha256(BASELINE) != report['results']['baseline']['checkpoint_sha256']
            or file_sha256(candidate) != report['results']['candidate']['checkpoint_sha256']):
        raise ValueError('Frozen comparison model changed')
    _, moving = tasks(SOURCE, AUGMENT)
    selected = [{**task, 'id':'rest_'+task['id'],
                 'velocities':np.zeros_like(task['velocities']),
                 'accelerations':np.zeros_like(task['accelerations'])} for task in moving]
    policies = {'baseline':DiffusionPolicy(BASELINE,'cuda',42,cuda_graph=True),
                'candidate':TransformerPolicy(candidate,'cuda',42)}
    for name, policy in policies.items():
        policy.generator.manual_seed(202610088)
        measured = motion_suite(policy,selected,120)
        fixed_duration_metrics(measured,selected,120)
        report['results'][name]['moving_from_rest'] = measured
        write_json(output.with_name(output.stem+f'_{name}_moving_from_rest.json'),measured)
        print(json.dumps({'method':name,'suite':'moving_from_rest','summary':measured['summary']}),flush=True)
    report['checks']['moving_from_rest_success_not_lower'] = (report['results']['candidate']['moving_from_rest']['summary']['successes']
        >= report['results']['baseline']['moving_from_rest']['summary']['successes'])
    report['checks']['tracking_error_not_higher'] = (report['results']['candidate']['moving_from_rest']['summary']['mean_tracking_rmse_mm']
        <= report['results']['baseline']['moving_from_rest']['summary']['mean_tracking_rmse_mm'])
    report['offline_replacement_gates_passed'] = all(report['checks'].values())
    report['configuration']['moving_from_rest_is_diagnostic'] = True
    report['limitations'] += ' Original moving suite rejected 17/24 shared initial braking continuations before policy evaluation; 16 historical incoming labels exceed the present conservative motion envelope. Moving-from-rest diagnostic evaluates all 24 unchanged target sequences with zero initial joint velocity/acceleration and unchanged controller checks.'
    report['environment_record'] = 'output/trajectory_transformer_environment_20261008.json'
    report['comparison_source_sha256'] = file_sha256(Path(__file__))
    write_json(output,report)
    return report


def compare(candidate,protected,output):
    output=Path(output)
    if output.exists():raise ValueError('Comparison output already exists')
    if file_sha256(BASELINE)!=PIN:raise ValueError('Latest hardware diffusion checkpoint changed')
    if not torch.cuda.is_available():raise RuntimeError('GPU comparison requires CUDA')
    torch.set_num_threads(4)
    fixed,moving=tasks(SOURCE,AUGMENT);extra=protected_tasks(protected)
    policies={'baseline':DiffusionPolicy(BASELINE,'cuda',42,cuda_graph=True),
              'candidate':TransformerPolicy(candidate,'cuda',42)}
    timing=latency(policies,fixed[0]);write_json(output.with_name(output.stem+'_latency.json'),timing)
    results={name:{'checkpoint_sha256':p.sha256,'training':p.training} for name,p in policies.items()}
    suites=[('fixed',fixed_motion_tasks(fixed),200),('moving',moving,120),('protected',fixed_motion_tasks(extra),200)]
    for index,(kind,selected,budget) in enumerate(suites):
        order=['baseline','candidate'] if index%2==0 else ['candidate','baseline']
        for name in order:
            policy=policies[name];policy.generator.manual_seed(202610085+index)
            measured=motion_suite(policy,selected,budget)
            fixed_duration_metrics(measured,selected,budget)
            measured['summary']['by_kind']={k:{'episodes':sum(r['kind']==k for r in measured['rows']),
                'successes':sum(r['kind']==k and r['success'] for r in measured['rows'])} for k in sorted(set(t['kind'] for t in selected))}
            results[name][kind]=measured
            write_json(output.with_name(output.stem+f'_{name}_{kind}.json'),measured)
            print(json.dumps({'method':name,'suite':kind,'summary':measured['summary']}),flush=True)
    # Four original and four new tasks span block/no-block/near-boundary families.
    selected=[fixed[i] for i in (0,16,32,48)]+[next(t for t in extra if t['kind']==kind) for kind in sorted(set(t['kind'] for t in extra))]
    pipeline=full_pipeline(policies,selected)
    checks={'success_not_lower':all(results['candidate'][k]['summary']['successes']>=results['baseline'][k]['summary']['successes'] for k,_,_ in suites),
            'executed_collision_free':all(results['candidate'][k]['summary']['execution_collisions']==0 for k,_,_ in suites),
            'motion_bounds':all(results['candidate'][k]['summary']['max_velocity_rad_s']<=1+1e-6 and results['candidate'][k]['summary']['max_acceleration_rad_s2']<=3+1e-6 for k,_,_ in suites),
            'c2_boundaries':all(results['candidate'][k]['summary']['max_boundary_velocity_error']<1e-6 and results['candidate'][k]['summary']['max_boundary_acceleration_error']<1e-6 for k,_,_ in suites),
            'complete_plan_success_not_lower':sum(r['candidate']['converged'] for r in pipeline)>=sum(r['baseline']['converged'] for r in pipeline),
            'gpu_inference_faster':timing['summary']['candidate']['median_ms']<timing['summary']['baseline']['median_ms']}
    report={'kind':'frozen_one_pass_transformer_vs_active_motion_v2','results':results,'gpu_inference_latency':timing,
            'full_pipeline':pipeline,'checks':checks,'offline_replacement_gates_passed':all(checks.values()),
            'dataset_sha256':{'original':file_sha256(SOURCE),'motion':file_sha256(AUGMENT),'protected':file_sha256(protected)},
            'configuration':{'samples':4,'baseline_ddpm_steps':32,'candidate_decoder_passes':1,'fixed_and_protected_decisions':200,'moving_decisions':120,'controller':'shared checked C2','deployment_changed':False},
            'limitations':'One candidate training seed; architecture AND training-data/budget differ. Held-out sets were not used for checkpoint selection. Reuses historical task corpus, whose old baseline results were known. Virtual kinematic control; compute delay excluded from tracking metrics. No physical qualification or hardware motion. Lower executed-collision count does not imply safe raw proposals.'}
    write_json(output,report);print(json.dumps({'checks':checks,'latency':timing['summary']}),flush=True)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,default=Path('simulation/artifacts/trajectory_transformer_v1/model/trajectory_transformer.pth'))
    parser.add_argument('--protected',type=Path,default=Path('simulation/artifacts/collision_protected_v1/training_1000.h5'))
    parser.add_argument('--output',type=Path,default=Path('output/trajectory_transformer_comparison_20261008.json'))
    parser.add_argument('--moving-from-rest-only',action='store_true',help='Append diagnostic moving targets starting from rest to an existing frozen comparison')
    args=parser.parse_args()
    if args.moving_from_rest_only: moving_from_rest(args.candidate,args.output)
    else:
        compare(args.candidate,args.protected,args.output)
        moving_from_rest(args.candidate,args.output)
