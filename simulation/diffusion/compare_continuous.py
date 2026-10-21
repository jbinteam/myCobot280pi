"""Paired model comparison under the production full-horizon C2 controller."""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import torch
from .common import file_sha256,write_json
from .policy import DiffusionPolicy
from .compare_motion import tasks,motion_suite,cpu_latency

SOURCE=Path('simulation/artifacts/diffusion_100k_v1/demonstrations.h5')
AUGMENTATION=Path('simulation/artifacts/diffusion_motion_v2/augmentation.h5')
BASELINE=Path('simulation/artifacts/diffusion_v1/model/diffusion_policy_6dof.pth')
CANDIDATE=Path('simulation/artifacts/diffusion_motion_v2/model/diffusion_policy_6dof.pth')
OUTPUT=Path('output/diffusion/motion_v2_continuous_comparison.json')


def fixed_motion_tasks(fixed):
    result=[]
    for task in fixed:
        result.append({**task,'goals':np.repeat(task['goal'][None],2,axis=0),'times':np.array([0.,.1]),
                       'velocities':np.zeros((2,6)),'accelerations':np.zeros((2,6)),
                       'target_velocities':np.zeros((2,3))})
    return result


def fixed_duration_metrics(result,tasks,max_decisions):
    """Compare equal observation duration; early successes keep holding the goal.

    Averaging only until each policy finishes penalizes faster successes by
    excluding their subsequent low-error hold. Preserve that diagnostic too.
    """
    from .eval import errors
    for row,task in zip(result['rows'],tasks):
        trace=list(row['tracking_error_mm'])
        row['until_completion_tracking_rmse_mm']=row['tracking_rmse_mm']
        if row['status']=='initial_continuation_rejected':
            trace=[errors(task['q'][0],task['goals'][min(int(round(step*.4/.1)),len(task['goals'])-1)])[0]
                   for step in range(max_decisions+1)]
        else:trace+=([row['final_error_mm']]*max(0,max_decisions+1-len(trace)))
        row['fixed_duration_tracking_error_mm']=trace
        row['tracking_rmse_mm']=float(np.sqrt(np.mean(np.square(trace))))
    summary=result['summary']
    summary['mean_until_completion_tracking_rmse_mm']=summary['mean_tracking_rmse_mm']
    summary['mean_tracking_rmse_mm']=float(np.mean([r['tracking_rmse_mm'] for r in result['rows']]))
    summary['tracking_measurement_duration_s']=max_decisions*.4
    summary['initial_continuation_rejections']=sum(r['status']=='initial_continuation_rejected' for r in result['rows'])


def run(name):
    fixed,moving=tasks(SOURCE,AUGMENTATION);checkpoint=BASELINE if name=='baseline' else CANDIDATE
    policy=DiffusionPolicy(checkpoint,'cuda',20261010)
    policy.generator.manual_seed(20261010);fixed_result=motion_suite(policy,fixed_motion_tasks(fixed),100)
    summary=fixed_result['summary'];rows=fixed_result['rows']
    summary['by_kind']={kind:{'episodes':sum(r['kind']==kind for r in rows),'successes':sum(r['success'] and r['kind']==kind for r in rows)} for kind in ('block','static_absent')}
    summary['blocked_successes']=sum(r['success'] and t.get('blocked',False) for r,t in zip(rows,fixed))
    policy.generator.manual_seed(20261011);motion_result=motion_suite(policy,moving,120)
    result={'checkpoint_sha256':policy.sha256,'observation_dim':policy.observation_dim,
            'source_sha256':file_sha256(SOURCE),'augmentation_sha256':file_sha256(AUGMENTATION),
            'fixed':fixed_result,'motion':motion_result}
    del policy;gc.collect();torch.cuda.empty_cache();result['cpu_inference_latency']=cpu_latency(checkpoint,fixed[0])
    write_json(OUTPUT.with_name(f'continuous_{name}_comparison.json'),result)
    print(json.dumps({kind:result[kind]['summary'] for kind in ('fixed','motion')}),flush=True);return result


def merge():
    fixed,moving=tasks(SOURCE,AUGMENTATION);results={}
    for name,checkpoint in [('baseline',BASELINE),('candidate',CANDIDATE)]:
        result=json.loads(OUTPUT.with_name(f'continuous_{name}_comparison.json').read_text())
        if (result['checkpoint_sha256']!=file_sha256(checkpoint) or result['source_sha256']!=file_sha256(SOURCE)
                or result['augmentation_sha256']!=file_sha256(AUGMENTATION)
                or [r['id'] for r in result['fixed']['rows']]!=[r['id'] for r in fixed]
                or [r['id'] for r in result['motion']['rows']]!=[r['id'] for r in moving]):raise ValueError('Continuous comparison provenance differs')
        fixed_duration_metrics(result['fixed'],fixed_motion_tasks(fixed),100)
        fixed_duration_metrics(result['motion'],moving,120)
        result['cpu_inference_latency']=cpu_latency(checkpoint,fixed[0]);results[name]=result
    old=results['baseline'];new=results['candidate'];checks={
        'fixed_success_not_lower':new['fixed']['summary']['successes']>=old['fixed']['summary']['successes'],
        'no_block_success_improves':new['fixed']['summary']['by_kind']['static_absent']['successes']>old['fixed']['summary']['by_kind']['static_absent']['successes'],
        'motion_success_not_lower':new['motion']['summary']['successes']>=old['motion']['summary']['successes'],
        'tracking_error_not_higher':new['motion']['summary']['mean_tracking_rmse_mm']<=old['motion']['summary']['mean_tracking_rmse_mm'],
        'executed_collision_free':all(new[k]['summary']['execution_collisions']==0 for k in ('fixed','motion')),
        'continuous_motion_limits':all(new[k]['summary']['max_velocity_rad_s']<=1+1e-6 and new[k]['summary']['max_acceleration_rad_s2']<=3+1e-6 for k in ('fixed','motion')),
        'continuous_boundaries_match':all(new[k]['summary']['max_boundary_velocity_error']<1e-6 and new[k]['summary']['max_boundary_acceleration_error']<1e-6 for k in ('fixed','motion'))}
    report={'kind':'paired_full_horizon_c2_comparison','source_sha256':file_sha256(SOURCE),'augmentation_sha256':file_sha256(AUGMENTATION),
            'configuration':{'fixed_seed':20261010,'motion_seed':20261011,'samples':4,'inference_steps':32,
                             'fixed_max_decisions':100,'motion_max_decisions':120,'execute_seconds':.4,'motion_controller':'collision_checked_bezier_c2'},
            'results':results,'deployment_checks':checks,'recommended_for_deployment':all(checks.values()),
            'legacy_controller_reference':'baseline_comparison.json and candidate_comparison.json measure the former two-reference fixed-goal loop. The current paired comparison uses the same full-horizon C2 controller for both checkpoints.',
            'limitations':'88 scenarios, one training seed. Moving goals follow expert routes. Tracking RMSE uses equal durations (40 s fixed, 48 s moving), retaining low-error goal holds after early completion. Curve execution includes retained braking tails and resting endpoints; compute delay is excluded from tracking RMSE/stalls. CPU latency measures denoising separately. Initial nonzero motion has a checked target-independent braking continuation. No hardware.'}
    write_json(OUTPUT,report);print(json.dumps({'checks':checks,'summaries':{name:{k:r[k]['summary'] for k in ('fixed','motion')} for name,r in results.items()}}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--only',choices=('baseline','candidate'))
    args=parser.parse_args();run(args.only) if args.only else merge()
