"""Offline, paired held-out evaluation of numerical versus learned IK starts.

This compares endpoints, not diffusion trajectories or physical arm execution.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
import numpy as np
from ..kinematics import ARM
from ..pose5 import axis_error
from .env import ArmEnv
from .endpoint_ik import IKSearchExhausted


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(dataset, checkpoint, output, limit=128, attempts=32):
    from .ik_seed_data import load_dataset
    arrays, metadata = load_dataset(dataset)
    checkpoint = Path(checkpoint).resolve()
    # Require an explicit artifact so a missing model cannot masquerade as a trial.
    checkpoint_hash = digest(checkpoint)
    keys = ['q_current_rad', 'q_goal_rad', 'target_xyz_m', 'target_axis_base', 'target_id']
    data = {k: arrays['test_' + k] for k in keys}
    count = len(data['q_current_rad'])
    indices = np.random.default_rng(20261007).permutation(count)[:min(limit, count)]
    if len(indices) < 1:
        raise ValueError('A positive held-out evaluation count is required')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    previous = os.environ.get('MYCOBOT_IK_SEED_MODEL')
    os.environ['MYCOBOT_IK_SEED_MODEL'] = str(checkpoint)
    rows = []
    try:
        with ArmEnv() as env:
            env.set_obstacle(None)
            for order, index in enumerate(indices):
                current = data['q_current_rad'][index]
                goal = data['q_goal_rad'][index]
                target = data['target_xyz_m'][index]
                axis = data['target_axis_base'][index]
                witness = ARM.transform(goal)
                if not (env.valid(current) and env.valid(goal)):
                    raise ValueError('Evaluation current/witness fails the deployment scene')
                if np.linalg.norm(witness[:3, 3]-target) > 1e-8 or axis_error(witness[:3, 2],axis)[2] > 1e-5:
                    raise ValueError('Evaluation witness does not match its target')
                row = {'index': int(index), 'target_id': int(data['target_id'][index])}
                # Alternate call order to reduce systematic warm-cache bias.
                for enabled in ([False, True] if order % 2 == 0 else [True, False]):
                    env.reset(current)
                    began = perf_counter()
                    try:
                        joints = env.inverse_kinematics(target,axis,current=current,seed=42,
                                                       attempts=attempts,robust=True,learned=enabled)
                        transform = ARM.transform(joints)
                        pos = float(np.linalg.norm(transform[:3,3]-target)*1000)
                        angle = float(axis_error(transform[:3,2],axis)[2])
                        safe = bool(env.valid(joints))
                        if not safe or pos > 1. or angle > 1.:
                            raise AssertionError('Search accepted an invalid endpoint')
                        result = {'success': True, 'position_error_mm': pos,
                                  'direction_error_deg': angle, 'joints_rad': joints.tolist()}
                    except IKSearchExhausted:
                        result = {'success': False}
                    result['elapsed_ms'] = (perf_counter()-began)*1000
                    result['diagnostics'] = dict(env.ik_diagnostics)
                    if enabled and result['diagnostics'].get('legacy_attempts',0) > 1:
                        status = result['diagnostics'].get('learned_status')
                        if status != 'ready':
                            raise AssertionError(f'Learned model was unavailable during evaluation: {status}')
                    row['learned' if enabled else 'baseline'] = result
                rows.append(row)
                if (order+1) % 16 == 0:
                    print(f'Compared {order+1}/{len(indices)} held-out pairs',flush=True)
    finally:
        if previous is None:
            os.environ.pop('MYCOBOT_IK_SEED_MODEL',None)
        else:
            os.environ['MYCOBOT_IK_SEED_MODEL'] = previous
    if digest(checkpoint) != checkpoint_hash:
        raise RuntimeError('Checkpoint changed during evaluation')
    summary = {'pairs':len(rows),'target_groups':len({r['target_id'] for r in rows}),
               'attempts':attempts,'baseline_success':sum(r['baseline']['success'] for r in rows),
               'learned_success':sum(r['learned']['success'] for r in rows),
               'recovered_indices':[r['index'] for r in rows if not r['baseline']['success'] and r['learned']['success']],
               'regressed_indices':[r['index'] for r in rows if r['baseline']['success'] and not r['learned']['success']],
               'learned_refinement_successes':sum(bool(r['learned']['diagnostics'].get('learned_refine_success')) for r in rows)}
    for mode in ['baseline','learned']:
        elapsed = [r[mode]['elapsed_ms'] for r in rows]
        summary[mode+'_median_ms'] = float(np.median(elapsed))
        summary[mode+'_p95_ms'] = float(np.percentile(elapsed,95))
        summary[mode+'_total_ms'] = float(sum(elapsed))
    summary['median_paired_ratio'] = float(np.median([r['baseline']['elapsed_ms']/r['learned']['elapsed_ms'] for r in rows]))
    report = {'summary':summary,'dataset':str(Path(dataset).resolve()),'dataset_sha256':digest(dataset),
              'checkpoint':str(checkpoint),'checkpoint_sha256':checkpoint_hash,
              'evaluation':'test split only; shuffled fixed indices; alternating method order; single paired run',
              'limitations':'Descriptive endpoint timing only. No repeated timing blocks, full path validation or physical execution.',
              'scene':env.metadata(),'rows':rows}
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=Path('simulation/artifacts/ik_seed_v1/dataset.npz'))
    parser.add_argument('--checkpoint',type=Path,default=Path('simulation/artifacts/ik_seed_v1/models.json'))
    parser.add_argument('--output',type=Path,default=Path('output/learned_ik_seed_20261007/evaluation.json'))
    parser.add_argument('--limit',type=int,default=128)
    args=parser.parse_args()
    evaluate(args.dataset,args.checkpoint,args.output,args.limit)


if __name__=='__main__':main()
