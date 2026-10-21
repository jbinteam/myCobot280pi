"""Prepare a pinned Cartesian candidate manifest without activating the pointer."""
import argparse
import copy
import json
from pathlib import Path

from .cartesian_conditioning import supports
from .common import file_sha256, write_json


def prepare_manifest(folder=Path('simulation/artifacts/diffusion_cartesian_v3'),
                     comparison=Path('output/diffusion/cartesian_v3_comparison.json'),
                     baseline_manifest=Path('simulation/artifacts/diffusion_motion_v2/webui.json'),
                     audit=Path('output/diffusion/cartesian_v3_dataset_validation.json')):
    """Preserve prior deployments; qualification is recorded, never fabricated."""
    from .policy import DiffusionPolicy
    folder = Path(folder); comparison = Path(comparison); baseline_manifest = Path(baseline_manifest)
    audit = Path(audit); validation = json.loads(audit.read_text())
    report = json.loads(comparison.read_text()); original = json.loads(baseline_manifest.read_text())
    training = json.loads((folder / 'model/training.json').read_text())
    checkpoint = folder / 'model/diffusion_policy_6dof.pth'
    candidate = report['results']['candidate']; baseline = report['results']['baseline']
    identity = training['data_identity']
    conditioning = identity['conditioning']
    raw_identity = identity.get('source_cache_identity', identity)
    if (not validation['passed'] or validation['conditioning'] != conditioning
            or validation['dataset_sha256'] != training['dataset_sha256']
            or validation['source_sha256'] != report['source_sha256']
            or validation['augmentation_sha256'] != report['augmentation_sha256']
            or report['kind'] != 'paired_cartesian_c2_comparison'
            or file_sha256(checkpoint) != candidate['checkpoint_sha256']
            or original['checkpoint_sha256'] != baseline['checkpoint_sha256']
            or raw_identity['source_sha256'] != report['source_sha256']
            or raw_identity['augmentation_sha256'] != report['augmentation_sha256']
            or conditioning.get('schema') not in (3, 4) or conditioning.get('goal_joint_input') is not False
            or training['kind'] != 'full' or training['completed_steps'] < 1
            or training.get('goal_joint_input') is not False
            or report['configuration']['joint_endpoint_capture'] is not False
            or not all(candidate[k]['ik_disabled'] and candidate[k]['goal_joint_labels_removed']
                       for k in ('fixed', 'motion'))):
        raise ValueError('Candidate checkpoint/data/evaluation/schema provenance differs')
    policy = DiffusionPolicy(checkpoint, 'cpu')
    if (not supports(policy) or policy.conditioning != conditioning
            or candidate.get('conditioning', conditioning) != conditioning
            or policy.training['dataset_sha256'] != training['dataset_sha256']
            or policy.training['completed_steps'] != training['completed_steps']):
        raise ValueError('Candidate model load/schema/training validation failed')
    # Source manifests contain target labels and goal joints for their legacy UI.
    # v3 examples expose only Cartesian targets and never suggest joint endpoints.
    manifest = copy.deepcopy(original)
    manifest['cases'] = [{key: value for key, value in case.items() if key != 'goal_deg'}
                         for case in original['cases']]
    for case in manifest['cases']:
        if 'name' in case: case['name'] = case['name'].replace('held-out', 'legacy example')
    fixed = candidate['fixed']['summary']; motion = candidate['motion']['summary']
    qualified = report['recommended_for_deployment'] and all(report['deployment_checks'].values())
    manifest.update(model_name=folder.name, checkpoint_sha256=file_sha256(checkpoint),
                    dataset_sha256=training['dataset_sha256'], conditioning=conditioning,
                    goal_joint_input=False, inverse_kinematics_required=False,
                    no_block_trained=True, moving_targets_trained=True,
                    dataset_episodes=sum(validation['episodes'].values()),
                    dataset_episode_count_scope='All preserved source/augmentation episodes; test excluded from training',
                    held_out_evaluation={**fixed, 'success_rate': fixed['successes'] / fixed['episodes']},
                    motion_evaluation=motion, recommended_for_deployment=qualified,
                    comparison={'path': str(comparison), 'sha256': file_sha256(comparison)},
                    dataset_audit={'path': str(audit), 'sha256': file_sha256(audit)},
                    deployment_checks=report['deployment_checks'],
                    training_provenance={'path': str(folder / 'model/training.json'),
                                         'sha256': file_sha256(folder / 'model/training.json')})
    manifest_path = folder / 'webui.json'; write_json(manifest_path, manifest)
    return {'manifest': str(manifest_path), 'manifest_sha256': file_sha256(manifest_path),
            'checkpoint_sha256': policy.sha256, 'recommended_for_deployment': qualified,
            'activated': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=Path('simulation/artifacts/diffusion_cartesian_v3'))
    parser.add_argument('--comparison', type=Path, default=Path('output/diffusion/cartesian_v3_comparison.json'))
    parser.add_argument('--baseline-manifest', type=Path, default=Path('simulation/artifacts/diffusion_motion_v2/webui.json'))
    parser.add_argument('--audit', type=Path, default=Path('output/diffusion/cartesian_v3_dataset_validation.json'))
    print(json.dumps(prepare_manifest(**vars(parser.parse_args()))), flush=True)
