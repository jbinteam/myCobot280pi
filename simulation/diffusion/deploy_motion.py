"""Activate a verified simulation candidate through an atomic pinned manifest."""
import json
from pathlib import Path
from . import service
from .common import file_sha256,write_json
from .motion_conditioning import CONDITIONING


def deploy(folder=Path('simulation/artifacts/diffusion_motion_v2'),
           comparison=Path('output/diffusion/motion_v2_continuous_comparison.json'),
           audit=Path('output/diffusion/motion_v2_dataset_validation.json')):
    folder=Path(folder);report=json.loads(Path(comparison).read_text());validation=json.loads(Path(audit).read_text())
    checkpoint=folder/'model/diffusion_policy_6dof.pth';training=json.loads((folder/'model/training.json').read_text())
    if not report['recommended_for_deployment'] or not all(report['deployment_checks'].values()):
        raise ValueError('Candidate did not meet paired comparison checks; current deployment retained')
    if (not validation['passed'] or validation['augmentation_sha256']!=report['augmentation_sha256']
            or validation['source_sha256']!=report['source_sha256']
            or training['data_identity']['source_sha256']!=report['source_sha256']
            or training['data_identity']['augmentation_sha256']!=report['augmentation_sha256']
            or file_sha256(checkpoint)!=report['results']['candidate']['checkpoint_sha256']
            or training['kind']!='full' or training['completed_steps']<5000):
        raise ValueError('Candidate checkpoint/data/evaluation provenance differs')
    current=service.info()
    if current['status']!='ready' or current['checkpoint_sha256']!=report['results']['baseline']['checkpoint_sha256']:
        raise ValueError('Active deployment changed since baseline comparison')
    from .policy import DiffusionPolicy
    policy=DiffusionPolicy(checkpoint,'cpu')
    if policy.observation_dim!=34 or policy.training['dataset_sha256']!=training['dataset_sha256']:
        raise ValueError('Candidate load/schema/provenance validation failed')
    original=json.loads(service.MANIFEST.read_text());fixed=report['results']['candidate']['fixed']['summary']
    original.update(checkpoint_sha256=file_sha256(checkpoint),dataset_sha256=training['dataset_sha256'],
                    model_name='diffusion_motion_v2',dataset_episodes=validation['episodes']+100000,
                    conditioning=CONDITIONING,no_block_trained=True,moving_targets_trained=True,
                    held_out_evaluation={**fixed,'success_rate':fixed['successes']/fixed['episodes']},
                    motion_evaluation=report['results']['candidate']['motion']['summary'],
                    comparison={'path':str(comparison),'sha256':file_sha256(comparison)},
                    augmentation_audit={'path':str(audit),'sha256':file_sha256(audit)})
    manifest=folder/'webui.json';write_json(manifest,original)
    pointer={'schema':1,'manifest':str(manifest.resolve().relative_to(service.ARTIFACTS.parent.resolve())),
             'manifest_sha256':file_sha256(manifest),'previous_manifest':'diffusion_v1/webui.json',
             'previous_checkpoint_sha256':current['checkpoint_sha256']}
    temporary=service.ACTIVE.with_suffix('.tmp');write_json(temporary,pointer);temporary.replace(service.ACTIVE)
    result=service.info()
    if result['status']!='ready' or result['checkpoint_sha256']!=policy.sha256:
        service.ACTIVE.unlink();raise RuntimeError('Activation failed; original deployment restored')
    return {'model_name':result['model_name'],'checkpoint_sha256':policy.sha256,'dataset_episodes':result['dataset_episodes']}


if __name__=='__main__':print(json.dumps(deploy()),flush=True)
