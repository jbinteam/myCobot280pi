"""Select a trained Seed5 on validation only, then publish its separate files."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
from time import perf_counter
from unittest.mock import patch
import numpy as np
from . import planning, seed5
from .prm import planning_space
from .seed5_data import load_dataset


def select(candidates,output,count=512):
    data,_=load_dataset(Path('simulation/artifacts/seed5_v1/dataset.npz'))
    indices=np.random.default_rng(1102).choice(len(data['validation_q_current_rad']),count,replace=False)
    models={str(Path(p)):seed5.Seed5Model(Path(p)/'models.json') for p in candidates}
    if any(m.data['training']['kind']!='full' for m in models.values()):raise ValueError('Selection requires full-trained candidates')
    space=planning_space('self',1.)
    config=planning.IKConfig(task='pose5',nn='seed5',numerical_fallback=False,max_candidates=1)
    rows={name:[] for name in models}
    for n,index in enumerate(indices):
        q=data['validation_q_current_rad'][index];xyz=data['validation_target_xyz_m'][index];axis=data['validation_target_axis_base'][index]
        for name,model in models.items():
            def proposal(q,p,u):
                tick=perf_counter();guesses=model.predict(q,p,u)
                return guesses,{'status':'ready','training_kind':'full'},(perf_counter()-tick)*1000
            with patch.object(seed5,'propose',side_effect=proposal):
                result=planning.generate_goals(q,xyz,space=space,config=config,target_axis=axis)
            rows[name].append({'validation_index':int(index),'success':bool(result['goals']),
                               'iterations':sum(a.get('iterations',0) for a in result['attempts']),
                               'refinement_ms':sum(a.get('refinement_ms',0) for a in result['attempts']),
                               'inference_ms':result['inference_ms'],'best_error':result['best_error']})
        if n%32==0 or n==count-1:print(f'Candidate selection {n+1}/{count}',flush=True)
    summaries={name:{'valid_success':sum(r['success'] for r in values),'tasks':count,
                    'mean_refinement_iterations':float(np.mean([r['iterations'] for r in values])),
                    'mean_refinement_ms':float(np.mean([r['refinement_ms'] for r in values])),
                    'mean_inference_ms':float(np.mean([r['inference_ms'] for r in values])),
                    'raw_validation':json.loads((Path(name)/'training.json').read_text())['validation_metrics']}
               for name,values in rows.items()}
    winner=min(summaries,key=lambda n:(-summaries[n]['valid_success'],summaries[n]['mean_refinement_iterations'],n))
    output=Path(output)
    if output.resolve()==(Path(__file__).parent/'artifacts').resolve():raise ValueError('Cannot publish over XYZ weights')
    import torch
    # Older local runs serialized torch.__version__ as its str subclass.
    # Allow only that known metadata type, retain safe checkpoint loading and
    # normalize it for ordinary weights_only=True consumers.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        payload=torch.load(Path(winner)/'seed5_v1.pt',map_location='cpu',weights_only=True)
    if payload['configuration']!=models[winner].data['configuration']:raise ValueError('PyTorch/NumPy checkpoint metadata mismatch')
    for index,(weight,bias) in zip((0,2,4),models[winner].layers):
        if not np.array_equal(payload['state_dict'][f'{index}.weight'].numpy(),weight) or not np.array_equal(payload['state_dict'][f'{index}.bias'].numpy(),bias):
            raise ValueError('PyTorch/NumPy checkpoint weights mismatch')
    payload['training']['torch_version']=str(payload['training']['torch_version'])
    output.mkdir(parents=True,exist_ok=True)
    for filename in ('models.json','training.json'):shutil.copyfile(Path(winner)/filename,output/filename)
    torch.save(payload,output/'seed5_v1.pt')
    report={'selection':'Validation only: maximize finite/limit/self-collision-valid success within 1 mm AND 1 degree; tie-break with fewer total numerical refinement iterations',
            'validation_subset_seed':1102,'validation_tasks':count,'numerical_fallback':False,'max_candidates':1,
            'selected':winner,'summary':summaries,'rows':rows,'published':str(output/'models.json'),
            'limitation':'Best of these trained candidates on this validation sample; no claim of globally optimal model. Held-out benchmark remains separate.'}
    (output/'selection.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({'selected':winner,'summary':summaries}),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidates',type=Path,nargs='+');parser.add_argument('--count',type=int,default=512)
    parser.add_argument('--output',type=Path,default=Path('simulation/artifacts/seed5_v1'))
    args=parser.parse_args();select(args.candidates,args.output,args.count)


if __name__=='__main__':main()
