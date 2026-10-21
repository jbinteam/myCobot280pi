"""Train virtual numerical-IK guesses; held-out test targets never select weights."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from torch import nn
from ..kinematics import ARM
from ..train import network
from ..train_seed5 import task_costs
from .ik_seed import configuration, IKSeedModel
from .ik_seed_data import load_dataset


def train(dataset,output,steps=8000,batch_size=512,seed=202610072,device='cuda',witness_weight=.05):
    if steps<1 or batch_size<1 or not np.isfinite(witness_weight) or witness_weight<0:raise ValueError('Invalid training options')
    arrays,metadata=load_dataset(dataset,check_scene=True)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.manual_seed(seed);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    if device=='cuda':torch.cuda.manual_seed_all(seed)
    center=torch.tensor(ARM.center,dtype=torch.float32,device=device)
    radius=torch.tensor(ARM.radius,dtype=torch.float32,device=device)
    def tensors(split):
        return tuple(torch.tensor(arrays[split+'_'+key],dtype=torch.float32,device=device) for key in ('q_current_rad','target_xyz_m','target_axis_base','q_goal_rad'))
    training=tensors('train');validation=tensors('validation')
    model=network(12,48,256).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=max(1,steps//3),gamma=.45)
    def objective(batch,diagnostics=False):
        current,xyz,axis,witness=batch
        inputs=torch.cat((xyz/ARM.scale,axis,(current-center)/radius),dim=-1)
        guess=center+radius*torch.tanh(model(inputs).reshape(-1,8,6))
        fk=ARM.torch_transform(guess)
        task=task_costs(fk[...,:3,3],fk[...,:3,2],xyz,axis)
        # Local-Z spin is unconstrained. Witness auxiliary supervision excludes
        # q6, whose random FK-witness spin is not identifiable from the target.
        witness_cost=((guess[...,:5]-witness[:,None,:5])/radius[:5]).square().sum(-1)
        travel=((guess-current[:,None,:])/radius).square().sum(-1)
        value=(task+1e-5*travel).min(-1).values.mean()+witness_weight*witness_cost.min(-1).values.mean()
        if diagnostics:
            position=torch.linalg.vector_norm(fk[...,:3,3]-xyz[:,None],dim=-1)*1000
            dot=(fk[...,:3,2]*axis[:,None]).sum(-1).clamp(-1,1)
            angle=torch.rad2deg(torch.acos(dot))
            scores=task.argmin(-1);rows=torch.arange(len(current),device=device)
            return value,{'mean_pose_best_position_mm':float(position[rows,scores].mean()),
                          'mean_pose_best_direction_deg':float(angle[rows,scores].mean()),
                          'raw_pose_success':int(((position<=1)&(angle<=1)).any(-1).sum())}
        return value
    began=perf_counter();best=float('inf');state=None;history=[];selected=0
    for step in range(steps):
        index=torch.randint(len(training[0]),(batch_size,),device=device)
        batch=tuple(value[index] for value in training)
        optimizer.zero_grad(set_to_none=True);loss=objective(batch)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite IK seed training loss')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();scheduler.step()
        if step%400==0 or step==steps-1:
            with torch.no_grad():val,diagnostic=objective(validation,True)
            row={'step':step+1,'train_loss':float(loss.detach()),'validation_loss':float(val),**diagnostic}
            history.append(row);print(json.dumps(row),flush=True)
            if float(val)<best:best=float(val);state=copy.deepcopy(model.state_dict());selected=step+1
    model.load_state_dict(state);model.cpu();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    provenance={'kind':'full','completed':True,'completed_steps':steps,'selected_step':selected,'seed':seed,
        'dataset':str(dataset),'dataset_sha256':hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
        'dataset_generation':metadata['generation'],'scene':metadata['scene'],'device':device,'torch_version':str(torch.__version__),
        'batch_size':batch_size,'seconds':perf_counter()-began,'best_validation_loss':best,'history':history,
        'objective':'Best-of-eight directed FK task loss plus separate best-of-eight normalized q1-q5 witness loss; tiny travel regularizer',
        'witness_weight':witness_weight,'test_used_for_selection':False,
        'scope':'Virtual initial guesses only; raw network outputs never certify reachability or collision freedom'}
    layers=[{'weight':layer.weight.detach().tolist(),'bias':layer.bias.detach().tolist()} for layer in model if isinstance(layer,nn.Linear)]
    checkpoint={'configuration':configuration(),'training':provenance,'layers':layers}
    checkpoint_path=output/'models.json';checkpoint_path.write_text(json.dumps(checkpoint,separators=(',',':'),allow_nan=False))
    torch.save({'configuration':configuration(),'training':provenance,'state_dict':model.state_dict()},output/'ik_seed_v1.pt')
    (output/'training.json').write_text(json.dumps(provenance,indent=2,allow_nan=False))
    # Serving parity verifies export only; no held-out-test scoring/selection here.
    serving=IKSeedModel(checkpoint_path)
    q,xyz,axis=(arrays['validation_'+key][0] for key in ('q_current_rad','target_xyz_m','target_axis_base'))
    model=model.to(device)
    with torch.no_grad():
        input_value=torch.tensor(np.r_[xyz/ARM.scale,axis,(q-ARM.center)/ARM.radius],dtype=torch.float32,device=device)
        expected=(center+radius*torch.tanh(model(input_value).reshape(8,6))).cpu().numpy()
    np.testing.assert_allclose(serving.predict(q,xyz,axis),expected,atol=3e-6,rtol=0)
    print(json.dumps({'checkpoint':str(checkpoint_path),'steps':steps,'seconds':provenance['seconds'],'selected_step':selected}),flush=True)
    return provenance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=Path('simulation/artifacts/ik_seed_v1/dataset.npz'))
    parser.add_argument('--output',type=Path,default=Path('simulation/artifacts/ik_seed_v1'))
    parser.add_argument('--steps',type=int,default=8000);parser.add_argument('--batch-size',type=int,default=512)
    parser.add_argument('--seed',type=int,default=202610072);parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--witness-weight',type=float,default=.05)
    args=parser.parse_args();train(args.dataset,args.output,args.steps,args.batch_size,args.seed,args.device,args.witness_weight)


if __name__=='__main__':main()
