"""Train five small myCobot-specific networks; export NumPy inference weights."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from .kinematics import ARM
from .solvers import ARTIFACTS


def network(inputs,outputs,hidden=128):return nn.Sequential(nn.Linear(inputs,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,outputs))


def dataset(count,seed,device):
    rng=np.random.default_rng(seed)
    q=ARM.center+ARM.radius*rng.uniform(-.9,.9,(count,6))
    goal=ARM.center+ARM.radius*rng.uniform(-.9,.9,(count,6))
    nearby=rng.random(count)<.75
    goal[nearby]=ARM.clip(q[nearby]+rng.normal(0,.35,(nearby.sum(),6)))
    q=torch.tensor(q,dtype=torch.float32,device=device)
    goal=torch.tensor(goal,dtype=torch.float32,device=device)
    with torch.no_grad():
        target,_=ARM.torch_fk_jacobian(goal)
        position,j=ARM.torch_fk_jacobian(q);j=j/ARM.scale
        error=(target-position)/ARM.scale
        velocity=3*(j.transpose(-1,-2)@torch.linalg.solve(j@j.transpose(-1,-2)+.04**2*torch.eye(3,device=device),error[...,None])).squeeze(-1)
        velocity*=torch.clamp(1.2/torch.linalg.vector_norm(velocity,dim=-1,keepdim=True).clamp_min(1e-9),max=1)
    return q,goal,target,error,velocity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='auto',choices=['auto','cpu','cuda'])
    parser.add_argument('--steps',type=int,default=1200)
    args=parser.parse_args()
    if args.steps<1:parser.error('steps must be positive')
    device='cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device)
    torch.manual_seed(42);torch.set_num_threads(4)
    if device=='cuda':torch.cuda.manual_seed_all(42)
    print(f'Training new 6-joint XYZ models on {device}; source URDF {ARM.fingerprint[:12]}',flush=True)
    center=torch.tensor(ARM.center,dtype=torch.float32,device=device)
    radius=torch.tensor(ARM.radius,dtype=torch.float32,device=device)
    low=center-radius;high=center+radius
    train=dataset(24000,42,device);validation=dataset(2048,43,device)
    ARTIFACTS.mkdir(exist_ok=True)
    np.savez_compressed(ARTIFACTS/'dataset.npz',q=train[0].cpu().numpy(),goal=train[1].cpu().numpy(),target=train[2].cpu().numpy(),validation_q=validation[0].cpu().numpy(),validation_target=validation[2].cpu().numpy())
    specs={'direct':(3,6),'conditional':(9,6),'differential':(9,6),'null':(6,6),'seed':(9,24)}
    exported={};metadata={};started=time.time()
    for name,(inputs,outputs) in specs.items():
        model=network(inputs,outputs).to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=max(1,args.steps//2),gamma=.4)
        def losses(data):
            q,goal,target,error,velocity=data;q_norm=(q-center)/radius
            if name=='direct':
                prediction=center+radius*torch.tanh(model(target/ARM.scale))
                fk,_=ARM.torch_fk_jacobian(prediction)
                # Direct inverse mapping is multi-valued; FK loss avoids averaging joint labels.
                return (((fk-target)/ARM.scale)**2).sum(-1).mean()+.002*(((prediction-center)/radius)**2).mean()
            if name in ('conditional','differential'):
                raw=model(torch.cat((q_norm,error),dim=-1))
                scale=.12 if name=='conditional' else 1.2
                expected=velocity*.05 if name=='conditional' else velocity
                return ((torch.tanh(raw)-expected/scale)**2).mean()
            if name=='null':
                prediction=1.2*torch.tanh(model(q_norm))
                return ((prediction+.6*q_norm)**2).mean()
            raw=model(torch.cat((target/ARM.scale,q_norm),dim=-1)).reshape(-1,4,6)
            candidates=torch.clamp(q[:,None,:]+radius*torch.tanh(raw),min=low,max=high)
            fk,_=ARM.torch_fk_jacobian(candidates)
            costs=(((fk-target[:,None,:])/ARM.scale)**2).sum(-1)+.003*((candidates-q[:,None,:])**2).sum(-1)
            return costs.min(dim=-1).values.mean()
        best=float('inf');best_state=None;history=[]
        steps=args.steps*2 if name=='seed' else (max(100,args.steps//3) if name=='null' else args.steps)
        for step in range(steps):
            index=torch.randint(len(train[0]),(512,),device=device)
            batch=tuple(value[index] for value in train)
            optimizer.zero_grad(set_to_none=True);loss=losses(batch);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5)
            optimizer.step();scheduler.step()
            if step%200==0 or step==steps-1:
                with torch.no_grad(): val=float(losses(validation))
                history.append({'step':step+1,'train_loss':float(loss),'validation_loss':val})
                if val<best:best=val;best_state=copy.deepcopy(model.state_dict())
                print(f'{name} {step+1}/{steps}: validation loss {val:.6f}',flush=True)
        model.load_state_dict(best_state);model.cpu()
        torch.save({'state_dict':model.state_dict(),'name':name,'seed':42,'urdf_sha256':ARM.fingerprint,
                    'limits_rad':{'min':ARM.lower.tolist(),'max':ARM.upper.tolist()}},ARTIFACTS/(name+'.pt'))
        layers=[{'weight':layer.weight.detach().tolist(),'bias':layer.bias.detach().tolist()} for layer in model if isinstance(layer,nn.Linear)]
        exported[name]={'layers':layers,'activation':'tanh','inputs':inputs,'outputs':outputs}
        metadata[name]={'steps':steps,'best_validation_loss':best,'history':history,'architecture':[inputs,128,128,outputs]}
    data={'urdf_sha256':ARM.fingerprint,'training_seed':42,'validation_seed':43,'task':'XYZ position, six-joint myCobot 280 Pi, no tool offset',
          'source_ideas':'/home/jb/codex/projects/nn_ik/rrr_neural_ik/architectures',
          'device':device,'torch_version':torch.__version__,'train_samples':24000,'validation_samples':2048,
          'seconds':time.time()-started,'models':exported,'training':metadata,
          'limits_rad':{'min':ARM.lower.tolist(),'max':ARM.upper.tolist()},'joint_limit_profile':ARM.limit_profile,'position_scale_m':ARM.scale}
    (ARTIFACTS/'models.json').write_text(json.dumps(data,separators=(',',':')))
    (ARTIFACTS/'training.json').write_text(json.dumps({k:v for k,v in data.items() if k!='models'},indent=2))
    print(f'Exported five trained models in {data["seconds"]:.1f}s',flush=True)


if __name__=='__main__':main()
