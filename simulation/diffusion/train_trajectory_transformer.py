"""GPU train a one-pass trajectory transformer; validation selects the model.

Reuse immutable motion-v2 caches and collision-protected train/validation data.
Test arrays and hard negatives are never loaded into the training process.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import h5py
import numpy as np
import torch
from .common import normalize, file_sha256, write_json
from .generate_collision_protected import audit
from .motion_conditioning import CONDITIONING
from .trajectory_transformer import TrajectoryTransformer, configuration


def protected_arrays(path,split):
    obs=[];q=[];ends=[];ranges=[];offset=0
    with h5py.File(path,'r') as data:
        metadata=json.loads(data.attrs['metadata_json'])
        if metadata['conditioning']!=CONDITIONING or not metadata.get('audit',{}).get('passed'):
            raise ValueError('Protected dataset lacks completed independent replay')
        for group in data['episodes'].values():
            if group.attrs['split']!=split:continue
            joints=group['q_rad'][:];goal=group['goal_rad'][:];size=len(joints)
            box=group['obstacle_m'][:] if group.attrs['obstacle_present'] else None
            geometry=np.zeros(7) if box is None else np.r_[box/.3,1.]
            features=np.c_[normalize(joints),normalize(goal),np.broadcast_to(geometry,(size,7)),
                           group['velocity_rad_s'][:],group['acceleration_rad_s2'][:]/3,
                           group['target_velocity_m_s'][:]/.5]
            obs.append(features);q.append(normalize(joints));ends.append(np.full(size,offset+size-1))
            ranges.append([offset,size]);offset+=size
    if not ranges:raise ValueError('Empty protected split')
    return {'obs':np.concatenate(obs).astype(np.float32),'q':np.concatenate(q).astype(np.float32),
            'ends':np.concatenate(ends).astype(np.int64)},np.array(ranges,np.int64)


def train(cache,protected,output,steps=12000,batch_size=256,seed=202610083):
    if steps<1 or batch_size<8:raise ValueError('Invalid training budgets')
    if not torch.cuda.is_available():raise RuntimeError('GPU training requested but CUDA is unavailable')
    output=Path(output)
    if output.exists():raise ValueError('Choose a new transformer model folder')
    cache=Path(cache);metadata=json.loads((cache/'cache.json').read_text())
    if metadata['identity']['conditioning']!=CONDITIONING:raise ValueError('Base cache conditioning changed')
    # Check the original cache sources against their recorded immutable hashes.
    identity=metadata['identity']
    for key in ('source','augmentation'):
        if file_sha256(identity[key])!=identity[key+'_sha256']:raise ValueError('Base cache data source changed')
    arrays={};ranges={}
    for split in ('train','validation'):
        base={key:np.load(cache/f'{split}_{key}.npy',mmap_mode='r') for key in ('obs','q','ends')}
        extra,extra_ranges=protected_arrays(protected,split)
        count=len(base['obs']);extra['ends']+=count
        arrays[split]={key:torch.as_tensor(np.concatenate((base[key],extra[key])),device='cuda') for key in base}
        ranges[split]=[torch.as_tensor(np.load(cache/f'{split}_ranges_{i}.npy'),device='cuda') for i in (0,1)]
        ranges[split].append(torch.as_tensor(extra_ranges+np.array([count,0]),device='cuda'))
    torch.manual_seed(seed);torch.set_num_threads(4)
    model=TrajectoryTransformer().cuda();ema=copy.deepcopy(model).eval()
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    rate=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,steps,eta_min=3e-6)
    generator=torch.Generator(device='cuda').manual_seed(seed)
    rng=np.random.default_rng(seed+1);validation_index=[]
    for group in ranges['validation']:
        cpu=group.cpu().numpy();size=1024 if len(validation_index)<2048 else 512
        selected=cpu[rng.integers(len(cpu),size=size)]
        validation_index.extend((selected[:,0]+(rng.random(size)*selected[:,1]).astype(np.int64)).tolist())
    validation_index=torch.tensor(validation_index,device='cuda');val_gen=torch.Generator(device='cuda').manual_seed(seed+1)
    val_latent=torch.randn((len(validation_index),4,16),device='cuda',generator=val_gen);val_latent[:,0]=0
    future=torch.arange(1,17,device='cuda')[None];best=float('inf');best_step=0;history=[];best_state=None;stale=0
    output.mkdir(parents=True);began=perf_counter();total=0.
    provenance={'kind':'full','completed':False,'seed':seed,'requested_steps':steps,'batch_size':batch_size,
                'gpu':torch.cuda.get_device_name(),'device':'cuda','conditioning':CONDITIONING,
                'base_cache':str(cache),'base_identity':identity,'protected_dataset':str(protected),
                'protected_sha256':file_sha256(protected),
                'sampling':'40% original expert / 40% motion augmentation / 20% collision-protected; uniform episode then frame',
                'objective':'normalized action MSE + 0.05 first-difference MSE + 0.0005 posterior KL',
                'selection':'lowest fixed-seed best-of-four prior validation action MSE; no test arrays loaded',
                'train_episode_counts':[len(r) for r in ranges['train']],
                'validation_episode_counts':[len(r) for r in ranges['validation']],
                'parameter_count':sum(p.numel() for p in model.parameters()),'history':history,
                'source_sha256':{name:file_sha256(Path(__file__).with_name(name)) for name in ('train_trajectory_transformer.py','trajectory_transformer.py')}}
    for step in range(1,steps+1):
        indices=[];sizes=[int(batch_size*.4),int(batch_size*.4)];sizes.append(batch_size-sum(sizes))
        for group,size in zip(ranges['train'],sizes):
            selected=group[torch.randint(len(group),(size,),device='cuda',generator=generator)]
            indices.append(selected[:,0]+(torch.rand(size,device='cuda',generator=generator)*selected[:,1]).long())
        index=torch.cat(indices);data=arrays['train'];actions=data['q'][torch.minimum(index[:,None]+future,data['ends'][index,None])]
        noise=torch.randn((len(index),16),device='cuda',generator=generator)
        pred,mean,logvar=model(data['obs'][index],actions,noise)
        mse=torch.nn.functional.mse_loss(pred,actions)
        smooth=torch.nn.functional.mse_loss(pred[:,1:]-pred[:,:-1],actions[:,1:]-actions[:,:-1])
        kl=.5*(mean.square()+logvar.exp()-1-logvar).mean();loss=mse+.05*smooth+.0005*kl
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite transformer training loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();rate.step();decay=min(.995,(1+step)/(10+step))
        with torch.no_grad():
            for target,current in zip(ema.parameters(),model.parameters()):target.mul_(decay).add_(current,alpha=1-decay)
        total+=float(loss.detach())
        if step%1000 and step!=steps:continue
        validation_loss=0.;val_data=arrays['validation']
        with torch.no_grad():
            for begin in range(0,len(validation_index),128):
                selected=validation_index[begin:begin+128];count=len(selected)
                clean=val_data['q'][torch.minimum(selected[:,None]+future,val_data['ends'][selected,None])]
                pred=ema.decode(val_data['obs'][selected].repeat_interleave(4,dim=0),val_latent[begin:begin+count].reshape(-1,16)).reshape(count,4,16,6)
                error=((pred-clean[:,None])**2).mean(dim=(2,3)).min(dim=1).values
                validation_loss+=float(error.sum())
        validation_loss/=len(validation_index)
        history.append({'step':step,'train_loss':total/min(1000,step),'validation_prior_best4_mse':validation_loss});total=0.
        if validation_loss<best-1e-7:
            best=validation_loss;best_step=step;stale=0;best_state={k:v.detach().cpu().clone() for k,v in ema.state_dict().items()}
        else:stale+=1
        provenance.update(completed_steps=step,selected_step=best_step,best_validation_mse=best,seconds=perf_counter()-began)
        write_json(output/'training_progress.json',provenance)
        print(json.dumps(history[-1]|{'best_step':best_step,'seconds':provenance['seconds']}),flush=True)
        if stale>=6:break
    provenance.update(completed=True,seconds=perf_counter()-began)
    torch.save({'configuration':configuration(),'training':provenance,'state_dict':best_state},output/'trajectory_transformer.pth')
    write_json(output/'training.json',provenance)
    return provenance

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=Path('simulation/artifacts/diffusion_motion_v2/cache'))
    parser.add_argument('--protected',type=Path,default=Path('simulation/artifacts/collision_protected_v1/training_1000.h5'))
    parser.add_argument('--output',type=Path,default=Path('simulation/artifacts/trajectory_transformer_v1/model'))
    parser.add_argument('--steps',type=int,default=12000);parser.add_argument('--batch-size',type=int,default=256)
    args=parser.parse_args();train(args.cache,args.protected,args.output,args.steps,args.batch_size)
