"""GPU-resident, episode-balanced fine-tuning on 100k + motion demonstrations."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from time import perf_counter
import h5py
import numpy as np
import torch
from .common import file_sha256, normalize, robot_schema, write_json
from .generate_motion import derivatives
from .motion_conditioning import CONDITIONING
from .model import ConditionalUNet1D, scheduler
from ..kinematics import ARM


def build_cache(source,augmentation,folder):
    """Store flat float32 inputs, keeping window endpoints and episode identity."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    identity={'source':str(source),'source_sha256':file_sha256(source),
              'augmentation':str(augmentation),'augmentation_sha256':file_sha256(augmentation),
              'robot':robot_schema(),'conditioning':CONDITIONING}
    identity['dataset_sha256']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    manifest=folder/'cache.json'
    if manifest.exists():
        saved=json.loads(manifest.read_text())
        if saved['identity']!=identity:raise ValueError('Cache sources/schema changed')
        return saved
    counts={split:0 for split in ('train','validation','test')};episode_counts=counts.copy()
    for path in (source,augmentation):
        with h5py.File(path,'r') as data:
            metadata=json.loads(data.attrs['metadata_json'])
            if metadata['completed_episodes']!=len(data['episodes']) or metadata['environment']['robot']!=robot_schema():raise ValueError('Incomplete or incompatible dataset')
            for group in data['episodes'].values():
                split=group.attrs['split'];counts[split]+=len(group['q_rad']);episode_counts[split]+=1
    arrays={};ranges={split:[[],[]] for split in counts};offsets=counts.copy()
    for split,count in counts.items():
        offsets[split]=0
        arrays[split]={key:np.lib.format.open_memmap(folder/f'{split}_{key}.npy',mode='w+',dtype=dtype,shape=shape)
                       for key,dtype,shape in [('obs',np.float32,(count,34)),('q',np.float32,(count,6)),('ends',np.int64,(count,))]}
    for source_index,path in enumerate((source,augmentation)):
        with h5py.File(path,'r') as data:
            for number,(name,group) in enumerate(data['episodes'].items()):
                q=group['q_rad'][:];times=group['time_s'][:];split=group.attrs['split'];count=len(q)
                if q.shape!=(count,6) or count<2 or not np.isfinite(q).all() or np.any(q<ARM.lower) or np.any(q>ARM.upper) or not np.allclose(np.diff(times),.1,atol=1e-9):raise ValueError('Invalid cache trajectory')
                if source_index==0:
                    goal=np.broadcast_to(group['q_goal_rad'][:],q.shape);box=group['obstacle_m'][:]
                    velocity,acceleration=derivatives(q,times);target_velocity=np.zeros((count,3))
                else:
                    goal=group['goal_rad'][:];box=group['obstacle_m'][:] if group.attrs['obstacle_present'] else None
                    velocity=group['velocity_rad_s'][:];acceleration=group['acceleration_rad_s2'][:];target_velocity=group['target_velocity_m_s'][:]
                if (goal.shape!=q.shape or not np.isfinite(goal).all() or np.any(goal<ARM.lower) or np.any(goal>ARM.upper)
                        or velocity.shape!=q.shape or acceleration.shape!=q.shape or target_velocity.shape!=(count,3)
                        or not np.isfinite(velocity).all() or not np.isfinite(acceleration).all() or not np.isfinite(target_velocity).all()
                        or np.max(np.linalg.norm(velocity,axis=1))>1+1e-6 or np.max(np.linalg.norm(acceleration,axis=1))>3+1e-6
                        or np.max(np.linalg.norm(target_velocity,axis=1))>.5+1e-6):raise ValueError('Invalid motion labels')
                geometry=np.zeros(7) if box is None else np.r_[box/ARM.scale,1.]
                obs=np.concatenate((normalize(q),normalize(goal),np.broadcast_to(geometry,(count,7)),velocity,acceleration/3,target_velocity/.5),axis=1)
                start=offsets[split];end=start+count
                arrays[split]['obs'][start:end]=obs;arrays[split]['q'][start:end]=normalize(q);arrays[split]['ends'][start:end]=end-1
                ranges[split][source_index].append([start,count]);offsets[split]=end
                if number%10000==0:print(f'Cache source {source_index}: {number} episodes',flush=True)
    for split in arrays:
        for value in arrays[split].values():value.flush()
        for index,value in enumerate(ranges[split]):np.save(folder/f'{split}_ranges_{index}.npy',np.asarray(value,np.int64))
    saved={'identity':identity,'frames':counts,'episodes':episode_counts,'sampling':'50% original / 50% augmented; uniform parent episodes then uniform frame',
           'test_isolation':'Only train/validation cache arrays are loaded by training; source split inherited by augmentation'}
    write_json(manifest,saved);return saved


def train(source,augmentation,cache,output,initial,steps=20000,batch_size=512,seed=20261007,validation_every=500,patience=10):
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('Choose a new model directory')
    torch.manual_seed(seed);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():raise RuntimeError('This full training run requires the available GPU')
    metadata=build_cache(source,augmentation,cache);device='cuda';cache=Path(cache)
    arrays={split:{key:torch.as_tensor(np.load(cache/f'{split}_{key}.npy').copy(),device=device)
                   for key in ('obs','q','ends')} for split in ('train','validation')}
    ranges=[torch.as_tensor(np.load(cache/f'train_ranges_{i}.npy'),device=device) for i in (0,1)]
    original=torch.load(initial,map_location='cpu',weights_only=True)
    if original['configuration']['robot']!=robot_schema() or original['configuration'].get('conditioning') is not None:raise ValueError('Expected original compatible v1 checkpoint')
    model=ConditionalUNet1D(original['configuration']['channels'],34).to(device)
    state=original['state_dict'];expanded=torch.zeros((128,34));expanded[:,:19]=state['observation.0.weight'];state['observation.0.weight']=expanded
    model.load_state_dict(state,strict=True);ema=copy.deepcopy(model).eval()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=1e-6)
    rate=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,steps,eta_min=.000003)
    noise_scheduler=scheduler();generator=torch.Generator(device=device).manual_seed(seed)
    validation_generator=torch.Generator(device=device).manual_seed(seed+1)
    validation_indices=[];rng=np.random.default_rng(seed+1)
    for index in (0,1):
        group=np.load(cache/f'validation_ranges_{index}.npy')
        selected=group[rng.integers(len(group),size=2048)]
        validation_indices.extend((selected[:,0]+(rng.random(len(selected))*selected[:,1]).astype(np.int64)).tolist())
    validation_index=torch.as_tensor(validation_indices,device=device)
    future=torch.arange(1,17,device=device)[None];history=[];best=float('inf');best_step=0;stale=0;total=0.;began=perf_counter()
    output.mkdir(parents=True,exist_ok=True)
    configuration={**original['configuration'],'conditioning':CONDITIONING}
    provenance={'kind':'full','seed':seed,'dataset':str(cache),'dataset_sha256':metadata['identity']['dataset_sha256'],
                'data_identity':metadata['identity'],'train_episodes':metadata['episodes']['train'],
                'validation_episodes':metadata['episodes']['validation'],'train_windows':metadata['frames']['train'],
                'validation_windows':len(validation_index),'sampling':metadata['sampling'],'batch_size':batch_size,
                'requested_steps':steps,'learning_rate':.0003,'initial_checkpoint_sha256':file_sha256(initial),
                'device':device,'gpu':torch.cuda.get_device_name(),'objective':'Gaussian noise MSE; inherited collision-checked expert trajectories',
                'selection':'Lowest fixed-seed EMA validation noise MSE; rollout test set never loaded',
                'history':history}
    validation_noise=torch.randn((len(validation_index),6,16),device=device,generator=validation_generator)
    validation_time=torch.randint(100,(len(validation_index),),device=device,generator=validation_generator)
    best_state=None
    for step in range(1,steps+1):
        indices=[]
        for group in ranges:
            selected=group[torch.randint(len(group),(batch_size//2,),device=device,generator=generator)]
            indices.append(selected[:,0]+(torch.rand(len(selected),device=device,generator=generator)*selected[:,1]).long())
        index=torch.cat(indices);data=arrays['train'];action_index=torch.minimum(index[:,None]+future,data['ends'][index,None])
        clean=data['q'][action_index].transpose(1,2);obs=data['obs'][index]
        noise=torch.randn(clean.shape,device=device,generator=generator);time=torch.randint(100,(len(index),),device=device,generator=generator)
        noisy=noise_scheduler.add_noise(clean,noise,time);loss=torch.nn.functional.mse_loss(model(noisy,time,obs),noise)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();rate.step();decay=min(.995,(1+step)/(10+step))
        with torch.no_grad():
            for target,current in zip(ema.parameters(),model.parameters()):target.mul_(decay).add_(current,alpha=1-decay)
        total+=float(loss.detach())
        if step%validation_every and step!=steps:continue
        validation_loss=0.;validation_data=arrays['validation']
        with torch.no_grad():
            for start in range(0,len(validation_index),batch_size):
                selected=validation_index[start:start+batch_size];size=len(selected)
                clean=validation_data['q'][torch.minimum(selected[:,None]+future,validation_data['ends'][selected,None])].transpose(1,2)
                noise=validation_noise[start:start+size];time=validation_time[start:start+size]
                loss=torch.nn.functional.mse_loss(ema(noise_scheduler.add_noise(clean,noise,time),time,validation_data['obs'][selected]),noise)
                validation_loss+=float(loss)*size
        validation_loss/=len(validation_index)
        history.append({'step':step,'train_noise_mse':total/min(validation_every,step),'validation_noise_mse':validation_loss});total=0
        if validation_loss<best-1e-7:
            best=validation_loss;best_step=step;stale=0
            best_state={name:value.detach().cpu().clone() for name,value in ema.state_dict().items()}
        else:stale+=1
        provenance.update(completed_steps=step,completed_epochs=max(1,int(step*batch_size/metadata['frames']['train'])),
                          selected_step=best_step,best_validation_noise_mse=best,seconds=perf_counter()-began)
        temporary=output/'diffusion_policy_6dof.tmp';torch.save({'configuration':configuration,'training':provenance,'state_dict':best_state},temporary)
        temporary.replace(output/'diffusion_policy_6dof.pth');write_json(output/'training.json',provenance)
        print(f'Motion diffusion {step}/{steps}: validation {validation_loss:.6f}, best {best_step}, seconds {provenance["seconds"]:.1f}',flush=True)
        if stale>=patience:break
    return provenance


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('simulation/artifacts/diffusion_100k_v1/demonstrations.h5'))
    parser.add_argument('--augmentation',type=Path,default=Path('simulation/artifacts/diffusion_motion_v2/augmentation.h5'))
    parser.add_argument('--cache',type=Path,default=Path('simulation/artifacts/diffusion_motion_v2/cache'))
    parser.add_argument('--output',type=Path,default=Path('simulation/artifacts/diffusion_motion_v2/model'))
    parser.add_argument('--initial',type=Path,default=Path('simulation/artifacts/diffusion_v1/model/diffusion_policy_6dof.pth'))
    parser.add_argument('--steps',type=int,default=20000);parser.add_argument('--batch-size',type=int,default=512)
    args=parser.parse_args();train(args.source,args.augmentation,args.cache,args.output,args.initial,args.steps,args.batch_size)
