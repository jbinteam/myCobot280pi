"""Train explicit Cartesian residual conditioning with clean expert FK supervision.

Original v3 checkpoint and datasets remain preserved. The auxiliary loss matches
actual future action labels, not distant/moving target endpoints, and only uses
noise levels with cumulative alpha>=0.25. Checkpoint selection is validation-only.
"""
import argparse
import copy
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from .common import file_sha256, robot_schema, write_json
from .cartesian_conditioning import CONDITIONING as ABSOLUTE_CONDITIONING, RELATIVE_CONDITIONING as CONDITIONING
from .model import ConditionalUNet1D, scheduler
from .cartesian_relative_cache import build_cache
from ..kinematics import ARM


def resume_state(checkpoint):
    config=checkpoint['configuration']
    if config['robot']!=robot_schema() or config.get('conditioning')!=ABSOLUTE_CONDITIONING:
        raise ValueError('Expected compatible absolute Cartesian v3 checkpoint')
    state={name:value.clone() for name,value in checkpoint['state_dict'].items()}
    if state['observation.0.weight'].shape!=(128,34):raise ValueError('Expected v3 observation weights')
    state['observation.0.weight'][:,6:12]=0
    return state


def clean_fk_loss(noisy,prediction,clean,timestep,alphas,alpha_min=.25):
    """Differentiable expert-label FK loss, zero gradients at excluded noise levels.

    Averaging the masked per-example loss over the full batch keeps the weight
    consistent when the eligible fraction fluctuates between batches.
    """
    alpha=alphas[timestep].to(device=noisy.device,dtype=noisy.dtype)
    eligible=alpha>=alpha_min
    fraction=eligible.to(noisy.dtype).mean()
    if not eligible.any():return prediction.sum()*0,fraction
    signal=alpha[eligible,None,None]
    reconstructed=(noisy[eligible]-(1-signal).sqrt()*prediction[eligible])/signal.sqrt()
    center=torch.as_tensor(ARM.center,device=noisy.device,dtype=noisy.dtype)
    radius=torch.as_tensor(ARM.radius,device=noisy.device,dtype=noisy.dtype)
    q_pred=center+reconstructed.transpose(1,2)*radius
    q_clean=center+clean[eligible].transpose(1,2)*radius
    predicted=ARM.torch_transform(q_pred)
    with torch.no_grad():expected=ARM.torch_transform(q_clean)
    position=((predicted[...,:3,3]-expected[...,:3,3])/.03).square().mean()
    axis=((predicted[...,:3,2]-expected[...,:3,2])/.2).square().mean()
    return (position+axis)*fraction,fraction


@torch.no_grad()
def validation_metrics(model,data,index,fixed_noise,fixed_time,future,noise_scheduler,alphas,fk_weight,batch_size):
    noise_total=fk_total=0.
    for start in range(0,len(index),batch_size):
        selected=index[start:start+batch_size];size=len(selected)
        action_index=torch.minimum(selected[:,None]+future,data['ends'][selected,None])
        clean=data['q'][action_index].transpose(1,2)
        noise=fixed_noise[start:start+size];time=fixed_time[start:start+size]
        noisy=noise_scheduler.add_noise(clean,noise,time)
        prediction=model(noisy,time,data['obs'][selected])
        noise_loss=torch.nn.functional.mse_loss(prediction,noise)
        fk_loss,_=clean_fk_loss(noisy,prediction,clean,time,alphas)
        noise_total+=float(noise_loss)*size;fk_total+=float(fk_loss)*size
    noise_total/=len(index);fk_total/=len(index)
    return {'validation_noise_mse':noise_total,'validation_gated_fk_loss':fk_total,
            'validation_composite':noise_total+fk_weight*fk_total}


def refine(source,augmentation,cache,output,initial,steps=20000,batch_size=512,seed=20261014,validation_every=500,patience=10,fk_weight=.02,learning_rate=.0003):
    if steps<1 or batch_size<2 or batch_size%2 or validation_every<1 or patience<1:
        raise ValueError('Need positive steps, even batch size, positive validation interval/patience')
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('Choose a new model directory')
    torch.manual_seed(seed);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():raise RuntimeError('This full training run requires the available GPU')
    metadata=build_cache(source,augmentation,cache,device="cuda");device='cuda';cache=Path(cache)
    arrays={split:{key:torch.as_tensor(np.load(cache/f'{split}_{key}.npy').copy(),device=device)
                   for key in ('obs','q','ends')} for split in ('train','validation')}
    ranges=[torch.as_tensor(np.load(cache/f'train_ranges_{i}.npy'),device=device) for i in (0,1)]
    original=torch.load(initial,map_location='cpu',weights_only=True)
    state=resume_state(original)
    model=ConditionalUNet1D(original['configuration']['channels'],34).to(device)
    model.load_state_dict(state,strict=True);ema=copy.deepcopy(model).eval()
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=1e-6)
    rate=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,steps,eta_min=learning_rate/100)
    noise_scheduler=scheduler();alphas=noise_scheduler.alphas_cumprod.to(device);generator=torch.Generator(device=device).manual_seed(seed)
    validation_generator=torch.Generator(device=device).manual_seed(seed+1)
    validation_indices=[];rng=np.random.default_rng(seed+1)
    for index in (0,1):
        group=np.load(cache/f'validation_ranges_{index}.npy')
        selected=group[rng.integers(len(group),size=2048)]
        validation_indices.extend((selected[:,0]+(rng.random(len(selected))*selected[:,1]).astype(np.int64)).tolist())
    validation_index=torch.as_tensor(validation_indices,device=device)
    future=torch.arange(1,17,device=device)[None];history=[];best=float('inf');best_step=0;stale=0;total=0.;total_noise=total_fk=total_fraction=total_gradient=0.;count_since_validation=0;began=perf_counter()
    output.mkdir(parents=True,exist_ok=True)
    configuration={**original['configuration'],'conditioning':CONDITIONING}
    provenance={'kind':'full','seed':seed,'dataset':str(cache),'dataset_sha256':metadata['identity']['dataset_sha256'],
                'data_identity':metadata['identity'],'train_episodes':metadata['episodes']['train'],
                'validation_episodes':metadata['episodes']['validation'],'train_windows':metadata['frames']['train'],
                'validation_windows':len(validation_index),'sampling':metadata['sampling'],'batch_size':batch_size,
                'requested_steps':steps,'learning_rate':learning_rate,'initial_checkpoint_sha256':file_sha256(initial),
                'device':device,'gpu':torch.cuda.get_device_name(),'objective':'noise MSE + low-noise reconstructed clean expert FK loss',
                'selection':'Lowest fixed-seed EMA validation composite (noise + FK weight * gated FK); test set never loaded',
                'warm_start':'Refined v3 EMA; absolute-target columns 6:12 ZEROed for Cartesian residual basis',
                'auxiliary':{'weight':fk_weight,'alpha_min':.25,'position_scale_m':.03,'axis_scale':.2,'label':'actual clean expert future FK; never requested target'},
                'goal_joint_input':False,'test_isolation':metadata['test_isolation'],'history':history}
    validation_noise=torch.randn((len(validation_index),6,16),device=device,generator=validation_generator)
    validation_time=torch.randint(100,(len(validation_index),),device=device,generator=validation_generator)
    baseline=validation_metrics(ema,arrays['validation'],validation_index,validation_noise,validation_time,
                                future,noise_scheduler,alphas,fk_weight,batch_size)
    if not all(np.isfinite(value) for value in baseline.values()):raise RuntimeError('Nonfinite initial validation objective')
    history.append({'step':0,**baseline})
    best=baseline['validation_composite'];best_step=0
    best_state={name:value.detach().cpu().clone() for name,value in ema.state_dict().items()}
    provenance.update(initial_validation=baseline,initial_selected_training_step=original['training']['selected_step'],
                      selected_validation_noise_mse=baseline['validation_noise_mse'],
                      selected_validation_gated_fk_loss=baseline['validation_gated_fk_loss'])
    print(f'Relative Cartesian baseline: noise {baseline["validation_noise_mse"]:.6f}, FK {baseline["validation_gated_fk_loss"]:.6f}, composite {best:.6f}',flush=True)
    for step in range(1,steps+1):
        indices=[]
        for group in ranges:
            selected=group[torch.randint(len(group),(batch_size//2,),device=device,generator=generator)]
            indices.append(selected[:,0]+(torch.rand(len(selected),device=device,generator=generator)*selected[:,1]).long())
        index=torch.cat(indices);data=arrays['train'];action_index=torch.minimum(index[:,None]+future,data['ends'][index,None])
        clean=data['q'][action_index].transpose(1,2);obs=data['obs'][index]
        noise=torch.randn(clean.shape,device=device,generator=generator);time=torch.randint(100,(len(index),),device=device,generator=generator)
        noisy=noise_scheduler.add_noise(clean,noise,time);prediction=model(noisy,time,obs)
        noise_loss=torch.nn.functional.mse_loss(prediction,noise)
        fk_loss,fraction=clean_fk_loss(noisy,prediction,clean,time,alphas)
        loss=noise_loss+fk_weight*fk_loss
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();rate.step();decay=min(.995,(1+step)/(10+step))
        with torch.no_grad():
            for target,current in zip(ema.parameters(),model.parameters()):target.mul_(decay).add_(current,alpha=1-decay)
        total+=float(loss.detach());total_noise+=float(noise_loss.detach());total_fk+=float(fk_loss.detach())
        total_fraction+=float(fraction);total_gradient+=float(gradient);count_since_validation+=1
        if step%validation_every and step!=steps:continue
        measured=validation_metrics(ema,arrays['validation'],validation_index,validation_noise,validation_time,
                                     future,noise_scheduler,alphas,fk_weight,batch_size)
        validation_loss=measured['validation_composite']
        validation_noise_loss=measured['validation_noise_mse'];validation_fk_loss=measured['validation_gated_fk_loss']
        history.append({'step':step,'train_composite':total/count_since_validation,'train_noise_mse':total_noise/count_since_validation,
                        'train_gated_fk_loss':total_fk/count_since_validation,'train_eligible_fraction':total_fraction/count_since_validation,
                        'train_gradient_norm_before_clip':total_gradient/count_since_validation,'validation_composite':validation_loss,
                        'validation_noise_mse':validation_noise_loss,'validation_gated_fk_loss':validation_fk_loss})
        total=total_noise=total_fk=total_fraction=total_gradient=0.;count_since_validation=0
        if validation_loss<best-1e-7:
            best=validation_loss;best_step=step;stale=0
            best_state={name:value.detach().cpu().clone() for name,value in ema.state_dict().items()}
            provenance['selected_validation_noise_mse']=validation_noise_loss
            provenance['selected_validation_gated_fk_loss']=validation_fk_loss
        else:stale+=1
        provenance.update(completed_steps=step,completed_epochs=max(1,int(step*batch_size/metadata['frames']['train'])),
                          selected_step=best_step,best_validation_composite=best,seconds=perf_counter()-began)
        temporary=output/'diffusion_policy_6dof.tmp';torch.save({'configuration':configuration,'training':provenance,'state_dict':best_state},temporary)
        temporary.replace(output/'diffusion_policy_6dof.pth');write_json(output/'training.json',provenance)
        print(f'Relative Cartesian diffusion {step}/{steps}: validation {validation_loss:.6f}, best {best_step}, seconds {provenance["seconds"]:.1f}',flush=True)
        if stale>=patience:break
    return provenance


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('simulation/artifacts/diffusion_100k_v1/demonstrations.h5'))
    parser.add_argument('--augmentation',type=Path,default=Path('simulation/artifacts/diffusion_motion_v2/augmentation.h5'))
    parser.add_argument('--cache',type=Path,default=Path('simulation/artifacts/diffusion_cartesian_relative_v4/cache'))
    parser.add_argument('--output',type=Path,default=Path('simulation/artifacts/diffusion_cartesian_relative_v4/model'))
    parser.add_argument('--initial',type=Path,default=Path('simulation/artifacts/diffusion_cartesian_v3_refined/model/diffusion_policy_6dof.pth'))
    parser.add_argument('--steps',type=int,default=20000);parser.add_argument('--batch-size',type=int,default=512)
    parser.add_argument('--fk-weight',type=float,default=.02);parser.add_argument('--learning-rate',type=float,default=.0003)
    args=parser.parse_args()
    if not np.isfinite(args.fk_weight) or args.fk_weight<0 or not np.isfinite(args.learning_rate) or args.learning_rate<=0:
        raise ValueError('Need finite nonnegative FK weight and positive learning rate')
    refine(args.source,args.augmentation,args.cache,args.output,args.initial,args.steps,args.batch_size,
           fk_weight=args.fk_weight,learning_rate=args.learning_rate)
