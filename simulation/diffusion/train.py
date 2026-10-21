"""Train conditional action-noise prediction with DDPM, AdamW and EMA."""
import argparse
import copy
from importlib.metadata import version
import math
import os
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from .common import ARTIFACTS, file_sha256, robot_schema, write_json
from .dataset import TrajectoryDataset
from .model import ConditionalUNet1D, scheduler


def train(dataset,output,epochs=80,batch_size=512,learning_rate=.001,seed=42,device='auto',
          channels=32,patience=12,smoke=False,validation_windows=2048):
    """Batch tensors: actions/noise [B,6,16], observations [B,19], times [B]."""
    if epochs<1 or batch_size<1 or patience<1 or validation_windows<1 or not math.isfinite(learning_rate) or learning_rate<=0:
        raise ValueError('Positive epochs, batch size, patience, validation size and learning rate required')
    output=Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Choose an empty model output directory; existing artifacts are preserved')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.manual_seed(seed);torch.set_num_threads(4)
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.use_deterministic_algorithms(True)
    training=TrajectoryDataset(dataset,'train');validation=TrajectoryDataset(dataset,'validation')
    assert not set(training.episode_ids)&set(validation.episode_ids)
    indices=np.random.default_rng(seed+1).choice(len(validation),min(validation_windows,len(validation)),replace=False)
    loader=DataLoader(training,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=device=='cuda',
                      generator=torch.Generator().manual_seed(seed))
    validation_loader=DataLoader(Subset(validation,indices.tolist()),batch_size=batch_size,shuffle=False,num_workers=0)
    model=ConditionalUNet1D(channels).to(device);ema=copy.deepcopy(model).eval()
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=1e-6)
    rate=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=learning_rate*.01)
    noise_scheduler=scheduler();began=perf_counter();history=[];best=float('inf');best_state=None;best_epoch=0;stale=0;steps=0
    output.mkdir(parents=True,exist_ok=True)
    configuration={'robot':robot_schema(),'channels':channels,'diffusion_timesteps':100,
                   'scheduler':{'beta_schedule':'squaredcos_cap_v2','prediction_type':'epsilon',
                                'clip_sample':True,'timestep_spacing':'linspace'}}
    provenance={'kind':'smoke' if smoke else 'full','seed':seed,'dataset':str(dataset),
                'dataset_sha256':file_sha256(dataset),'train_episodes':len(training.episode_ids),
                'validation_episodes':len(validation.episode_ids),'train_windows':len(training),
                'validation_windows':len(indices),'batch_size':batch_size,'requested_epochs':epochs,
                'learning_rate':learning_rate,'optimizer':'AdamW + cosine learning-rate decay',
                'objective':'MSE(predicted diffusion noise, sampled Gaussian noise)',
                'ema':'warm decay min(0.995, (1+steps)/(10+steps))',
                'device':device,'gpu':torch.cuda.get_device_name() if device=='cuda' else None,
                'versions':{name:version(name) for name in ('torch','diffusers','numpy','h5py')},
                'selection':'Lowest fixed-seed validation noise MSE using EMA; test episodes never loaded',
                'history':history}
    for epoch in range(epochs):
        model.train();total=0.;count=0
        for batch in loader:
            action=batch['action'].to(device);obs=batch['obs'].to(device)
            noise=torch.randn_like(action);times=torch.randint(100,(len(action),),device=device)
            noisy=noise_scheduler.add_noise(action,noise,times)
            prediction=model(noisy,times,obs);loss=torch.nn.functional.mse_loss(prediction,noise)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite diffusion training loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();steps+=1
            decay=min(.995,(1+steps)/(10+steps))
            with torch.no_grad():
                for target,source in zip(ema.parameters(),model.parameters()):target.mul_(decay).add_(source,alpha=1-decay)
            total+=float(loss.detach())*len(action);count+=len(action)
        rate.step();generator=torch.Generator().manual_seed(seed+2);validation_sum=0.;validation_count=0
        with torch.no_grad():
            for batch in validation_loader:
                clean=batch['action'].to(device);obs=batch['obs'].to(device)
                noise=torch.randn(clean.shape,generator=generator).to(device)
                times=torch.randint(100,(len(clean),),generator=generator).to(device)
                noisy=noise_scheduler.add_noise(clean,noise,times)
                loss=torch.nn.functional.mse_loss(ema(noisy,times,obs),noise)
                validation_sum+=float(loss)*len(clean);validation_count+=len(clean)
        value=validation_sum/validation_count
        if not math.isfinite(value):raise RuntimeError('Nonfinite validation loss')
        history.append({'epoch':epoch+1,'train_noise_mse':total/count,'validation_noise_mse':value})
        if value<best-1e-7:
            best=value;best_epoch=epoch+1;stale=0
            best_state={name:value.detach().cpu().clone() for name,value in ema.state_dict().items()}
        else:stale+=1
        provenance.update(completed_epochs=epoch+1,completed_steps=steps,selected_epoch=best_epoch,
                          best_validation_noise_mse=best,seconds=perf_counter()-began)
        temporary=output/'diffusion_policy_6dof.tmp'
        torch.save({'configuration':configuration,'training':provenance,'state_dict':best_state},temporary)
        temporary.replace(output/'diffusion_policy_6dof.pth');write_json(output/'training.json',provenance)
        print(f'Diffusion {epoch+1}/{epochs}: train {total/count:.6f} validation {value:.6f}, best epoch {best_epoch}',flush=True)
        if stale>=patience:break
    return provenance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=ARTIFACTS/'demonstrations.h5')
    parser.add_argument('--output',type=Path,default=ARTIFACTS/'model')
    parser.add_argument('--epochs',type=int,default=80);parser.add_argument('--batch-size',type=int,default=512)
    parser.add_argument('--learning-rate',type=float,default=.001);parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='auto');parser.add_argument('--channels',type=int,default=32)
    parser.add_argument('--patience',type=int,default=12);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();train(args.dataset,args.output,args.epochs,args.batch_size,args.learning_rate,args.seed,
        args.device,args.channels,args.patience,args.smoke)


if __name__=='__main__':main()
