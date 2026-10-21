"""Experimental one-pass conditional-VAE transformer trajectory proposals.

The existing collision/continuous-curve gates must authorize every proposal.
This module is not selected by the active WebUI manifest or hardware adapter.
"""
import os
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
from ..kinematics import ARM
from .common import robot_schema, file_sha256, denormalize
from .motion_conditioning import CONDITIONING


def configuration():
    return {'kind':'trajectory_transformer_cvae_v1','robot':robot_schema(),
            'conditioning':CONDITIONING,'width':64,'heads':4,'latent_dim':16,
            'posterior_layers':1,'decoder_layers':2,'horizon':16,
            'output':'absolute operating-box-normalized joint positions; residual to current joints',
            'input':'current6/goal6/scene7/motion15; four conditioning tokens',
            'inference':'one decoder pass; first candidate zero latent, remaining Gaussian latent'}


class TrajectoryTransformer(nn.Module):
    def __init__(self):
        super().__init__();width=64
        self.features=nn.ModuleList(nn.Linear(n,width) for n in (6,6,7,15))
        self.feature_position=nn.Parameter(torch.randn(4,width)*.02)
        self.posterior_action=nn.Linear(6,width)
        self.posterior_position=nn.Parameter(torch.randn(17,width)*.02)
        self.cls=nn.Parameter(torch.zeros(1,1,width))
        encoder=nn.TransformerEncoderLayer(width,4,256,dropout=0.,batch_first=True,norm_first=True)
        self.posterior=nn.TransformerEncoder(encoder,1,enable_nested_tensor=False)
        self.latent_parameters=nn.Linear(width,32)
        self.latent_projection=nn.Linear(16,width)
        self.queries=nn.Parameter(torch.randn(16,width)*.02)
        decoder=nn.TransformerDecoderLayer(width,4,256,dropout=0.,batch_first=True,norm_first=True)
        self.decoder=nn.TransformerDecoder(decoder,2,norm=nn.LayerNorm(width))
        self.output=nn.Linear(width,6)
        nn.init.normal_(self.output.weight,std=.001);nn.init.zeros_(self.output.bias)

    def memory(self,obs):
        return torch.stack([layer(part) for layer,part in zip(self.features,torch.split(obs,[6,6,7,15],dim=-1))],dim=1)+self.feature_position

    def decode(self,obs,latent):
        memory=torch.cat((self.memory(obs),self.latent_projection(latent)[:,None]),dim=1)
        hidden=self.decoder(self.queries[None].expand(len(obs),-1,-1),memory)
        return (obs[:,:6,None].transpose(1,2)+self.output(hidden)).clamp(-1,1)

    def forward(self,obs,actions,noise):
        tokens=torch.cat((self.cls.expand(len(obs),-1,-1),self.posterior_action(actions)),dim=1)+self.posterior_position
        # Current/goal/scene/motion are also available to the posterior.
        hidden=self.posterior(torch.cat((tokens,self.memory(obs)),dim=1))[:,0]
        mean,logvar=self.latent_parameters(hidden).chunk(2,dim=-1);logvar=logvar.clamp(-8,4)
        latent=mean+torch.exp(.5*logvar)*noise
        return self.decode(obs,latent),mean,logvar


class TransformerPolicy:
    observation_dim=34
    conditioning=CONDITIONING
    def __init__(self,path,device='cuda',seed=42):
        self.path=Path(path);self.device=device;self.sha256=file_sha256(path)
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if payload.get('configuration')!=configuration():raise ValueError('Transformer schema/robot mismatch')
        self.training=payload.get('training',{})
        if not self.training.get('completed') or self.training.get('completed_steps',0)<1:
            raise ValueError('Transformer checkpoint is not trained')
        self.model=TrajectoryTransformer();self.model.load_state_dict(payload['state_dict'],strict=True)
        if not all(torch.isfinite(p).all() for p in self.model.parameters()):raise ValueError('Nonfinite transformer weights')
        self.model.to(device).eval();self.generator=torch.Generator(device=device).manual_seed(seed)

    @torch.inference_mode()
    def sample(self,observations,samples=4,inference_steps=32):
        obs=np.asarray(observations,np.float32)
        if obs.ndim!=2 or obs.shape[1]!=34 or not len(obs) or not np.isfinite(obs).all():
            raise ValueError('Expected finite [N,34] observations')
        if isinstance(samples,bool) or not isinstance(samples,int) or not 1<=samples<=64:
            raise ValueError('Samples must be an integer 1..64')
        condition=torch.as_tensor(obs,device=self.device).repeat_interleave(samples,dim=0)
        latent=torch.randn((len(condition),16),device=self.device,generator=self.generator)
        latent[::samples]=0
        value=self.model.decode(condition,latent)
        if not torch.isfinite(value).all():raise RuntimeError('Nonfinite transformer actions')
        return ARM.clip(denormalize(value.cpu().numpy().reshape(len(obs),samples,16,6)))
