"""Small conditioned temporal U-Net predicting DDPM noise in [B,6,16]."""
import math
import torch
from torch import nn
from diffusers import DDPMScheduler
from .common import HORIZON, OBS_DIM


def scheduler(timesteps=100):
    return DDPMScheduler(num_train_timesteps=timesteps,beta_schedule='squaredcos_cap_v2',
                         prediction_type='epsilon',clip_sample=True,timestep_spacing='linspace')


class Residual(nn.Module):
    """Conv1d residual block with time/observation FiLM; [B,C,T] -> [B,D,T]."""
    def __init__(self,inputs,outputs,condition=128):
        super().__init__()
        self.first=nn.Sequential(nn.Conv1d(inputs,outputs,3,padding=1),nn.GroupNorm(8,outputs),nn.SiLU())
        self.film=nn.Linear(condition,outputs*2)
        self.second=nn.Sequential(nn.Conv1d(outputs,outputs,3,padding=1),nn.GroupNorm(8,outputs),nn.SiLU())
        self.skip=nn.Conv1d(inputs,outputs,1) if inputs!=outputs else nn.Identity()

    def forward(self,x,condition):
        scale,bias=self.film(condition).chunk(2,dim=-1)
        value=self.first(x)*(1+scale[:,:,None])+bias[:,:,None]
        return self.skip(x)+self.second(value)


class ConditionalUNet1D(nn.Module):
    """Inputs: noisy joints [B,6,16], diffusion time [B], observation [B,19]."""
    def __init__(self,channels=32,observation_dim=OBS_DIM):
        super().__init__()
        if channels<8 or channels%8:raise ValueError('Channel width must be a positive multiple of eight')
        self.channels=channels
        if observation_dim not in (19,34):raise ValueError('Unsupported observation dimension')
        self.observation_dim=observation_dim
        self.time=nn.Sequential(nn.Linear(64,128),nn.SiLU(),nn.Linear(128,128))
        self.observation=nn.Sequential(nn.Linear(observation_dim,128),nn.SiLU(),nn.Linear(128,128))
        self.down0=Residual(6,channels)
        self.stride0=nn.Conv1d(channels,channels*2,4,stride=2,padding=1)
        self.down1=Residual(channels*2,channels*2)
        self.stride1=nn.Conv1d(channels*2,channels*4,4,stride=2,padding=1)
        self.middle=nn.ModuleList([Residual(channels*4,channels*4),Residual(channels*4,channels*4)])
        self.up1=nn.ConvTranspose1d(channels*4,channels*2,4,stride=2,padding=1)
        self.decode1=Residual(channels*4,channels*2)
        self.up0=nn.ConvTranspose1d(channels*2,channels,4,stride=2,padding=1)
        self.decode0=Residual(channels*2,channels)
        self.output=nn.Conv1d(channels,6,1)

    def forward(self,noisy,timestep,obs):
        if noisy.ndim!=3 or noisy.shape[1:]!=(6,HORIZON) or obs.shape!=(len(noisy),self.observation_dim):
            raise ValueError(f'Expected noisy [B,6,16] and observation [B,{self.observation_dim}]')
        timestep=torch.as_tensor(timestep,device=noisy.device).reshape(-1)
        if len(timestep)==1:timestep=timestep.expand(len(noisy))
        if len(timestep)!=len(noisy):raise ValueError('Expected one timestep per sample')
        frequency=torch.exp(torch.arange(32,device=noisy.device,dtype=noisy.dtype)*(-math.log(10000)/31))
        phase=timestep.to(noisy.dtype)[:,None]*frequency
        condition=self.time(torch.cat((phase.sin(),phase.cos()),dim=-1))+self.observation(obs)
        skip0=self.down0(noisy,condition);skip1=self.down1(self.stride0(skip0),condition)
        value=self.stride1(skip1)
        for block in self.middle:value=block(value,condition)
        value=self.decode1(torch.cat((self.up1(value),skip1),dim=1),condition)
        value=self.decode0(torch.cat((self.up0(value),skip0),dim=1),condition)
        return self.output(value)
