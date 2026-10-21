"""Compatible DDPM inference and independently collision-gated action chunks."""
import os
from pathlib import Path
import numpy as np
import torch
from ..kinematics import ARM
from .common import HORIZON, OBS_DIM, denormalize, robot_schema, file_sha256
from .model import ConditionalUNet1D, scheduler
from .motion_conditioning import CONDITIONING
from .cartesian_conditioning import supports_conditioning


class DiffusionPolicy:
    """Generate joint trajectories with versioned v1/v2/v3/v4 observations."""
    def __init__(self,path,device='auto',seed=42,allow_smoke=False,*,cuda_graph=False,scheduler_cache=True):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
        self.device=device;self.path=Path(path);self.sha256=file_sha256(path)
        payload=torch.load(path,map_location='cpu',weights_only=True)
        config=payload['configuration'];training=payload['training']
        if config['robot']!=robot_schema():raise ValueError('Policy URDF/geometry/limits/observation mismatch')
        if training.get('kind') not in ('full','smoke') or training.get('completed_epochs',0)<1:
            raise ValueError('Policy has no completed training')
        if training['kind']=='smoke' and not allow_smoke:raise ValueError('Smoke weights require explicit allow_smoke')
        expected={'beta_schedule':'squaredcos_cap_v2','prediction_type':'epsilon','clip_sample':True,'timestep_spacing':'linspace'}
        if config['scheduler']!=expected or config['diffusion_timesteps']!=100:raise ValueError('Unsupported diffusion scheduler')
        conditioning=config.get('conditioning')
        if conditioning is not None and conditioning!=CONDITIONING and not supports_conditioning(conditioning):raise ValueError('Unsupported motion conditioning schema')
        self.conditioning=conditioning
        self.observation_dim=19 if conditioning is None else conditioning['observation_dim']
        self.model=ConditionalUNet1D(config['channels'],self.observation_dim)
        self.model.load_state_dict(payload['state_dict'],strict=True)
        if not all(torch.isfinite(value).all() for value in self.model.parameters()):raise ValueError('Nonfinite policy weights')
        self.model.to(device).eval();self.training=training;self.scheduler=scheduler()
        self.generator=torch.Generator(device=device).manual_seed(seed)
        self.cuda_graph=bool(cuda_graph and torch.device(device).type=='cuda')
        self._replay=None
        self.scheduler_cache=bool(scheduler_cache)
        self._scheduler_cache_hits=self._scheduler_cache_misses=0

    @torch.inference_mode()
    def sample(self,observations,samples=4,inference_steps=32):
        """Denoise Gaussian [N*samples,6,16]; return joint targets [N,samples,16,6]."""
        obs=np.asarray(observations,np.float32)
        if obs.ndim!=2 or obs.shape[1]!=self.observation_dim or not len(obs) or not np.isfinite(obs).all():raise ValueError(f'Expected finite [N,{self.observation_dim}] observations')
        if isinstance(samples,bool) or not isinstance(samples,int) or not 1<=samples<=64:raise ValueError('Samples must be 1..64')
        if isinstance(inference_steps,bool) or not isinstance(inference_steps,int) or not 1<=inference_steps<=100:raise ValueError('Inference steps must be 1..100')
        condition=torch.as_tensor(obs,device=self.device).repeat_interleave(samples,dim=0)
        value=torch.randn((len(condition),6,HORIZON),device=self.device,generator=self.generator)
        self.scheduler.set_timesteps(inference_steps,device=self.device)
        denoise=self.model
        if self.cuda_graph and condition.shape==(4,34):
            from .cuda_replay import DenoiserReplay
            if self._replay is None or self._replay.model is not self.model:
                self._replay=DenoiserReplay(self.model)
            denoise=self._replay
        from .timestep_cache import PreviousTimestepCache
        cache=PreviousTimestepCache(self.scheduler,enabled=self.scheduler_cache)
        try:
            with cache:
                for timestep in cache:
                    noise=denoise(value,timestep,condition)
                    value=self.scheduler.step(noise,timestep,value,generator=self.generator).prev_sample
        finally:
            self._scheduler_cache_hits+=cache.hits
            self._scheduler_cache_misses+=cache.misses
        if not torch.isfinite(value).all():raise RuntimeError('Nonfinite denoised actions')
        normalized=value.clamp(-1,1).cpu().numpy().transpose(0,2,1).reshape(len(obs),samples,HORIZON,6)
        return ARM.clip(denormalize(normalized))


def _project_chunk(current,chunk,dt,speed):
    result=[];previous=current
    for target in chunk:
        delta=target-previous;distance=np.linalg.norm(delta)
        target=ARM.clip(previous+delta*min(1.,speed*dt/max(distance,1e-12)))
        result.append(target);previous=target
    return np.asarray(result)


def projected_chunk(current,chunk,dt=.1,speed=1.):
    """Cheap ranking geometry only; this never authorizes a candidate route."""
    current=np.asarray(current,float);chunk=np.asarray(chunk,float)
    if current.shape!=(6,) or chunk.shape!=(HORIZON,6) or not np.isfinite(chunk).all():return None
    if not np.isfinite(dt) or dt<=0 or not np.isfinite(speed) or speed<=0:raise ValueError('Positive dt and speed required')
    return _project_chunk(current,chunk,dt,speed)


def checked_chunk(env,current,chunk,dt=.1,speed=1.,max_step_rad=.04):
    """Reject unsafe raw paths; speed-project and recheck the complete [16,6] chunk.

    A rejection never changes the execution pose. These reference checks bound
    speed, not acceleration or real actuator dynamics.
    """
    current=np.asarray(current,float);chunk=np.asarray(chunk,float)
    if current.shape!=(6,) or chunk.shape!=(HORIZON,6) or not np.isfinite(chunk).all():return None,'malformed'
    if not np.isfinite(dt) or dt<=0 or not np.isfinite(speed) or speed<=0:raise ValueError('Positive dt and speed required')
    if not env.path_valid(np.vstack((current,chunk)),max_step_rad):return None,'raw_collision'
    result=_project_chunk(current,chunk,dt,speed)
    if not env.path_valid(np.vstack((current,result)),max_step_rad):return None,'projected_collision'
    return result,'accepted'
