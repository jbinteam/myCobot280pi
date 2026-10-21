"""Fine-tune Cartesian-conditioned joint diffusion without inference-time IK.

Goal joint labels in preserved expert datasets are used only to derive XYZ and
flange-Z by shared URDF FK. They never appear in model observations. Repeated FK
labels are deduplicated in large batches before GPU transforms.
"""
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
from .cartesian_conditioning import CONDITIONING
from .motion_conditioning import CONDITIONING as MOTION_CONDITIONING
from .model import ConditionalUNet1D, scheduler
from ..kinematics import ARM


def warm_start_state(checkpoint):
    """Drop the six old goal-joint feature weights, retaining shared motion priors."""
    config = checkpoint['configuration']
    if config['robot'] != robot_schema() or config.get('conditioning') != MOTION_CONDITIONING:
        raise ValueError('Expected compatible motion v2 checkpoint')
    state = {name: value.clone() for name, value in checkpoint['state_dict'].items()}
    weight = state['observation.0.weight']
    if weight.shape != (128, 34):
        raise ValueError('Expected 34-feature v2 observation weights')
    weight[:, 6:12] = 0
    return state


def cartesian_labels(goals, device='cpu', batch_size=32768):
    """Exact source float64 labels, deduplicated before shared batched URDF FK."""
    goals = np.asarray(goals, np.float64)
    if goals.ndim != 2 or goals.shape[1] != 6 or not np.isfinite(goals).all():
        raise ValueError('Expected finite [N,6] FK labels')
    unique, inverse = np.unique(goals, axis=0, return_inverse=True)
    labels = np.empty((len(unique), 6), np.float32)
    with torch.no_grad():
        for start in range(0, len(unique), batch_size):
            transform = ARM.torch_transform(torch.as_tensor(unique[start:start+batch_size], device=device))
            value = torch.cat((transform[:, :3, 3] / ARM.scale, transform[:, :3, 2]), dim=1)
            labels[start:start+len(value)] = value.cpu().numpy()
    return labels[inverse]


def _split(group):
    split = group.attrs['split']
    if split not in ('train', 'validation', 'test'):
        raise ValueError(f'Unexpected dataset split: {split}')
    return split


def build_cache(source, augmentation, folder, device='cpu', chunk_frames=100000):
    """Cache schema-identified Cartesian observations, retaining parent splits."""
    source = Path(source); augmentation = Path(augmentation); folder = Path(folder)
    if chunk_frames < 1:
        raise ValueError('Cache chunk_frames must be positive')
    folder.mkdir(parents=True, exist_ok=True)
    identity = {'source': str(source), 'source_sha256': file_sha256(source),
                'augmentation': str(augmentation), 'augmentation_sha256': file_sha256(augmentation),
                'robot': robot_schema(), 'conditioning': CONDITIONING, 'cache_schema': 1,
                'label_transform': 'float64 shared URDF FK; XYZ/ARM.scale + directed flange-Z'}
    identity['dataset_sha256'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest = folder / 'cache.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved['identity'] != identity:
            raise ValueError('Cache sources/schema changed')
        for split in ('train', 'validation', 'test'):
            for key, shape, dtype in (('obs', (saved['frames'][split], 34), np.float32),
                                      ('q', (saved['frames'][split], 6), np.float32),
                                      ('ends', (saved['frames'][split],), np.int64)):
                array = np.load(folder / f'{split}_{key}.npy', mmap_mode='r')
                if array.shape != shape or array.dtype != dtype:
                    raise ValueError('Incomplete or incompatible Cartesian cache')
            for index in (0, 1):
                np.load(folder / f'{split}_ranges_{index}.npy', mmap_mode='r')
        return saved
    # A partial cache is never accepted; it is rebuilt before the final manifest.
    counts = {split: 0 for split in ('train', 'validation', 'test')}
    episode_counts = counts.copy(); parents = {}; seen_parents = set()
    for source_index, path in enumerate((source, augmentation)):
        with h5py.File(path, 'r') as data:
            metadata = json.loads(data.attrs['metadata_json'])
            if metadata['completed_episodes'] != len(data['episodes']) or metadata['environment']['robot'] != robot_schema():
                raise ValueError('Incomplete or incompatible dataset')
            if source_index and metadata.get('source_sha256') != identity['source_sha256']:
                raise ValueError('Augmentation does not derive from this source')
            for name, group in data['episodes'].items():
                split = _split(group)
                counts[split] += len(group['q_rad']); episode_counts[split] += 1
                if not source_index:
                    parents[name] = split
                else:
                    parent = group.attrs['parent_episode']
                    if parent not in parents or parents[parent] != split or parent in seen_parents:
                        raise ValueError('Augmentation parent missing, duplicated, or split leakage')
                    seen_parents.add(parent)
    arrays = {}; ranges = {split: [[], []] for split in counts}; offsets = {split: 0 for split in counts}
    for split, count in counts.items():
        if not count:
            raise ValueError(f'Expected nonempty {split} dataset')
        arrays[split] = {key: np.lib.format.open_memmap(folder / f'{split}_{key}.npy', mode='w+', dtype=dtype, shape=shape)
                         for key, dtype, shape in [('obs', np.float32, (count, 34)),
                                                   ('q', np.float32, (count, 6)), ('ends', np.int64, (count,))]}
    pending = []; pending_frames = 0

    def flush():
        nonlocal pending_frames
        labels = cartesian_labels(np.concatenate([record[3] for record in pending]), device)
        label_offset = 0
        for split, source_index, q, goal, geometry, velocity, acceleration, target_velocity in pending:
            count = len(q); start = offsets[split]; end = start + count
            obs = np.concatenate((normalize(q), labels[label_offset:label_offset+count],
                                  np.broadcast_to(geometry, (count, 7)), velocity, acceleration / 3., target_velocity / .5), axis=1)
            arrays[split]['obs'][start:end] = obs
            arrays[split]['q'][start:end] = normalize(q)
            arrays[split]['ends'][start:end] = end - 1
            ranges[split][source_index].append([start, count]); offsets[split] = end; label_offset += count
        pending.clear(); pending_frames = 0

    for source_index, path in enumerate((source, augmentation)):
        with h5py.File(path, 'r') as data:
            for number, group in enumerate(data['episodes'].values()):
                q = group['q_rad'][:]; times = group['time_s'][:]; count = len(q); split = _split(group)
                if (q.shape != (count, 6) or count < 2 or not np.isfinite(q).all() or np.any(q < ARM.lower)
                        or np.any(q > ARM.upper) or times.shape != (count,) or not np.isfinite(times).all()
                        or not np.allclose(np.diff(times), .1, atol=1e-9)):
                    raise ValueError('Invalid cache trajectory')
                if source_index == 0:
                    goal = np.broadcast_to(group['q_goal_rad'][:], q.shape); box = group['obstacle_m'][:]
                    velocity, acceleration = derivatives(q, times); target_velocity = np.zeros((count, 3))
                else:
                    goal = group['goal_rad'][:]
                    box = group['obstacle_m'][:] if group.attrs['obstacle_present'] else None
                    velocity = group['velocity_rad_s'][:]; acceleration = group['acceleration_rad_s2'][:]
                    target_velocity = group['target_velocity_m_s'][:]
                if (goal.shape != q.shape or not np.isfinite(goal).all() or np.any(goal < ARM.lower) or np.any(goal > ARM.upper)
                        or velocity.shape != q.shape or acceleration.shape != q.shape or target_velocity.shape != (count, 3)
                        or not np.isfinite(velocity).all() or not np.isfinite(acceleration).all() or not np.isfinite(target_velocity).all()
                        or np.max(np.linalg.norm(velocity, axis=1)) > 1+1e-6
                        or np.max(np.linalg.norm(acceleration, axis=1)) > 3+1e-6
                        or np.max(np.linalg.norm(target_velocity, axis=1)) > .5+1e-6):
                    raise ValueError('Invalid motion labels')
                if box is not None and (box.shape != (6,) or not np.isfinite(box).all() or np.any(box[3:] <= 0)):
                    raise ValueError('Invalid obstacle geometry')
                geometry = np.zeros(7) if box is None else np.r_[box / ARM.scale, 1.]
                pending.append((split, source_index, q, goal, geometry, velocity, acceleration, target_velocity))
                pending_frames += count
                if pending_frames >= chunk_frames:
                    flush()
                if number % 10000 == 0:
                    print(f'Cartesian cache source {source_index}: {number} episodes', flush=True)
    if pending:
        flush()
    for split in arrays:
        for value in arrays[split].values():
            value.flush()
        for index, value in enumerate(ranges[split]):
            np.save(folder / f'{split}_ranges_{index}.npy', np.asarray(value, np.int64).reshape(-1, 2))
    saved = {'identity': identity, 'frames': counts, 'episodes': episode_counts,
             'sampling': '50% original / 50% augmented; uniform parent episodes then uniform frame',
             'test_isolation': 'Only train/validation cache arrays loaded by training; all augmentation parent splits verified',
             'target_label_source': 'Expert joint goal -> URDF FK only; goal joints absent from observations'}
    write_json(manifest, saved)
    return saved


def train(source,augmentation,cache,output,initial,steps=20000,batch_size=512,seed=20261008,validation_every=500,patience=10):
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
    state=warm_start_state(original)
    model=ConditionalUNet1D(original['configuration']['channels'],34).to(device)
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
                'device':device,'gpu':torch.cuda.get_device_name(),'objective':'Gaussian noise MSE; Cartesian target FK labels, collision-checked joint actions',
                'selection':'Lowest fixed-seed EMA validation noise MSE; rollout test set never loaded',
                'warm_start':'v2 EMA weights; obsolete goal-joint columns 6:12 zeroed',
                'goal_joint_input':False,'test_isolation':metadata['test_isolation'],'history':history}
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
        print(f'Cartesian diffusion {step}/{steps}: validation {validation_loss:.6f}, best {best_step}, seconds {provenance["seconds"]:.1f}',flush=True)
        if stale>=patience:break
    return provenance


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('simulation/artifacts/diffusion_100k_v1/demonstrations.h5'))
    parser.add_argument('--augmentation',type=Path,default=Path('simulation/artifacts/diffusion_motion_v2/augmentation.h5'))
    parser.add_argument('--cache',type=Path,default=Path('simulation/artifacts/diffusion_cartesian_v3/cache'))
    parser.add_argument('--output',type=Path,default=Path('simulation/artifacts/diffusion_cartesian_v3/model'))
    parser.add_argument('--initial',type=Path,default=Path('simulation/artifacts/diffusion_motion_v2/model/diffusion_policy_6dof.pth'))
    parser.add_argument('--steps',type=int,default=20000);parser.add_argument('--batch-size',type=int,default=512)
    args=parser.parse_args();train(args.source,args.augmentation,args.cache,args.output,args.initial,args.steps,args.batch_size)
