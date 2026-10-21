"""Episode-separated HDF5 demonstrations and [6,16] future-position windows."""
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from ..kinematics import ARM
from .common import HORIZON, normalize, observation, robot_schema


def read_episodes(path,split=None):
    """Read validated episodes; return metadata and dicts containing [T,6] q."""
    if split not in (None,'train','validation','test'):raise ValueError('Unknown episode split')
    episodes=[]
    with h5py.File(Path(path),'r') as data:
        metadata=json.loads(data.attrs['metadata_json'])
        if metadata.get('completed_episodes')!=len(data['episodes']):raise ValueError('Dataset is incomplete')
        if metadata['environment']['robot']!=robot_schema():raise ValueError('Dataset URDF/geometry/normalization mismatch')
        for name,group in data['episodes'].items():
            stored_split=group.attrs['split']
            if stored_split not in ('train','validation','test'):raise ValueError('Invalid stored split')
            if split is not None and stored_split!=split:continue
            q=group['q_rad'][:];goal=group['q_goal_rad'][:];box=group['obstacle_m'][:];times=group['time_s'][:]
            if q.ndim!=2 or q.shape[1]!=6 or len(q)<2 or times.shape!=(len(q),) or not np.isfinite(q).all() or not np.isfinite(times).all():
                raise ValueError('Invalid trajectory shapes or nonfinite values')
            if np.any(q<ARM.lower) or np.any(q>ARM.upper) or np.any(np.diff(times)<=0):raise ValueError('Invalid trajectory limits/timing')
            if goal.shape!=(6,) or not np.array_equal(q[-1],goal) or box.shape!=(6,) or not np.isfinite(box).all() or np.any(box[3:]<=0):
                raise ValueError('Invalid goal or obstacle')
            transform=ARM.transform(goal)
            if not np.allclose(transform[:3,3],group['goal_xyz_m'][:],atol=1e-9,rtol=0) or not np.allclose(transform[:3,2],group['goal_axis_base'][:],atol=1e-9,rtol=0):
                raise ValueError('Goal labels differ from shared 5D FK')
            episodes.append({'id':name,'split':stored_split,'q':q,'goal':goal,'obstacle':box,'times':times,
                             'record':json.loads(group.attrs['record_json'])})
    if not episodes:raise ValueError(f'No episodes in requested split {split}')
    return metadata,episodes


class TrajectoryDataset(Dataset):
    """Each item has obs [19] and absolute normalized future actions [6,16].

    The first action is q[t+1], never the current observation q[t]. Short tails
    repeat the exact terminal state. Windows never cross an episode boundary.
    Operating limits define normalization, independent of the held-out data.
    """
    def __init__(self,path,split='train'):
        self.metadata,episodes=read_episodes(path,split)
        self.episode_ids=[e['id'] for e in episodes];observations=[];actions=[];indices=[]
        for e in episodes:
            q=e['q'];future=np.minimum(np.arange(len(q))[:,None]+np.arange(1,HORIZON+1),len(q)-1)
            actions.append(np.clip(normalize(q[future]),-1,1).transpose(0,2,1).astype(np.float32))
            observations.append(np.stack([observation(current,e['goal'],e['obstacle']) for current in q]))
            indices.extend((e['id'],i) for i in range(len(q)))
        self.observations=torch.from_numpy(np.concatenate(observations))
        self.actions=torch.from_numpy(np.concatenate(actions));self.window_indices=indices

    def __len__(self):return len(self.actions)

    def __getitem__(self,index):
        return {'obs':self.observations[index],'action':self.actions[index]}
