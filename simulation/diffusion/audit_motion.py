"""Independent replay and split/provenance audit of motion augmentation."""
import json
from pathlib import Path
import h5py
import numpy as np
from ..kinematics import ARM
from .common import file_sha256,write_json
from .env import ArmEnv


def audit(source,augmentation,output,sample_size=1000):
    counts={};parents=set();frames=0;replayed=0;max_v=max_a=0.
    with h5py.File(source,'r') as original,h5py.File(augmentation,'r') as data,ArmEnv() as env:
        metadata=json.loads(data.attrs['metadata_json'])
        if metadata['source_sha256']!=file_sha256(source) or metadata['environment']!=env.metadata():raise ValueError('Augmentation provenance/environment mismatch')
        if metadata['completed_episodes']!=len(data['episodes']):raise ValueError('Incomplete augmentation')
        names=sorted(data['episodes']);selected=set(np.random.default_rng(20261009).choice(names,min(sample_size,len(names)),replace=False))
        for number,name in enumerate(names):
            group=data['episodes'][name];parent=group.attrs['parent_episode']
            if parent in parents or group.attrs['split']!=original['episodes'][parent].attrs['split']:raise ValueError('Duplicate parent or split leakage')
            parents.add(parent);counts[group.attrs['split']]=counts.get(group.attrs['split'],0)+1
            q=group['q_rad'][:];goals=group['goal_rad'][:];times=group['time_s'][:]
            v=group['velocity_rad_s'][:];a=group['acceleration_rad_s2'][:];tv=group['target_velocity_m_s'][:]
            if (q.shape!=goals.shape or q.shape!=v.shape or q.shape!=a.shape or tv.shape!=(len(q),3)
                    or not all(np.isfinite(value).all() for value in (q,goals,times,v,a,tv))
                    or np.any(q<ARM.lower) or np.any(q>ARM.upper) or np.any(goals<ARM.lower) or np.any(goals>ARM.upper)
                    or not np.array_equal(q[-1],goals[-1]) or not np.allclose(np.diff(times),.1,atol=1e-9)):
                raise ValueError('Invalid augmented labels')
            max_v=max(max_v,float(np.linalg.norm(v,axis=1).max()));max_a=max(max_a,float(np.linalg.norm(a,axis=1).max()))
            if max_v>1+1e-6 or max_a>3+1e-6 or np.linalg.norm(tv,axis=1).max()>.5+1e-6:raise ValueError('Motion bounds exceeded')
            if name in selected:
                env.set_obstacle(group['obstacle_m'][:] if group.attrs['obstacle_present'] else None)
                if not env.path_valid(q) or not all(env.valid(goal) for goal in goals):raise ValueError('Independent collision replay failed')
                replayed+=len(q)
            frames+=len(q)
            if number%5000==0:print(f'Motion audit {number}/{len(names)}',flush=True)
    report={'kind':'independent_motion_dataset_audit','episodes':len(names),'frames':frames,'unique_source_episodes':len(parents),
            'all_parent_splits_match':True,'split_counts':counts,'replayed_episodes':len(selected),'replayed_states':replayed,
            'max_velocity_rad_s':max_v,'max_acceleration_rad_s2':max_a,'source_sha256':file_sha256(source),
            'augmentation_sha256':file_sha256(augmentation),'passed':True}
    write_json(output,report);print(json.dumps(report),flush=True);return report


if __name__=='__main__':
    audit(Path('simulation/artifacts/diffusion_100k_v1/demonstrations.h5'),Path('simulation/artifacts/diffusion_motion_v2/augmentation.h5'),
          Path('output/diffusion/motion_v2_dataset_validation.json'))
