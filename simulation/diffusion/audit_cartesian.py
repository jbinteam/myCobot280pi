"""Independent Cartesian cache verification against preserved source FK labels."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from ..kinematics import ARM
from .common import file_sha256, write_json


def audit(cache=Path('simulation/artifacts/diffusion_cartesian_v3/cache'),
          output=Path('output/diffusion/cartesian_v3_dataset_validation.json'), samples=256):
    cache=Path(cache);metadata=json.loads((cache/'cache.json').read_text());identity=metadata['identity']
    schema=identity['conditioning']['schema']
    assert schema in (3,4) and identity['conditioning']['goal_joint_input'] is False
    for kind in ('source','augmentation'):
        assert file_sha256(Path(identity[kind]))==identity[f'{kind}_sha256']
    checked=0;max_position=max_axis=0.;rng=np.random.default_rng(20261005)
    for split in ('train','validation','test'):
        observations=np.load(cache/f'{split}_obs.npy',mmap_mode='r')
        actions=np.load(cache/f'{split}_q.npy',mmap_mode='r')
        ends=np.load(cache/f'{split}_ends.npy',mmap_mode='r')
        assert len(observations)==metadata['frames'][split] and observations.shape[1]==34
        for start in range(0,len(observations),100000):
            obs=observations[start:start+100000];q=actions[start:start+100000]
            assert np.isfinite(obs).all() and np.isfinite(q).all()
            assert np.abs(q).max()<=1+1e-6 and np.abs(obs[:,:6]).max()<=1+1e-6
            if schema==3:assert np.allclose(np.linalg.norm(obs[:,9:12],axis=1),1,atol=2e-6)
            else:assert np.linalg.norm(obs[:,9:12],axis=1).max()<=2+2e-6
            assert np.isin(obs[:,18],[0.,1.]).all()
            assert np.all(obs[obs[:,18]==0,12:18]==0)
            assert np.linalg.norm(obs[:,19:25],axis=1).max()<=1+1e-6
            assert np.linalg.norm(obs[:,25:31],axis=1).max()<=1+1e-6
            assert np.linalg.norm(obs[:,31:34],axis=1).max()<=1+1e-6
        ranges=[np.load(cache/f'{split}_ranges_{i}.npy') for i in (0,1)]
        combined=np.vstack(ranges)
        assert len(combined)==metadata['episodes'][split]
        assert combined[0,0]==0 and combined[-1].sum()==len(observations)
        assert np.array_equal(combined[1:,0],combined[:-1].sum(axis=1))
        for offset,count in combined:
            assert np.all(ends[offset:offset+count]==offset+count-1)
        for source_index,kind in enumerate(('source','augmentation')):
            path=Path(identity[kind])
            with h5py.File(path,'r') as data:
                names=[name for name,group in data['episodes'].items() if group.attrs['split']==split]
                assert len(names)==len(ranges[source_index])
                for index in rng.choice(len(names),size=min(samples//6,len(names)),replace=False):
                    group=data['episodes'][names[index]];offset,count=ranges[source_index][index]
                    frame=int(rng.integers(count));q=group['q_rad'][frame]
                    goal=group['q_goal_rad'][:] if source_index==0 else group['goal_rad'][frame]
                    transform=ARM.transform(goal);obs=observations[offset+frame]
                    current=ARM.transform(q)
                    expected_position=transform[:3,3]-(current[:3,3] if schema==4 else 0)
                    expected_axis=transform[:3,2]-(current[:3,2] if schema==4 else 0)
                    position_error=float(np.linalg.norm(obs[6:9]*ARM.scale-expected_position))
                    axis_error=float(np.linalg.norm(obs[9:12]-expected_axis))
                    max_position=max(max_position,position_error);max_axis=max(max_axis,axis_error)
                    assert position_error<2e-7 and axis_error<1e-6
                    assert np.allclose(obs[:6],(q-ARM.center)/ARM.radius,atol=1e-6)
                    assert np.allclose(actions[offset+frame],(q-ARM.center)/ARM.radius,atol=1e-6)
                    checked+=1
    source=Path(identity['source']);augmentation=Path(identity['augmentation']);parents=set()
    with h5py.File(source,'r') as original,h5py.File(augmentation,'r') as data:
        for group in data['episodes'].values():
            parent=group.attrs['parent_episode'];assert parent not in parents;parents.add(parent)
            assert original['episodes'][parent].attrs['split']==group.attrs['split']
    if schema==4:
        parent_cache=Path(identity['source_cartesian_cache'])
        assert file_sha256(parent_cache/'cache.json')==identity['source_cache_manifest_sha256']
        for filename,digest in identity['source_cache_arrays_sha256'].items():
            assert file_sha256(parent_cache/filename)==digest
            if not filename.endswith('_obs.npy'):assert file_sha256(cache/filename)==digest
        for split in ('train','validation','test'):
            before=np.load(parent_cache/f'{split}_obs.npy',mmap_mode='r')
            after=np.load(cache/f'{split}_obs.npy',mmap_mode='r')
            for start in range(0,len(before),100000):
                assert np.array_equal(before[start:start+100000,:6],after[start:start+100000,:6])
                assert np.array_equal(before[start:start+100000,12:],after[start:start+100000,12:])
    result={'passed':True,'conditioning':identity['conditioning'],'source_sha256':identity['source_sha256'],
            'augmentation_sha256':identity['augmentation_sha256'],'dataset_sha256':identity['dataset_sha256'],
            'cache_manifest_sha256':file_sha256(cache/'cache.json'),'episodes':metadata['episodes'],
            'frames':metadata['frames'],'sampled_fk_labels':checked,'max_target_position_error_m':max_position,
            'max_target_axis_vector_error':max_axis,'distinct_augmented_parents':len(parents),
            'checks':['all observation/action values finite and within normalization bounds',
                      'all directed target axes unit length' if schema==3 else 'all tool-axis residual norms bounded by two',
                      'absent block geometry zero',
                      'all motion feature vector bounds','all episode boundaries/ranges contiguous',
                      'sampled original + augmented labels match independent NumPy URDF FK',
                      'all augmentation parent splits match original with no duplicate parent'],
            'collision_validation':'Underlying joint action trajectories unchanged; inherited motion_v2 dataset audit, no new collision replay claimed'}
    if schema==4:result['checks'].append('all parent cache hashes and copied actions/boundaries/ranges preserved; non-target inputs byte-identical')
    write_json(output,result);print(json.dumps(result),flush=True);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=Path('simulation/artifacts/diffusion_cartesian_v3/cache'))
    parser.add_argument('--output',type=Path,default=Path('output/diffusion/cartesian_v3_dataset_validation.json'))
    args=parser.parse_args();audit(args.cache,args.output)
