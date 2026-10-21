"""Data contracts ensure known reachability and target-level held-out separation."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from simulation.diffusion.ik_seed_data import generate,load_dataset
from simulation.kinematics import ARM


class IKSeedDataTest(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory();self.addCleanup(self.folder.cleanup)
        self.path=Path(self.folder.name)/'dataset.npz'
        with patch('builtins.print'):
            generate(self.path,train_targets=8,validation_targets=3,test_targets=3,
                     currents_per_target=3,seed=931,current_pool=8)

    def rewrite(self,arrays):
        np.savez_compressed(self.path,**arrays)
        metadata=json.loads(self.path.with_suffix('.json').read_text())
        metadata['dataset_sha256']=hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.path.with_suffix('.json').write_text(json.dumps(metadata))

    def test_fk_validity_and_target_group_separation(self):
        arrays,metadata=load_dataset(self.path,check_scene=True)
        ids=[]
        for prefix,n in [('train',8),('validation',3),('test',3)]:
            values,counts=np.unique(arrays[prefix+'_target_id'],return_counts=True)
            self.assertEqual(len(values),n);np.testing.assert_array_equal(counts,3)
            ids.append(set(values.tolist()))
        self.assertFalse(ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2])
        self.assertEqual(metadata['obstacle'],None)

    def test_rejects_overlapping_target_ids(self):
        arrays,_=load_dataset(self.path)
        arrays['test_target_id'][:3]=arrays['train_target_id'][0]
        self.rewrite(arrays)
        with self.assertRaisesRegex(ValueError,'Target leakage'):load_dataset(self.path)

    def test_rejects_same_target_with_new_id_in_another_split(self):
        arrays,_=load_dataset(self.path)
        for key in ('q_goal_rad','target_xyz_m','target_axis_base'):
            arrays['test_'+key][:3]=arrays['train_'+key][0]
        self.rewrite(arrays)
        with self.assertRaisesRegex(ValueError,'Target leakage'):load_dataset(self.path)

    def test_rejects_fk_mismatch_even_with_updated_content_hash(self):
        arrays,_=load_dataset(self.path);arrays['train_target_xyz_m'][0,0]+=.001
        self.rewrite(arrays)
        with self.assertRaisesRegex(ValueError,'FK labels'):load_dataset(self.path)

    def test_rejects_unknown_schema(self):
        metadata=json.loads(self.path.with_suffix('.json').read_text())
        metadata['configuration']['fk']['urdf_sha256']='changed'
        self.path.with_suffix('.json').write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError,'schema/FK'):load_dataset(self.path)

    def test_rejects_hash_change(self):
        with self.path.open('ab') as file:file.write(b'changed')
        with self.assertRaisesRegex(ValueError,'content hash'):load_dataset(self.path)


if __name__=='__main__':unittest.main()
