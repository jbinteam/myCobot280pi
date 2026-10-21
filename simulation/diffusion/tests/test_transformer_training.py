import json
from pathlib import Path
import tempfile
import unittest
import h5py
import numpy as np
from simulation.diffusion.train_trajectory_transformer import protected_arrays
from simulation.diffusion.motion_conditioning import CONDITIONING
from simulation.kinematics import ARM

class TransformerDataTest(unittest.TestCase):
    def test_test_and_negative_arrays_excluded_and_episode_ends_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'data.h5'
            with h5py.File(path,'w') as data:
                data.attrs['metadata_json']=json.dumps({'conditioning':CONDITIONING,'audit':{'passed':True}})
                root=data.create_group('episodes')
                for name,split in [('0','train'),('1','test')]:
                    g=root.create_group(name);q=np.array([ARM.center,ARM.center+.05])
                    if split=='test':q[:]=np.nan
                    g.attrs['split']=split;g.attrs['obstacle_present']=False
                    for key,values in [('q_rad',q),('goal_rad',q),('velocity_rad_s',np.zeros((2,6))),
                                       ('acceleration_rad_s2',np.zeros((2,6))),('target_velocity_m_s',np.zeros((2,3)))]:g.create_dataset(key,data=values)
                data.create_group('hard_negatives').create_dataset('poison',data=np.full((9,6),np.nan))
            arrays,ranges=protected_arrays(path,'train')
            self.assertTrue(np.isfinite(arrays['obs']).all());self.assertEqual(arrays['obs'].shape,(2,34))
            np.testing.assert_array_equal(arrays['ends'],[1,1]);np.testing.assert_array_equal(ranges,[[0,2]])
            np.testing.assert_array_equal(arrays['obs'][:,12:19],0)

if __name__=='__main__':unittest.main()
