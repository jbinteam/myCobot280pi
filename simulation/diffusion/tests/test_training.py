import tempfile
from pathlib import Path
import unittest
import torch
from simulation.diffusion.model import ConditionalUNet1D
from simulation.diffusion.train import train
from simulation.diffusion.tests.helpers import demonstrations


class TrainingTest(unittest.TestCase):
    def test_optimizer_validation_and_safe_checkpoint_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'model'
            report=train(demonstrations(),output,epochs=2,
                         batch_size=64,channels=16,device='cpu',smoke=True,validation_windows=32)
            self.assertEqual(report['completed_epochs'],2)
            self.assertEqual(report['kind'],'smoke')
            payload=torch.load(output/'diffusion_policy_6dof.pth',weights_only=True,map_location='cpu')
            model=ConditionalUNet1D(16);model.load_state_dict(payload['state_dict'])
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
            self.assertEqual(payload['training']['selected_epoch'],report['selected_epoch'])
            self.assertGreater(report['completed_steps'],0)
            with self.assertRaises(ValueError):train('missing.h5',output)
            existing=Path(tmp)/'existing';existing.mkdir()
            (existing/'training.json').write_text('existing Seed5 metadata')
            with self.assertRaises(ValueError):train('missing.h5',existing)
            self.assertEqual((existing/'training.json').read_text(),'existing Seed5 metadata')


if __name__=='__main__':unittest.main()
