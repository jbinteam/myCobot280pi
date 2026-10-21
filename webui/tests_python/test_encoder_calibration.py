"""Verified encoder-sign reuse near zero; fake hardware only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from webui.encoder_calibration import EncoderCalibration
from webui import encoder_drag,hardware_diffusion
from simulation.kinematics import ARM
from test_encoder_drag import EncoderRobot
from server import RobotController

KEY=('192.168.77.225',9000,6.5)


class EncoderCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.robot=EncoderRobot()
        self.robot.angles=[20.,-35.,-90.,40.,10.,150.]
        self.anchor=encoder_drag.capture(self.robot,self.robot.angles[:])

    def test_verified_sign_is_reused_at_zero_and_preflight_sends_no_motion(self):
        self.robot.angles[1]=0.
        unknown=encoder_drag.capture(self.robot,self.robot.angles[:])
        self.assertEqual(unknown['encoder_directions'][1],0)
        known=encoder_drag.capture(self.robot,self.robot.angles[:],verified_mapping=self.anchor)
        self.assertEqual(known['encoder_directions'][1],-1)
        self.assertEqual(known['reused_direction_joints'],[2])
        self.assertEqual(known['encoder_direction_evidence'][1],self.anchor['encoder_direction_evidence'][1])
        encoder_drag.verify(self.robot,self.robot.angles[:],known)
        self.assertEqual(self.robot.calls,[])

    def test_legacy_paired_mapping_can_supply_evidence_but_a_claimed_sign_cannot(self):
        legacy={k:self.anchor[k] for k in ['start_deg','start_encoders','encoder_directions']}
        self.robot.angles[1]=0.
        known=encoder_drag.capture(self.robot,self.robot.angles[:],verified_mapping=legacy)
        self.assertEqual(known['encoder_directions'][1],-1)
        ambiguous=encoder_drag.capture(self.robot,self.robot.angles[:])
        ambiguous['encoder_directions'][1]=-1
        result=encoder_drag.capture(self.robot,self.robot.angles[:],verified_mapping=ambiguous)
        self.assertEqual(result['encoder_directions'][1],0)
        bad=copy.deepcopy(self.anchor)
        bad['encoder_direction_evidence'][1]={'angle_deg':0.,'encoder':2048,'direction':-1}
        with self.assertRaisesRegex(ValueError,'unambiguous'):encoder_drag.direction_evidence(bad)

    def test_persistence_is_scoped_to_endpoint_and_firmware(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'calibration.json'
            store=EncoderCalibration(path);store.remember(*KEY,self.anchor)
            original=path.read_bytes()
            store.remember(*KEY,self.anchor)
            self.assertEqual(path.read_bytes(),original)
            loaded=EncoderCalibration(path)
            self.robot.angles[1]=0.
            known=encoder_drag.capture(self.robot,self.robot.angles[:],verified_mapping=loaded.lookup(*KEY))
            self.assertEqual(known['encoder_directions'][1],-1)
            for key in [('192.168.77.226',9000,6.5),('192.168.77.225',9001,6.5),('192.168.77.225',9000,6.6)]:
                self.assertIsNone(loaded.lookup(*key))
            loaded.forget(*KEY)
            self.assertIsNone(EncoderCalibration(path).lookup(*KEY))

    def test_conflicting_live_sign_invalidates_cached_evidence(self):
        controller=RobotController(lambda *_:self.robot)
        self.addCleanup(controller.shutdown)
        controller.connect(KEY[0],KEY[1]);controller.ik.encoder_preflight({})
        key=(controller.host,controller.port,controller.firmware)
        self.assertIsNotNone(controller.encoder_calibration.lookup(*key))
        self.robot.directions=self.robot.directions.copy();self.robot.directions[1]=1
        with self.assertRaisesRegex(encoder_drag.EncoderDirectionChanged,'J2'):
            controller.ik.encoder_preflight({})
        self.assertIsNone(controller.encoder_calibration.lookup(*key))
        self.assertEqual(self.robot.calls,[])

    def test_corrupt_persistence_and_contradictory_records_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'calibration.json';path.write_text('{broken')
            with self.assertRaisesRegex(RuntimeError,'invalid'):EncoderCalibration(path).lookup(*KEY)
        store=EncoderCalibration();store.remember(*KEY,self.anchor)
        reverse=copy.deepcopy(self.anchor)
        evidence=reverse['encoder_direction_evidence'][1]
        evidence['direction']=1;evidence['encoder']=round(2048+evidence['angle_deg']*encoder_drag.STEPS_PER_DEG)
        with self.assertRaisesRegex(ValueError,'contradictory'):store.remember(*KEY,reverse)

    def test_profile_planning_near_zero_succeeds_with_persisted_paired_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'calibration.json'
            store=EncoderCalibration(path);store.remember(*KEY,self.anchor)
            self.robot.angles=[-7.,0.,-12.,-6.,22.,150.]
            controller=RobotController(lambda *_:self.robot,encoder_calibration_path=path)
            self.addCleanup(controller.shutdown)
            controller.connect(KEY[0],KEY[1])
            # Fake SDK firmware is an independent fixture value; seed the
            # matching endpoint/firmware record used by this controller.
            controller.encoder_calibration.remember(controller.host,controller.port,controller.firmware,self.anchor)
            start=self.robot.angles[:];end=start.copy();end[1]+=.5
            pose=ARM.transform(np.deg2rad(end));deployment=hardware_diffusion.verified_status()
            body={'target_mm':(pose[:3,3]*1000).tolist(),'target_axis':pose[:3,2].tolist(),'speed':5,'joint_command':'encoder_drag_profile'}
            result={'method':'diffusion','task':'pose5','converged':True,'validation':'joint_limits_self_table_block',
                'obstacle_mm':None,'checkpoint_sha256':hardware_diffusion.V2_SHA256,'environment':deployment['environment'],
                'motion_conditioning':{'schema':2},'target_mm':body['target_mm'],'target_axis':body['target_axis'],
                'final_error_mm':0.,'final_orientation_error_deg':0.,
                'frames':[{'q':start},{'q':end}], 'smoothing':{'segments':[{'start_deg':start,'end_deg':end}]}}
            with patch.object(hardware_diffusion.service,'plan',return_value=result):
                data=controller.ik.prepare_diffusion(body)
            self.assertEqual(data['plan']['encoder_drag']['encoder_directions'][1],-1)
            self.assertEqual(data['plan']['encoder_drag']['reused_direction_joints'],[2])
            self.assertEqual(self.robot.calls,[])


if __name__=='__main__':unittest.main()
