"""Headless object-following contracts using only an in-memory transport."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation

import test_v2_mouse as lifecycle
from simulation.kinematics import ARM
from simulation.tracking_geometry import region_for


class ObjectFollowAPITest(unittest.TestCase):
    ENGINE = 'learned_rrt'
    PROFILE = {'kind':'smooth_trapezoid', 'rise_percent':5., 'fall_percent':10.}
    REGION = {'mode':'cone_disk', 'distance_mm':350., 'half_angle_deg':30.}
    tearDown = lifecycle.V2MouseTest.tearDown

    def setUp(self):
        lifecycle.V2MouseTest.setUp(self)
        # Keep the 350 mm object above the table and in modeled line of sight.
        self.start[3] = np.deg2rad(-40.)
        self.robot.angles = np.rad2deg(self.start).tolist()
        with self.controller.lock:
            self.controller._read()
        self.goal = self.start.copy(); self.goal[0] += np.deg2rad(4.)

    def start_ros(self):
        self.controller.ik.stop_mouse_aim(self.session)
        state = self.controller.ik.start_mouse_aim({'armed':True, 'frame_verified':True,
            'scene_verified':True, 'tracking_engine':self.ENGINE, 'speed':30,
            'tracking_region':self.REGION, 'velocity_shape':self.PROFILE,
            'client_id':'ros2_object_follow'})
        self.session = state['session_id']
        return state

    def proposal(self, body):
        result = lifecycle.V2MouseTest.proposal(self, body)
        if 'tracking_region' in body:
            result['tracking_region'] = region_for(body['cursor_mm'], body['target_axis'], body['tracking_region'])
        return result

    def plan(self):
        pose = ARM.transform(self.goal)
        return self.controller.ik.prepare_mouse_aim({'session_id':self.session,
            'cursor_mm':(pose[:3, 3]*1000+350*pose[:3, 2]).tolist(),
            'tracking_region':self.REGION})

    def test_ui_and_ros_start_cannot_take_an_active_session(self):
        state = self.controller.snapshot()['ik']
        self.assertEqual(state['mouse_aim_client_id'], 'webui')
        original = self.session
        with self.assertRaisesRegex(RuntimeError, 'already active for webui'):
            self.controller.ik.start_mouse_aim({'armed':True, 'frame_verified':True,
                'client_id':'ros2_object_follow'})
        self.assertEqual(self.controller.ik.mouse_session, original)
        state = self.start_ros()['ik']
        self.assertEqual(state['mouse_aim_client_id'], 'ros2_object_follow')
        self.assertEqual(state['mouse_session_owner_protocol'], 1)
        with self.assertRaisesRegex(RuntimeError, 'already active for ros2_object_follow'):
            self.controller.ik.start_mouse_aim({'armed':True, 'frame_verified':True})
        self.assertEqual(self.robot.calls, [])

    def test_old_session_stop_and_heartbeat_cannot_affect_new_owner(self):
        old = self.session
        self.start_ros()
        self.controller.ik.stop_mouse_aim(old)
        self.assertEqual(self.controller.ik.mouse_session, self.session)
        self.assertEqual(self.controller.ik.mouse_client_id, 'ros2_object_follow')
        with self.assertRaises(ValueError):
            self.controller.ik.mouse_aim_heartbeat(old)
        self.controller.ik.stop_mouse_aim(self.session)
        self.assertIsNone(self.controller.snapshot()['ik']['mouse_aim_client_id'])
        self.assertEqual(self.robot.calls, [])

    def test_execute_must_match_explicit_headless_session(self):
        self.start_ros()
        data = self.plan()
        with self.assertRaises(ValueError):
            self.controller.ik.execute(data['plan']['id'], 30, True, True, True,
                session_id='some_other_owner')
        self.assertEqual(self.controller.ik.plan['id'], data['plan']['id'])
        self.assertEqual(self.robot.calls, [])
        with patch.object(self.controller.ik.v2_mouse, 'execute', return_value={'accepted':True}) as execute:
            accepted = self.controller.ik.execute(data['plan']['id'], 30, True, True, True,
                session_id=self.session)
        self.assertEqual(accepted, {'accepted':True})
        self.assertEqual(execute.call_args.args[0]['mouse_aim_session_id'], self.session)

    def test_350mm_region_is_pinned_without_changing_ui_default(self):
        default = region_for([0, 0, 100], [0, 0, 1], {})
        self.assertEqual(default['distance_mm'], 100.)
        self.assertEqual(default['half_angle_deg'], 20.)
        state = self.start_ros()['ik']
        self.assertEqual(state['tracking_region_settings'], self.REGION)
        data = self.plan()
        region = data['result']['tracking_region']
        self.assertEqual(region['distance_mm'], 350.)
        self.assertEqual(region['half_angle_deg'], 30.)
        self.assertAlmostEqual(region['radius_mm'], 350*np.tan(np.deg2rad(30)))
        self.assertEqual(self.robot.calls, [])
        for distance in (501., True, float('nan')):
            with self.assertRaises(ValueError):
                region_for([0, 0, 100], [0, 0, 1], {'distance_mm':distance})

    def test_invalid_client_labels_do_not_mutate_session(self):
        for client in (None, True, '', 'a'*65, 'owner\nother'):
            with self.subTest(client=client), self.assertRaises(ValueError):
                self.controller.ik.start_mouse_aim({'client_id':client})
            self.assertEqual(self.controller.ik.mouse_session, self.session)

    def test_pause_invalidates_old_plan_keeps_owner_and_resumes_from_rest(self):
        self.start_ros()
        data=self.plan()
        old_plan=self.controller.ik.plan
        mouse=self.controller.ik.v2_mouse
        state=self.controller.ik.pause_mouse_aim(self.session)
        self.assertEqual(state['ik']['state'],'paused')
        self.assertEqual(state['ik']['mouse_aim_client_id'],'ros2_object_follow')
        self.assertEqual(state['ik']['mouse_pause_protocol'],1)
        self.assertIs(self.controller.ik.v2_mouse,mouse)
        self.assertIsNone(self.controller.ik.plan)
        self.assertEqual(self.robot.calls[-1][0],'stop')
        count=len(self.robot.calls)
        self.controller.ik.pause_mouse_aim(self.session)
        self.assertEqual(len(self.robot.calls),count)
        with self.assertRaises(lifecycle.StaleMousePlan):
            mouse.execute(old_plan,30,True,True,True)
        with self.assertRaisesRegex(RuntimeError,'already active'):
            self.controller.ik.start_mouse_aim({'armed':True,'frame_verified':True})
        self.robot.angles[0]+=.5  # Resume anchor must come from new feedback.
        data=self.plan()
        np.testing.assert_allclose(self.planner.call_args.args[0]['start_deg'],self.robot.angles)
        np.testing.assert_allclose(self.planner.call_args.args[0]['start_velocity_rad_s'],np.zeros(6))
        self.assertIsNone(self.controller.ik.plan['join'])
        self.controller.ik.execute(data['plan']['id'],30,True,True,True,session_id=self.session)
        self.controller.ik.mouse_aim_heartbeat(self.session,data['plan']['id'])
        state=lifecycle.V2MouseTest.wait(self,{'completed'})
        self.assertEqual(state['state'],'completed')

    def test_old_pause_cannot_affect_new_owner(self):
        old=self.session;self.start_ros()
        with self.assertRaises(ValueError):self.controller.ik.pause_mouse_aim(old)
        self.assertEqual(self.controller.ik.mouse_session,self.session)
        self.assertEqual(self.robot.calls,[])

    def test_recovery_waits_for_measured_stationary_arm(self):
        self.start_ros();self.controller.ik.pause_mouse_aim(self.session)
        with patch.object(self.robot,'is_moving',return_value=1):
            with self.assertRaises(lifecycle.StaleMousePlan):self.plan()
        self.assertEqual(self.controller.snapshot()['ik']['state'],'paused')
        self.assertEqual(self.controller.ik.mouse_session,self.session)
        self.assertIsNone(self.controller.ik.plan)

    def test_pause_during_plan_discards_candidate(self):
        import threading
        self.start_ros();entered=threading.Event();release=threading.Event();errors=[]
        original=self.planner.side_effect
        def blocked(body):
            entered.set();release.wait(2)
            return original(body)
        self.planner.side_effect=blocked
        def plan():
            try:self.plan()
            except Exception as exc:errors.append(exc)
        thread=threading.Thread(target=plan);thread.start()
        try:
            self.assertTrue(entered.wait(1))
            self.controller.ik.pause_mouse_aim(self.session)
        finally:
            release.set();thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertTrue(errors and isinstance(errors[0],lifecycle.StaleMousePlan))
        self.assertIsNone(self.controller.ik.plan)
        self.assertEqual(self.controller.snapshot()['ik']['state'],'paused')

    def test_feedback_timestamp_tracks_angle_reply_not_snapshot_polling(self):
        with self.controller.lock:
            with patch('server.time.time', side_effect=[1001., 1002.]):
                self.controller._read()
            first = self.controller.snapshot()
            second = self.controller.snapshot()
        pose = first['feedback_pose']
        self.assertEqual(pose['stamp_unix_s'], 1001.)
        self.assertEqual(first['updated'], 1002.)
        self.assertEqual(pose, second['feedback_pose'])
        self.assertEqual(pose['stamp_source'], 'angle_reply_received')
        self.assertEqual(pose['frame_id'], ARM.base_frame)
        self.assertEqual(pose['child_frame_id'], ARM.tcp)
        expected = ARM.transform(np.deg2rad(first['angles']))
        np.testing.assert_allclose(pose['transform_m'], expected)
        np.testing.assert_allclose(pose['translation_m'], expected[:3, 3])
        np.testing.assert_allclose(Rotation.from_quat(pose['quaternion_xyzw']).as_matrix(), expected[:3, :3], atol=1e-12)
        self.controller.disconnect()
        self.assertIsNone(self.controller.snapshot()['feedback_pose'])


if __name__ == '__main__':
    unittest.main()
