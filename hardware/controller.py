"""Reusable arm controller shared by standalone ROS and the optional web UI.

This module owns hardware I/O, feedback and checked execution. It opens no HTTP
listener and connects only after an explicit connect() call.
"""
from collections import deque
import copy
import fcntl
import ipaddress
import math
import os
from pathlib import Path
import stat
import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation

from control_mycobot import REST_ANGLES
from pymycobot.common import ProtocolCode
from simulation.kinematics import ARM
from webui.motion_settings import MAX_SDK_SPEED
from webui.hardware_ik import HardwareIK
from webui.robot_transport import connect_robot
from webui.encoder_calibration import EncoderCalibration

LIMITS = {'angles_min': np.round(np.rad2deg(ARM.lower),1).tolist(),
          'angles_max': np.round(np.rad2deg(ARM.upper),1).tolist()}


def number(value, name, low, high, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if not low <= value <= high or (integer and int(value) != value):
        raise ValueError(f'{name} must be between {low} and {high}' + (' (integer)' if integer else ''))
    return int(value) if integer else value


class RobotController:
    def __init__(self, factory=connect_robot, *, encoder_calibration_path=None):
        self.factory = factory
        self.visibility_camera = None  # Optional immutable ROS runtime override; otherwise load configured mount per session.
        self._ownership_fd = None
        calibration_path=(Path(__file__).resolve().parent.parent/'hardware/encoder_direction_calibration.json'
                          if factory is connect_robot else None)
        self.encoder_calibration=EncoderCalibration(encoder_calibration_path if encoder_calibration_path is not None else calibration_path)
        self.lock = threading.RLock()
        # Pose readers never wait behind serial I/O or health checks. Only
        # validated angle replies populate this cache; GET never restamps it.
        self.feedback_lock = threading.Lock()
        self.feedback_poses = deque(maxlen=32)
        self.robot = None
        self.timer = None
        self.generation = 0
        self.motion = None
        self.target = None
        self.angles = None
        self.moving = False
        self.updated = None
        self.feedback_stamp_unix_s = None
        self.error = None
        self.host = '192.168.77.225'
        self.port = 9000
        self.firmware = None
        self.ik = HardwareIK(self)
        self.closed = threading.Event()
        self.poller = threading.Thread(target=self._poll, daemon=True)
        self.poller.start()

    def feedback_snapshot(self):
        with self.feedback_lock:
            poses = copy.deepcopy(list(self.feedback_poses))
        return {'connected': bool(poses), 'feedback_protocol': 1,
                'feedback_pose': poses[-1] if poses else None, 'feedback_poses': poses}

    def _publish_feedback(self, angles, stamp):
        transform = ARM.transform(np.deg2rad(angles))
        pose = {'stamp_unix_s':stamp, 'angles_deg':[float(a) for a in angles],
            'joint_names':list(ARM.names), 'frame_id':ARM.base_frame,
            'child_frame_id':ARM.tcp, 'transform_m':transform.tolist(),
            'translation_m':transform[:3, 3].tolist(),
            'quaternion_xyzw':Rotation.from_matrix(transform[:3, :3]).as_quat().tolist(),
            'stamp_source':'angle_reply_received'}
        with self.feedback_lock:
            self.feedback_poses.append(pose)

    def snapshot(self):
        with self.lock:
            feedback_pose = None
            if self.robot is not None and self.angles is not None and self.feedback_stamp_unix_s is not None:
                transform = ARM.transform(np.deg2rad(self.angles))
                feedback_pose = {'stamp_unix_s':self.feedback_stamp_unix_s,
                    'angles_deg':[float(a) for a in self.angles], 'joint_names':list(ARM.names),
                    'frame_id':ARM.base_frame, 'child_frame_id':ARM.tcp,
                    'transform_m':transform.tolist(), 'translation_m':transform[:3, 3].tolist(),
                    'quaternion_xyzw':Rotation.from_matrix(transform[:3, :3]).as_quat().tolist(),
                    'stamp_source':'angle_reply_received'}
            return dict(connected=self.robot is not None, host=self.host, port=self.port,
                        angles=self.angles, moving=self.moving, motion=self.motion,
                        updated=self.updated, error=self.error, firmware=self.firmware,
                        feedback_pose=feedback_pose,
                        directional_jog_blocked=self.firmware == 6.5,
                        rest=REST_ANGLES, limits={'min': LIMITS['angles_min'], 'max': LIMITS['angles_max']}, ik=self.ik.snapshot())

    def _read(self):
        angles = self.robot.get_angles()
        capture_stamp = time.time()
        if not isinstance(angles, list) or len(angles) != 6 or any(
            isinstance(a, bool) or not isinstance(a, (int, float)) or not math.isfinite(a) for a in angles
        ):
            received = f'{len(angles)} angle values' if isinstance(angles,list) else f'{type(angles).__name__} instead of an angle list'
            if isinstance(angles,(int,float)) and not isinstance(angles,bool) and angles == -1:
                received = 'SDK status -1 (no decoded angle reply)'
            elif isinstance(angles,list) and len(angles) == 6:
                received = 'six values with invalid numeric data'
            diagnostic = getattr(getattr(self.robot, '_read', None), 'failure_detail', None)
            detail = diagnostic(ProtocolCode.GET_ANGLES) if callable(diagnostic) else ''
            raise RuntimeError(f'Robot did not return six valid joint angles ({received}{"; "+detail if detail else ""})')
        self._publish_feedback(angles, capture_stamp)
        moving = self.robot.is_moving()
        if moving not in (0, 1):
            raise RuntimeError('Robot did not return a valid motion status')
        self.angles = angles
        self.feedback_stamp_unix_s = capture_stamp
        self.moving = bool(moving)
        self.updated = time.time()
        if self.motion == 'pose' and not self.moving and all(abs(a-b) <= 2 for a,b in zip(angles,self.target)):
            self._cancel_timer()
            self.motion = self.target = None

    def _cancel_timer(self):
        self.generation += 1
        if self.timer:
            self.timer.cancel()
            self.timer = None

    def _stop(self):
        self.ik.cancel()
        self._cancel_timer()
        if self.robot:
            # Old Atom firmware does not acknowledge movement/stop packets.
            # Send STOP once without blocking or repeating it for an ACK.
            self.robot._mesg(ProtocolCode.STOP, _async=True)
        self.motion = self.target = None
        self.ik.release_mouse_mode()

    def _close(self):
        with self.feedback_lock:
            self.feedback_poses.clear()
        self.ik.cancel('disconnected')
        self._cancel_timer()
        try:
            if self.robot:
                self.robot.close()
        finally:
            self.robot = None
            self.feedback_stamp_unix_s = None
            self.ik.mouse_original_mode = self.ik.refresh_mode = None
            self.ik.mouse_mode_changed = False
            self.motion = self.target = None
            self.moving = False
            self._release_ownership()

    def _acquire_ownership(self, host, port):
        """Exclude other local controllers before opening the same Pi bridge.

        flock is released by the kernel on process exit. The file is retained
        so competing processes always lock the same inode. Fake factories do
        not lock physical hardware addresses.
        """
        if self.factory is not connect_robot:
            return
        directory = Path('/tmp') / f'mycobot280-{os.getuid()}'
        directory.mkdir(mode=0o700, exist_ok=True)
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise RuntimeError(f'Unsafe robot ownership directory: {directory}')
        path = directory / f'{host}-{port}.lock'
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                raise RuntimeError(f'Unsafe robot ownership file: {path}')
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f'Robot {host}:{port} is already owned by another local controller; disconnect it from the WebUI or stop the other ROS controller first') from exc
            os.ftruncate(fd, 0)
            os.write(fd, f'{os.getpid()}\n'.encode())
            self._ownership_fd = fd
        except BaseException:
            os.close(fd)
            raise

    def _release_ownership(self):
        if self._ownership_fd is not None:
            fd, self._ownership_fd = self._ownership_fd, None
            os.close(fd)

    def connect(self, host, port):
        host = str(ipaddress.IPv4Address(host))
        port = number(port, 'Port', 1, 65535, True)
        with self.lock:
            if self.robot:
                if self.host == host and self.port == port:
                    return self.snapshot()
                raise RuntimeError('Disconnect the current robot before changing its address')
            self.host, self.port = host, port
            self.error = None
            try:
                self._acquire_ownership(host, port)
                self.robot = self.factory(host, port)
                self._read()
                self.firmware = self.robot.get_system_version()
                try:
                    self.ik.read_refresh_mode()
                except Exception:
                    # A firmware without this getter can still use manual
                    # controls, but mouse tracking fails closed at Start.
                    self.ik.refresh_mode = None
            except Exception as exc:
                self._close()
                self.error = str(exc)
                raise RuntimeError(f'Could not connect: {exc}') from exc
            return self.snapshot()

    def disconnect(self):
        with self.lock:
            try:
                if self.motion or self.ik.mouse_mode_changed:
                    self._stop()
                else:
                    self.ik.release_mouse_mode()
            finally:
                self._close()
            return self.snapshot()

    def _ready(self, armed, mouse_session_id=None):
        if armed is not True: raise ValueError('Enable movement after clearing the robot path')
        if self.ik.mouse_session and self.ik.mouse_session != mouse_session_id:
            raise RuntimeError('Stop mouse aim before another movement command')
        if not self.robot: raise RuntimeError('Connect to the robot first')
        self._read()
        if self.motion or self.moving: raise RuntimeError('Robot is moving; stop it before another command')
        self._health()

    def _health(self):
        if self.robot.is_power_on() != 1: raise RuntimeError('Robot is not powered on')
        if self.robot.is_all_servo_enable() != 1: raise RuntimeError('Robot servos are not enabled')
        if self.robot.is_paused() != 0: raise RuntimeError('Robot is paused')
        error = self.robot.get_error_information()
        if error != 0: raise RuntimeError(f'Robot reports error {error}')

    def _send_motion(self, send):
        try:
            send()
        except Exception as exc:
            was_ik = self.motion == 'ik'
            try: self._stop()
            except Exception: pass
            self._close()
            self.error = f'Motion delivery failed: {exc}. Check the arm before reconnecting.'
            if was_ik: self.ik.status.update(state='failed', message=self.error)
            raise RuntimeError(self.error) from exc

    def _deadline(self, delay, timeout_error=None):
        generation = self.generation
        def finish():
            with self.lock:
                if generation != self.generation or not self.robot: return
                try:
                    self._stop()
                    if timeout_error: self.error = timeout_error
                except Exception as exc:
                    self.error = f'Stop could not be delivered: {exc}'
                    self._close()
        self.timer = threading.Timer(delay, finish)
        self.timer.daemon = True
        self.timer.start()

    def move(self, angles, speed, armed):
        speed = number(speed, 'Speed', 1, MAX_SDK_SPEED, True)
        if not isinstance(angles, list) or len(angles) != 6: raise ValueError('Provide six target angles')
        angles = [number(a, f'Joint {i+1}', LIMITS['angles_min'][i], LIMITS['angles_max'][i]) for i,a in enumerate(angles)]
        with self.lock:
            self._ready(armed)
            self._cancel_timer()
            self.error = None
            self._send_motion(lambda: self.robot.send_angles(angles, speed, _async=True))
            self.motion, self.target = 'pose', angles
            self._deadline(90, 'Move timed out after 90 seconds; stop command sent')
            return self.snapshot()

    def jog(self, joint, direction, speed, duration, armed):
        joint = number(joint, 'Joint', 1, 6, True)
        direction = number(direction, 'Direction', 0, 1, True)
        speed = number(speed, 'Speed', 1, MAX_SDK_SPEED, True)
        duration = number(duration, 'Jog duration', 0.1, 2)
        with self.lock:
            if self.firmware == 6.5:
                raise ValueError('Directional jog is disabled on Atom 6.5 after an excessive-travel trial. Firmware command validation and a Pi-side Stop watchdog are required.')
            self._ready(armed)
            self._cancel_timer()
            self.error = None
            self._send_motion(lambda: self.robot.jog_angle(joint, direction, speed, _async=True))
            self.motion = 'jog'
            self._deadline(duration)
            return self.snapshot()

    def stop(self):
        with self.lock:
            if not self.robot: raise RuntimeError('No active robot connection')
            try: self._stop()
            except Exception as exc:
                self.error = f'Stop could not be delivered: {exc}'
                self._close()
                raise RuntimeError(self.error) from exc
            return self.snapshot()

    def _poll(self):
        while not self.closed.wait(0.5):
            with self.lock:
                try: self.ik.expire_mouse_aim()
                except Exception:
                    self._close()
                if not self.robot: continue
                # The executing IK worker already reads both angles and motion
                # status. Avoid competing SDK reads between its short commands.
                if self.motion == 'ik' and self.ik.heartbeat_confirmed: continue
                try: self._read()
                except Exception as exc:
                    was_ik = self.motion == 'ik'
                    try:
                        if self.motion: self._stop()
                    except Exception: pass
                    self._close()
                    self.error = f'Robot connection lost: {exc}. Check the arm and reconnect.'
                    if was_ik: self.ik.status.update(state='failed', message=self.error)

    def shutdown(self):
        self.closed.set()
        try:
            self.disconnect()
        finally:
            if threading.current_thread() is not self.poller:
                self.poller.join(timeout=2.)
