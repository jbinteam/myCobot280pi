"""Audit actual rounded SDK target increments using monotonic send times.

These are interval-average command derivatives, not motor measurements or a
proof of the firmware's interpolation. Reject before sending; commit only
after a successful SDK call. No failed candidate changes the prior history.
"""
import numpy as np
from simulation.kinematics import ARM
from simulation.motion_limits import envelope, scale_needed


class CommandProfile:
    def __init__(self, start_deg, timestamp, joint_speed_limit=None):
        self.q = np.deg2rad(np.asarray(start_deg, float))
        self.timestamp = float(timestamp)
        self.velocity = np.zeros(6)
        self.interval = 0.
        if joint_speed_limit is not None and (not np.isfinite(joint_speed_limit) or joint_speed_limit<=0):
            raise ValueError('Joint command speed allowance must be positive and finite')
        self.joint_speed_limit=joint_speed_limit

    def candidate(self, target_deg, timestamp):
        q = np.deg2rad(np.asarray(target_deg, float))
        dt = float(timestamp)-self.timestamp
        if (q.shape != (6,) or not np.isfinite(q).all() or not np.isfinite(dt) or dt <= 0
                or np.any(q < ARM.lower-1e-9) or np.any(q > ARM.upper+1e-9)):
            raise ValueError('Invalid hardware command profile sample')
        velocity = (q-self.q)/dt
        acceleration = (velocity-self.velocity)/((dt+self.interval)/2)
        bounds = envelope(np.abs(velocity), np.abs(acceleration))
        return {'q': q, 'time': float(timestamp), 'dt': dt, 'velocity': velocity,
                'acceleration': acceleration, 'bounds': bounds,
                'accepted': scale_needed(bounds) <= 1.+1e-8 and (self.joint_speed_limit is None
                    or bounds['joint_velocity_rad_s']<=self.joint_speed_limit+1e-8)}

    def commit(self, sample):
        if not sample['accepted']:
            raise RuntimeError('Hardware command profile exceeds the arm motion specification')
        self.q = sample['q']; self.timestamp = sample['time']
        self.velocity = sample['velocity']; self.interval = sample['dt']
        return {'command_velocity_deg_s': np.rad2deg(self.velocity).tolist(),
                'command_acceleration_deg_s2': np.rad2deg(sample['acceleration']).tolist(),
                'command_motion_bounds': sample['bounds'], 'command_profile_checked': True,
                'command_profile_scope': 'send-interval averages; physical servo dynamics unverified'}
