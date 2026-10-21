"""Fixed-v2 trial target poses derived from captured live joints; no motion."""
import numpy as np
from simulation.kinematics import ARM
from simulation.pose5 import xy_from_axis

PRESETS = {
    'j4_minus_2': (3, -2.), 'j4_plus_2': (3, 2.),
    'j4_minus_1_5': (3, -1.5), 'j4_plus_1_5': (3, 1.5),
    'j3_minus_1_5': (2, -1.5), 'j3_plus_1_5': (2, 1.5),
    'j1_minus_1_5': (0, -1.5), 'j1_plus_1_5': (0, 1.5),
    'j3_minus_6': (2, -6.), 'j3_plus_6': (2, 6.),
    'j1_minus_10': (0, -10.), 'j1_plus_10': (0, 10.),
}
STAGED_PRESETS = frozenset(('j3_minus_6', 'j3_plus_6', 'j1_minus_10', 'j1_plus_10'))


def target(start_deg, preset):
    if not isinstance(preset, str) or preset not in PRESETS:
        raise ValueError('Choose a listed target preset')
    start = np.asarray(start_deg, float)
    joint, delta = PRESETS[preset]
    goal = start.copy(); goal[joint] += delta
    if np.any(np.deg2rad(goal) < ARM.lower) or np.any(np.deg2rad(goal) > ARM.upper):
        raise ValueError('Target preset exceeds a joint limit; choose the opposite direction or another joint')
    pose = ARM.transform(np.deg2rad(goal))
    staged = preset in STAGED_PRESETS
    return {'preset': preset, 'start_deg': start.tolist(), 'staged': staged,
            'recommended_joint_command': 'encoder_drag_staged' if staged else None,
            'target_mm': np.round(pose[:3, 3]*1000, 3).tolist(),
            'orientation_xy_deg': np.round(xy_from_axis(pose[:3, 2]), 4).tolist(),
            'eef_offset_mm': [0, 0, 0], 'maximum_speed_percent': 5,
            'requires_planning': True, 'motion_sent': False}
