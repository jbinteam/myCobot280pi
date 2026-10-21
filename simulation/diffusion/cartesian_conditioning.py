"""Versioned Cartesian motion inputs; goal joint angles are never observations."""
import copy
import numpy as np
from .common import normalize
from .motion_conditioning import bounded_vector
from ..kinematics import ARM

CARTESIAN_CONDITIONING = {
    'schema': 3, 'observation_dim': 34, 'base_observation_dim': 19,
    'target_input': 'Cartesian XYZ + directed flange local-Z unit vector; spin free',
    'observation': ['current_joint_normalized'] * 6 +
                   ['target_x_scaled', 'target_y_scaled', 'target_z_scaled'] +
                   ['target_tool_z_x', 'target_tool_z_y', 'target_tool_z_z'] +
                   ['box_x_scaled', 'box_y_scaled', 'box_z_scaled',
                    'box_half_x_scaled', 'box_half_y_scaled', 'box_half_z_scaled', 'box_present'] +
                   ['joint_velocity_rad_s'] * 6 + ['joint_acceleration_rad_s2'] * 6 +
                   ['target_velocity_m_s'] * 3,
    'normalization': {'Cartesian_position': ARM.scale, 'obstacle_geometry': ARM.scale,
                      'joint_velocity': 1., 'joint_acceleration': 3., 'target_velocity': .5},
    'base_frame': ARM.base_frame, 'tcp': ARM.tcp, 'tcp_offset_m': [0., 0., 0.],
    'target_velocity': 'Cartesian target point in g_base; vector clipped to 0.5 m/s',
    'goal_joint_input': False,
}
CONDITIONING = CARTESIAN_CONDITIONING
RELATIVE_CONDITIONING = copy.deepcopy(CARTESIAN_CONDITIONING)
RELATIVE_CONDITIONING.update(schema=4,
    target_input='Cartesian target-minus-current XYZ and directed flange-Z residual; spin free',
    current_FK='shared URDF current joints -> XYZ and directed flange local-Z',
    target_axis_residual='normalized target tool-Z minus current tool-Z')
RELATIVE_CONDITIONING['observation'][6:12] = [
    'target_minus_current_x_scaled', 'target_minus_current_y_scaled', 'target_minus_current_z_scaled',
    'target_minus_current_tool_z_x', 'target_minus_current_tool_z_y', 'target_minus_current_tool_z_z']


def supports(policy):
    """Dimension alone cannot distinguish v2 joint goals from v3 Cartesian goals."""
    return supports_conditioning(getattr(policy, 'conditioning', None))


def supports_conditioning(conditioning):
    return conditioning in (CARTESIAN_CONDITIONING, RELATIVE_CONDITIONING)


def observe(q, target_position, tool_axis, obstacle=None, velocity=None,
            acceleration=None, target_velocity=None):
    q = np.asarray(q, float)
    position = np.asarray(target_position, float)
    axis = np.asarray(tool_axis, float)
    if q.shape != (6,) or not np.isfinite(q).all() or np.any(q < ARM.lower) or np.any(q > ARM.upper):
        raise ValueError('Expected six current joints inside the operating limits')
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError('Expected finite Cartesian target XYZ in meters')
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-9:
        raise ValueError('Expected nonzero finite directed tool-Z axis')
    axis = axis / np.linalg.norm(axis)
    geometry = np.zeros(7)
    if obstacle is not None:
        box = np.asarray(obstacle, float)
        if box.shape != (6,) or not np.isfinite(box).all() or np.any(box[3:] <= 0):
            raise ValueError('Expected finite box XYZ and positive half-extents')
        geometry = np.r_[box / ARM.scale, 1.]
    v = bounded_vector(np.zeros(6) if velocity is None else velocity, 6, 1., 'joint velocity')
    a = bounded_vector(np.zeros(6) if acceleration is None else acceleration, 6, 3., 'joint acceleration')
    tv = bounded_vector(np.zeros(3) if target_velocity is None else target_velocity,
                        3, .5, 'target velocity', clip=True)
    return np.r_[normalize(q), position / ARM.scale, axis, geometry, v, a / 3., tv / .5].astype(np.float32)


def observe_relative(q, target_position, tool_axis, obstacle=None, velocity=None,
                     acceleration=None, target_velocity=None):
    value = observe(q, target_position, tool_axis, obstacle, velocity, acceleration, target_velocity)
    transform = ARM.transform(q)
    value[6:9] = (np.asarray(target_position, float)-transform[:3, 3]) / ARM.scale
    axis = np.asarray(tool_axis, float)
    value[9:12] = axis / np.linalg.norm(axis)-transform[:3, 2]
    return value


def observe_for_policy(policy, q, target_position, tool_axis, obstacle=None,
                       velocity=None, acceleration=None, target_velocity=None):
    if not supports(policy):
        raise ValueError('Expected known Cartesian policy conditioning schema')
    encoder = observe_relative if policy.conditioning == RELATIVE_CONDITIONING else observe
    return encoder(q, target_position, tool_axis, obstacle, velocity, acceleration, target_velocity)
