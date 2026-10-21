"""Versioned motion inputs, separate from the immutable v1 robot/data schema."""
import numpy as np

CONDITIONING = {
    'schema':2, 'observation_dim':34, 'base_observation_dim':19,
    'extra_features':['joint_velocity_rad_s']*6+['joint_acceleration_rad_s2']*6+
                     ['target_velocity_m_s']*3,
    'normalization':{'joint_velocity':1.,'joint_acceleration':3.,'target_velocity':.5},
    'target_velocity':'Cartesian target point in g_base; vector clipped to 0.5 m/s',
}


def bounded_vector(value, size, limit, name, clip=False):
    value=np.asarray(value,float)
    if value.shape!=(size,) or not np.isfinite(value).all():raise ValueError(f'Invalid {name}')
    length=np.linalg.norm(value)
    if length>limit+1e-6:
        if not clip:raise ValueError(f'{name} exceeds its vector limit')
        value=value*(limit/length)
    return value


def extend(base, velocity=None, acceleration=None, target_velocity=None):
    base=np.asarray(base,np.float32)
    if base.shape!=(19,) or not np.isfinite(base).all():raise ValueError('Expected finite base observation [19]')
    v=bounded_vector(np.zeros(6) if velocity is None else velocity,6,1.,'joint velocity')
    a=bounded_vector(np.zeros(6) if acceleration is None else acceleration,6,3.,'joint acceleration')
    tv=bounded_vector(np.zeros(3) if target_velocity is None else target_velocity,3,.5,'target velocity',clip=True)
    return np.r_[base,v,a/3.,tv/.5].astype(np.float32)


def encode(policy, base, velocity=None, acceleration=None, target_velocity=None):
    """Retain exact v1 inputs when comparing or rolling back to old weights."""
    conditioning=getattr(policy,'conditioning',None)
    if conditioning is not None and conditioning!=CONDITIONING:
        raise ValueError('Use schema-specific Cartesian conditioning; joint-goal encoding is incompatible')
    return extend(base,velocity,acceleration,target_velocity) if policy.observation_dim==34 else base
