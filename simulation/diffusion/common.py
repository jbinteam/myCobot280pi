"""Shared schema and bounded-joint normalization; meters, radians and seconds."""
import hashlib
import json
from pathlib import Path
import numpy as np
from ..kinematics import ARM
from ..prm import geometry_digest

HORIZON = 16
OBS_DIM = 19
SCHEMA = 1
ARTIFACTS = Path(__file__).resolve().parents[1] / 'artifacts/diffusion_v1'


def normalize(q):
    """Map joint arrays [..., 6] to the operating box [-1, 1]."""
    return (np.asarray(q)-ARM.center)/ARM.radius


def denormalize(q):
    """Map normalized joint arrays [..., 6] back to radians."""
    return ARM.center+np.asarray(q)*ARM.radius


def observation(q, goal, obstacle):
    """Return [19]: current joints, selected goal joints, box XYZ/half-size/mask.

    Cartesian 5D goals are converted to valid goal joints by environment IK.
    A missing obstacle has six zero geometry features and a zero presence mask.
    """
    box=np.zeros(7) if obstacle is None else np.r_[np.asarray(obstacle)/ARM.scale,1.]
    value=np.r_[normalize(q),normalize(goal),box].astype(np.float32)
    if value.shape!=(OBS_DIM,) or not np.isfinite(value).all():
        raise ValueError('Expected finite current/goal joints and one axis-aligned box')
    return value


def robot_schema():
    """Compatibility identity independent of any saved training weights."""
    return {'schema':SCHEMA,'robot':'myCobot 280 Pi','urdf_sha256':ARM.fingerprint,
            'geometry_sha256':geometry_digest(),'joint_names':ARM.names,
            'limits_rad':{'min':ARM.lower.tolist(),'max':ARM.upper.tolist()},
            'base_frame':ARM.base_frame,'tcp':ARM.tcp,'units':'meters/radians/seconds',
            'goal':'selected joint goal from XYZ + directed flange Z; local Z spin free',
            'observation':['current_joint_normalized']*6+['goal_joint_normalized']*6+
                          ['box_x_scaled','box_y_scaled','box_z_scaled','box_half_x_scaled',
                           'box_half_y_scaled','box_half_z_scaled','box_present'],
            'action':'absolute operating-box-normalized joint positions',
            'horizon':HORIZON,'observation_dim':OBS_DIM}


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
