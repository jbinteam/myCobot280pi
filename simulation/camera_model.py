"""Immutable optical model shared by tracking planning and execution.

The configured mount may be assumed. The view angles remain synthetic until
current camera intrinsics are supplied and validated; a mount flag alone never
qualifies the complete visibility model as calibrated.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class CameraModel:
    horizontal_fov_deg: float = 80.
    vertical_fov_deg: float = 80.
    angular_margin_deg: float = 0.
    near_m: float = .005
    eef_to_optical_translation_m: tuple = (0., 0., 0.)
    eef_to_optical_quaternion_xyzw: tuple = (0., 0., 0., 1.)
    mount_assumed: bool = True
    mount_calibrated: bool = False

    def __post_init__(self):
        values = [self.horizontal_fov_deg, self.vertical_fov_deg, self.angular_margin_deg, self.near_m]
        if not np.isfinite(values).all():
            raise ValueError('Finite camera parameters required')
        if not (0 < self.horizontal_fov_deg < 180 and 0 < self.vertical_fov_deg < 180
                and 0 <= self.angular_margin_deg < min(self.horizontal_fov_deg, self.vertical_fov_deg)/2
                and self.near_m > 0):
            raise ValueError('Invalid camera view or near distance')
        for key, size in (('eef_to_optical_translation_m', 3), ('eef_to_optical_quaternion_xyzw', 4)):
            value = np.asarray(getattr(self, key), float)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f'{key} requires {size} finite values')
            object.__setattr__(self, key, tuple(float(x) for x in value))
        if abs(sum(x*x for x in self.eef_to_optical_quaternion_xyzw)-1.) > 1e-5:
            raise ValueError('Optical quaternion must be normalized [x,y,z,w]')
        if (type(self.mount_assumed) is not bool or type(self.mount_calibrated) is not bool
                or self.mount_assumed and self.mount_calibrated):
            raise ValueError('Camera mount flags must distinguish assumed from calibrated')

    @classmethod
    def from_ros_settings(cls, settings):
        if settings.get('eef_frame', 'joint6_flange') != 'joint6_flange':
            raise ValueError('Visibility mount requires the joint6_flange EEF frame')
        keys = ('horizontal_fov_deg', 'vertical_fov_deg', 'angular_margin_deg', 'near_m',
                'eef_to_optical_translation_m', 'eef_to_optical_quaternion_xyzw',
                'mount_assumed', 'mount_calibrated')
        return cls(**{key: settings[key] for key in keys if key in settings})

    def configuration(self):
        result = asdict(self)
        for key in ('eef_to_optical_translation_m', 'eef_to_optical_quaternion_xyzw'):
            result[key] = list(result[key])
        return result

    def optical_transform(self, eef_transform):
        from scipy.spatial.transform import Rotation
        mount = np.eye(4)
        mount[:3, :3] = Rotation.from_quat(self.eef_to_optical_quaternion_xyzw).as_matrix()
        mount[:3, 3] = self.eef_to_optical_translation_m
        return np.asarray(eef_transform) @ mount

    def project(self, camera_point):
        point = np.asarray(camera_point, float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError('Expected finite camera-frame XYZ')
        angles = np.abs(np.rad2deg(np.arctan2(point[:2], point[2])))
        margins = np.array([self.horizontal_fov_deg, self.vertical_fov_deg])/2 - angles
        in_view = bool(point[2] > self.near_m and np.all(margins >= self.angular_margin_deg))
        return {'in_view': in_view, 'depth_m': float(point[2]),
                'horizontal_deg': float(angles[0]), 'vertical_deg': float(angles[1]),
                'margin_deg': float(margins.min())}


def configured_camera(path=None):
    """Read the server's current ROS view/mount configuration without hardware I/O."""
    import yaml
    source = Path(path) if path is not None else (
        Path(__file__).resolve().parents[1]/'object_tracking_ros2/config/object_follow.yaml')
    settings = yaml.safe_load(source.read_text())['object_follow']['ros__parameters']
    return CameraModel.from_ros_settings(settings)
