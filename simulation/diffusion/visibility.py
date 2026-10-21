"""Point-object visibility using a pinned optical mount and assumed view angles."""
import numpy as np
import pybullet as p
from ..kinematics import ARM
from ..camera_model import CameraModel
from .env import ArmEnv


class VisibilityEnv(ArmEnv):
    """Use view and line-of-sight checks for every sampled RRT/smoothing edge.

    The optical pose is flange FK composed with the pinned mount. Ray begins near_m
    toward the point object. Filled URDF hulls may conservatively occlude it.
    No requirement that intermediate EEF positions lie inside a cone.
    """
    def __init__(self, object_m=(0., 0., .5), camera=CameraModel(), cancel_check=None, **kwargs):
        self.cancel_check = cancel_check
        self.camera = camera
        self._visibility_cache = {}
        self.visibility_evaluations = 0
        self.ray_queries = 0
        self.set_object(object_m)
        super().__init__(**kwargs)

    def set_object(self, object_m):
        value = np.asarray(object_m, float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError('Expected finite object XYZ')
        self.object_m = value.copy()
        self._visibility_cache.clear()

    def set_obstacle(self, box):
        super().set_obstacle(box)
        self._visibility_cache.clear()

    def inspect(self, q):
        if self.cancel_check is not None:
            self.cancel_check()
        q = np.asarray(q, float)
        key = q.tobytes()
        if key in self._visibility_cache:
            return self._visibility_cache[key]
        self.visibility_evaluations += 1
        t = self.camera.optical_transform(ARM.transform(q))
        vector = self.object_m - t[:3, 3]
        projection = self.camera.project(t[:3, :3].T @ vector)
        result = {**projection, 'visible': False, 'occluded': False}
        if projection['in_view']:
            try:
                self._set_joints(q)
                origin = t[:3, 3] + self.camera.near_m * vector / np.linalg.norm(vector)
                hit = p.rayTest(origin, self.object_m, physicsClientId=self.client)[0]
                self.ray_queries += 1
                result.update(visible=hit[0] < 0, occluded=hit[0] >= 0,
                              occluding_body=int(hit[0]), occluding_link=int(hit[1]))
                result['occluding_object'] = ('table' if hit[0] == self.table else
                    'robot' if hit[0] == self.robot else 'block' if hit[0] == self.box else None)
            finally:
                self._set_joints(self._display_q)
        if len(self._visibility_cache) > 200000:
            self._visibility_cache.clear()
        self._visibility_cache[key] = result
        return result

    def valid(self, q):
        if self.cancel_check is not None:
            self.cancel_check()
        return bool(super().valid(q) and self.inspect(q)['visible'])

    def audit(self, path, step=.005):
        if not np.isfinite(step) or step <= 0:
            raise ValueError('Positive replay step required')
        path = np.asarray(path, float)
        if path.ndim != 2 or path.shape[1] != 6 or not len(path) or not np.isfinite(path).all():
            raise ValueError('Expected finite nonempty joint path')
        samples = [path[0]]
        for a, b in zip(path[:-1], path[1:]):
            samples.extend(self.space.interpolate(a, b, step)[1:])
        checks = [self.inspect(q) for q in samples]
        return {'passed': all(c['visible'] for c in checks), 'samples': len(checks),
                'outside_view_samples': sum(not c['in_view'] for c in checks),
                'occluded_samples': sum(c['occluded'] for c in checks),
                'minimum_view_margin_deg': min(c['margin_deg'] for c in checks),
                'max_step_rad': step}

    def visibility_metadata(self):
        return {'camera': self.camera.configuration(), 'calibrated': False,
                'object_representation': 'center point',
                'camera_mount': 'flange-to-optical transform',
                'intrinsics_assumed': True,
                'ray_start': 'near distance toward object',
                'cone_path_constraint': False}
