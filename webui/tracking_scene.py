"""Server-owned visibility context shared by curve and encoder validation.

The object coordinate is installed by V2Mouse from its accepted detection;
clients cannot supply this field or camera assumptions through the motion API.
Each plan carries its session's configured mount and assumed camera view angles.
"""
import numpy as np
from simulation.diffusion.env import ArmEnv
from simulation.diffusion.visibility import VisibilityEnv
from simulation.camera_model import CameraModel
from simulation.kinematics import ARM
from simulation.tracking_geometry import region_metrics


def region_acceptance(q, plan, position_tolerance_mm, angle_tolerance_deg):
    """Check a plan's pinned endpoint contract; scene checks remain independent.

    Relaxing tool pointing never relaxes the frozen disk or rectangular image
    bounds. Callers also validate collision/occlusion for the complete path and
    every measured-to-command segment with scene_for().
    """
    mode = plan.get('orientation_mode', 'point_at_object')
    if mode not in ('point_at_object', 'camera_fov'):
        raise ValueError('Endpoint acceptance requires an effective orientation mode')
    region = plan['tracking_region']
    metrics = region_metrics(q, region, position_tolerance_mm,
                             180. if mode == 'camera_fov' else angle_tolerance_deg)
    metrics['orientation_mode'] = mode
    if mode == 'camera_fov':
        obj = plan.get('visibility_object_m')
        if (obj is None or not np.allclose(np.asarray(obj)*1000, region['cursor_mm'], atol=1e-8, rtol=0)
                or plan.get('visibility_camera') is None):
            raise ValueError('Camera-FOV endpoint omitted its pinned camera/object')
        camera = CameraModel(**plan['visibility_camera'])
        optical = camera.optical_transform(ARM.transform(q))
        projection = camera.project(optical[:3, :3].T @ (np.asarray(obj)-optical[:3, 3]))
        metrics['camera_projection'] = projection
        metrics['accepted'] = metrics['tracking_accepted'] = bool(metrics['accepted'] and projection['in_view'])
    return metrics


def scene_for(plan):
    obj = plan.get('visibility_object_m')
    if obj is None:
        return ArmEnv()
    if 'visibility_camera' not in plan:
        raise ValueError('Visible tracking plan omitted its pinned camera')
    return VisibilityEnv(object_m=obj, camera=CameraModel(**plan['visibility_camera']))


def step_for(scene):
    return .005 if isinstance(scene, VisibilityEnv) else .01


def set_candidate_object(scene, candidate):
    """Temporarily switch a running visibility scene before checking a join."""
    obj = candidate.get('visibility_object_m')
    if isinstance(scene, VisibilityEnv):
        if obj is None:
            raise ValueError('Visible tracking continuation omitted its object')
        if candidate.get('visibility_camera') != scene.camera.configuration():
            raise ValueError('Tracking continuation changed its pinned camera')
        previous = scene.object_m.copy()
        scene.set_object(np.asarray(obj, float))
        return previous
    if obj is not None:
        raise ValueError('Tracking continuation changed its visibility contract')
    return None
