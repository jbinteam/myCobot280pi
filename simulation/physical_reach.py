"""Conservative, URDF-derived physical reach checks before bounded IK searches.

The sphere encloses every flange position: its center is the first revolute
joint origin, and its radius is the sum of downstream translation lengths.
Rotations preserve those lengths. Passing is not a certificate of joint-limit,
orientation, collision, or path feasibility; failing proves geometric exclusion.
All public positions and tolerances are in g_base millimeters.
"""
from dataclasses import dataclass
import numpy as np
from .kinematics import ARM
from .pose5 import unit_axis


def _vector(value):
    value = np.asarray(value, float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError('Reach check requires three finite millimeter coordinates')
    return value


def _allowance(value):
    if isinstance(value, bool) or not np.isfinite(value) or value < 0:
        raise ValueError('Reach allowance must be finite and nonnegative')
    return float(value)


@dataclass(frozen=True)
class ReachEnvelope:
    center_mm: tuple
    radius_mm: float

    @classmethod
    def from_arm(cls, arm):
        transform = np.eye(4)
        center = None
        radius = 0.
        for origin, axis in arm.chain:
            if center is None:
                transform = transform @ origin
                if axis is not None:
                    center = tuple((transform[:3, 3]*1000).tolist())
            else:
                radius += float(np.linalg.norm(origin[:3, 3]))*1000
        if center is None:
            raise ValueError('Reach envelope requires a revolute arm chain')
        return cls(center, radius)

    def _result(self, minimum, kind, allowance):
        # Roundoff at the enclosing sphere boundary must not discard a pose.
        possible = bool(minimum <= self.radius_mm+1e-7)
        return {'possible': possible, 'kind': kind,
                'minimum_distance_mm': float(minimum),
                'reach_center_mm': list(self.center_mm), 'reach_radius_mm': self.radius_mm,
                'position_allowance_mm': allowance,
                'reason': 'not_excluded_by_outer_bound' if possible else 'outside_outer_bound'}

    def point(self, position_mm, allowance_mm=0.):
        allowance = _allowance(allowance_mm)
        minimum = max(0., float(np.linalg.norm(_vector(position_mm)-self.center_mm))-allowance)
        return self._result(minimum, 'point', allowance)

    def disk(self, region, allowance_mm=0.):
        """Exact minimum distance from the sphere center to a disk/slab.

        Include the axial planning allowance but preserve the disk radial bound.
        Checking just its center, or just a finite sample, cannot exclude a disk.
        """
        allowance = _allowance(allowance_mm)
        radius = _allowance(region['radius_mm'])
        axis = unit_axis(region['axis'])
        offset = np.asarray(self.center_mm)-_vector(region['center_mm'])
        axial = float(offset @ axis)
        radial = float(np.linalg.norm(offset-axial*axis))
        minimum = float(np.hypot(max(0., abs(axial)-allowance), max(0., radial-radius)))
        return self._result(minimum, 'cone_disk', allowance)


ENVELOPE = ReachEnvelope.from_arm(ARM)


def target_precheck(target_m, region=None, allowance_mm=0.):
    """Check an exact EEF target or the entire canonical cone disk."""
    if region is not None:
        return ENVELOPE.disk(region, allowance_mm)
    return ENVELOPE.point(_vector(target_m)*1000, allowance_mm)
