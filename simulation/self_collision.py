"""Conservative URDF-mesh self-collision checking, independent of transport.

Each <collision> mesh is enclosed by a convex hull. FCL tests solid convex
objects, including containment. Only directly connected URDF link pairs are
excluded; all other pairs are checked with configurable clearance. This may
reject free configurations because concavities are filled. It does not model
external obstacles, tools, cables, dynamics or continuous articulated motion.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import io
from importlib.metadata import version
from itertools import combinations
import json
from pathlib import Path
import threading
import xml.etree.ElementTree as ET
import numpy as np
from .kinematics import ARM, URDF, origin_matrix


@dataclass(frozen=True)
class CollisionConfig:
    clearance_mm: float = 1.

    def __post_init__(self):
        if (isinstance(self.clearance_mm, bool) or not isinstance(self.clearance_mm, (int, float))
                or not np.isfinite(self.clearance_mm) or not 0 <= self.clearance_mm <= 10):
            raise ValueError('Self-collision clearance must be between 0 and 10 mm')


def collision_mesh(path, scale):
    """Load all scene geometry in original URDF coordinates, in meters.

    COLLADA up-axis is retained, matching URDFLoader's reset of the COLLADA
    root rotation. Scene node transforms and asset-unit scaling are applied.
    Correct the bundled joint5 material reference as the viewer does. Broken
    geometry or other references fail closed, never silently disappear.
    """
    import trimesh
    path = Path(path); root = ET.parse(path).getroot()
    ns = {'c': 'http://www.collada.org/2005/11/COLLADASchema'}
    unit_node = root.find('c:asset/c:unit', ns)
    unit = float(unit_node.get('meter', '1')) if unit_node is not None else 1.
    if not np.isfinite(unit) or unit <= 0: raise ValueError('Invalid COLLADA asset unit')
    materials = {m.get('id') for m in root.findall('.//c:library_materials/c:material', ns)}
    for binding in root.findall('.//c:instance_material', ns):
        if binding.get('target', '').lstrip('#') not in materials:
            if len(materials) != 1: raise ValueError(f'Ambiguous broken material in {path.name}')
            binding.set('target', '#'+next(iter(materials)))
    scene = trimesh.load_scene(io.BytesIO(ET.tostring(root)), file_type='dae',
                               process=False, ignore_broken=False)
    mesh = scene.to_mesh()
    if not len(mesh.vertices) or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError(f'Missing or invalid collision mesh: {path.name}')
    scale = np.asarray(scale, float)
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Collision mesh requires three positive scale factors')
    mesh.vertices *= unit*scale
    # Remove duplicated positions (normals/UV seams) before hull construction.
    hull = trimesh.convex.convex_hull(np.unique(mesh.vertices, axis=0))
    if not hull.is_watertight or hull.volume <= 0 or not np.isfinite(hull.vertices).all():
        raise ValueError(f'Invalid convex collision hull: {path.name}')
    return hull, {'file': path.name, 'asset_unit_m': unit, 'scale': scale.tolist(),
                  'source_triangles': len(mesh.faces), 'hull_triangles': len(hull.faces),
                  'local_bounds_m': hull.bounds.tolist()}


class SelfCollisionChecker:
    def __init__(self, config=None):
        try:
            import fcl
            import trimesh
            import collada
            import scipy
        except ImportError as exc:
            raise RuntimeError('Self-collision dependencies missing; run .venv/bin/python -m pip install -r simulation/requirements-collision.txt') from exc
        self.fcl = fcl; self.config = config or CollisionConfig(); self.lock = threading.RLock()
        if hashlib.sha256(URDF.read_bytes()).hexdigest() != ARM.fingerprint:
            raise ValueError('Robot URDF changed; restart the process before collision checking')
        self.bodies = []; model_info = []; root = ET.parse(URDF).getroot()
        links = {ARM.base_frame, *ARM.link_names}
        self.excluded_pairs = sorted(tuple(sorted((j.find('parent').get('link'), j.find('child').get('link'))))
                                     for j in root.findall('joint'))
        for link in root.findall('link'):
            name = link.get('name')
            if name not in links: raise ValueError(f'Collision FK unsupported for link {name}')
            collisions = link.findall('collision')
            if not collisions: raise ValueError(f'Missing collision geometry for link {name}')
            for collision in collisions:
                geometry = collision.find('geometry'); mesh_node = geometry.find('mesh') if geometry is not None else None
                if mesh_node is None: raise ValueError('This robot collision checker requires URDF mesh geometry')
                filename = mesh_node.get('filename', '')
                prefix = 'package://mycobot_description/'
                if not filename.startswith(prefix): raise ValueError('Unsupported collision mesh path')
                path = URDF.parents[2]/filename[len(prefix):]
                if not path.resolve().is_relative_to(URDF.parents[2].resolve()): raise ValueError('Collision mesh escapes robot package')
                scale = list(map(float, mesh_node.get('scale', '1 1 1').split()))
                hull, info = collision_mesh(path, scale)
                vertices = np.asarray(hull.vertices, dtype=np.float64)
                faces = np.column_stack([np.full(len(hull.faces), 3), hull.faces]).ravel().tolist()
                geom = fcl.Convex(vertices, len(hull.faces), faces)
                bounds = np.array(np.meshgrid(*zip(hull.bounds[0], hull.bounds[1]), indexing='ij')).reshape(3, -1).T
                self.bodies.append({'link': name, 'origin': origin_matrix(collision.find('origin')),
                                    'object': fcl.CollisionObject(geom), 'corners': bounds})
                model_info.append({'link': name, **info})
        exclusions = set(self.excluded_pairs)
        self.pairs = [(i, j) for i, j in combinations(range(len(self.bodies)), 2)
                      if self.bodies[i]['link'] != self.bodies[j]['link']
                      and tuple(sorted((self.bodies[i]['link'], self.bodies[j]['link']))) not in exclusions]
        self._batch_origins=np.stack([body['origin'] for body in self.bodies])
        self._batch_corners=np.stack([body['corners'] for body in self.bodies])
        self._pair_left,self._pair_right=np.asarray(self.pairs,dtype=int).T
        digest = hashlib.sha256()
        for body in self.bodies:
            digest.update(body['origin'].tobytes()); digest.update(body['corners'].tobytes())
        self.description = {'checker': 'urdf_convex_fcl_v1', 'environment': 'self_only_no_external_obstacles',
                            'representation': 'convex_hull_per_URDF_collision_mesh',
                            'clearance_mm': self.config.clearance_mm,
                            'excluded_pairs': [list(pair) for pair in self.excluded_pairs],
                            'exclusion_policy': 'direct_URDF_parent_child_pairs_only',
                            'checked_link_pairs': [sorted((self.bodies[i]['link'], self.bodies[j]['link'])) for i, j in self.pairs],
                            'mesh_info': model_info, 'derived_bounds_sha256': digest.hexdigest(),
                            'versions': {name: version(name) for name in ('python-fcl', 'trimesh', 'pycollada', 'scipy')}}

    def _check(self, q, details):
        transforms = ARM.link_transforms(q); boxes = []
        clearance = self.config.clearance_mm/1000; conflicts = []
        for body in self.bodies:
            t = transforms[body['link']]@body['origin']
            body['object'].setTransform(self.fcl.Transform(t[:3, :3], t[:3, 3]))
            corners = body['corners']@t[:3, :3].T+t[:3, 3]
            boxes.append((corners.min(axis=0), corners.max(axis=0)))
        for i, j in self.pairs:
            lo_a, hi_a = boxes[i]; lo_b, hi_b = boxes[j]
            gap = np.maximum(0., np.maximum(lo_a-hi_b, lo_b-hi_a))
            if np.linalg.norm(gap) > clearance: continue
            a, b = self.bodies[i]['object'], self.bodies[j]['object']
            result = self.fcl.CollisionResult()
            hit = bool(self.fcl.collide(a, b, self.fcl.CollisionRequest(), result))
            distance = 0. if hit else float(self.fcl.distance(a, b, self.fcl.DistanceRequest(), self.fcl.DistanceResult()))
            if not np.isfinite(distance): raise RuntimeError('Nonfinite FCL distance; configuration validity unknown')
            if hit or distance <= clearance:
                if not details: return False
                conflicts.append({'links': sorted((self.bodies[i]['link'], self.bodies[j]['link'])),
                                  'intersecting': hit, 'distance_mm': max(0., distance*1000)})
        return {'valid': not conflicts, 'conflicts': conflicts, 'clearance_mm': self.config.clearance_mm} if details else True

    def valid(self, q):
        # Native objects have mutable transforms: serialize all per-pose queries.
        with self.lock: return self._check(q, False)

    def valid_many(self, states):
        """Batch FK and AABB screening, retaining scalar FCL narrow-phase checks.

        The same eight bounding-box corners, excluded pairs and clearance are
        used. Near a clearance boundary, fall back to the original scalar
        predicate to avoid changing decisions through floating-point rounding.
        Native object transforms remain protected by the checker lock.
        """
        states=np.asarray(states,float)
        if states.ndim!=2 or states.shape[1]!=6 or not np.isfinite(states).all():
            raise ValueError('Expected finite [N,6] collision states')
        result=np.ones(len(states),dtype=bool)
        clearance=self.config.clearance_mm/1000
        boundary_epsilon=1e-10  # Numeric fallback band, not extra clearance.
        with self.lock:
            # Bound temporary geometry storage for very long sampled paths.
            for start in range(0,len(states),256):
                block=states[start:start+256]
                links=ARM.link_transforms_batch(block)
                transforms=np.stack([links[body['link']] for body in self.bodies],axis=1)@self._batch_origins
                corners=self._batch_corners@np.swapaxes(transforms[...,:3,:3],-1,-2)+transforms[...,:3,3][...,None,:]
                lower=corners.min(axis=-2);upper=corners.max(axis=-2)
                gaps=np.maximum(0.,np.maximum(lower[:,self._pair_left]-upper[:,self._pair_right],
                                               lower[:,self._pair_right]-upper[:,self._pair_left]))
                distances=np.linalg.norm(gaps,axis=-1)
                for row,q in enumerate(block):
                    if np.any(np.abs(distances[row]-clearance)<=boundary_epsilon):
                        result[start+row]=self._check(q,False)
                        continue
                    candidates=np.flatnonzero(distances[row]<=clearance)
                    updated=set()
                    for pair in candidates:
                        i,j=self.pairs[pair]
                        # Only bodies involved in a narrow-phase query need
                        # native transform updates; never reuse a stale body.
                        for body_index in (i,j):
                            if body_index not in updated:
                                t=transforms[row,body_index]
                                self.bodies[body_index]['object'].setTransform(self.fcl.Transform(t[:3,:3],t[:3,3]))
                                updated.add(body_index)
                        a,b=self.bodies[i]['object'],self.bodies[j]['object']
                        hit=bool(self.fcl.collide(a,b,self.fcl.CollisionRequest(),self.fcl.CollisionResult()))
                        distance=0. if hit else float(self.fcl.distance(a,b,self.fcl.DistanceRequest(),self.fcl.DistanceResult()))
                        if not np.isfinite(distance):
                            raise RuntimeError('Nonfinite FCL distance; configuration validity unknown')
                        if not hit and abs(distance-clearance)<=boundary_epsilon:
                            result[start+row]=self._check(q,False)
                            break
                        if hit or distance<=clearance:
                            result[start+row]=False
                            break
        return result

    def check(self, q):
        with self.lock: return self._check(q, True)

    def assumptions(self):
        return json.loads(json.dumps(self.description))


@lru_cache(maxsize=8)
def _checker(clearance_mm, geometry_hash):
    return SelfCollisionChecker(CollisionConfig(clearance_mm))


def checker(clearance_mm=1.):
    from .prm import geometry_digest
    config = CollisionConfig(clearance_mm)
    return _checker(config.clearance_mm, geometry_digest())


def self_collision_space(clearance_mm=1.):
    from .prm import JointSpace
    collision = checker(clearance_mm)
    space = JointSpace(collision.valid, collision.assumptions(), validation='joint_limits_and_self_collision')
    space.collision_checker = collision
    return space
