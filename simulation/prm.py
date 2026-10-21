"""Reusable joint-space PRM; NumPy graph core and optional collision backend.

All six installed joints are bounded revolute joints. Their numeric intervals
are not circles: +177 to -177 degrees must pass through zero, not a limit.
Distance and Dijkstra cost are Euclidean joint displacement divided by each
joint's operating range. Costs are nonnegative; edges are symmetric linear
joint motions, independent of the learned controller.
"""
import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import heapq
import json
from pathlib import Path
import threading
from time import perf_counter
import xml.etree.ElementTree as ET
import numpy as np
from .kinematics import ARM, URDF
from .workspace import transforms
from .asset_watch import AssetWatch

_GEOMETRY_WATCH=AssetWatch()
_GEOMETRY_LOCK=threading.RLock()


@dataclass(frozen=True)
class PRMConfig:
    samples: int = 400
    neighbors: int = 12
    max_step_rad: float = .04
    seed: int = 9001
    sampling_attempt_factor: int = 20

    def __post_init__(self):
        for key, low, high in [('samples', 0, 20000), ('neighbors', 1, 1000),
                               ('seed', 0, 2**32-1), ('sampling_attempt_factor', 1, 1000)]:
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise ValueError(f'{key} must be an integer between {low} and {high}')
        if isinstance(self.max_step_rad, bool) or not np.isfinite(self.max_step_rad) or not 0 < self.max_step_rad <= .5:
            raise ValueError('max_step_rad must be finite and between 0 and 0.5')


def _asset_signature(path):
    path=Path(path).resolve();stat=path.stat()
    # ctime catches same-size edits even if mtime is restored; inode/device
    # also distinguish replacements and symlink target changes.
    return (str(path),stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)


@lru_cache(maxsize=8)
def _geometry_assets(urdf_signature,revision):
    path=Path(urdf_signature[0]);package=path.parents[2]
    filenames=sorted({m.get('filename') for m in ET.parse(path).getroot().iter('mesh')})
    assets=[]
    for filename in filenames:
        if not filename.startswith('package://mycobot_description/'):
            raise ValueError(f'Unsupported model mesh path: {filename}')
        assets.append((filename,package/filename.split('package://mycobot_description/',1)[1]))
    if _asset_signature(path)!=urdf_signature:
        raise ValueError('Robot URDF changed during geometry fingerprinting')
    return tuple(assets)


@lru_cache(maxsize=8)
def _geometry_digest(urdf_signature,mesh_signatures,revision):
    digest=hashlib.sha256(Path(urdf_signature[0]).read_bytes())
    for filename,signature in mesh_signatures:
        digest.update(filename.encode());digest.update(Path(signature[0]).read_bytes())
    # A changing asset set must never acquire a reusable validation identity.
    if (_asset_signature(urdf_signature[0])!=urdf_signature or
            any(_asset_signature(signature[0])!=signature for _,signature in mesh_signatures)):
        raise ValueError('Robot geometry changed during fingerprinting')
    paths=[urdf_signature[0],*[signature[0] for _,signature in mesh_signatures]]
    if _GEOMETRY_WATCH.revision(paths)!=revision:
        raise ValueError('Robot geometry changed during fingerprinting')
    return digest.hexdigest()


def geometry_digest():
    """Cache identical bytes only with working event and stat invalidation."""
    with _GEOMETRY_LOCK:
        signature=_asset_signature(URDF)
        revision=_GEOMETRY_WATCH.revision([signature[0]])
        assets=(_geometry_assets.__wrapped__ if revision is None else _geometry_assets)(signature,revision)
        paths=[signature[0],*[path for _,path in assets],*[path.resolve() for _,path in assets]]
        current=_GEOMETRY_WATCH.revision(paths)
        if revision is not None and current is not None and revision!=current:
            raise ValueError('Robot geometry changed during fingerprinting')
        revision=current
        meshes=tuple((filename,_asset_signature(path)) for filename,path in assets)
        digest=(_geometry_digest.__wrapped__ if revision is None else _geometry_digest)(signature,meshes,revision)
        if _GEOMETRY_WATCH.revision(paths)!=revision:
            raise ValueError('Robot geometry changed during fingerprinting')
        return digest


class JointSpace:
    def __init__(self, validity=None, assumptions=None, validation=None):
        """validity(q_rad) is a deterministic, symmetric state-validity predicate.

        It must describe its checker revision, environment, base and tool in
        assumptions. Changing any of these requires a new cache. The default
        JointSpace() is the limit-only primitive; Roadmap() uses the installed
        self-collision checker by default. Neither imports robot transport.
        """
        if validity is not None and (not isinstance(assumptions, dict) or
                                    not assumptions.get('checker') or not assumptions.get('environment')):
            raise ValueError('A validity callback requires checker and environment cache identifiers')
        self.validity = validity
        self.assumptions = json.loads(json.dumps(assumptions or {'checker': None, 'environment': 'unspecified'}))
        self.lower = ARM.lower.copy(); self.upper = ARM.upper.copy()
        self.scale = self.upper-self.lower
        self.validation = 'joint_limits_and_callback' if validity is not None else 'joint_limits_only'
        if validation is not None:
            if validity is None or validation != 'joint_limits_and_self_collision':
                raise ValueError('Self-collision validation requires a collision checker')
            self.validation = validation
        self._metadata = {'schema': 1, 'robot': 'myCobot 280 Pi', 'urdf_sha256': ARM.fingerprint,
                          'geometry_sha256': geometry_digest(), 'joint_names': ARM.names,
                          'joint_types': ['bounded_revolute']*6,
                          'limits_rad': {'min': self.lower.tolist(), 'max': self.upper.tolist()},
                          'units': {'joints': 'radians', 'position': 'meters'},
                          'base_frame': 'g_base', 'base_transform': np.eye(4).tolist(),
                          'tcp': ARM.tcp, 'tool_offset_m': [0, 0, 0],
                          'validation': self.validation, 'assumptions': self.assumptions,
                          'metric': 'Euclidean displacement / operating joint range'}

    def metadata(self):
        return json.loads(json.dumps(self._metadata))

    def diagnostic(self, q):
        if not self.valid(q):
            q = np.asarray(q, float)
            if q.shape != (6,) or not np.isfinite(q).all(): return {'reason': 'Expected six finite joint angles'}
            if np.any(q < self.lower) or np.any(q > self.upper): return {'reason': 'Joint operating limits violated'}
            if hasattr(self, 'collision_checker'):
                return {'reason': 'Self-collision or clearance violation', 'self_collision': self.collision_checker.check(q)}
            return {'reason': 'Configuration validity callback rejected the state'}
        return {'reason': 'valid'}

    def valid(self, q):
        q = np.asarray(q, float)
        return bool(q.shape == (6,) and np.isfinite(q).all() and
                    np.all(q >= self.lower) and np.all(q <= self.upper) and
                    (self.validity is None or self.validity(q.copy())))

    def delta(self, a, b):
        # Deliberately no modulo 2*pi for these bounded revolute joints.
        return np.asarray(b, float)-np.asarray(a, float)

    def distance(self, a, b):
        return np.linalg.norm(self.delta(a, b)/self.scale, axis=-1)

    def interpolate(self, a, b, max_step_rad):
        if not np.isfinite(max_step_rad) or max_step_rad <= 0:
            raise ValueError('Interpolation step must be positive')
        delta = self.delta(a, b)
        pieces = max(1, int(np.ceil(np.max(np.abs(delta))/max_step_rad)))
        points = np.asarray(a)+np.linspace(0, 1, pieces+1)[:, None]*delta
        points[0] = a; points[-1] = b
        return points

    def edge_valid(self, a, b, max_step_rad):
        if not self.valid(a) or not self.valid(b): return False
        if self.validity is None:
            # The operating box is convex: its entire linear segment is valid.
            return True
        return all(self.valid(q) for q in self.interpolate(a, b, max_step_rad))


def fk_samples(path):
    """Reuse training FK configurations only after checking their provenance.

    Position-only workspace.json lacks q, and cannot be reused as PRM nodes.
    FK metadata is recomputed from q rather than trusting old Cartesian labels.
    """
    path = Path(path)
    metadata = json.loads(path.with_name('training.json').read_text())
    if metadata.get('urdf_sha256') != ARM.fingerprint or metadata.get('limits_rad') != {
            'min': ARM.lower.tolist(), 'max': ARM.upper.tolist()}:
        raise ValueError('FK sample geometry or joint limits mismatch')
    with np.load(path, allow_pickle=False) as data:
        arrays = [data[k] for k in ('q', 'goal') if k in data]
        if not arrays: raise ValueError('FK sample file must retain joint configurations')
        if any(a.ndim != 2 or a.shape[1] != 6 or not np.isfinite(a).all() for a in arrays):
            raise ValueError('Invalid FK joint sample array')
        return np.vstack(arrays)


def planning_space(collision='self', clearance_mm=1.):
    """PRM entry points default to self-collision; explicit limit-only ablation."""
    if collision == 'self':
        from .self_collision import self_collision_space
        return self_collision_space(clearance_mm)
    if collision == 'limits': return JointSpace()
    raise ValueError('Collision mode must be self or limits')


class Roadmap:
    def __init__(self, space=None, config=None):
        self.space = space if space is not None else planning_space()
        self.config = config or PRMConfig()
        self.q = np.empty((0, 6)); self.poses = np.empty((0, 4, 4))
        self.edges = {}; self.stats = {}

    def grow(self, count=None, samples=None):
        """Add nodes and connect nearest neighbors; retain previous valid edges.

        count is the number of additional nodes, not the total. A finite sample
        budget may leave fewer nodes when validity rejection is frequent.
        """
        began = perf_counter(); count = self.config.samples if count is None else count
        if isinstance(count, bool) or not isinstance(count, int) or count < 0 or count+len(self.q) > 20000:
            raise ValueError('Roadmap must contain at most 20000 nodes')
        rng = np.random.default_rng(np.random.SeedSequence([self.config.seed, len(self.q)]))
        added = []; seen = {tuple(q) for q in self.q}; reused = 0
        if samples is not None and count//2 > 0:
            samples = np.asarray(samples, float)
            if samples.ndim != 2 or samples.shape[1] != 6: raise ValueError('Expected Nx6 FK joint samples')
            # Leave half the requested additions for full-range uniform samples.
            for i in rng.permutation(len(samples)):
                q = samples[i]
                if self.space.valid(q) and tuple(q) not in seen:
                    added.append(q.copy()); seen.add(tuple(q)); reused += 1
                if reused >= count//2: break
        attempts = 0
        while len(added) < count and attempts < count*self.config.sampling_attempt_factor:
            q = rng.uniform(self.space.lower, self.space.upper); attempts += 1
            if self.space.valid(q) and tuple(q) not in seen:
                added.append(q); seen.add(tuple(q))
        added = added[:count]
        if added:
            self.q = np.vstack([self.q, added]); self.poses = transforms(self.q)
        for i, q in enumerate(self.q):
            self.edges.setdefault(i, {})
            order = np.argsort(self.space.distance(q, self.q), kind='stable')
            for j in order[order != i][:self.config.neighbors]:
                j = int(j)
                if j in self.edges[i]: continue
                if self.space.edge_valid(q, self.q[j], self.config.max_step_rad):
                    cost = float(self.space.distance(q, self.q[j]))
                    self.edges[i][j] = cost; self.edges.setdefault(j, {})[i] = cost
        self.stats = {'nodes': len(self.q), 'edges': sum(map(len, self.edges.values()))//2,
                      'added': len(added), 'reused_fk_samples': min(reused, len(added)),
                      'sampling_attempts': attempts, 'build_ms': (perf_counter()-began)*1000}
        return self

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        payload = {'metadata': self.space.metadata(), 'config': asdict(self.config), 'stats': self.stats,
                   'q_rad': self.q.tolist(), 'fk_transforms_m': self.poses.tolist(),
                   'edges': [[i, j, cost] for i, neighbors in self.edges.items()
                             for j, cost in neighbors.items() if i < j]}
        temporary = path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(payload, allow_nan=False)); temporary.replace(path)

    @classmethod
    def load(cls, path, space=None, config=None):
        data = json.loads(Path(path).read_text()); result = cls(space, config or PRMConfig(**data['config']))
        if data.get('metadata') != result.space.metadata():
            raise ValueError('Stale roadmap: robot, limits, units or validity/environment assumptions changed')
        if config is not None and data.get('config') != asdict(config):
            raise ValueError('Stale roadmap: build configuration changed')
        result.q = np.asarray(data['q_rad'], float).reshape(-1, 6)
        if len(result.q) > 20000 or not all(result.space.valid(q) for q in result.q):
            raise ValueError('Invalid cached roadmap nodes')
        result.poses = transforms(result.q)
        saved = np.asarray(data['fk_transforms_m'], float).reshape(-1, 4, 4)
        if saved.shape != result.poses.shape or not np.allclose(saved, result.poses, atol=1e-10, rtol=0):
            raise ValueError('Invalid cached FK metadata')
        result.edges = {i: {} for i in range(len(result.q))}
        # Recheck cache edges too: metadata equality alone is not data validation.
        for i, j, cost in data['edges']:
            if (isinstance(i, bool) or isinstance(j, bool) or not isinstance(i, int) or not isinstance(j, int)
                    or not 0 <= i < j < len(result.q) or not np.isfinite(cost) or cost < 0
                    or not np.isclose(cost, result.space.distance(result.q[i], result.q[j]), atol=1e-12, rtol=0)
                    or not result.space.edge_valid(result.q[i], result.q[j], result.config.max_step_rad)):
                raise ValueError('Invalid cached roadmap edge')
            result.edges[i][j] = cost; result.edges[j][i] = cost
        result.stats = data.get('stats', {})
        return result

    def plan(self, start, goals, shortcut=False):
        """Attach temporary start/goals without modifying the reusable graph."""
        began = perf_counter(); space = self.space; step = self.config.max_step_rad
        def outcome(status, reason, path=None):
            path = np.asarray(path, float).reshape(-1, 6) if path is not None else np.empty((0, 6))
            return {'status': status, 'reason': reason, 'validation': space.validation,
                    'waypoints_rad': path.tolist(), 'waypoints_deg': np.rad2deg(path).tolist(),
                    'fk_transforms_m': transforms(path).tolist(),
                    'path_cost': float(sum(space.distance(a, b) for a, b in zip(path, path[1:]))),
                    'path_length_rad': float(sum(np.linalg.norm(space.delta(a, b)) for a, b in zip(path, path[1:]))),
                    'planning_ms': (perf_counter()-began)*1000}
        if not space.valid(start):
            return {**outcome('invalid_start', 'Start violates configuration validity'), **space.diagnostic(start)}
        goals = np.asarray(goals, float).reshape(-1, 6)
        if not len(goals) or not all(space.valid(g) for g in goals):
            return outcome('invalid_goal', 'Planner requires validated joint goal configurations')
        order = np.argsort(space.distance(start, goals), kind='stable')
        for index in order:
            if space.edge_valid(start, goals[index], step):
                return outcome('success', 'direct_connection', [start, goals[index]])
        n = len(self.q); points = np.vstack([self.q, start, goals])
        edges = {i: dict(neighbors) for i, neighbors in self.edges.items()}
        for i in range(n, len(points)):
            edges[i] = {}
            order = np.argsort(space.distance(points[i], self.q), kind='stable')[:self.config.neighbors]
            for j in order:
                j = int(j)
                if space.edge_valid(points[i], points[j], step):
                    cost = float(space.distance(points[i], points[j]))
                    edges[i][j] = cost; edges[j][i] = cost
        if not edges[n]: return outcome('roadmap_connection_not_found', 'start_not_connected')
        if not any(edges[i] for i in range(n+1, len(points))):
            return outcome('roadmap_connection_not_found', 'goals_not_connected')
        goal_ids = set(range(n+1, len(points))); queue = [(0., n)]; best = {n: 0.}; parents = {}
        found = None
        while queue:
            cost, i = heapq.heappop(queue)
            if cost > best[i]: continue
            if i in goal_ids: found = i; break
            for j, weight in edges[i].items():
                candidate = cost+weight
                if candidate < best.get(j, float('inf')):
                    best[j] = candidate; parents[j] = i; heapq.heappush(queue, (candidate, j))
        if found is None: return outcome('path_not_found', 'disconnected_components')
        indices = [found]
        while indices[-1] != n: indices.append(parents[indices[-1]])
        path = points[indices[::-1]]
        if shortcut:
            kept = [path[0]]; i = 0
            while i < len(path)-1:
                j = len(path)-1
                while j > i+1 and not space.edge_valid(path[i], path[j], step): j -= 1
                kept.append(path[j]); i = j
            path = np.asarray(kept)
        # Final verification covers shortcut segments and protects callers from
        # accidentally mutating an in-memory roadmap's q or edges.
        if not all(space.edge_valid(a, b, step) for a, b in zip(path, path[1:])):
            return outcome('path_not_found', 'final_path_validation_failed')
        return outcome('success', 'roadmap_path', path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('simulation/artifacts/roadmap.json'))
    parser.add_argument('--samples', type=int, default=400)
    parser.add_argument('--neighbors', type=int, default=12)
    parser.add_argument('--max-step-rad', type=float, default=.04)
    parser.add_argument('--seed', type=int, default=9001)
    parser.add_argument('--dataset', type=Path, help='Existing dataset.npz with adjacent training.json')
    parser.add_argument('--grow', action='store_true', help='Load output and add --samples nodes')
    parser.add_argument('--collision', choices=['self', 'limits'], default='self')
    parser.add_argument('--self-clearance-mm', type=float, default=1.)
    args = parser.parse_args()
    config = PRMConfig(samples=args.samples, neighbors=args.neighbors, max_step_rad=args.max_step_rad, seed=args.seed)
    space = planning_space(args.collision, args.self_clearance_mm)
    roadmap = Roadmap.load(args.output, space=space, config=config) if args.grow else Roadmap(space, config)
    roadmap.grow(samples=fk_samples(args.dataset) if args.dataset else None).save(args.output)
    print(json.dumps({**roadmap.stats, 'validation': roadmap.space.validation, 'output': str(args.output)}))


if __name__ == '__main__': main()
