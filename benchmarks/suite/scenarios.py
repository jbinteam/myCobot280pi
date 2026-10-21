"""Frozen, self-contained offline requests; importing this module loads no robot code.

Known-success labels describe saved seed-42 runs only. New exact-FK goals have
checked start/end geometry, but deliberately do not assert planner convergence.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

MANIFEST = Path(__file__).with_name('cases.json')
CONFIG = {'seed': 42, 'samples': 4, 'inference_steps': 32, 'max_decisions': 200,
          'speed': 70, 'velocity_shape': {'kind': 'smooth_trapezoid',
                                        'rise_percent': 5, 'fall_percent': 10}}
EXPLORATORY_STATUSES = ['success', 'decision_limit', 'no_safe_chunk']
REJECTION_STATUSES = {'invalid_start', 'invalid_goal', 'ik_failed'}


def _vector(value, size, name):
    if (not isinstance(value, list) or len(value) != size or
            any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                not math.isfinite(v) for v in value)):
        raise ValueError(f'{name} must contain {size} finite numbers')


def validate_cases(cases):
    """Reject ambiguous/malformed benchmark definitions before starting workers.

    Structural validation only. CPU geometry tests independently check the
    frozen FK endpoints, actual operating limits and scene rejection fixtures.
    """
    if not isinstance(cases, list) or not cases:
        raise ValueError('cases must be a nonempty list')
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError('case must be an object')
        name = case.get('id')
        if not isinstance(name, str) or not re.fullmatch(r'[a-z0-9_]+', name) or name in seen:
            raise ValueError('case id must be unique lowercase letters, digits and underscores')
        seen.add(name)
        for field in ('category', 'description'):
            if not isinstance(case.get(field), str) or not case[field].strip():
                raise ValueError(f'{name}: {field} must be nonempty text')
        if type(case.get('quick')) is not bool or type(case.get('qualified')) is not bool:
            raise ValueError(f'{name}: quick and qualified must be booleans')
        r = case.get('request')
        if not isinstance(r, dict):
            raise ValueError(f'{name}: request must be an object')
        for field, size in (('start_deg', 6), ('target_mm', 3), ('target_axis', 3)):
            _vector(r.get(field), size, f'{name}.{field}')
        if 'goal_deg' in r:
            _vector(r['goal_deg'], 6, f'{name}.goal_deg')
        endpoint_selection = ('explicit_joint_goal_skips_ik' if 'goal_deg' in r
                              else 'cartesian_ik_when_start_valid')
        if case.get('endpoint_selection') != endpoint_selection:
            raise ValueError(f'{name}: endpoint_selection must describe the actual IK workload')
        if max(abs(v) for v in r['target_mm']) > 600:
            raise ValueError(f'{name}: target must lie within the API ±600 mm coordinate bounds')
        if abs(sum(v * v for v in r['target_axis']) - 1) > 1e-9:
            raise ValueError(f'{name}: target_axis must be a unit directed flange Z axis')
        if 'obstacle_mm' not in r:
            raise ValueError(f'{name}: obstacle_mm must explicitly be null or a box')
        box = r['obstacle_mm']
        if box is not None:
            _vector(box, 6, f'{name}.obstacle_mm')
            if (max(abs(v) for v in box[:2]) > 400 or
                    any(not 5 <= v <= 250 for v in box[3:]) or
                    abs(box[2] - box[5] + 35) > 1e-6):
                raise ValueError(f'{name}: block must match the table-mounted API box family')
        for field, value in CONFIG.items():
            actual = r.get(field)
            if actual != value or type(actual) is not type(value):
                raise ValueError(f'{name}: frozen request must explicitly set {field}={value}')
        expected = case.get('expected')
        if not isinstance(expected, dict) or expected.get('kind') not in {'success', 'reject', 'allowed'}:
            raise ValueError(f'{name}: expected kind must be success, reject or allowed')
        statuses = expected.get('statuses')
        if (not isinstance(statuses, list) or not statuses or
                any(not isinstance(s, str) for s in statuses) or len(set(statuses)) != len(statuses)):
            raise ValueError(f'{name}: expected statuses must be a nonempty unique string list')
        kind = expected['kind']
        if kind == 'success':
            if statuses != ['success'] or case.get('qualified_seeds') != [42] or not case['qualified']:
                raise ValueError(f'{name}: saved success qualification is seed 42 only')
            if case.get('exploratory_statuses') != EXPLORATORY_STATUSES:
                raise ValueError(f'{name}: other seeds must carry explicit exploratory outcomes')
        elif kind == 'reject':
            if not set(statuses) <= REJECTION_STATUSES or not case['qualified']:
                raise ValueError(f'{name}: rejects must name explicit scene/IK rejection statuses')
        elif case['qualified'] or statuses != EXPLORATORY_STATUSES:
            raise ValueError(f'{name}: exploratory cases allow bounded outcomes, never timeouts or errors')
        if not isinstance(case.get('evidence'), dict) or not case['evidence'].get('kind'):
            raise ValueError(f'{name}: evidence provenance is required')
        if case.get('encoder_profile'):
            if kind != 'success' or box is not None or not isinstance(case.get('encoder_mapping'), dict):
                raise ValueError(f'{name}: encoder replay requires a qualified no-block case and frozen mapping')
    return cases


def load_manifest(path=MANIFEST):
    """Read and validate static fixtures without Torch, PyBullet or transport IO."""
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict) or value.get('schema') != 1:
        raise ValueError('unsupported benchmark manifest schema')
    validate_cases(value.get('cases'))
    return value


def load_cases(mode='quick'):
    """Return independent request dictionaries in stable manifest order."""
    if mode not in {'quick', 'full'}:
        raise ValueError('mode must be quick or full')
    cases = load_manifest()['cases']
    return deepcopy([case for case in cases if mode == 'full' or case['quick']])


def manifest_sha256():
    return hashlib.sha256(MANIFEST.read_bytes()).hexdigest()


def expected_for_seed(case, seed):
    """Do not extrapolate historical seed-42 convergence to unqualified seeds."""
    expected = deepcopy(case['expected'])
    if expected['kind'] == 'success' and seed not in case['qualified_seeds']:
        return {'kind': 'allowed', 'statuses': deepcopy(case['exploratory_statuses'])}
    return expected
