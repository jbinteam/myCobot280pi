"""Read trusted local cProfile artifacts without changing planner function bindings.

Exclusive-time buckets are disjoint source ownership, not reconstructed complete
pipeline stages. Shared native/NumPy costs remain ambiguous; caller edges supply
context without adding nested cumulative times.
"""
from collections import defaultdict
import html
import json
from pathlib import Path
import pstats

CAVEATS = [
    'cProfile measures observed elapsed time in instrumented calls, not CPU cycles; GPU synchronization, native waits and profiling overhead can appear in those times.',
    'Exclusive tottime is summed once per function across disjoint source categories. Inclusive cumulative times and caller-edge cumulative times overlap and must never be added.',
    'A local collision-function bucket is not total collision cost: shared NumPy/FK/native functions serve multiple callers; unprofiled extension work can remain inside its Python caller.',
    'NumPy and other shared builtins remain ambiguous. Direct caller edges provide context but do not recover complete stack attribution or distinguish every use of a shared native function.',
    'Counts are exact for calls visible to cProfile, not for internal C/CUDA operations. An absent FCL/native entry means not separately observed, not zero work.',
    'This profile window covers service.plan generation only, inside timing_ms.total; surrounding synchronization and offline encoder preparation are excluded. Existing planner probe overhead inside service.plan is included. The profiled/control wall-time ratio includes profiling overhead and host variation; it cannot isolate either contribution and is never a speedup claim.',
    'Reading pstats uses the trusted-local Python profiling format; do not load untrusted uploaded profiles.',
]


def classify_function(filename, name):
    """Conservative static source attribution; shared routines stay shared."""
    path = filename.replace('\\', '/').lower()
    lowered = name.lower()
    if '/benchmarks/suite/' in path:
        return 'benchmark_observer'
    if 'pybullet' in lowered or 'pybullet' in path:
        if any(word in lowered for word in ('connect', 'disconnect', 'loadurdf', 'createcollision',
                                            'createvisual', 'createmulti', 'setgravity', 'setphysics',
                                            'getjointinfo', 'getnumjoints')):
            return 'scene_setup_provenance'
        return 'collision_bullet_operations'
    if '/fcl/' in path or '/fcl.' in path or 'fcl.' in lowered or 'fcl/' in lowered:
        return 'collision_fcl_visible'
    if path.endswith('/simulation/self_collision.py'):
        return ('scene_setup_provenance' if name in {'__init__', 'collision_mesh', '_checker',
                'checker', 'self_collision_space', 'assumptions', '__post_init__'}
                else 'self_collision_geometry')
    if path.endswith('/simulation/kinematics.py'):
        return ('scene_setup_provenance' if name in {'__init__', 'origin_matrix'}
                else 'forward_kinematics_shared')
    if path.endswith('/simulation/diffusion/env.py'):
        if name in {'valid', 'valid_many', 'edge_valid', '_edge_valid', 'path_valid', '_set_joints'}:
            return 'collision_bullet_operations'
        if name == 'inverse_kinematics':
            return 'inverse_kinematics'
        if name in {'forward_kinematics', 'bullet_transform'}:
            return 'forward_kinematics_shared'
        if name in {'bullet_urdf', '_bullet_urdf_unlocked', '__init__', 'close', '__enter__',
                    '__exit__', 'set_obstacle', 'metadata'}:
            return 'scene_setup_provenance'
        return 'planner_orchestration'
    if path.endswith('/simulation/prm.py'):
        if name in {'valid', 'diagnostic', 'edge_valid'}:
            return 'collision_validation'
        if name in {'fk_samples'}:
            return 'forward_kinematics_shared'
        if name in {'_asset_signature', '_geometry_assets', '_geometry_digest', 'geometry_digest',
                    '__init__', '__post_init__', 'metadata', 'planning_space'}:
            return 'scene_setup_provenance'
        return 'trajectory_work'
    if path.endswith('/simulation/diffusion/service.py'):
        if name in {'info', 'load_policy'}:
            return 'scene_setup_provenance'
        if name in {'errors', 'cartesian_score'}:
            return 'task_error_evaluation'
        return 'planner_orchestration'
    if path.endswith('/simulation/diffusion/policy.py'):
        if name in {'sample'}:
            return 'neural_inference_scheduler'
        return 'scene_setup_provenance' if name == '__init__' else 'trajectory_work'
    if (any(part in path for part in ('/torch/', '/diffusers/')) or
            path.endswith(('/diffusion/cuda_replay.py', '/diffusion/model.py'))):
        return 'neural_inference_scheduler'
    if path.endswith(('/simulation/pose5.py', '/simulation/ik.py')) or '/scipy/optimize/' in path:
        return 'inverse_kinematics'
    if path.endswith('/diffusion/common.py'):
        return 'trajectory_work' if name in {'observation', 'normalize', 'denormalize'} else 'scene_setup_provenance'
    if path.endswith('/simulation/robot_spec.py'):
        return 'scene_setup_provenance'
    if (path.endswith(('/diffusion/continuous_rollout.py', '/diffusion/smoothing.py',
                       '/diffusion/motion_conditioning.py', '/diffusion/cartesian_conditioning.py',
                       '/simulation/continuous.py', '/simulation/motion_limits.py')) or
            any(part in path for part in ('/simulation/c2', '/simulation/trajectory',
                                         '/webui/encoder_profile.py', '/webui/hardware_diffusion.py'))):
        return 'trajectory_work'
    if '/numpy/' in path or any(part in lowered for part in ('numpy.', 'numpy/', 'numpy.ndarray')):
        return 'numpy_shared_ambiguous'
    return 'other_shared_ambiguous'


def _identity(key):
    filename, line, name = key
    return {'filename': filename, 'line': line, 'name': name,
            'function': f'{filename}:{line}({name})'}


def _row(key, values):
    primitive, total, exclusive, cumulative, _ = values
    return {**_identity(key), 'category': classify_function(key[0], key[2]),
            'primitive_calls': primitive, 'total_calls': total,
            'exclusive_s': exclusive, 'cumulative_s': cumulative}


def _target_keys():
    targets = {
        'pybullet.resetJointState': lambda f, n: 'pybullet' in (f+' '+n) and 'resetJointState' in n,
        'pybullet.getClosestPoints': lambda f, n: 'pybullet' in (f+' '+n) and 'getClosestPoints' in n,
        'env.valid_many': lambda f, n: f.replace('\\', '/').endswith('/diffusion/env.py') and n == 'valid_many',
        'env._set_joints': lambda f, n: f.replace('\\', '/').endswith('/diffusion/env.py') and n == '_set_joints',
        'self_collision.valid_many': lambda f, n: f.replace('\\', '/').endswith('/self_collision.py') and n == 'valid_many',
        'self_collision._check': lambda f, n: f.replace('\\', '/').endswith('/self_collision.py') and n == '_check',
        'kinematics.link_transforms_batch': lambda f, n: f.replace('\\', '/').endswith('/kinematics.py') and n == 'link_transforms_batch',
        'kinematics.link_transforms': lambda f, n: f.replace('\\', '/').endswith('/kinematics.py') and n == 'link_transforms',
        'FCL.visible': lambda f, n: classify_function(f, n) == 'collision_fcl_visible',
    }
    for module, names in {
        'simulation/continuous.py': ('prepare', 'bound', 'sample_motion'),
        'simulation/motion_limits.py': ('derivative_hull', 'curve_envelope', 'scale_needed'),
        'simulation/diffusion/common.py': ('file_sha256',),
        'simulation/diffusion/service.py': ('info',),
    }.items():
        for name in names:
            label = Path(module).stem+'.'+name
            targets[label] = lambda f, n, module=module, name=name: f.replace('\\', '/').endswith('/'+module) and n == name
    return targets


def analyze_profile(path):
    """Analyze a trusted local .pstats file; never import simulation or Torch."""
    stats = pstats.Stats(str(Path(path)))
    rows = [_row(key, values) for key, values in stats.stats.items()]
    exclusive = sorted(rows, key=lambda row: (-row['exclusive_s'], row['function']))
    cumulative = sorted(rows, key=lambda row: (-row['cumulative_s'], row['function']))
    total = sum(row['exclusive_s'] for row in rows)
    groups = defaultdict(float)
    for row in rows:
        groups[row['category']] += row['exclusive_s']
    targeted = {}
    hotspot_functions = set()
    for label, matches in _target_keys().items():
        found = [r for r in cumulative if matches(r['filename'], r['name'])]
        hotspot_functions.update(row['function'] for row in found)
        targeted[label] = {'observed': bool(found), 'functions': found,
                           'primitive_calls': sum(row['primitive_calls'] for row in found),
                           'total_calls': sum(row['total_calls'] for row in found),
                           'exclusive_s': sum(row['exclusive_s'] for row in found)}
    # Context covers major hot functions as well as every targeted collision call.
    context_functions = hotspot_functions | {row['function'] for row in exclusive[:20]} | {
        row['function'] for row in cumulative[:20]}
    callers = []
    for key, values in stats.stats.items():
        identity = _identity(key)
        if identity['function'] not in context_functions:
            continue
        edges = []
        for caller, value in values[4].items():
            if isinstance(value, tuple) and len(value) == 4:
                # Caller edges reverse the first two fields relative to stats:
                # cProfile.snapshot_stats stores (nc, cc, tt, ct) for each edge.
                nc, cc, tt, ct = value
                edge = {**_identity(caller), 'primitive_calls': cc, 'total_calls': nc,
                        'exclusive_s': tt, 'cumulative_s': ct}
            else:
                # Legacy profile format has only a count: do not invent edge times.
                edge = {**_identity(caller), 'primitive_calls': None, 'total_calls': value,
                        'exclusive_s': None, 'cumulative_s': None}
            edge['category'] = classify_function(caller[0], caller[2])
            edges.append(edge)
        edges.sort(key=lambda e: (-(e['cumulative_s'] or 0), e['function']))
        callers.append({**identity, 'callers': edges})
    callers.sort(key=lambda e: e['function'])
    return {'total_exclusive_s': total, 'total_calls': stats.total_calls,
            'primitive_calls': stats.prim_calls, 'function_count': len(rows),
            'exclusive_groups': [{'category': category, 'exclusive_s': elapsed,
                                  'percent_of_profile': 100*elapsed/total if total else 0}
                                 for category, elapsed in sorted(groups.items(), key=lambda item: (-item[1], item[0]))],
            'top_exclusive': exclusive[:30], 'top_cumulative': cumulative[:30],
            'collision_hotspots': [r for r in cumulative if r['function'] in hotspot_functions
                                   and r['category'] in {'collision_bullet_operations', 'collision_fcl_visible',
                                                         'self_collision_geometry', 'forward_kinematics_shared'}],
            'target_hotspots': [r for r in cumulative if r['function'] in hotspot_functions],
            'targeted_calls': targeted, 'caller_context': callers,
            'attribution_policy': 'Disjoint static-source exclusive tottime groups; cumulative times are nonadditive. '
                                  'Shared NumPy/FK/native costs are not reassigned to callers or claimed as total collision cost.',
            'source_context': {
                'batched_collision': 'ArmEnv.valid_many caches/deduplicates poses, calls SelfCollisionChecker.valid_many '
                    'for batched FK/AABB screening and visible FCL narrow-phase, then _set_joints and getClosestPoints '
                    'for accepted poses versus table/block, restoring displayed joints afterward.',
                'scalar_collision': 'ArmEnv.valid reaches JointSpace.valid and SelfCollisionChecker.valid/_check; '
                    'scalar checks can occur outside existing measured collision stages, including execution gates and IK.',
                'remaining_cpu': 'Look for continuous.prepare/bound, motion_limits.curve_envelope/derivative_hull, '
                    'candidate projection and execution gates, smoothing/motion sampling, frame FK, and scene/provenance '
                    'setup. These are source-based investigation candidates, not assumed measured bottlenecks.',
                'fast_path_preserved': 'cProfile observes calls without replacing ArmEnv.valid or JointSpace.validity; '
                    'bound-method identity checks and batched collision dispatch remain unchanged.'},
            'caveats': list(CAVEATS)}


def _escape(value):
    return html.escape(str(value), quote=True)


def _json(value):
    return _escape(json.dumps(value, indent=2, allow_nan=False))


def _table(rows, columns):
    def display(value):
        if isinstance(value, float):
            return '<0.0001' if 0 < value < .0001 else f'{value:.4f}'
        return value
    heading = '<tr>'+''.join('<th>'+_escape(label)+'</th>' for _, label in columns)+'</tr>'
    body = ''.join('<tr>'+''.join('<td>'+_escape(display(row.get(key, '—')))+'</td>' for key, _ in columns)+'</tr>'
                   for row in rows)
    return '<div class="table-scroll"><table><thead>'+heading+'</thead><tbody>'+body+'</tbody></table></div>'


def write_report(directory, report):
    """Write self-contained diagnostic JSON/HTML, escaping every report string."""
    directory = Path(directory)
    (directory/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    sections = []
    columns = [('function', 'Function'), ('category', 'Attribution'), ('total_calls', 'Total calls'),
               ('primitive_calls', 'Primitive calls'), ('exclusive_s', 'Exclusive seconds'),
               ('cumulative_s', 'Cumulative seconds (nonadditive)')]
    for row in report.get('records', []):
        control = row.get('control', {}).get('timing_ms', {}).get('total')
        profiled = row.get('profiled', {}).get('timing_ms', {}).get('total')
        title = f"{row.get('case_id', '?')} / seed {row.get('seed', '?')} / {row.get('backend', '?')} / iteration {row.get('iteration', '?')}"
        sections.append('<section><h2>'+_escape(title)+'</h2>')
        sections.append(_table([{'control': control, 'profiled': profiled,
                                'overhead': row.get('observer_overhead_ratio'),
                                'exact': row.get('signatures_exact'),
                                'reference': row.get('expected_reference_exact')}],
                               [('control', 'Control planner ms'), ('profiled', 'Profiled planner ms'),
                                ('overhead', 'Profiled/control wall-time ratio (diagnostic)'),
                                ('exact', 'Control/profile signatures exact'), ('reference', 'Reference signatures exact')]))
        analysis = row.get('analysis', {})
        sections.append('<h3>Exclusive source attribution</h3>'+_table(analysis.get('exclusive_groups', []),
                        [('category', 'Source category'), ('exclusive_s', 'Exclusive seconds'),
                         ('percent_of_profile', '% of observed profile')]))
        for field, title in (('top_exclusive', 'Top exclusive functions'),
                             ('top_cumulative', 'Top cumulative functions — nonadditive'),
                             ('target_hotspots', 'Targeted collision/FK and remaining CPU hotspots')):
            sections.append('<h3>'+title+'</h3>'+_table(analysis.get(field, []), columns))
        sections.append('<details><summary>Exact visible call counts and direct callers (inclusive times overlap)</summary><pre>'+_json({
            'targeted_calls': analysis.get('targeted_calls', {}),
            'caller_context': analysis.get('caller_context', [])})+'</pre></details>')
        sections.append('<details><summary>Source context and attribution limits</summary><pre>'+_json(analysis.get('source_context', {}))+'</pre></details>')
        sections.append('<details><summary>CPU telemetry</summary><pre>'+_json({
            'summary': row.get('cpu_telemetry', {}), 'phases': row.get('cpu_telemetry_phases', {})})+'</pre></details></section>')
    status = 'PASS' if report.get('summary', {}).get('passed') else 'FAIL / INCOMPLETE'
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Offline collision and CPU profile</title><style>body{font:15px system-ui;margin:clamp(.75rem,3vw,2rem);color:#18222c}.table-scroll{max-width:100%;overflow-x:auto}table{border-collapse:collapse;width:100%;min-width:760px;margin:1rem 0}th,td{padding:.5rem;border:1px solid #ccd3da;text-align:left;overflow-wrap:anywhere}th{background:#edf2f6}h1,h2,h3{overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere}section{margin:2rem 0}li,p{max-width:110ch}</style>
<h1>Offline collision and remaining CPU profile: '''+status+'''</h1><p>Diagnostic attribution only. The profiled/control wall-time ratio includes profiling overhead and host variation; it cannot isolate either contribution and is not an optimization or speedup. No hardware execution or physical accuracy is qualified.</p>
<p>Tables of exclusive source time are disjoint. Cumulative times overlap. The collision-related rows do not by themselves measure total collision cost; inspect direct callers and shared NumPy/native work.</p><ul>'''+''.join('<li>'+_escape(caveat)+'</li>' for caveat in CAVEATS)+'''</ul>
<h2>Validation failures</h2><pre>'''+_json(report.get('summary', {}))+'''</pre>'''+''.join(sections)+'''<details><summary>Complete reproducibility record</summary><pre>'''+_json(report)+'''</pre></details></html>'''
    (directory/'report.html').write_text(page)
