"""Measure mouse-aim latency without changing its IK or validation policy.

Run in a fresh process with OPENBLAS_NUM_THREADS=1. Unprofiled wall-clock
measurements are separate from instrumented/cProfile hotspot measurements.
This module does not import hardware transport or execute robot motion.
"""
import argparse
from contextlib import contextmanager
import cProfile
import functools
import io
import json
from pathlib import Path
import platform
import pstats
from statistics import median
from time import perf_counter


class StageTimers:
    """Report nested inclusive/exclusive times; never add inclusive totals."""
    def __init__(self):
        self.rows = {}
        self.stack = []

    def wrap(self, name, function):
        @functools.wraps(function)
        def measured(*args, **kwargs):
            entry = [perf_counter(), 0.]
            self.stack.append(entry)
            try:
                return function(*args, **kwargs)
            finally:
                elapsed = perf_counter()-entry[0]
                self.stack.pop()
                if self.stack:
                    self.stack[-1][1] += elapsed
                row = self.rows.setdefault(name, {'calls': 0, 'inclusive_ms': 0., 'exclusive_ms': 0.})
                row['calls'] += 1
                row['inclusive_ms'] += elapsed*1000
                row['exclusive_ms'] += (elapsed-entry[1])*1000
        return measured


@contextmanager
def instrument(timers, patches):
    originals = []
    try:
        for owner, attribute, label in patches:
            original = getattr(owner, attribute)
            originals.append((owner, attribute, original))
            setattr(owner, attribute, timers.wrap(label, original))
        yield
    finally:
        for owner, attribute, original in reversed(originals):
            setattr(owner, attribute, original)


def summary(values, np):
    return {'min': float(np.min(values)), 'median': float(np.median(values)),
            'max': float(np.max(values)), 'p90': float(np.percentile(values, 90))}


def profile_call(name, call, patches, output):
    timers = StageTimers()
    profiler = cProfile.Profile()
    began = perf_counter()
    with instrument(timers, patches):
        result = profiler.runcall(call)
    elapsed = (perf_counter()-began)*1000
    profiler.dump_stats(str(output/f'{name}.pstats'))
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative').print_stats(40)
    stats.sort_stats('tottime').print_stats(30)
    (output/f'{name}.txt').write_text(stream.getvalue())
    def hotspot(key, value):
        primitive, calls, exclusive, inclusive, _ = value
        path, line, function = key
        return {'file': path, 'line': line, 'function': function, 'calls': calls,
                'primitive_calls': primitive, 'exclusive_ms': exclusive*1000,
                'inclusive_ms': inclusive*1000}
    return {'instrumented_elapsed_ms': elapsed, 'status': result.get('status'),
            'stages': timers.rows,
            'top_cumulative': [hotspot(k,v) for k,v in sorted(stats.stats.items(), key=lambda item: item[1][3], reverse=True)[:30]],
            'top_exclusive': [hotspot(k,v) for k,v in sorted(stats.stats.items(), key=lambda item: item[1][2], reverse=True)[:30]]}


def run(output, repeats):
    output.mkdir(parents=True, exist_ok=True)
    began = perf_counter()
    import numpy as np
    from . import service, planning, prm, pose5, seed5
    from .kinematics import ARM, Arm
    from .self_collision import SelfCollisionChecker
    import_ms = (perf_counter()-began)*1000
    initial = [0., -40., -130., 80., 0., 50.]
    t = ARM.transform(np.deg2rad(initial))
    anchor = t[:3,3]*1000+50*t[:3,2]
    fixtures = json.loads((Path(__file__).parent/'fixtures/mouse_aim_recovery_cases.json').read_text())
    cases = [
        {'id': 'already_at_target', 'start_deg': initial, 'cursor_mm': anchor.tolist()},
        {'id': 'nearby_direct', 'start_deg': initial, 'cursor_mm': (anchor+[4,3,0]).tolist()},
    ]
    for case_id, label in [('start-1-cursor--60-120', 'recovered_edge'), ('start-0-cursor-120-120', 'failed_5d')]:
        case = next(c for c in fixtures['cases'] if c['id']==case_id)
        cases.append({'id': label, **{key:case[key] for key in ('start_deg','cursor_mm')}})
    patches = [(prm, 'planning_space', 'space_construction'),
               (service, 'load_roadmap', 'roadmap_load_or_cache'),
               (planning, 'generate_goals', 'goal_generation'),
               (seed5, 'propose', 'nn_proposals'),
               (pose5, 'solve', 'numerical_refinement'),
               (prm.Roadmap, 'plan', 'direct_or_prm_routing'),
               (planning, 'trajectory', 'trajectory_generation_validation'),
               (planning, 'joint_waypoints', 'controller_waypoint_conversion'),
               (SelfCollisionChecker, '_check', 'collision_checks'),
               (Arm, 'transform', 'fk_or_jacobian'),
               (Arm, 'link_transforms', 'collision_fk')]
    cold_timers = StageTimers()
    cold_patches = [(prm, 'planning_space', 'space_construction'),
                    (service, 'load_roadmap', 'roadmap_load_or_cache'),
                    (SelfCollisionChecker, '__init__', 'collision_checker_initialization')]
    began = perf_counter()
    with instrument(cold_timers, cold_patches):
        cold_result = service.mouse_aim_plan(cases[1])
    cold_ms = (perf_counter()-began)*1000
    print(f'Cold query: {cold_ms:.1f} ms; {cold_result["status"]}', flush=True)
    road = service.load_roadmap(service.ARTIFACTS/'roadmap.json', prm.planning_space('self',1.))
    assert cold_result['converged'] and cold_result['cursor_acceptance']['accepted']
    report = {'simulation_only': True, 'platform': platform.platform(), 'python': platform.python_version(),
              'numpy': np.__version__, 'repeats': repeats, 'imports_ms': import_ms,
              'cold_query': {'elapsed_ms': cold_ms, 'ik_ms': cold_result['ik_ms'],
                             'planning_ms': cold_result['planning_ms'], 'stages': cold_timers.rows},
              'roadmap': {'nodes': len(road.q), 'edges': sum(map(len,road.edges.values()))//2,
                          'step_rad': road.config.max_step_rad}, 'cases': []}
    for case in cases:
        samples = []
        for _ in range(repeats):
            began = perf_counter()
            result = service.mouse_aim_plan(case)
            elapsed_ms = (perf_counter()-began)*1000
            generated = result.get('ik', {})
            attempts = [a for a in generated.get('attempts',[]) if 'refinement_method' in a]
            samples.append({'elapsed_ms': elapsed_ms, 'ik_ms': result['ik_ms'], 'planning_ms': result['planning_ms'],
                            'outside_reported_compute_ms': elapsed_ms-result['ik_ms']-result['planning_ms'],
                            'inference_ms': generated.get('inference_ms',0.),
                            'refinement_ms': sum(a['refinement_ms'] for a in attempts),
                            'refinements': len(attempts), 'iterations': sum(a['iterations'] for a in attempts),
                            'accepted_goals': len(generated.get('goals',[])), 'frames': len(result.get('frames',[])),
                            'status': result['status'], 'planning_outcome': result['planning_outcome']})
            if result.get('converged'):
                assert result['validation']=='joint_limits_and_self_collision'
                assert result['cursor_acceptance']['accepted']
        began = perf_counter()
        payload = json.dumps(result, allow_nan=False).encode()
        serialization_ms = (perf_counter()-began)*1000
        row = {**case, 'samples': samples, 'timing_ms': {key:summary([s[key] for s in samples],np)
               for key in ('elapsed_ms','ik_ms','planning_ms','outside_reported_compute_ms','inference_ms','refinement_ms')},
               'json_serialization_ms': serialization_ms, 'response_bytes': len(payload)}
        row['profile'] = profile_call(case['id'], lambda:service.mouse_aim_plan(case), patches, output)
        report['cases'].append(row)
        print(f'{case["id"]}: median {row["timing_ms"]["elapsed_ms"]["median"]:.1f} ms; '
              f'IK {row["timing_ms"]["ik_ms"]["median"]:.1f}; planning {row["timing_ms"]["planning_ms"]["median"]:.1f}', flush=True)
        (output/'profile.json').write_text(json.dumps(report,indent=2)+'\n')
    fixture = json.loads((Path(__file__).parent/'fixtures/self_collision_path.json').read_text())
    start, goal = (np.deg2rad(fixture[key]) for key in ('start_deg','goal_deg'))
    def blocked_route():
        result = road.plan(start, [goal], shortcut=True)
        if result['status']=='success':
            planning.trajectory(result['waypoints_rad'], ARM.fk(goal), road.space,
                                road.config.max_step_rad, target_axis=ARM.transform(goal)[:3,2])
        return result
    samples=[]
    for _ in range(repeats):
        began=perf_counter();result=blocked_route()
        samples.append({'elapsed_ms':(perf_counter()-began)*1000,'status':result['status'],'reason':result['reason']})
    report['blocked_joint_route']={'samples':samples,'timing_ms':summary([s['elapsed_ms'] for s in samples],np),
                                   'profile':profile_call('blocked_joint_route',blocked_route,patches,output),
                                   'scope':'Known joint endpoints: routing/trajectory only; no IK.'}
    report['microbenchmarks']={}
    for name, call in [
        ('fk', lambda:ARM.transform(np.deg2rad(initial))),
        ('fk_with_jacobian', lambda:ARM.spatial_jacobian(np.deg2rad(initial))),
        ('state_validity', lambda:road.space.valid(np.deg2rad(initial))),
        ('warm_nn_proposals', lambda:seed5.propose(np.deg2rad(initial),anchor/1000,t[:3,2])),
        ('warm_space_construction', lambda:prm.planning_space('self',1.)),
    ]:
        durations=[]
        for _ in range(100):
            began=perf_counter();call();durations.append((perf_counter()-began)*1000)
        report['microbenchmarks'][name]=summary(durations,np)
    report['measurement_notes']=[
        'Warm latency samples are unprofiled. Cold query has only three low-frequency timer wrappers.',
        'cProfile/instrumented durations are for hotspots, not production latency or speedup claims.',
        'Inclusive stage times overlap; exclusive times exclude other wrapped stages.',
        'Four selected tasks and one blocked joint route are diagnostic cases, not workspace-wide statistics.',
        'Other applications were running; absolute times depend on CPU load/power policy.',
    ]
    (output/'profile.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Profile written to {output}/profile.json',flush=True)
    return report


def http_measurements(output, repeats):
    """Measure the existing local simulation endpoint, without serial access."""
    import urllib.request
    base = 'http://127.0.0.1:8000'
    state = json.load(urllib.request.urlopen(base+'/api/state', timeout=5))
    if (state.get('hardware_enabled') is not False or state.get('connected')
            or state.get('motion') or state.get('moving')):
        raise RuntimeError('HTTP measurements require idle, hardware-disabled simulation')
    profile = json.loads((output/'profile.json').read_text())
    warmup_body={key:profile['cases'][0][key] for key in ('start_deg','cursor_mm')}
    warmup_request=urllib.request.Request(base+'/api/sim/mouse-aim',data=json.dumps(warmup_body).encode(),
                   headers={'Content-Type':'application/json','X-Robot-Token':state['token']})
    began=perf_counter()
    with urllib.request.urlopen(warmup_request,timeout=180) as response:
        warmup=json.load(response)
    warmup_ms=(perf_counter()-began)*1000
    assert warmup['converged'] and warmup['cursor_acceptance']['accepted']
    rows = []
    for case in profile['cases']:
        samples = []
        for _ in range(repeats):
            body = {key:case[key] for key in ('start_deg','cursor_mm')}
            request = urllib.request.Request(base+'/api/sim/mouse-aim', data=json.dumps(body).encode(),
                      headers={'Content-Type':'application/json','X-Robot-Token':state['token']})
            began = perf_counter()
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = response.read()
            elapsed = (perf_counter()-began)*1000
            result = json.loads(payload)
            if result.get('converged'):
                assert result['cursor_acceptance']['accepted']
                assert result['validation']=='joint_limits_and_self_collision'
            samples.append({'elapsed_ms':elapsed,'ik_ms':result['ik_ms'],'planning_ms':result['planning_ms'],
                            'outside_reported_compute_ms':elapsed-result['ik_ms']-result['planning_ms'],
                            'status':result['status'],'response_bytes':len(payload)})
        rows.append({'id':case['id'],'samples':samples})
        print(f'{case["id"]}: HTTP median {median(s["elapsed_ms"] for s in samples):.1f} ms',flush=True)
    result = {'simulation_only':True,'warmup_ms':warmup_ms,'rows':rows,
              'note':'Warm HTTP queries on the already-running simulation-only server. '
                     'Wall time includes service setup, JSON and transport; excludes browser scheduling/playback.'}
    (output/'http.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('output/performance/mouse_aim'))
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--http-only',action='store_true',help='Replay existing profile cases through the running simulation server')
    args=parser.parse_args()
    if not 1<=args.repeats<=100:parser.error('--repeats must be between 1 and 100')
    (http_measurements if args.http_only else run)(args.output,args.repeats)
