"""Opt-in offline CPU/collision cProfile diagnostics, never a speed benchmark.

Only service.plan is profiled; checked encoder preparation remains outside
cProfile. Existing probe wrappers and all production collision predicates stay
as they are. The process never connects to hardware.
"""
import argparse
import cProfile
from datetime import datetime, timezone
import fcntl
import functools
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

from .metrics import digest, outcome_errors
from .runner import (ROOT, CHILD_THREAD_ENV, TIMING_CASES, create_output,
                     source_inventory, frozen_inventory, gpu_snapshot, read_events)
from .scenarios import load_cases, load_manifest

DEFAULT_REFERENCE = ROOT/'output/benchmark_reliable_timing_20261007/qualified/report.json'


def select_cases(names):
    catalog = {c['id']: c for c in load_cases('full')}
    if not names or len(names) != len(set(names)):
        raise ValueError('Selected cases must be nonempty and unique')
    unknown = [name for name in names if name not in catalog]
    if unknown: raise ValueError('Unknown cases: '+', '.join(unknown))
    return [catalog[name] for name in names]


def reference_signatures(reference, cases, seed, backend, sources, manifest):
    """Use prior evidence for exact correctness only, never absolute timing."""
    if reference.get('summary', {}).get('passed') is not True:
        raise ValueError('Reference report did not pass correctness checks')
    for field in ('checkpoint_sha256', 'urdf_sha256'):
        if reference.get('inputs', {}).get('expected_'+field) != manifest[field]:
            raise ValueError('Reference '+field+' differs from scenario manifest')
    previous = reference.get('source_sha256', {})
    production = {k:v for k,v in sources.items() if k.startswith(('simulation/', 'webui/'))}
    prior_production = {k:v for k,v in previous.items() if k.startswith(('simulation/', 'webui/'))}
    if not production or production != prior_production:
        raise ValueError('Production simulation/webui source differs from reference report')
    expected = {}
    for case in cases:
        rows = [r for r in reference.get('records', []) if r.get('case_id') == case['id']
                and r.get('seed') == seed and r.get('backend') == backend]
        if not rows: raise ValueError(f"No reference records for {case['id']} seed {seed} backend {backend}")
        request = {**case['request'], 'seed': seed}
        signatures = []
        for row in rows:
            if row.get('request') != request or outcome_errors(row, case):
                raise ValueError('Reference request/outcome differs or is invalid for '+case['id'])
            signature = row.get('signatures')
            if not isinstance(signature, dict) or not all(signature.get(k) for k in
                    ('full_reference','sampled_predictions','rng_after')) or 'encoder_profile' not in signature:
                raise ValueError('Reference signatures missing for '+case['id'])
            if row.get('device') != 'cuda':
                raise ValueError('Reference is not a CUDA run')
            if backend == 'cuda_graph' and row.get('metrics', {}).get('sample_calls', 0) and not row.get('graph_active'):
                raise ValueError('Reference graph replay was inactive')
            signatures.append(signature)
        if any(s != signatures[0] for s in signatures):
            raise ValueError('Reference signatures are inconsistent for '+case['id'])
        expected[case['id']] = signatures[0]
    return expected


def run_profiled_case(case, seed, backend, service, torch, path, run=None, profiler_factory=cProfile.Profile):
    """Always restore service.plan; encoder replay inside run_case is excluded."""
    if run is None:
        from .worker import run_case
        run = run_case
    profiler = profiler_factory()
    original = service.plan
    @functools.wraps(original)
    def measured(*args, **kwargs):
        return profiler.runcall(original, *args, **kwargs)
    service.plan = measured
    try:
        return run(case, seed, backend, service, torch)
    finally:
        service.plan = original
        profiler.dump_stats(str(path))


def diagnostic_errors(record, case, expected, backend):
    if record.get('error'): return [record['error']]
    phases = ('warmup', 'control', 'profiled')
    signatures = [record.get(phase, {}).get('signatures') for phase in phases]
    equal = all(s is not None and s == signatures[0] for s in signatures)
    errors = []
    if not equal: errors.append('Warmup/control/profiled signatures differ or are missing')
    if any(s != expected for s in signatures): errors.append('Signatures differ from qualified reference')
    for phase in phases:
        row = record.get(phase, {})
        if not row:
            errors.append(phase+': missing record'); continue
        errors.extend(phase+': '+e for e in outcome_errors({**row, 'warmup_exact': equal}, case))
        if row.get('device') != 'cuda': errors.append(phase+': CUDA backend unavailable')
        if backend == 'cuda_graph' and row.get('metrics', {}).get('sample_calls', 0) and not row.get('graph_active'):
            errors.append(phase+': requested graph replay inactive')
        if backend == 'eager' and row.get('graph_active'):
            errors.append(phase+': eager backend unexpectedly used graph replay')
    return errors


def worker_main(input_path, output_path):
    from .worker import offline_guard, run_case
    offline_guard()
    spec = json.loads(Path(input_path).read_text())
    backend = spec['backend']
    os.environ['MYCOBOT_CUDA_GRAPH'] = '1' if backend == 'cuda_graph' else '0'
    with Path(output_path).open('x') as output:
        def emit(event):
            event['monotonic_s'] = time.monotonic()
            output.write(json.dumps(event, allow_nan=False)+'\n'); output.flush()
        began = time.perf_counter()
        try:
            import torch
            from simulation.diffusion import service
            from simulation.kinematics import ARM
            from webui.hardware_diffusion import verified_status
            status = verified_status()
            if status['checkpoint_sha256'] != spec['expected_checkpoint_sha256'] or ARM.fingerprint != spec['expected_urdf_sha256']:
                raise RuntimeError('Runtime checkpoint/URDF differs from scenario manifest')
            with service._lock: policy = service.load_policy(status, spec['seed'])
            if policy.device != 'cuda': raise RuntimeError('CUDA is required for the selected diagnostic backend')
            torch.cuda.synchronize()
            emit({'type':'setup', 'cold_import_model_setup_ms':(time.perf_counter()-began)*1000,
                  'device':str(policy.device), 'backend':backend, 'checkpoint_sha256':status['checkpoint_sha256'],
                  'urdf_sha256':ARM.fingerprint, 'environment':status['environment'],
                  'deployment_sha256':hashlib.sha256((Path(status['checkpoint_path']).parent.parent/'webui.json').read_bytes()).hexdigest(),
                  'torch':torch.__version__, 'cuda_runtime':torch.version.cuda, 'gpu':torch.cuda.get_device_name(),
                  'torch_threads':torch.get_num_threads(), 'torch_interop_threads':torch.get_num_interop_threads(),
                  'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
                  'thread_environment':{k:os.environ.get(k) for k in CHILD_THREAD_ENV},
                  'dependencies':{name:importlib.metadata.version(name) for name in
                                  ('numpy','scipy','torch','diffusers','pybullet','python-fcl')}})
            for case in spec['cases']:
                for iteration in range(spec['repeats']):
                    key = {'case_id':case['id'], 'seed':spec['seed'], 'backend':backend, 'iteration':iteration}
                    emit({'type':'started', **key})
                    profile_path = Path(output_path).parent/f"{case['id']}-seed{spec['seed']}-{backend}-{iteration:02d}.pstats"
                    try:
                        emit({'type':'phase_started', 'phase':'warmup', **key})
                        warmup = run_case(case, spec['seed'], backend, service, torch)
                        emit({'type':'phase_finished', 'phase':'warmup', **key})
                        emit({'type':'phase_started', 'phase':'control', **key})
                        control = run_case(case, spec['seed'], backend, service, torch)
                        emit({'type':'phase_finished', 'phase':'control', **key})
                        emit({'type':'phase_started', 'phase':'profiled', **key})
                        profiled = run_profiled_case(case, spec['seed'], backend, service, torch, profile_path)
                        emit({'type':'phase_finished', 'phase':'profiled', **key})
                        record = {**key, 'warmup':warmup, 'control':control, 'profiled':profiled,
                                  'profiler_pstats':profile_path.name,
                                  'signatures_exact':warmup['signatures'] == control['signatures'] == profiled['signatures'],
                                  'expected_reference_exact':all(r['signatures'] == spec['expected_signatures'][case['id']]
                                                                 for r in (warmup,control,profiled)),
                                  'observer_overhead_ratio':profiled['timing_ms']['total']/control['timing_ms']['total']}
                        record['errors'] = diagnostic_errors(record, case, spec['expected_signatures'][case['id']], backend)
                        emit({'type':'finished', **record})
                    except Exception as exc:
                        emit({'type':'finished', **key, 'error':f'{type(exc).__name__}: {exc}',
                              'traceback':traceback.format_exc(),
                              'profiler_pstats':profile_path.name if profile_path.exists() else None})
        except Exception as exc:
            emit({'type':'fatal', 'error':f'{type(exc).__name__}: {exc}', 'traceback':traceback.format_exc()})
            return 2
    return 0


def launch_diagnostic(directory, timeout):
    from .cpu_telemetry import sample_cpu, summarize_cpu
    output = directory/'worker.jsonl'
    samples = []
    observed = 0
    began = last_progress = next_sample = time.monotonic()
    timed_out = False
    with (directory/'worker.log').open('x') as log, (directory/'cpu-telemetry.jsonl').open('x') as telemetry:
        child = subprocess.Popen([sys.executable,'-m','benchmarks.suite.profile_cpu',
                    '--worker-input',str(directory/'inputs.json'),'--worker-output',str(output)],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    env={**os.environ,**CHILD_THREAD_ENV})
        try:
            while child.poll() is None:
                now = time.monotonic()
                if now >= next_sample:
                    sample = sample_cpu(child.pid)
                    samples.append(sample)
                    telemetry.write(json.dumps(sample,allow_nan=False)+'\n'); telemetry.flush()
                    next_sample = now+.5
                events = read_events(output)
                if len(events) > observed:
                    observed = len(events); last_progress = time.monotonic()
                if time.monotonic()-last_progress > timeout:
                    timed_out=True; child.kill(); break
                time.sleep(.05)
            child.wait()
        finally:
            if child.poll() is None: child.kill(); child.wait()
    events = read_events(output)
    starts = {}
    phase_starts, phase_summaries = {}, {}
    for event in events:
        key = event.get('case_id'),event.get('seed'),event.get('iteration')
        if event['type'] == 'started': starts[key] = event['monotonic_s']
        elif event['type'] == 'phase_started': phase_starts[(*key,event['phase'])] = event['monotonic_s']
        elif event['type'] == 'phase_finished':
            phase_key = (*key,event['phase'])
            phase_summaries.setdefault(key,{})[event['phase']] = summarize_cpu(
                samples,phase_starts.get(phase_key,event['monotonic_s']),event['monotonic_s'],child.pid)
        elif event['type'] == 'finished':
            event['cpu_telemetry'] = summarize_cpu(samples,starts.get(key,event['monotonic_s']),event['monotonic_s'],child.pid)
            event['cpu_telemetry_phases'] = phase_summaries.get(key,{})
    return {'pid':child.pid,'returncode':child.returncode,'timed_out':timed_out,
            'wall_ms':(time.monotonic()-began)*1000,'events':events,'cpu_telemetry_samples':samples,
            'cpu_telemetry_file':'cpu-telemetry.jsonl','data':'worker.jsonl','log':'worker.log'}


def parser():
    p = argparse.ArgumentParser(description='Offline diagnostic CPU/collision cProfile; no speed claims or hardware IO.')
    p.add_argument('--backend',choices=('eager','cuda_graph'),default='cuda_graph')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--repeats',type=int,default=2,help='Warmup/control/profile triples per case, 1..10')
    p.add_argument('--cases',nargs='+',default=TIMING_CASES)
    p.add_argument('--reference',type=Path,default=DEFAULT_REFERENCE,help='Prior passing report used only for exact correctness')
    p.add_argument('--output',type=Path,help='New output directory; protected frozen paths forbidden')
    p.add_argument('--timeout',type=float,default=300,help='Seconds without child progress')
    p.add_argument('--worker-input',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--worker-output',type=Path,help=argparse.SUPPRESS)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.worker_input or args.worker_output:
        if not (args.worker_input and args.worker_output): parser().error('Both worker paths required')
        return worker_main(args.worker_input,args.worker_output)
    if not 1 <= args.repeats <= 10 or not 0 <= args.seed < 2**31 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser().error('repeats must be 1..10, seed 0..2**31-1, timeout finite and positive')
    try: cases = select_cases(args.cases)
    except ValueError as exc: parser().error(str(exc))
    lock_path = ROOT/'tmp/benchmark-suite.lock'
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('Another benchmark owns the GPU lock.',file=sys.stderr); return 2
        try:
            sources = source_inventory()
            frozen = frozen_inventory()
            manifest = load_manifest()
            reference_bytes = args.reference.read_bytes()
            reference = json.loads(reference_bytes)
            expected = reference_signatures(reference,cases,args.seed,args.backend,sources,manifest)
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
            directory=create_output(args.output or ROOT/'output/cpu-profile'/stamp)
        except (OSError, ValueError) as exc:
            print(str(exc),file=sys.stderr); return 2
        spec={'cases':cases,'seed':args.seed,'backend':args.backend,'repeats':args.repeats,
              'expected_checkpoint_sha256':manifest['checkpoint_sha256'],'expected_urdf_sha256':manifest['urdf_sha256'],
              'expected_signatures':expected}
        (directory/'inputs.json').write_text(json.dumps(spec,indent=2)+'\n')
        report={'schema_version':1,'created_utc':stamp,'inputs':spec,'input_sha256':digest(spec),
                'scope':'DIAGNOSTIC cProfile of service.plan only; observer overhead prevents performance claims or absolute comparisons to prior timing',
                'hardware_qualification':'UNTESTED; no robot/network/serial IO',
                'reference':{'path':str(args.reference.resolve()),'sha256':hashlib.sha256(reference_bytes).hexdigest(),
                             'use':'Exact request/source/model/output correctness only; prior timings are not compared'},
                'source_sha256':sources,'source_closure_sha256':digest(sources),'frozen_reference_sha256':frozen,
                'runtime':{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
                           'enforced_child_thread_environment':CHILD_THREAD_ENV},
                'measurement_policy':{'per_case_iteration':['warmup','unprofiled_control','profiled'],
                       'profile_scope':'Only service.plan; encoder preparation excluded; existing stage probes retained; ArmEnv.valid identity unchanged',
                       'cpu_poll_interval_s':.5,'cpu_window':'Warmup, control, profiled plan and optional encoder preparation',
                       'isolation':'One fresh sequential child process, shared benchmark GPU lock',
                       'timeout_seconds_without_progress':args.timeout},
                'gpu_before':gpu_snapshot(),'workers':[],'records':[]}
        failures=[]
        try:
            process=launch_diagnostic(directory,args.timeout)
            report['workers'].append(process)
            for event in process['events']:
                if event['type']=='finished': report['records'].append(event)
                elif event['type']=='fatal': failures.append(event['error'])
            if process['returncode'] or process['timed_out']:
                failures.append(f'Worker failed or timed out (exit {process["returncode"]})')
        except KeyboardInterrupt: failures.append('Diagnostic interrupted')
        except Exception as exc: failures.append(f'Orchestration error: {type(exc).__name__}: {exc}')
        from .profile_analysis import analyze_profile, write_report
        for case in cases:
            rows=[r for r in report['records'] if r['case_id']==case['id']]
            if sorted(r['iteration'] for r in rows) != list(range(args.repeats)):
                failures.append(case['id']+': missing or duplicate iterations')
            for row in rows:
                errors=diagnostic_errors(row,case,expected[case['id']],args.backend)
                if row.get('seed') != args.seed or row.get('backend') != args.backend:
                    errors.append('Record seed/backend differs from declared diagnostic inputs')
                row['errors']=errors
                failures.extend(f"{case['id']} iteration {row['iteration']}: {e}" for e in errors)
                if row.get('profiler_pstats'):
                    try: row['analysis']=analyze_profile(directory/row['profiler_pstats'])
                    except Exception as exc: failures.append(f"{case['id']} profile analysis failed: {exc}")
                else: failures.append(case['id']+': raw profile missing')
        if len(report['records']) != len(cases)*args.repeats: failures.append('Incorrect total record count')
        report['gpu_after']=gpu_snapshot()
        report['source_unchanged']=source_inventory()==sources
        report['frozen_unchanged']=frozen_inventory()==frozen
        try: report['reference_unchanged']=args.reference.read_bytes()==reference_bytes
        except OSError: report['reference_unchanged']=False
        if not report['source_unchanged']: failures.append('Source changed during profiling')
        if not report['frozen_unchanged']: failures.append('Frozen benchmark changed during profiling')
        if not report['reference_unchanged']: failures.append('Reference report changed during profiling')
        report['summary']={'passed':not failures,'failures':failures,'records':len(report['records']),
                           'expected_records':len(cases)*args.repeats,'performance_claim':'None; diagnostic profiling only'}
        write_report(directory,report)
        print(f'{"PASS" if not failures else "FAIL"}: {directory}/report.html',flush=True)
        return 0 if not failures else 1


if __name__=='__main__': raise SystemExit(main())
