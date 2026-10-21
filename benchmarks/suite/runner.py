"""Sequential isolated ABBA orchestration, immutable inputs, reproducible reports."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from . import SCHEMA_VERSION
from .metrics import digest, summarize
from .report import write_report
from .telemetry import POLL_INTERVAL_S, MAX_COVERAGE_GAP_S, sample_gpu, attach_windows

ROOT = Path(__file__).resolve().parents[2]
CHILD_THREAD_ENV = {name: '1' for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}
TIMING_CASES = ['recent_live_start', 'heldout_000062_block', 'heldout_000590_block']


def selected_cases(mode, requested=None):
    from .scenarios import load_cases
    catalog = load_cases('full' if mode == 'timing' else mode)
    names = requested if requested is not None else (TIMING_CASES if mode == 'timing' else [c['id'] for c in catalog])
    lookup = {case['id']: case for case in catalog}
    if not names or len(names) != len(set(names)):
        raise ValueError('Selected cases must be nonempty and unique')
    unknown = [name for name in names if name not in lookup]
    if unknown:
        raise ValueError(f'Unknown cases in {mode} catalog: '+', '.join(unknown))
    return [lookup[name] for name in names]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_inventory(root=ROOT):
    paths = set()
    for folder in ('simulation', 'webui', 'benchmarks/suite'):
        paths.update((root/folder).rglob('*.py'))
    # URDF/convex geometry and active manifests affect reproducibility too.
    for folder in ('simulation/assets', 'webui/public/models'):
        if (root/folder).exists():
            paths.update(p for p in (root/folder).rglob('*') if p.is_file())
    for name in ('simulation/artifacts/diffusion_active.json', 'benchmarks/suite/cases.json',
                 'webui/motion_settings.py'):
        if (root/name).exists(): paths.add(root/name)
    return {str(p.relative_to(root)): sha256(p) for p in sorted(paths)
            if '__pycache__' not in p.parts and 'node_modules' not in p.parts}


def frozen_inventory(root=ROOT):
    paths = [root/'benchmarks/current.json']
    paths.extend(p for d in (root/'benchmarks').glob('fixed-*') if d.is_dir()
                 for p in d.rglob('*') if p.is_file())
    paths.extend((root/'backups').glob('fixed-*.tar.gz'))
    return {str(p.relative_to(root)): sha256(p) for p in paths if p.exists()}


def create_output(path, root=ROOT):
    path = Path(path).resolve()
    protected = [root/'backups'] + [p for p in (root/'benchmarks').iterdir()
                                    if p.is_dir() and p.name.startswith('fixed-')]
    if any(path == p.resolve() or path.is_relative_to(p.resolve()) for p in protected):
        raise ValueError('Output must not be inside a frozen benchmark or backup directory')
    path.mkdir(parents=True, exist_ok=False)
    return path


def gpu_snapshot():
    try:
        proc = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,driver_version,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw',
                               '--format=csv'], capture_output=True, text=True, timeout=10)
        return {'returncode': proc.returncode, 'csv': proc.stdout, 'stderr': proc.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'unavailable': str(exc)}


def read_events(path):
    if not path.exists(): return []
    events = []
    for line in path.read_text().splitlines():
        try: events.append(json.loads(line))
        except json.JSONDecodeError: break  # Writer may be midway through its last line.
    return events


def launch_worker(directory, index, backend, timeout):
    output = directory/f'worker-{index:02d}-{backend}.jsonl'
    log = directory/f'worker-{index:02d}-{backend}.log'
    began = time.monotonic()
    last_progress = began
    observed = 0
    timed_out = False
    samples = []
    telemetry_path = directory/f'worker-{index:02d}-{backend}-telemetry.jsonl'
    next_poll = began
    with log.open('x') as stream, telemetry_path.open('x') as telemetry_stream:
        process = subprocess.Popen([sys.executable, '-m', 'benchmarks.suite.worker',
                    '--input', str(directory/'inputs.json'), '--output', str(output), '--backend', backend],
                    cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                    env={**os.environ, **CHILD_THREAD_ENV})
        try:
            while process.poll() is None:
                if time.monotonic() >= next_poll:
                    sample = sample_gpu()
                    samples.append(sample)
                    telemetry_stream.write(json.dumps(sample, allow_nan=False)+'\n'); telemetry_stream.flush()
                    next_poll = sample['query_started_monotonic_s']+POLL_INTERVAL_S
                events = read_events(output)
                if len(events) > observed:
                    last_progress = time.monotonic(); observed = len(events)
                if time.monotonic()-last_progress > timeout:
                    timed_out = True
                    process.kill()
                    break
                time.sleep(.1)
            process.wait()
        finally:
            if process.poll() is None:
                process.kill(); process.wait()
    return {'backend': backend, 'execution_position': index, 'pid': process.pid,
            'returncode': process.returncode, 'timed_out': timed_out,
            'process_wall_ms': (time.monotonic()-began)*1000,
            'events': attach_windows(read_events(output), samples, process.pid),
            'gpu_telemetry_samples': samples, 'gpu_telemetry_file': telemetry_path.name,
            'log': log.name, 'data': output.name}


def parser():
    p = argparse.ArgumentParser(description='Offline same-source eager/CUDA graph ABBA benchmark; never moves the robot.')
    p.add_argument('--mode', choices=('quick', 'full', 'timing'), default='quick')
    p.add_argument('--repeats', type=int, help='ABBA blocks (2 warm trials per backend per block; default quick=1/full=2/timing=10)')
    p.add_argument('--seeds', nargs='+', type=int)
    p.add_argument('--cases', nargs='+', help='Unique case IDs from the mode catalog (timing uses full catalog)')
    p.add_argument('--output', type=Path, help='New run directory; must not already exist')
    p.add_argument('--timeout', type=float, default=240, help='Seconds without worker progress (setup or warmup+trial pair)')
    p.add_argument('--allow-long-run', action='store_true', help='Required explicit opt-in for full and timing modes')
    p.add_argument('--list-cases', action='store_true', help='Print selected cases without loading Torch or running inference')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    from .scenarios import load_manifest
    try: cases = selected_cases(args.mode, args.cases)
    except ValueError as exc: parser().error(str(exc))
    if args.list_cases:
        print(json.dumps(cases, indent=2)); return 0
    if args.mode in ('full', 'timing') and not args.allow_long_run:
        parser().error(f'{args.mode} mode requires --allow-long-run')
    repeats = args.repeats if args.repeats is not None else {'quick':1, 'full':2, 'timing':10}[args.mode]
    seeds = args.seeds if args.seeds is not None else ([42, 43] if args.mode == 'full' else [42])
    if repeats < 1 or repeats > 100 or not math.isfinite(args.timeout) or args.timeout <= 0 or not seeds or len(seeds) != len(set(seeds)) or any(s < 0 or s >= 2**31 for s in seeds):
        parser().error('repeats must be 1..100, timeout positive, seeds unique integers in 0..2**31-1')
    # Global per-project nonblocking lock prevents two benchmark suites sharing the GPU.
    lock_path = ROOT/'tmp/benchmark-suite.lock'
    lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Another benchmark suite owns the GPU lock.', file=sys.stderr); return 2
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        try: directory = create_output(args.output or ROOT/'output/benchmark-suite'/stamp)
        except (ValueError, FileExistsError) as exc:
            print(str(exc), file=sys.stderr); return 2
        sources = source_inventory()
        frozen = frozen_inventory()
        manifest = load_manifest()
        spec = {'cases': cases, 'seeds': seeds, 'mode': args.mode, 'repeats': repeats,
                'expected_checkpoint_sha256': manifest['checkpoint_sha256'],
                'expected_urdf_sha256': manifest['urdf_sha256']}
        (directory/'inputs.json').write_text(json.dumps(spec, indent=2)+'\n')
        report = {'schema_version': SCHEMA_VERSION, 'created_utc': stamp,
                  'baseline_kind': 'same_source_eager', 'candidate_kind': 'same_source_cuda_graph',
                  'hardware_qualification': 'UNTESTED; physical endpoint convergence unresolved',
                  'scope': 'Offline pinned v2 planning and selected pure encoder preparation; no robot IO',
                  'input_sha256': digest(spec), 'scenario_sha256': digest(cases), 'inputs': spec,
                  'source_sha256': sources, 'source_closure_sha256': digest(sources),
                  'frozen_reference_sha256': frozen, 'frozen_reference_executed': False,
                  'runtime': {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
                              'cpu': platform.processor(), 'cpu_count': os.cpu_count(),
                              'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                              'thread_environment': {k: os.environ.get(k) for k in
                                  ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG')},
                              'enforced_child_thread_environment': CHILD_THREAD_ENV},
                  'gpu_before': gpu_snapshot(),
                  'measurement_policy': {'order': ['eager', 'cuda_graph', 'cuda_graph', 'eager']*repeats,
                        'warmup': 'One complete case+seed run immediately before every measured run in each fresh backend process',
                        'cold': 'Each process import/model setup and each case warmup separately recorded; graph capture in warmup',
                        'timing': 'Equal probes on both backends; total includes prediction hashing; profile/encoder preparation excluded from generation total',
                        'timeout_seconds_without_progress': args.timeout,
                        'reliable_timing_requested': args.mode == 'timing',
                        'timing_minimum_abba_blocks': 10,
                        'telemetry': {'poll_interval_s': POLL_INTERVAL_S, 'maximum_coverage_gap_s': MAX_COVERAGE_GAP_S,
                            'query_location': 'parent process outside worker planning timers',
                            'window': 'started warmup through measured trial completion, including encoder preparation',
                            'claim_limit': 'No competing compute process observed; short activity between samples may be missed'},
                        'isolation': 'Sequential fresh processes; no concurrent benchmark GPU workers'},
                  'workers': [], 'records': []}
        infrastructure_errors = []
        try:
            for index, backend in enumerate(report['measurement_policy']['order']):
                print(f'[{index+1}/{4*repeats}] {backend}: {len(cases)} cases × {len(seeds)} seeds', flush=True)
                process = launch_worker(directory, index, backend, args.timeout)
                report['workers'].append(process)
                for event in process['events']:
                    if event['type'] == 'trial':
                        report['records'].append({**event, 'execution_position': index,
                                                  'repetition': index//4, 'abba_slot': index%4, 'pid': process['pid']})
                    if event['type'] == 'fatal': infrastructure_errors.append(event['error'])
                if process['returncode'] or process['timed_out']:
                    infrastructure_errors.append(f'worker {index} failed or timed out (exit {process["returncode"]})')
                (directory/'progress.json').write_text(json.dumps(report, indent=2)+'\n')
                # Device/setup failure cannot be repaired by repeating equivalent processes.
                if any(e['type'] == 'fatal' for e in process['events']): break
        except KeyboardInterrupt:
            infrastructure_errors.append('benchmark interrupted')
        except Exception as exc:
            infrastructure_errors.append(f'orchestration error: {type(exc).__name__}: {exc}')
        report['gpu_after'] = gpu_snapshot()
        report['source_unchanged'] = source_inventory() == sources
        report['frozen_unchanged'] = frozen_inventory() == frozen
        if not report['source_unchanged']: infrastructure_errors.append('source/config changed during benchmark')
        if not report['frozen_unchanged']: infrastructure_errors.append('frozen benchmark changed during benchmark')
        report['summary'] = summarize(report['records'], cases, seeds, repeats, reliable_timing=args.mode == 'timing')
        report['summary']['failures'].extend(infrastructure_errors)
        report['summary']['passed'] = not report['summary']['failures']
        if not report['summary']['passed']:
            for group in report['summary']['groups']:
                group['speedup_ratio'] = None
                if group.get('timing_status') == 'supported_speedup':
                    group['timing_status'] = 'inconclusive'
                    group.setdefault('timing_reasons', []).append('Global correctness or infrastructure failure')
        write_report(directory, report)
        print(f'{"PASS" if report["summary"]["passed"] else "FAIL"}: {directory}/report.html', flush=True)
        return 0 if report['summary']['passed'] else 1
