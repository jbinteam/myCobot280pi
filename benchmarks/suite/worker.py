"""One isolated backend process. No robot/network/serial calls are allowed."""
import argparse
import copy
import functools
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid
from collections import defaultdict
from .metrics import digest, stable_result


def offline_guard():
    """Audit network connection and serial-device opens before optional imports."""
    def audit(event, args):
        if event in ('socket.connect', 'socket.connect_ex', 'socket.bind', 'socket.sendto'):
            raise RuntimeError('Offline benchmark forbids network IO: '+event)
        if event == 'open' and args and isinstance(args[0], (str, bytes)):
            name = os.fsdecode(args[0])
            if name.startswith(('/dev/tty', '/dev/serial', '/dev/cu.')):
                raise RuntimeError('Offline benchmark forbids serial IO')
    sys.addaudithook(audit)


class Probe:
    def __init__(self, policy):
        self.policy = policy
        self.patches = []
        self.stack = []
        self.times = defaultdict(float)
        self.calls = defaultdict(int)
        self.predictions = []

    def wrap(self, obj, name, label, sample=False):
        original = getattr(obj, name)
        @functools.wraps(original)
        def measured(*args, **kwargs):
            node = [0.]
            self.stack.append(node)
            start = time.perf_counter()
            try:
                result = original(*args, **kwargs)
            finally:
                elapsed = (time.perf_counter()-start)*1000
                self.stack.pop()
                self.times[label] += elapsed-node[0]
                self.calls[label] += 1
                if self.stack:
                    self.stack[-1][0] += elapsed
            if sample:
                self.predictions.append({'shape': list(result.shape), 'dtype': str(result.dtype),
                                         'sha256': hashlib.sha256(result.tobytes()).hexdigest()})
            return result
        setattr(obj, name, measured)
        self.patches.append((obj, name, original))

    def __enter__(self):
        from simulation.diffusion import env, smoothing
        self.wrap(self.policy, 'sample', 'inference', sample=True)
        self.wrap(env.ArmEnv, 'inverse_kinematics', 'ik')
        # Never wrap valid: valid_many checks that method's identity.
        for name in ('path_valid', 'valid_many', '_edge_valid'):
            self.wrap(env.ArmEnv, name, 'collision')
        self.wrap(smoothing, 'prepare', 'smoothing')
        return self

    def __exit__(self, *exc):
        for obj, name, original in reversed(self.patches):
            setattr(obj, name, original)


def profile_packets(case, request, result):
    if not case.get('encoder_profile') or not result.get('converged'):
        return None, None
    import numpy as np
    from simulation.diffusion import service
    from webui import hardware_diffusion, encoder_profile
    mapping = copy.deepcopy(case['encoder_mapping'])
    body = {k: request[k] for k in ('target_mm', 'target_axis', 'seed', 'samples',
            'inference_steps', 'max_decisions', 'speed', 'velocity_shape')}
    body['joint_command'] = 'encoder_drag_profile'
    # Reuse exactly the generated result; exercise production validation/retiming.
    # This substitution is confined to the isolated worker and always restored.
    original = service.plan
    def replay(replayed):
        expected = {k: v for k, v in request.items() if k not in ('speed', 'velocity_shape')}
        if replayed != expected:
            raise ValueError('Offline profile request does not match generated reference')
        return copy.deepcopy(result)
    began = time.perf_counter()
    service.plan = replay
    try:
        prepared = hardware_diffusion.prepare(request['start_deg'], body)
    finally:
        service.plan = original
    encoder_profile.prepare(prepared, mapping, body['speed'])
    curve = prepared['result']['smoothing']
    # Include cadence points, exact corner boundaries and uniform coverage.
    times = sorted(set([float(x) for x in np.arange(0, curve['duration_s'], .2)] +
                       [float(x) for x in np.linspace(0, curve['duration_s'], 101)] +
                       [s['start_time_s'] for s in curve['segments']] + [curve['duration_s']]))
    packets = [encoder_profile.packet(curve, t, mapping, body['speed']) for t in times]
    return {'prepared': digest(stable_result(prepared)), 'packets': digest(packets),
            'packet_count': len(packets), 'mapping': digest(mapping)}, (time.perf_counter()-began)*1000


def run_case(case, seed, backend, service, torch):
    import numpy as np
    request = copy.deepcopy(case['request'])
    request['seed'] = seed
    planner_request = {k: v for k, v in request.items() if k not in ('speed', 'velocity_shape')}
    policy = service._policy
    began = time.perf_counter()
    result = None
    caught = None
    with Probe(policy) as probe:
        try:
            if policy.device == 'cuda': torch.cuda.synchronize()
            began = time.perf_counter()
            result = service.plan(planner_request)
            if policy.device == 'cuda': torch.cuda.synchronize()
        except ValueError as exc:
            # Only explicitly listed input validation exceptions can pass a rejection gate.
            caught = {'type': type(exc).__name__, 'message': str(exc)}
            result = {'converged': False, 'status': 'ValueError', 'exception': caught}
        elapsed = (time.perf_counter()-began)*1000
    rng = hashlib.sha256(policy.generator.get_state().cpu().numpy().tobytes()).hexdigest()
    profiles, preparation_ms = profile_packets(case, request, result)
    frames = result.get('frames', [])
    q = np.deg2rad(np.asarray([f['q'] for f in frames])) if frames else np.empty((0, 6))
    replay = policy._replay
    metrics = {'frame_count': len(frames), 'sample_calls': len(probe.predictions),
               'endpoint_error_mm': result.get('final_error_mm'),
               'endpoint_direction_error_deg': result.get('final_orientation_error_deg'),
               'endpoint_measurement': 'URDF FK estimate; no external measurement',
               'decisions': result.get('decisions', 0), 'proposed_chunks': result.get('proposed_chunks', 0),
               'rejections': result.get('rejections', {}),
               'joint_route_length_rad': float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum()) if len(q)>1 else 0.,
               'duration_s': result.get('smoothing', {}).get('duration_s'),
               'max_velocity_rad_s': result.get('max_velocity_rad_s'),
               'max_acceleration_rad_s2': result.get('max_acceleration_rad_s2'),
               'clearance': 'not measured; existing collision checks retained'}
    return {'case_id': case['id'], 'seed': seed, 'backend': backend, 'request': request,
            'status': result['status'], 'converged': result.get('converged', False),
            'exception': caught, 'device': str(policy.device),
            'graph_active': bool(replay is not None and replay.graph is not None and not replay.disabled),
            'graph_requested': backend == 'cuda_graph',
            'timing_ms': {'total': elapsed, **{k: probe.times[k] for k in ('inference','ik','collision','smoothing')},
                          'offline_encoder_preparation': preparation_ms},
            'timing_note': 'Exclusive nested stage timers; collision omits scalar validity outside measured methods; total includes probes/prediction hashing. Encoder preparation separate.',
            'stage_calls': dict(probe.calls), 'metrics': metrics,
            'signatures': {'full_reference': digest(stable_result(result)),
                           'sampled_predictions': digest(probe.predictions), 'rng_after': rng,
                           'encoder_profile': profiles}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--backend', choices=('eager','cuda_graph'), required=True)
    args = parser.parse_args()
    offline_guard()
    os.environ['MYCOBOT_CUDA_GRAPH'] = '1' if args.backend == 'cuda_graph' else '0'
    spec = json.loads(Path(args.input).read_text())
    with Path(args.output).open('x') as out:
        def emit(value):
            value['monotonic_s'] = time.monotonic()
            out.write(json.dumps(value, allow_nan=False)+'\n'); out.flush()
        began = time.perf_counter()
        try:
            import torch
            from simulation.diffusion import service
            from webui.hardware_diffusion import verified_status
            from simulation.kinematics import ARM
            status = verified_status()
            if status['checkpoint_sha256'] != spec['expected_checkpoint_sha256'] or ARM.fingerprint != spec['expected_urdf_sha256']:
                raise RuntimeError('Scenario manifest checkpoint/URDF hashes differ from runtime')
            with service._lock:
                policy = service.load_policy(status, spec['seeds'][0])
            if policy.device == 'cuda': torch.cuda.synchronize()
            device_uuid = None
            if policy.device == 'cuda':
                device_uuid = getattr(torch.cuda.get_device_properties(policy.device), 'uuid', None)
                if isinstance(device_uuid, bytes):
                    device_uuid = uuid.UUID(bytes=device_uuid) if len(device_uuid) == 16 else device_uuid.decode('ascii')
                if device_uuid is not None: device_uuid = str(device_uuid)
            metadata = {'type': 'setup', 'backend': args.backend,
                        'cold_import_model_setup_ms': (time.perf_counter()-began)*1000,
                        'checkpoint_sha256': status['checkpoint_sha256'], 'model_name': status['model_name'],
                        'urdf_sha256': ARM.fingerprint,
                        'environment': status['environment'], 'torch': torch.__version__,
                        'cuda_runtime': torch.version.cuda, 'device': str(policy.device),
                        'torch_threads': torch.get_num_threads(),
                        'torch_interop_threads': torch.get_num_interop_threads(),
                        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                        'deployment_sha256': hashlib.sha256((Path(status['checkpoint_path']).parent.parent/'webui.json').read_bytes()).hexdigest(),
                        'gpu': torch.cuda.get_device_name() if policy.device == 'cuda' else None,
                        'gpu_uuid': device_uuid,
                        'thread_environment': {k: os.environ.get(k) for k in
                                               ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
                        'dependencies': {name: importlib.metadata.version(name) for name in
                                         ('numpy','scipy','torch','diffusers','pybullet','python-fcl')}}
            emit(metadata)
            if policy.device != 'cuda':
                emit({'type': 'fatal', 'error': 'CUDA unavailable: eager/graph comparison is non-comparable; no speedup measured'})
                return 2
            for case in spec['cases']:
                for seed in spec['seeds']:
                    emit({'type': 'started', 'case_id': case['id'], 'seed': seed})
                    try:
                        cold = run_case(case, seed, args.backend, service, torch)
                        emit({'type': 'warmup_complete', 'case_id': case['id'], 'seed': seed})
                        record = run_case(case, seed, args.backend, service, torch)
                        record.update(type='trial', warmup_ms=cold['timing_ms'],
                                      warmup_exact=record['signatures'] == cold['signatures'])
                        emit(record)
                    except Exception as exc:
                        emit({'type':'trial', 'case_id':case['id'], 'seed':seed,
                              'backend':args.backend, 'error':f'{type(exc).__name__}: {exc}',
                              'traceback':traceback.format_exc()})
        except Exception as exc:
            emit({'type': 'fatal', 'error': f'{type(exc).__name__}: {exc}', 'traceback':traceback.format_exc()})
            return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
