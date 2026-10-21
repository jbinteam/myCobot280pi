"""Generate independently replayed single-box motion demonstrations, offline only.

Four equally represented families: no block, diverse box, direct-blocked box,
and collision-boundary endpoints. Existing datasets/models are never overwritten.
Motion episode fields match train_motion's augmentation reader. Rejected poses
live separately under hard_negatives and are never trajectory action labels.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import multiprocessing
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np
from ..kinematics import ARM
from .common import file_sha256, write_json
from .env import ArmEnv
from .motion_conditioning import CONDITIONING
from .planner import RRTConfig, plan, smooth

KINDS = ('no_block', 'diverse_box', 'blocked_box', 'near_boundary')
REPLAY_DT = .01
EDGE_STEP = .01
DT = .1


def random_box(env, rng):
    """Broad table-mounted geometry within the existing service input bounds."""
    angle = rng.uniform(-np.pi, np.pi)
    radius = rng.uniform(.10, .32)
    half = rng.uniform(.008, .075, 2)
    height = rng.uniform(.045, .40)
    return np.r_[radius*np.cos(angle), radius*np.sin(angle),
                 env.table_top+height/2, half, height/2]


def boundary_pair(env, q, rng, attempts=24):
    """A valid pose and nearby collision-invalid witness, both within limits.

    Near means joint-space distance <= 0.02 rad, not measured spatial clearance.
    Keep a 0.005 rad inward retreat from the numerically located boundary.
    """
    for _ in range(attempts):
        direction = rng.normal(size=6)
        direction /= np.linalg.norm(direction)
        bad = q + .15*direction
        if np.any(bad < ARM.lower) or np.any(bad > ARM.upper) or env.valid(bad):
            continue
        lo, hi = 0., .15
        for _ in range(12):
            at = (lo+hi)/2
            if env.valid(q+at*direction): lo = at
            else: hi = at
        safe = q + max(0., lo-.005)*direction
        bad = q + hi*direction
        if env.valid(safe) and not env.valid(bad) and np.linalg.norm(bad-safe) <= .02:
            return safe, bad
    return None


def sample_task(env, rng, kind, attempts=300):
    if kind not in KINDS: raise ValueError('Unknown collision task kind')
    for _ in range(attempts):
        env.set_obstacle(None)
        start = rng.uniform(ARM.lower*.95, ARM.upper*.95)
        if not env.valid(start): continue
        goal = (ARM.clip(start+rng.normal(0, .35, 6)) if rng.random() < .7
                else rng.uniform(ARM.lower*.95, ARM.upper*.95))
        if not env.valid(goal) or np.linalg.norm(goal-start) < .2: continue
        if kind == 'blocked_box':
            mid = start+rng.uniform(.25, .75)*(goal-start)
            point = ARM.fk(mid)
            if np.linalg.norm(point[:2]) < .095 or point[2] < .025: continue
            half = rng.uniform(.008, .05, 2)
            top = min(.46, max(point[2]+rng.uniform(.01, .04), .06))
            height = top-env.table_top
            box = np.r_[point[:2]+rng.normal(0, .004, 2), env.table_top+height/2, half, height/2]
        else:
            box = None if kind == 'no_block' else random_box(env, rng)
        env.set_obstacle(box)
        if not env.valid(start) or not env.valid(goal): continue
        bad = None
        if kind == 'near_boundary':
            pair = boundary_pair(env, goal, rng)
            if pair is None: continue
            goal, bad = pair
            if np.linalg.norm(goal-start) < .2: continue
        blocked = not env.edge_valid(start, goal, EDGE_STEP)
        if kind == 'blocked_box' and not blocked: continue
        return start, goal, box, bad, blocked
    raise RuntimeError('Collision task sampling budget exhausted')


def timed_route(path, dt=DT, replay_dt=REPLAY_DT, speed=1., acceleration=3.):
    """Sample the SAME analytic quintics at learning and dense replay times.

    Leg durations are rounded once to the learning grid. Dense replay never
    retimes the route; original knots are included exactly in both grids.
    """
    path = np.asarray(path, float)
    if (path.ndim != 2 or path.shape[1] != 6 or len(path) < 2
            or not np.isfinite(path).all() or np.any(path < ARM.lower) or np.any(path > ARM.upper)):
        raise ValueError('Expected finite bounded [N,6] route')
    if not all(np.isfinite(x) and x > 0 for x in (dt, replay_dt, speed, acceleration)):
        raise ValueError('Positive timing settings required')
    if replay_dt > dt: raise ValueError('Replay grid must be at least as dense as learning grid')
    segments = []
    now = 0.
    for a, b in zip(path[:-1], path[1:]):
        distance = float(np.linalg.norm(b-a))
        if distance < 1e-12: continue
        minimum = max(1.875*distance/speed, np.sqrt(5.773503*distance/acceleration))*1.001
        duration = int(np.ceil(minimum/dt))*dt
        segments.append({'start_rad':a.tolist(), 'end_rad':b.tolist(),
                         'start_time_s':now, 'duration_s':duration})
        now += duration
    if not segments: raise ValueError('Stationary route is not a trajectory example')

    def sample(step):
        qs, vs, accs, ts = [path[0]], [np.zeros(6)], [np.zeros(6)], [0.]
        for segment in segments:
            a = np.array(segment['start_rad']); b = np.array(segment['end_rad'])
            duration = segment['duration_s']; delta = b-a
            pieces = max(1, int(np.ceil(duration/step-1e-10)))
            u = np.arange(1, pieces+1)/pieces
            q = a+(10*u**3-15*u**4+6*u**5)[:,None]*delta
            v = (30*u**2*(1-u)**2/duration)[:,None]*delta
            acc = (60*u*(1-u)*(1-2*u)/duration**2)[:,None]*delta
            q[-1] = b; v[-1] = 0.; acc[-1] = 0.
            qs.extend(q); vs.extend(v); accs.extend(acc)
            ts.extend(segment['start_time_s']+np.arange(1,pieces+1)*duration/pieces)
        return tuple(np.asarray(x) for x in (qs, vs, accs, ts))
    return sample(dt), sample(replay_dt), segments


def final_valid(env, sparse, dense, goal):
    q, v, a, times = sparse
    replay, _, _, replay_times = dense
    return (len(q) >= 2 and np.allclose(q[-1], goal, atol=1e-12, rtol=0)
            and np.all(np.diff(times) > 0) and np.all(np.diff(replay_times) > 0)
            and np.max(np.linalg.norm(v, axis=1)) <= 1.+1e-6
            and np.max(np.linalg.norm(a, axis=1)) <= 3.+1e-6
            and env.path_valid(q, EDGE_STEP) and env.path_valid(replay, EDGE_STEP))


def worker(job):
    output, kinds, seed, config = job
    rng = np.random.default_rng(seed); failures = Counter()
    with ArmEnv() as env, h5py.File(output, 'x') as data:
        root = data.create_group('episodes'); negatives = data.create_group('hard_negatives')
        for number, kind in enumerate(kinds):
            for attempt in range(100):
                try: start, goal, box, bad, blocked = sample_task(env, rng, kind)
                except RuntimeError: failures['sampling_'+kind] += 1; continue
                planner_seed = int(rng.integers(2**31))
                route, diagnostics = plan(env, start, goal, planner_seed, config)
                if route is None: failures['planner_'+kind] += 1; continue
                route = smooth(env, route, EDGE_STEP)
                sparse, dense, segments = timed_route(route)
                if not final_valid(env, sparse, dense, goal):
                    failures['final_replay_'+kind] += 1; continue
                q, v, a, times = sparse
                group = root.create_group(f'{number:06d}')
                for key, value in [('q_rad',q), ('time_s',times), ('goal_rad',np.repeat(goal[None],len(q),axis=0)),
                                   ('velocity_rad_s',v), ('acceleration_rad_s2',a),
                                   ('target_velocity_m_s',np.zeros((len(q),3))),
                                   ('replay_q_rad',dense[0]), ('replay_time_s',dense[3])]:
                    group.create_dataset(key,data=value,compression='gzip',compression_opts=1)
                group.create_dataset('obstacle_m',data=np.zeros(6) if box is None else box)
                group.attrs['obstacle_present'] = box is not None
                record = {'kind':kind, 'direct_blocked':blocked, 'planner':diagnostics,
                          'segments':segments, 'replay_dt_s':REPLAY_DT, 'edge_step_rad':EDGE_STEP}
                group.attrs['record_json'] = json.dumps(record)
                if bad is not None:
                    neg = negatives.create_group(f'{number:06d}')
                    neg.create_dataset('q_rad', data=bad)
                    neg.create_dataset('safe_q_rad', data=goal)
                    neg.attrs['episode'] = f'{number:06d}'
                    neg.attrs['label'] = 'collision_invalid_within_joint_limits'
                print(json.dumps({'shard':Path(output).name,'completed':number+1,'kind':kind,
                                  'frames':len(q),'replay_frames':len(dense[0])}), flush=True)
                break
            else: raise RuntimeError(f'Could not generate protected episode {number}/{kind}')
        data.attrs['failures_json'] = json.dumps(dict(failures))
        data.attrs['environment_json'] = json.dumps(env.metadata())
    return output


def audit(path):
    """Reopen in a fresh scene; validate every stored state and connecting edge."""
    counts = Counter(); splits = Counter(); signatures = {}; negatives_checked = 0
    with h5py.File(path,'r') as data, ArmEnv() as env:
        metadata = json.loads(data.attrs['metadata_json'])
        if metadata['environment'] != env.metadata() or metadata['conditioning'] != CONDITIONING:
            raise ValueError('Replay schema/scene mismatch')
        for name, group in data['episodes'].items():
            box = group['obstacle_m'][:] if group.attrs['obstacle_present'] else None
            env.set_obstacle(box)
            q = group['q_rad'][:]; v = group['velocity_rad_s'][:]; a = group['acceleration_rad_s2'][:]
            times = group['time_s'][:]; goal = group['goal_rad'][:]
            replay = group['replay_q_rad'][:]; replay_times = group['replay_time_s'][:]
            if (q.ndim != 2 or q.shape[1] != 6 or v.shape != q.shape or a.shape != q.shape
                    or goal.shape != q.shape or times.shape != (len(q),)
                    or group['target_velocity_m_s'].shape != (len(q),3)
                    or replay_times.shape != (len(replay),)
                    or not np.isfinite(np.r_[q.ravel(),v.ravel(),a.ravel(),goal.ravel(),times,replay.ravel(),replay_times]).all()
                    or not np.allclose(np.diff(times), DT, atol=1e-9, rtol=0)
                    or np.max(np.diff(replay_times)) > REPLAY_DT+1e-9
                    or not np.array_equal(goal,np.repeat(q[-1][None],len(q),axis=0))
                    or not np.array_equal(group['target_velocity_m_s'][:],np.zeros((len(q),3)))):
                raise ValueError('Protected episode labels/timing invalid: '+name)
            record = json.loads(group.attrs['record_json'])
            route = np.array([record['segments'][0]['start_rad'],*[s['end_rad'] for s in record['segments']]])
            expected_sparse, expected_dense, expected_segments = timed_route(route)
            if record['segments'] != expected_segments or any(not np.allclose(x,y,atol=1e-12,rtol=0)
                    for x,y in zip((q,v,a,times),expected_sparse)) or any(not np.allclose(x,y,atol=1e-12,rtol=0)
                    for x,y in zip((replay,replay_times),(expected_dense[0],expected_dense[3]))):
                raise ValueError('Replay no longer matches the same analytic curve: '+name)
            if not final_valid(env,(q,v,a,times),(replay,None,None,replay_times),q[-1]):
                raise ValueError('Independent collision replay failed: '+name)
            actual_blocked = not env.edge_valid(q[0],q[-1],EDGE_STEP)
            if actual_blocked != record['direct_blocked'] or (record['kind']=='blocked_box' and not actual_blocked):
                raise ValueError('Blocked-route label incorrect')
            if record['kind'] not in KINDS or group.attrs['obstacle_present'] != (record['kind']!='no_block'):
                raise ValueError('Task kind/scene inconsistent')
            split = group.attrs['split']
            if split not in ('train','validation','test'): raise ValueError('Invalid split')
            identity = hashlib.sha256(np.r_[q[0],q[-1],np.zeros(6) if box is None else box].tobytes()).hexdigest()
            if identity in signatures: raise ValueError('Duplicate task across protected dataset')
            signatures[identity] = split
            counts[record['kind']] += 1; counts['learning_states'] += len(q); counts['replay_states'] += len(replay)
            splits[split] += 1
        for group in data['hard_negatives'].values():
            episode = data['episodes'][group.attrs['episode']]
            env.set_obstacle(episode['obstacle_m'][:] if episode.attrs['obstacle_present'] else None)
            bad, safe = group['q_rad'][:], group['safe_q_rad'][:]
            if (bad.shape != (6,) or not np.isfinite(bad).all() or np.any(bad<ARM.lower)
                    or np.any(bad>ARM.upper) or env.valid(bad) or not env.valid(safe)
                    or not np.array_equal(safe,episode['q_rad'][-1]) or np.linalg.norm(bad-safe) > .02):
                raise ValueError('Invalid hard-negative collision evidence')
            negatives_checked += 1
        if metadata['completed_episodes'] != len(data['episodes']): raise ValueError('Incomplete protected dataset')
        if counts['near_boundary'] != negatives_checked: raise ValueError('Missing collision-boundary evidence')
    return {'passed':True,'episodes':sum(splits.values()),'counts':dict(counts),'split_counts':dict(splits),
            'hard_negatives_checked':negatives_checked,'fresh_scene_replay':True,
            'checks':'every analytic 100 Hz pose + connecting joint edges at 0.01 rad; bounds/self/table/optional box',
            'limits':'Finite sampled kinematic checks; one box; no physical dynamics or added tool geometry'}


def generate(output, episodes=1000, workers=4, seed=202610081, iterations=1200):
    if any(isinstance(n,bool) or not isinstance(n,int) or n<1 for n in (episodes,workers,iterations)):
        raise ValueError('Positive integer episode/worker/planner budgets required')
    if episodes < 40 or episodes % 4: raise ValueError('Use at least 40 episodes, divisible by four (balanced families)')
    output = Path(output)
    folder = output.parent/(output.stem+'_shards')
    temporary = output.with_suffix('.partial.h5')
    if output.exists() or temporary.exists() or folder.exists() or output.with_suffix('.json').exists():
        raise ValueError('Choose a fresh output path; existing artifacts are preserved')
    output.parent.mkdir(parents=True,exist_ok=True); folder.mkdir()
    began = perf_counter(); workers = min(workers,episodes)
    config = RRTConfig(iterations=iterations,post_solution=20,max_step_rad=EDGE_STEP)
    kinds = [KINDS[i%4] for i in range(episodes)]
    jobs = [(str(folder/f'{i:03d}.h5'),kinds[i::workers],seed+i,config) for i in range(workers)]
    if workers == 1: worker(jobs[0])
    else:
        with ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            list(pool.map(worker,jobs))
    failures = Counter(); strata = {}; scene = None; number = 0
    with h5py.File(temporary,'x') as data:
        root = data.create_group('episodes'); negatives = data.create_group('hard_negatives')
        for job in jobs:
            with h5py.File(job[0],'r') as shard:
                failures.update(json.loads(shard.attrs['failures_json']))
                current_scene = json.loads(shard.attrs['environment_json'])
                if scene is not None and scene != current_scene: raise ValueError('Shard scene changed')
                scene = current_scene
                for name, group in shard['episodes'].items():
                    target = f'{number:06d}'; shard.copy(group,root,name=target)
                    kind = json.loads(group.attrs['record_json'])['kind']
                    strata.setdefault(kind,[]).append(target)
                    if name in shard['hard_negatives']:
                        shard.copy(shard['hard_negatives'][name],negatives,name=target)
                        negatives[target].attrs['episode'] = target
                    number += 1
        rng = np.random.default_rng(seed); splits = Counter()
        for kind, names in strata.items():
            rng.shuffle(names); train_end = int(.8*len(names)); valid_end = train_end+max(1,int(.1*len(names)))
            for i, name in enumerate(names):
                split = 'train' if i<train_end else 'validation' if i<valid_end else 'test'
                root[name].attrs['split'] = split; splits[split] += 1
        metadata = {'kind':'collision_protected_motion_demonstrations_v1','completed_episodes':number,
                    'seed':seed,'environment':scene,'conditioning':CONDITIONING,'planner':asdict(config),
                    'dt':DT,'replay_dt':REPLAY_DT,'edge_step_rad':EDGE_STEP,'clearance_mm':1.,
                    'task_counts':{k:len(v) for k,v in strata.items()},'split_counts':dict(splits),'failures':dict(failures),
                    'split':'episode-level 80/10/10 stratified by family; duplicate tasks rejected',
                    'training_reader':'train_motion augmentation input; existing [34] one-box conditioning',
                    'hard_negatives':'separate root; excluded from positive action windows',
                    'source_sha256':{p.name:file_sha256(p) for p in (Path(__file__),Path(__file__).with_name('planner.py'),Path(__file__).with_name('env.py'))},
                    'safety':'Offline only; no hardware/robot IO; final publication requires independent replay',
                    'seconds_before_audit':perf_counter()-began}
        data.attrs['metadata_json'] = json.dumps(metadata)
    report = audit(temporary)
    metadata.update(audit=report,seconds=perf_counter()-began)
    with h5py.File(temporary,'r+') as data: data.attrs['metadata_json'] = json.dumps(metadata)
    temporary.replace(output)
    metadata['sha256'] = file_sha256(output); write_json(output.with_suffix('.json'),metadata)
    print(json.dumps(metadata),flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('simulation/artifacts/collision_protected_v1/demonstrations.h5'))
    parser.add_argument('--episodes',type=int,default=1000); parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--seed',type=int,default=202610081); parser.add_argument('--iterations',type=int,default=1200)
    parser.add_argument('--audit',type=Path,help='Read-only independent replay of a generated dataset')
    args = parser.parse_args()
    if args.audit: print(json.dumps(audit(args.audit)))
    else: generate(args.output,args.episodes,args.workers,args.seed,args.iterations)

if __name__ == '__main__': main()
