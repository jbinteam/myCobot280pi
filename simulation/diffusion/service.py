"""Simulation-only WebUI adapter; bounded, serialized, checked DDPM rollouts.

Status does not import Torch or PyBullet. Inference runs lazily in an isolated
scene and only returns playback frames after a complete successful validation.
"""
from collections import Counter
from importlib.util import find_spec
import json
import os
import threading
from time import perf_counter
import numpy as np
from ..kinematics import ARM
from ..pose5 import axis_error, unit_axis
from .common import ARTIFACTS, file_sha256, observation, robot_schema

CHECKPOINT = ARTIFACTS/'model/diffusion_policy_6dof.pth'
MANIFEST = ARTIFACTS/'webui.json'
ACTIVE = ARTIFACTS.parent/'diffusion_active.json'
_lock = threading.Lock()
_policy = None


class PlannerBusy(RuntimeError):
    pass


def load_policy(status, seed):
    """Caller must own _lock: scheduler and generator are mutable."""
    global _policy
    from .policy import DiffusionPolicy
    if _policy is None or _policy.sha256 != status['checkpoint_sha256']:
        _policy = DiffusionPolicy(status.get('checkpoint_path',CHECKPOINT),
            cuda_graph=status.get('model_name')=='diffusion_motion_v2' and os.environ.get('MYCOBOT_CUDA_GRAPH','1')!='0',
            scheduler_cache=os.environ.get('MYCOBOT_SCHEDULER_CACHE','1')!='0')
    if _policy.sha256 != status['checkpoint_sha256']:
        raise RuntimeError('Diffusion checkpoint changed during loading')
    if _policy.training['dataset_sha256'] != status['dataset_sha256']:
        raise RuntimeError('Diffusion model training provenance differs from deployment manifest')
    if _policy.observation_dim != status.get('observation_dim',19):
        raise RuntimeError('Diffusion conditioning differs from deployment manifest')
    if getattr(_policy,'conditioning',None) != status.get('conditioning'):
        raise RuntimeError('Diffusion conditioning schema differs from deployment manifest')
    _policy.generator.manual_seed(seed)
    return _policy


def info():
    result = {'status':'unavailable', 'simulation_only':True, 'cases':[]}
    try:
        manifest_path=MANIFEST;checkpoint=CHECKPOINT
        if ACTIVE.exists():
            active=json.loads(ACTIVE.read_text())
            manifest_path=(ARTIFACTS.parent/active['manifest']).resolve()
            if not manifest_path.is_relative_to(ARTIFACTS.parent.resolve()) or file_sha256(manifest_path)!=active['manifest_sha256']:
                raise ValueError('Active diffusion manifest is outside artifacts or changed')
            checkpoint=manifest_path.parent/'model/diffusion_policy_6dof.pth'
        manifest = json.loads(manifest_path.read_text())
        if manifest['schema'] != 1 or manifest['environment']['robot'] != robot_schema():
            raise ValueError('Diffusion deployment URDF/geometry/schema mismatch')
        from .motion_conditioning import CONDITIONING
        from .cartesian_conditioning import supports_conditioning
        conditioning=manifest.get('conditioning')
        if conditioning is not None and conditioning!=CONDITIONING and not supports_conditioning(conditioning):raise ValueError('Unsupported deployed conditioning')
        digest = file_sha256(checkpoint)
        if digest != manifest['checkpoint_sha256']:
            raise ValueError('Diffusion checkpoint differs from the deployed version')
        missing = [name for name in ('torch','diffusers','pybullet','fcl') if find_spec(name) is None]
        if missing: raise ValueError('Install simulation/diffusion/requirements.txt; missing '+', '.join(missing))
        result.update(status='ready', checkpoint_sha256=digest,
                      checkpoint_path=str(checkpoint),observation_dim=19 if conditioning is None else 34,
                      conditioning=conditioning,model_name=manifest.get('model_name','diffusion_v1'),
                      dataset_episodes=manifest.get('dataset_episodes',1000),
                      dataset_sha256=manifest['dataset_sha256'],
                      cases=manifest['cases'], environment=manifest['environment'],
                      obstacle_modes=['block','no_block'],
                      no_block_note='' if manifest.get('no_block_trained') else 'Presence mask supports an absent block; deployed checkpoint was trained with blocks.',
                      evaluation=manifest['held_out_evaluation'],
                      defaults={'seed':42,'samples':4,'inference_steps':32,'max_decisions':200,
                                'dt':.1,'speed_rad_s':1.,'horizon':16,'execute_steps':None if conditioning else 2,
                                'execute_seconds':.4 if conditioning else .2,
                                'rollout_controller':'full_horizon_c2' if conditioning else 'two_reference_speed_projection',
                                'playback_dt':.02,'acceleration_rad_s2':3.})
    except (OSError, ValueError, KeyError) as exc:
        result['reason'] = str(exc)
    return result


def vector(value, size, name):
    if not isinstance(value, list) or len(value) != size or any(
            isinstance(v,bool) or not isinstance(v,(int,float)) for v in value):
        raise ValueError(f'{name} must contain {size} finite numbers')
    result = np.asarray(value,float)
    if not np.isfinite(result).all(): raise ValueError(f'{name} must be finite')
    return result


def integer(body, name, default, low, high):
    value = body.get(name,default)
    if isinstance(value,bool) or not isinstance(value,int) or not low <= value <= high:
        raise ValueError(f'{name} must be an integer between {low} and {high}')
    return value


def request(body,ignore_goal=False):
    start = np.deg2rad(vector(body.get('start_deg'),6,'start_deg'))
    if np.any(start < ARM.lower) or np.any(start > ARM.upper): raise ValueError('Starting joints outside operating limits')
    target = vector(body.get('target_mm'),3,'target_mm')/1000
    if np.any(np.abs(target) > .6): raise ValueError('Target coordinates must be within ±600 mm')
    axis = unit_axis(vector(body.get('target_axis'),3,'target_axis'))
    if 'obstacle_mm' not in body:raise ValueError('Include obstacle_mm as six numbers or null for no block')
    box = None if body['obstacle_mm'] is None else vector(body['obstacle_mm'],6,'obstacle_mm')/1000
    # Same table-mounted axis-aligned block family as training; finite UI bounds.
    if box is not None and (np.any(np.abs(box[:2]) > .4) or np.any(box[3:] < .005) or np.any(box[3:] > .25)):
        raise ValueError('Block center X/Y must be within ±400 mm and half-sizes 5..250 mm')
    goal = None
    if 'goal_deg' in body and not ignore_goal:
        goal = np.deg2rad(vector(body['goal_deg'],6,'goal_deg'))
        if np.any(goal < ARM.lower) or np.any(goal > ARM.upper): raise ValueError('Goal joints outside operating limits')
        transform = ARM.transform(goal)
        if np.linalg.norm(transform[:3,3]-target) > 1e-6 or axis_error(transform[:3,2],axis)[2] > .001:
            raise ValueError('Selected goal joints do not match the Cartesian target/direction')
    config = {'seed':integer(body,'seed',42,0,2**31-1),
              'samples':integer(body,'samples',4,1,8),
              'inference_steps':integer(body,'inference_steps',32,8,100),
              'max_decisions':integer(body,'max_decisions',200,1,200)}
    return start,target,axis,box,goal,config


def errors(q,target,axis,*,transform=None):
    if transform is None:transform = ARM.transform(q)
    return {'error_mm':float(np.linalg.norm(transform[:3,3]-target)*1000),
            'orientation_error_deg':float(axis_error(transform[:3,2],axis)[2])}


def condition(policy,q,goal,target,axis,box,velocity=None,acceleration=None,target_velocity=None):
    """Select semantic conditioning by schema, never by the shared 34 width."""
    from .cartesian_conditioning import supports,observe_for_policy
    if supports(policy):
        return observe_for_policy(policy,q,target,axis,box,velocity,acceleration,target_velocity)
    from .motion_conditioning import encode
    return encode(policy,observation(q,goal,box),velocity,acceleration,target_velocity)


def cartesian_score(q,target,axis):
    """Directed flange axis and Cartesian distance; no endpoint joint solution."""
    transform=ARM.transform(q)
    position=np.linalg.norm(transform[:3,3]-target)/ARM.scale
    angle=np.deg2rad(axis_error(transform[:3,2],axis)[2])
    return float(position+.35*angle)


def candidate_score(policy,chunk,goal,target,axis):
    from .cartesian_conditioning import supports
    if supports(policy):
        return cartesian_score(chunk[-1],target,axis)+.25*cartesian_score(chunk[min(1,len(chunk)-1)],target,axis)
    return float(np.linalg.norm((chunk[1]-goal)/ARM.radius)+.25*np.linalg.norm((chunk[-1]-goal)/ARM.radius))


def static_capture_prefix(chunk,target,axis,target_velocity=None,position_tol_mm=1.,angle_tol_deg=1.,accept=None):
    """Stop at an already checked neural pose that reaches a stationary target.

    Call only after checking the complete raw proposal and projected chunk.
    This selects an existing neural reference; it never synthesizes an IK goal.
    Moving targets retain the full horizon and ordinary candidate ranking.
    An optional accept(q) predicate adds task-specific checks such as cursor gap
    and pointing direction; it cannot replace the strict Cartesian pose checks.
    """
    if target_velocity is not None and np.linalg.norm(target_velocity)>1e-6:
        return chunk,False
    for index,q in enumerate(chunk):
        error=errors(q,target,axis)
        if (error['error_mm']<=position_tol_mm and error['orientation_error_deg']<=angle_tol_deg
                and (accept is None or accept(q))):
            return chunk[:index+1],True
    return chunk,False


def rollout(env,policy,start,goal,target,axis,box,config,deadline):
    """No repair planner: select checked chunks, advance two references, repeat."""
    from .policy import checked_chunk
    q = start.copy(); points = [q.copy()]; rejects = Counter(); stalls = 0
    velocity=np.zeros(6);acceleration=np.zeros(6)
    decisions = proposals = capture = 0; status = 'decision_limit'
    for _ in range(config['max_decisions']):
        error = errors(q,target,axis)
        if error['error_mm'] <= 1. and error['orientation_error_deg'] <= 1.:
            status = 'success'; break
        if perf_counter() >= deadline: status = 'time_limit'; break
        from .cartesian_conditioning import supports
        if not supports(policy) and goal is not None and np.linalg.norm(goal-q) <= .1 and env.execute_step(q,goal,.1,1.):
            q = goal.copy(); points.append(q); capture += 1; status = 'success'; break
        from .motion_conditioning import bounded_vector
        chunks = policy.sample(condition(policy,q,goal,target,axis,box,velocity,acceleration)[None],config['samples'],config['inference_steps'])[0]
        proposals += len(chunks); decisions += 1; candidates = []
        for raw in chunks:
            chunk,reason = checked_chunk(env,q,raw)
            if chunk is None: rejects[reason] += 1; continue
            score = candidate_score(policy,chunk,goal,target,axis)
            candidates.append((score,chunk))
        if not candidates:
            velocity=np.zeros(6);acceleration=np.zeros(6)
            stalls += 1
            if stalls >= 20: status = 'no_safe_chunk'; break
            continue
        stalls = 0
        for step in min(candidates,key=lambda item:item[0])[1][:2]:
            if not env.execute_step(q,step,.1,1.):
                return points,'execution_gate_rejected',decisions,proposals,dict(rejects),capture
            previous=q;q = env.current(); points.append(q.copy())
            next_velocity=(q-previous)/.1
            acceleration=bounded_vector((next_velocity-velocity)/.1,6,3.,'virtual reference acceleration',clip=True)
            velocity=next_velocity
    error = errors(q,target,axis)
    if error['error_mm'] <= 1. and error['orientation_error_deg'] <= 1.: status = 'success'
    return points,status,decisions,proposals,dict(rejects),capture


def plan(body):
    global _policy
    if not _lock.acquire(blocking=False): raise PlannerBusy('Diffusion planner busy; wait for the current preview')
    began = perf_counter()
    try:
        status = info()
        if status['status'] != 'ready': raise RuntimeError(status.get('reason','Diffusion checkpoint unavailable'))
        from .cartesian_conditioning import supports_conditioning,supports
        start,target,axis,box,goal,config = request(body,ignore_goal=supports_conditioning(status.get('conditioning')))
        from .env import ArmEnv
        load_policy(status, config['seed'])
        cartesian=supports(_policy)
        with ArmEnv() as env:
            if env.metadata() != status['environment']: raise RuntimeError('Diffusion scene/checker differs from training')
            if box is not None and abs(box[2]-box[5]-env.table_top) > 1e-6:
                raise ValueError('Block must rest on the diffusion table (top −35 mm)')
            env.set_obstacle(box); env.reset(start)
            result = {'method':'diffusion', 'task':'pose5', 'simulation_only':True,
                      'validation':'joint_limits_self_table_block', 'converged':False,'frames':[],
                      'start_deg':np.rad2deg(start).tolist(),'target_mm':(target*1000).tolist(),
                      'target_axis':axis.tolist(),'obstacle_mm':None if box is None else (box*1000).tolist(),
                      'scene_mode':'no_block' if box is None else 'block',
                      'collision_checks':['joint_limits','self','table']+([] if box is None else ['block']),
                      'conditioning_note':status['no_block_note'] if box is None else '',
                      'motion_conditioning':{'schema':getattr(_policy,'conditioning',None)['schema'] if getattr(_policy,'conditioning',None) else 1,
                          'observation_dim':_policy.observation_dim,
                          'target_kind':'cartesian_pose5' if cartesian else 'joint_goal'},
                      'checkpoint_sha256':_policy.sha256,'configuration':config,
                      'environment':status['environment'], 'dt':.1,'speed_rad_s':1.,
                      'limitations':'Kinematic references; 1 mm convex clearance and 0.04 rad path sampling. '
                                    'Final playback uses checked quintics with vector speed ≤1 rad/s and acceleration ≤3 rad/s². '
                                    'Reference limits only; simulation only.'}
            if not env.valid(start):
                result.update(status='invalid_start',reason='Starting pose collides with itself, the table or the block')
                return result
            if goal is None and not cartesian:
                from .endpoint_ik import IKSearchExhausted
                try: goal = env.inverse_kinematics(target,axis,current=start,seed=config['seed'],attempts=32,robust=True)
                except IKSearchExhausted:
                    result.update(status='ik_failed',reason='No collision-free IK endpoint found within the search budget',
                                  ik_diagnostics=env.ik_diagnostics,elapsed_ms=(perf_counter()-began)*1000)
                    return result
                if getattr(env,'ik_diagnostics',None) is not None:
                    result['ik_diagnostics']=env.ik_diagnostics
            if not cartesian and not env.valid(goal):
                result.update(status='invalid_goal',reason='Goal pose collides with itself, the table or the block')
                return result
            goal_error=None if cartesian else errors(goal,target,axis)
            if goal_error is not None and (goal_error['error_mm']>1. or goal_error['orientation_error_deg']>1.):
                result.update(status='final_validation_failed',reason='IK endpoint misses the requested Cartesian pose')
                return result
            # Motion-conditioned policies learn physically small first steps
            # from rest. Advance the checked full horizon through C2 timing.
            from .continuous_rollout import rollout as continuous_rollout
            runner=continuous_rollout if _policy.observation_dim==34 else rollout
            points,reason,decisions,proposals,rejections,capture = runner(
                env,_policy,start,goal,target,axis,box,config,began+90.)
            result.update(status=reason,reason=reason.replace('_',' '),decisions=decisions,
                          proposed_chunks=proposals,rejections=rejections,endpoint_capture_steps=capture,
                          goal_deg=None if cartesian else np.rad2deg(goal).tolist(),elapsed_ms=(perf_counter()-began)*1000,
                          rollout_controller='full_horizon_c2' if _policy.observation_dim==34 else 'two_reference_speed_projection',
                          candidate_validation='ranked_until_accepted' if _policy.observation_dim==34 and not cartesian else 'all_proposals',
                          iterations=len(points)-1,**{f'final_{k}':v for k,v in errors(points[-1],target,axis).items()})
            if reason != 'success': return result
            if (result['final_error_mm'] > 1. or result['final_orientation_error_deg'] > 1.
                    or not env.path_valid(points)
                    or np.any(np.linalg.norm(np.diff(points,axis=0),axis=1) > .1+1e-10)):
                result.update(status='final_validation_failed',reason='Complete path rejected by final validation')
                return result
            from .smoothing import prepare
            try:points,times,smoothing=prepare(env,points)
            except ValueError as exc:
                result.update(status='smoothing_validation_failed',reason=str(exc))
                return result
            frames = []
            from ..trajectory_math import frame_transforms
            for q,time,transform in zip(points,times,frame_transforms(points)):
                frames.append({'q':np.rad2deg(q).tolist(),'time':float(time),
                               'position':(transform[:3,3]*1000).tolist(),'tool_axis':transform[:3,2].tolist(),
                               **errors(q,target,axis,transform=transform)})
            result.update(converged=True,frames=frames,stop_reason='checked_smoothed_diffusion_rollout',
                          smoothing=smoothing,dt=smoothing['dt'],acceleration_rad_s2=3.,
                          rollout_reference_steps=result['iterations'],iterations=len(frames)-1,
                          elapsed_ms=(perf_counter()-began)*1000,
                          joint_travel_deg=float(np.rad2deg(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())),
                          max_velocity_rad_s=smoothing['max_velocity_rad_s'],
                          max_acceleration_rad_s2=smoothing['max_acceleration_rad_s2'])
            return result
    finally:
        _lock.release()
