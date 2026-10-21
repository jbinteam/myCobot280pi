"""Single-decision, local-first diffusion mouse tracking. No hardware imports.

The next checked references are smoothed and retimed before playback. A chunk does not imply
that the cursor goal has been reached. All local and neural paths use one scene.
"""
from collections import Counter
from time import perf_counter
import numpy as np
from ..kinematics import ARM
from .. import pose5
from ..planning import IKConfig, checked_goal
from ..tracking import local_pose_plan
from ..tracking_geometry import region_for, region_metrics, candidate_poses, gaze_score, annotate_gaze
from ..physical_reach import ENVELOPE, target_precheck
from . import service

_env = None  # Used only while holding service._lock; one bounded scene cache.


class SceneSpace:
    validation = 'joint_limits_self_table_block'

    def __init__(self, env): self.env = env
    def valid(self, q): return self.env.valid(q)
    def edge_valid(self, a, b, step): return self.env.edge_valid(a, b, step)
    def delta(self, a, b): return self.env.space.delta(a, b)
    def distance(self, a, b): return self.env.space.distance(a, b)
    def metadata(self): return {'assumptions': self.env.metadata()}
    def diagnostic(self, q): return {'reason': 'Joint limits, self, table or block clearance violation'}


def cursor_metrics(q, cursor, position_tol, angle_tol, region=None):
    if region is not None:
        return region_metrics(q, region, position_tol, angle_tol)
    transform = ARM.transform(q); delta = cursor-transform[:3,3]*1000
    distance = float(np.linalg.norm(delta))
    angle = pose5.axis_error(transform[:3,2], delta/distance)[2] if distance > 1e-6 else 180.
    return {'accepted': abs(distance-50) <= position_tol and angle <= angle_tol,
            'distance_mm': distance, 'gap_error_mm': abs(distance-50), 'direction_error_deg': angle}


def plan(body):
    global _env
    began = perf_counter()
    if 'obstacle_mm' not in body:raise ValueError('Include obstacle_mm as six numbers or null for no block')
    cursor = service.vector(body.get('cursor_mm'),3,'cursor_mm')
    start_deg = service.vector(body.get('start_deg'),6,'start_deg')
    start = np.deg2rad(start_deg)
    axis = pose5.unit_axis(service.vector(body['target_axis'],3,'target_axis')) if 'target_axis' in body else pose5.unit_axis(cursor-ARM.fk(start)*1000)
    region = region_for(cursor, axis, body.get("tracking_region"))
    target = (cursor-(region["distance_mm"] if region else 50)*axis)/1000
    # Reuse strict scene/config validation, without accepting an arbitrary goal.
    start,target,axis,box,_,sampling = service.request({
        'start_deg':start_deg.tolist(),'target_mm':(target*1000).tolist(),
        'target_axis':axis.tolist(),'obstacle_mm':body.get('obstacle_mm'),
        **{k:body[k] for k in ('seed','samples','inference_steps') if k in body}})
    config = IKConfig(task='pose5',nn='none',refine=False,numerical_fallback=False,
        random_seeds=0,roadmap_seeds=0,max_candidates=1,
        pose5_method=body.get('pose5_method','bounded_dls'),steps=body.get('steps',240),
        speed_rad_s=body.get('speed_rad_s',1.),damping=body.get('damping',.04),
        tolerance_mm=body.get('tolerance_mm',1.),
        orientation_tolerance_deg=body.get('orientation_tolerance_deg',1.))
    speed = min(config.speed_rad_s,1.)
    from ..continuous import initial_motion, prepare as continuous_prepare
    continuous,velocity,acceleration = initial_motion(body,speed)
    from ..reference_shape import settings as shape_settings
    velocity_shape=shape_settings(body["velocity_shape"]) if "velocity_shape" in body else None
    target_velocity=service.vector(body.get('target_velocity_m_s',[0.,0.,0.]),3,'target_velocity_m_s')
    from .motion_conditioning import bounded_vector
    target_velocity=bounded_vector(target_velocity,3,.5,'target velocity',clip=True)
    accept = lambda q: cursor_metrics(q,cursor,config.tolerance_mm,config.orientation_tolerance_deg,region)['accepted']
    if not service._lock.acquire(blocking=False):
        raise service.PlannerBusy('Diffusion planner busy; retry the latest mouse target')
    try:
        status = service.info()
        if status['status'] != 'ready': raise RuntimeError(status.get('reason','Diffusion unavailable'))
        from .cartesian_conditioning import supports_conditioning
        cartesian=supports_conditioning(status.get('conditioning'))
        hint=None
        if 'goal_hint_deg' in body and not cartesian:
            hint = np.deg2rad(service.vector(body['goal_hint_deg'],6,'goal_hint_deg'))
            if np.any(hint<ARM.lower) or np.any(hint>ARM.upper):raise ValueError('Goal hint outside operating limits')
        from .env import ArmEnv
        if _env is None or _env.closed:
            _env = ArmEnv()
        env = _env
        if env.metadata() != status['environment']:
            env.close(); _env = None
            raise RuntimeError('Mouse tracking scene differs from diffusion training')
        if box is not None and abs(box[2]-box[5]-env.table_top) > 1e-6:
            raise ValueError('Block must rest on the diffusion table (top −35 mm)')
        if (env.obstacle is None)!=(box is None) or (box is not None and not np.array_equal(env.obstacle,box)):
            env.set_obstacle(box)
        env.reset(start); space = SceneSpace(env)
        result = {'method':'local_diffusion','task':'pose5','simulation_only':True,
            'mouse_aim':True,'tracking_strategy':'local_then_diffusion',
            'tracking_chunk':True,'playable':False,'converged':False,'frames':[],
            'validation':space.validation,'coordinate_frame':'g_base','tool_frame':ARM.tcp,
            'gap_mm':region['distance_mm'] if region else 50.,'cursor_mm':cursor.tolist(),'start_deg':start_deg.tolist(),
            'target_mm':(target*1000).tolist(),'target_axis':axis.tolist(),
            'obstacle_mm':None if box is None else (box*1000).tolist(),'speed_rad_s':speed,
            'scene_mode':'no_block' if box is None else 'block',
            'collision_checks':['joint_limits','self','table']+([] if box is None else ['block']),
            'conditioning_note':status['no_block_note'] if box is None else '',
            'cursor_tolerance_mm':config.tolerance_mm,'cursor_direction_tolerance_deg':config.orientation_tolerance_deg,
            'checkpoint_sha256':status['checkpoint_sha256'], 'configuration':sampling,
            'replan_required':False,'rejections':{},'proposed_chunks':0,
            'limitations':'Kinematic simulation only. '
                         'Final references use checked quintics, speed ≤1 rad/s and acceleration ≤3 rad/s². '
                         'Sampled convex collision checks; preview interruptions may stop early.'}
        result['continuous_motion']=continuous
        if region is not None:
            result['tracking_region'] = region
            result['tracking_objective'] = 'gaze_first'
        if cartesian:
            result['motion_conditioning']={'schema':status['conditioning']['schema'],'observation_dim':34,
                'target_kind':'cartesian_pose5','target_velocity_m_s':target_velocity.tolist()}
        def finish(reason, text=None):
            result.update(status=reason,reason=text or reason.replace('_',' '),
                          server_compute_ms=(perf_counter()-began)*1000)
            return result
        if not env.valid(start): return finish('invalid_start','Starting joints collide with self, table or block; pose held')
        result['physical_reach_precheck'] = target_precheck(target, region, config.tolerance_mm)
        result['region_candidates_filtered'] = 0
        result['region_candidates_tested'] = 0
        if not result['physical_reach_precheck']['possible']:
            return finish('outside_workspace', 'EEF goal region is outside the URDF physical reach bound; pose held')
        start_error=service.errors(start,target,axis)
        static_hold=(np.linalg.norm(target_velocity)<=1e-6 and accept(start)
            and (region is not None or (cartesian and start_error['error_mm']<=config.tolerance_mm
            and start_error['orientation_error_deg']<=config.orientation_tolerance_deg)))
        if static_hold and region is not None:
            target = ARM.fk(start)
            axis = pose5.unit_axis(cursor-target*1000)
            start_error = service.errors(start,target,axis)
        if static_hold:
            local={'converged':True,'local_ik':{'status':'not_required'},'ik':None,
                'waypoints_rad':[start.copy()],
                'frames':[{'q':start_deg.tolist(),'time':0.,'position':(ARM.fk(start)*1000).tolist(),
                    'tool_axis':ARM.transform(start)[:3,2].tolist(),**start_error}]}
        else:
            candidates = list(candidate_poses(region, start))[:8] if region else [(target, axis)]
            solved = None
            local = {'converged':False, 'local_ik':{'status':'solver_failed'}}
            # Select a checked local route within the disk. If all local routes
            # fail, condition the neural fallback on the nearest disk point.
            fallback = None
            for candidate_target, candidate_axis in candidates:
                if not ENVELOPE.point(candidate_target*1000, config.tolerance_mm)['possible']:
                    result['region_candidates_filtered'] += 1
                    continue
                result['region_candidates_tested'] += 1
                try:
                    candidate_solved = pose5.solve(start,candidate_target,
                        method=config.pose5_method,target_axis=candidate_axis,
                        steps=min(config.steps,240),speed=speed,damping=config.damping,
                        tolerance=config.tolerance_mm,orientation_tolerance_deg=config.orientation_tolerance_deg,
                        goal_constraint=accept)
                    candidate_local = local_pose_plan(start,candidate_target,candidate_axis,
                        space,config,accept,solved=candidate_solved)
                except RuntimeError as exc:
                    if not cartesian: raise
                    candidate_solved = None
                    candidate_local = {'converged':False,'local_ik':{'status':'solver_failed','reason':str(exc)}}
                if fallback is None:
                    fallback = (candidate_target, candidate_axis, candidate_solved, candidate_local)
                if candidate_local['converged']:
                    target, axis, solved, local = candidate_target, candidate_axis, candidate_solved, candidate_local
                    break
            if not local['converged']:
                if fallback is None:
                    return finish('ik_failed', 'No sampled disk endpoint passed the physical reach precheck; the disk is not proven unreachable')
                target, axis, solved, local = fallback
        result.update(target_mm=(target*1000).tolist(), target_axis=axis.tolist())
        result['local_ik'] = local['local_ik']
        goal = None
        if local['converged']:
            result.update(tracking_source='diffusion_fallback' if static_hold else 'local_ik',
                          ik_outcome='not_required' if static_hold else 'accepted_goals',
                          planning_outcome='checked_cartesian_hold' if static_hold else 'checked_local_prefix',ik=local['ik'])
            goal = np.deg2rad(local['frames'][-1]['q'])
            frames = local['frames'] if continuous else [f for f in local['frames'] if f['time'] <= .2+1e-10]
        else:
            # A local solver may find a valid endpoint through a blocked path.
            # Only legacy schemas use an IK endpoint for conditioning.
            if not cartesian:
                candidate = np.deg2rad(solved['frames'][-1]['q'])
                if checked_goal(candidate,target,space,config,axis) is not None and accept(candidate):
                    goal = candidate
                elif hint is not None:
                    # A previous endpoint can preserve the selected IK branch.
                    if checked_goal(hint,target,space,config,axis) is not None and accept(hint):goal=hint
                    else:
                        refined=pose5.solve(hint,target,method='bounded_dls',target_axis=axis,
                            steps=min(config.steps,240),speed=speed,damping=config.damping,
                            tolerance=config.tolerance_mm,orientation_tolerance_deg=config.orientation_tolerance_deg,
                            goal_constraint=accept)
                        candidate=np.deg2rad(refined['frames'][-1]['q'])
                        if checked_goal(candidate,target,space,config,axis) is not None and accept(candidate):goal=candidate
                if goal is None:
                    try: goal = env.inverse_kinematics(target,axis,current=start,seed=sampling['seed'],attempts=3)
                    except RuntimeError: return finish('ik_failed','No scene-valid cursor goal found within the bounded IK search')
                if checked_goal(goal,target,space,config,axis) is None or not accept(goal):
                    return finish('goal_rejected','IK endpoint misses the cursor gap or direction tolerance')
            policy = service.load_policy(status,sampling['seed'])
            from .policy import checked_chunk
            denoise_at = perf_counter()
            condition=service.condition(policy,start,goal,target,axis,box,velocity,acceleration,target_velocity)
            chunks = policy.sample(condition[None],sampling['samples'],sampling['inference_steps'])[0]
            result['motion_conditioning']={'schema':getattr(policy,'conditioning',None)['schema'] if getattr(policy,'conditioning',None) else 1,
                'target_kind':'cartesian_pose5' if cartesian else 'joint_goal',
                'observation_dim':policy.observation_dim,
                'target_velocity_m_s':target_velocity.tolist()}
            result['denoise_ms'] = (perf_counter()-denoise_at)*1000
            candidates = []; rejects = Counter()
            for raw in chunks:
                chunk, reason = checked_chunk(env,start,raw,dt=.1,speed=speed)
                if chunk is None: rejects[reason] += 1; continue
                captured=False
                if cartesian:
                    chunk,captured=service.static_capture_prefix(chunk,target,axis,target_velocity,
                        config.tolerance_mm,config.orientation_tolerance_deg,accept=accept)
                score = gaze_score(chunk, region) if region is not None else service.candidate_score(policy,chunk,goal,target,axis)
                candidates.append((0 if captured else 1,score,chunk,captured))
            result.update(tracking_source='diffusion_fallback',ik_outcome='not_required' if cartesian else 'accepted_goals',
                          planning_outcome='checked_diffusion_chunk',rejections=dict(rejects),proposed_chunks=len(chunks))
            if not candidates: return finish('no_safe_chunk','All diffusion proposals rejected; pose held')
            _,_,selected,captured=min(candidates,key=lambda item:item[:2])
            if cartesian:result['static_cartesian_capture']=captured
            if not continuous and policy.observation_dim==34:
                # The learned first two time samples are near rest. For the
                # optional resting-chunk mode, take a bounded geometric prefix
                # of the checked horizon and retime that prefix from rest.
                from ..tracking import simplify_path
                route=simplify_path(np.vstack((start,selected)),space)
                delta=(route[1]-start) if len(route)>1 else np.zeros(6)
                delta*=min(1.,speed*.2/max(np.linalg.norm(delta),1e-12))
                points=np.vstack((start,start+delta*.5,start+delta))
            else:points = np.vstack((start,selected if continuous else selected[:2]))
            frames = [{'q':np.rad2deg(q).tolist(),'time':i*.1,
                'position':(ARM.fk(q)*1000).tolist(),'tool_axis':ARM.transform(q)[:3,2].tolist(),
                **service.errors(q,target,axis)} for i,q in enumerate(points)]
        from .smoothing import prepare_tracking,TRACKING_DURATION_LIMIT
        reference_steps=len(frames)-1
        try:
            if continuous:
                path=np.asarray(local['waypoints_rad']) if local['converged'] else np.deg2rad([f['q'] for f in frames])
                if velocity_shape is None:
                    states,times,smoothing=continuous_prepare(space,path,velocity,acceleration,speed)
                else:
                    from ..profile_tracking import prepare as profile_prepare
                    states,times,smoothing=profile_prepare(env,path,velocity,acceleration,speed,velocity_shape=velocity_shape)
            else:
                if velocity_shape is None:
                    states,times,smoothing=prepare_tracking(env,np.deg2rad([f['q'] for f in frames]),speed=speed)
                else:
                    from ..profile_tracking import prepare_tracking as profile_prepare_tracking
                    states,times,smoothing=profile_prepare_tracking(env,np.deg2rad([f['q'] for f in frames]),speed=speed,velocity_shape=velocity_shape)
        except ValueError as exc:return finish('smoothing_validation_failed',str(exc))
        if not continuous and times[-1]>TRACKING_DURATION_LIMIT+1e-9:
            return finish('smoothing_budget_exceeded','Smoothed tracking chunk exceeds the bounded playback horizon; pose held')
        frames=[{'q':np.rad2deg(q).tolist(),'time':float(time),'position':(ARM.fk(q)*1000).tolist(),
                 'tool_axis':ARM.transform(q)[:3,2].tolist(),**service.errors(q,target,axis)} for q,time in zip(states,times)]
        if not env.path_valid(states) or (len(states)>1 and np.any(
                np.linalg.norm(np.diff(states,axis=0)/np.diff(times)[:,None],axis=1)>speed+1e-9)):
            return finish('final_validation_failed','Returned tracking prefix failed scene or speed checks')
        acceptance = cursor_metrics(states[-1],cursor,config.tolerance_mm,config.orientation_tolerance_deg,region)
        result.update(playable=True,frames=frames,converged=acceptance['accepted'],
            smoothing=smoothing,dt=smoothing['dt'],acceleration_rad_s2=3.,reference_steps=reference_steps,
            max_velocity_rad_s=smoothing['max_velocity_rad_s'],max_acceleration_rad_s2=smoothing['max_acceleration_rad_s2'],
            replan_required=not acceptance['accepted'],cursor_acceptance=acceptance,
            goal_cursor_acceptance=None if cartesian else cursor_metrics(goal,cursor,config.tolerance_mm,config.orientation_tolerance_deg,region),
            goal_deg=None if cartesian else np.rad2deg(goal).tolist(),**{f'final_{k}':v for k,v in service.errors(states[-1],target,axis).items()})
        result['gaze_tracking'] = annotate_gaze(frames, cursor, region)
        return finish('target_reached' if result['converged'] else 'chunk_ready')
    finally:
        service._lock.release()
