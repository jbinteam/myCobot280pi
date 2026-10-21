"""Pure simulation API. This module has no robot transport or SDK imports."""
import json
from pathlib import Path
import threading
from time import perf_counter
import numpy as np
from .kinematics import ARM
from .solvers import ARTIFACTS, METHODS, cases, models, solve
from . import pose5, seed5
from .motion_limits import retime_result, profile, PREVIEW_SPEED

_PRM_CACHE = {}
_PRM_CACHE_LOCK = threading.RLock()


def load_roadmap(path, space):
    """Validate a persisted graph once; invalidate on file/profile changes.

    Every returned motion is still checked at query/trajectory resolution.
    Collision-space construction re-hashes model assets before this lookup.
    """
    from .prm import Roadmap
    path = Path(path).resolve(); stat = path.stat()
    profile = json.dumps(space.metadata(), sort_keys=True)
    key = (str(path), stat.st_mtime_ns, stat.st_size, profile)
    with _PRM_CACHE_LOCK:
        if key not in _PRM_CACHE:
            road = Roadmap.load(path, space)
            _PRM_CACHE.clear()  # Bounded memory; one current offline roadmap.
            _PRM_CACHE[key] = road
        return _PRM_CACHE[key]


def plan(body, roadmap=None, goal_constraint=None):
    """Dry-run PRM query, using the same target-offset and frame conventions.

    Roadmaps are built offline with simulation.prm. This endpoint does not
    prepare an executable hardware plan or accept arbitrary cache paths.
    """
    from .planning import IKConfig, query
    from .prm import planning_space
    task = body.get('task', 'position')
    start = np.deg2rad(np.asarray(body.get('start_deg'), float))
    base = np.asarray(body.get('target_mm'), float)
    offset = np.asarray(body.get('eef_offset_mm', [0, 0, 0]), float)
    if base.shape != (3,) or offset.shape != (3,) or not np.isfinite(base).all() or not np.isfinite(offset).all():
        return {'status': 'invalid_target', 'converged': False, 'simulation_only': True, 'reason': 'Expected finite target XYZ and local offset'}
    config = IKConfig(task=task, nn=body.get('nn', 'seed5' if task == 'pose5' else 'seed'),
                      pose5_method=body.get('pose5_method', 'bounded_dls'),
                      pose5_recovery_methods=body.get('pose5_recovery_methods', ()),
                      refine=body.get('refine', True), numerical_fallback=body.get('numerical_fallback', True),
                      random_seeds=body.get('random_seeds', 4), roadmap_seeds=body.get('roadmap_seeds', 4),
                      max_candidates=body.get('max_candidates', 4), seed=body.get('seed', 9001),
                      pose5_global_recovery=body.get('pose5_global_recovery', False),
                      recovery_seeds=body.get('recovery_seeds', 32),
                      recovery_evaluations=body.get('recovery_evaluations', 96),
                      recovery_max_candidates=body.get('recovery_max_candidates', 8),
                      guided_route_repair=body.get('guided_route_repair', False),
                      route_repair_samples=body.get('route_repair_samples', 32),
                      steps=body.get('steps', 240), speed_rad_s=body.get('speed_rad_s', 1.2),
                      tolerance_mm=body.get('tolerance_mm', 1.), damping=body.get('damping', .04),
                      orientation_tolerance_deg=body.get('orientation_tolerance_deg', 1.))
    if roadmap is None:
        space = planning_space(body.get('collision', 'self'), body.get('self_clearance_mm', 1.))
        roadmap = load_roadmap(ARTIFACTS/'roadmap.json', space)
    if not roadmap.space.valid(start):
        return query(start, base/1000, roadmap, config, body.get('orientation_xy_deg'),target_axis=body.get('target_axis'),goal_constraint=goal_constraint)
    rotation = ARM.transform(start)[:3, :3]
    result = query(start, (base+rotation@offset)/1000, roadmap, config,
                   body.get('orientation_xy_deg'), shortcut=body.get('shortcut', False),target_axis=body.get('target_axis'),goal_constraint=goal_constraint)
    result=retime_result(result, config.speed_rad_s)
    return {**result, 'eef_rotation': rotation.tolist(), 'base_target_mm': base.tolist(), 'eef_offset_mm': offset.tolist()}


def mouse_aim_plan(body, roadmap=None, *, space=None):
    """Try checked local IK from current joints, then the full 5D IK/PRM path.

    Cursor XYZ is in g_base millimeters. Tool Z is the joint6_flange local Z
    expressed in g_base. Freeze the captured start while computing this plan;
    Local iterates become motion only after complete path validation and checked
    simplification. Fallback refinement iterates are not motion. Spin is free.
    """
    engine = body.get('tracking_engine', 'classical')
    if engine not in ('classical', 'local_diffusion', 'learned_rrt'):
        raise ValueError('Unknown mouse tracking engine')
    if engine in ('local_diffusion', 'learned_rrt'):
        if roadmap is not None or space is not None:
            raise ValueError('Diffusion mouse tracking requires its isolated simulation scene')
        if engine == 'learned_rrt':
            from .diffusion.classical_tracking import plan as tracking_plan
        else:
            from .diffusion.tracking import plan as tracking_plan
        return tracking_plan(body)
    began=perf_counter()
    def numbers(value, count, name):
        if not isinstance(value, (list, tuple)) or len(value) != count or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) for v in value):
            raise ValueError(f'Expected {count} finite numeric {name}')
        return np.asarray(value, float)
    start = numbers(body.get('start_deg'), 6, 'starting joint angles in degrees')
    cursor = numbers(body.get('cursor_mm'), 3, 'cursor coordinates in g_base millimeters')
    if start.shape != (6,) or not np.isfinite(start).all():
        raise ValueError('Expected six finite starting joint angles in degrees')
    if cursor.shape != (3,) or not np.isfinite(cursor).all():
        raise ValueError('Expected three finite cursor coordinates in g_base millimeters')
    q = np.deg2rad(start)
    if np.any(q < ARM.lower) or np.any(q > ARM.upper):
        raise ValueError('Starting joints are outside the operating limits')
    delta = cursor - ARM.transform(q)[:3, 3]*1000
    if np.linalg.norm(delta) < 1e-6:
        raise ValueError('Cursor must be separated from the flange')
    axis = pose5.unit_axis(delta)
    from .continuous import initial_motion, prepare as continuous_prepare, frames_for
    continuous,velocity,acceleration = initial_motion(body,min(float(body.get('speed_rad_s',1.2)),1.))
    if continuous and 'target_axis' in body:
        axis=pose5.unit_axis(numbers(body['target_axis'],3,'target axis'))
    position_tol = float(body.get('tolerance_mm', 1.))
    angle_tol = float(body.get('orientation_tolerance_deg', 1.))
    if not np.isfinite(position_tol) or not .1 <= position_tol <= 10:
        raise ValueError('Position tolerance must be 0.1–10 mm')
    if not np.isfinite(angle_tol) or not .1 <= angle_tol <= 10:
        raise ValueError('Direction tolerance must be 0.1–10 degrees')
    def cursor_metrics(q):
        endpoint = ARM.transform(q)
        to_cursor = cursor-endpoint[:3, 3]*1000
        distance = float(np.linalg.norm(to_cursor))
        angular = pose5.axis_error(endpoint[:3, 2], pose5.unit_axis(to_cursor))[2] if distance > 1e-6 else 180.
        return {'accepted': abs(distance-50) <= position_tol and angular <= angle_tol,
                'distance_mm': distance, 'gap_error_mm': abs(distance-50), 'direction_error_deg': angular}
    # Test the actual cursor geometry instead of secretly tightening the UI's
    # tolerances. Apply this at raw acceptance, shortcuts, refinement termination
    # and goal admission, so an unsuitable goal cannot consume the goal budget.
    accept_cursor = lambda q: cursor_metrics(q)['accepted']
    request = {key: body[key] for key in ('steps', 'speed_rad_s', 'damping',
                                        'guided_route_repair', 'route_repair_samples') if key in body}
    request.update(start_deg=start.tolist(), target_mm=(cursor-50*axis).tolist(),
                   target_axis=axis.tolist(), eef_offset_mm=[0, 0, 0], task='pose5', nn='seed5',
                   pose5_method=body.get('pose5_method', 'dls'), pose5_recovery_methods=('dls',), refine=True, numerical_fallback=True,
                   random_seeds=4, roadmap_seeds=4, max_candidates=4, seed=9001,
                   pose5_global_recovery=True, recovery_seeds=32, recovery_evaluations=96,
                   recovery_max_candidates=8,
                   collision='self', self_clearance_mm=1., shortcut=True,
                   tolerance_mm=position_tol, orientation_tolerance_deg=angle_tol)
    if continuous:request['speed_rad_s']=min(float(request.get('speed_rad_s',1.2)),1.)
    from .planning import IKConfig
    from .prm import planning_space
    from .tracking import local_pose_plan
    space = roadmap.space if roadmap is not None else (space or planning_space('self', 1.))
    if space.validation != 'joint_limits_and_self_collision':
        raise ValueError('Mouse aim requires a self-collision-validated planning space')
    # Validate all settings before the fast path, including fallback-only options.
    config = IKConfig(task='pose5', nn='none', refine=False, numerical_fallback=False,
                      random_seeds=0, roadmap_seeds=0, max_candidates=1,
                      pose5_method=request['pose5_method'], steps=request.get('steps', 240),
                      speed_rad_s=request.get('speed_rad_s', 1.2), damping=request.get('damping', .04),
                      tolerance_mm=position_tol, orientation_tolerance_deg=angle_tol,
                      guided_route_repair=request.get('guided_route_repair', False),
                      route_repair_samples=request.get('route_repair_samples', 32))
    target = np.asarray(request['target_mm'])/1000
    result = local_pose_plan(q, target, axis, space, config, accept_cursor)
    local_attempt = result['local_ik']
    if result['converged']:
        result.update(tracking_source='local_ik', eef_rotation=ARM.transform(q)[:3, :3].tolist(),
                      base_target_mm=request['target_mm'], eef_offset_mm=[0, 0, 0],
                      guided_repair={'enabled': config.guided_route_repair, 'triggered': False, 'recovered': False})
    else:
        # Load the persisted graph only when the local attempt cannot supply a
        # fully accepted path. Keep all existing NN/recovery/repair behavior.
        if roadmap is None:
            roadmap = load_roadmap(ARTIFACTS/'roadmap.json', space)
        result = plan(request, roadmap, goal_constraint=accept_cursor)
        result.update(tracking_source='global_fallback', local_ik=local_attempt)
    result.update(mouse_aim=True, cursor_mm=cursor.tolist(), gap_mm=50.,
                  tracking_strategy='local_first',
                  cursor_tolerance_mm=position_tol, cursor_direction_tolerance_deg=angle_tol,
                  coordinate_frame='g_base', tool_frame=ARM.tcp,
                  speed_rad_s=request.get('speed_rad_s', 1.2))
    if result.get('converged'):
        result['cursor_acceptance'] = cursor_metrics(np.deg2rad(result['frames'][-1]['q']))
        if not result['cursor_acceptance']['accepted']:
            result.update(converged=False, frames=[], status='mouse_aim_goal_tolerance_not_met',
                          reason='Validated IK endpoint misses the requested cursor gap/direction tolerances',
                          stop_reason='cursor_tolerance_not_met')
    if continuous or 'velocity_shape' in body:
        result['continuous_motion']=continuous
        if result.get('converged'):
            path=np.asarray(result.get('waypoints_rad',np.deg2rad([f['q'] for f in result['frames']])),float)
            try:
                if 'velocity_shape' in body:
                    from .profile_tracking import prepare as profile_prepare
                    states,times,smoothing=profile_prepare(space,path,velocity,acceleration,min(request['speed_rad_s'],1.),velocity_shape=body['velocity_shape'])
                else:
                    states,times,smoothing=continuous_prepare(space,path,velocity,acceleration,request['speed_rad_s'])
                result.update(frames=frames_for(states,times,target,axis),smoothing=smoothing,
                    dt=.02,acceleration_rad_s2=3.,max_velocity_rad_s=smoothing['max_velocity_rad_s'],
                    max_acceleration_rad_s2=smoothing['max_acceleration_rad_s2'])
            except ValueError as exc:
                result.update(converged=False,frames=[],status='continuous_join_rejected',reason=str(exc))
    if not continuous and result.get('frames') and not result.get('smoothing'):
        result=retime_result(result,request.get('speed_rad_s',1.2))
    result['server_compute_ms']=(perf_counter()-began)*1000
    return result


def info():
    from .camera_model import configured_camera
    from .diffusion.service import info as diffusion_info
    from .diffusion.classical_service import info as classical_info
    ready=(ARTIFACTS/'models.json').exists()
    if ready:models()  # Validate the URDF fingerprint and warm inference outside rollouts.
    training=json.loads((ARTIFACTS/'training.json').read_text()) if (ARTIFACTS/'training.json').exists() else None
    benchmark=json.loads((ARTIFACTS/'benchmark.json').read_text()) if (ARTIFACTS/'benchmark.json').exists() else None
    benchmark_pose5=json.loads((ARTIFACTS/'benchmark_pose5.json').read_text()) if (ARTIFACTS/'benchmark_pose5.json').exists() else None
    goal=ARM.transform(np.deg2rad([20,-35,-90,40,10,0]))
    demo_start = [1.,-40.,-130.,-40.,1.,50.]
    camera = configured_camera()
    optical = camera.optical_transform(ARM.transform(np.deg2rad(demo_start)))
    tracking_demo = {'start_deg':demo_start, 'distance_mm':350., 'half_angle_deg':30.,
                     'cursor_mm':(1000*(optical[:3,3]+.350*optical[:3,2])).tolist()}
    return {'default_task':'pose5','task':'XYZ + flange Z direction; local Z spin free','benchmark_pose5':benchmark_pose5,
            'seed5_checkpoint':seed5.checkpoint_status(),'diffusion':diffusion_info(),'classical':classical_info(),'default_target_axis':goal[:3,2].tolist(),
            'pose5_methods':[{'id':key,'name':name,'description':description,'neural':False} for key,name,description in pose5.METHODS],'tcp':ARM.tcp,'models_ready':ready,'training':training,'benchmark':benchmark,
            'methods':[{'id':key,'name':name,'description':description,'neural':key in ('direct','conditional','differential','null_neural','seed_dls')} for key,name,description in METHODS],
            'default_start_deg':[0,-40,-130,80,0,50],
            'tracking_demo':tracking_demo, 'tracking_camera':camera.configuration(),
            'default_target_mm':(ARM.fk(np.deg2rad([20,-35,-90,40,10,0]))*1000).tolist(),
            'default_orientation_xy_deg':pose5.xy_from_axis(goal[:3,2]),'cases':pose5.cases(),'limits_deg':{'min':np.round(np.rad2deg(ARM.lower),1).tolist(),'max':np.round(np.rad2deg(ARM.upper),1).tolist()},
            'motion_specification':profile(),'max_preview_speed_rad_s':PREVIEW_SPEED,'joint_limit_profile':ARM.limit_profile,'urdf_sha256':ARM.fingerprint,'defaults':{'speed_rad_s':1.2,'dt':.05,'steps':240,'tolerance_mm':1.,'damping':.04,'orientation_tolerance_deg':1.}}


def run(body):
    start=np.deg2rad(np.asarray(body.get('start_deg'),dtype=float))
    base=np.asarray(body.get('target_mm'),dtype=float)
    offset=np.asarray(body.get('eef_offset_mm',[0,0,0]),dtype=float)
    if base.shape!=(3,) or offset.shape!=(3,) or not np.isfinite(base).all() or not np.isfinite(offset).all():
        raise ValueError('Expected three finite target coordinates and three finite EEF local offsets')
    # Use the full starting EEF orientation, including unconstrained wrist spin.
    # Offsets are vectors: rotate them without adding the flange translation.
    rotation=ARM.transform(start)[:3,:3]
    target=(base+rotation@offset)/1000
    task_mode=body.get('task','position')
    if task_mode not in ('position','pose5'):raise ValueError('Task must be position or pose5')
    requested=body.get('methods',['dls' if task_mode=='pose5' else 'seed_dls'])
    if not isinstance(requested,list) or not 1<=len(requested)<=len(METHODS) or len(set(requested))!=len(requested):raise ValueError('Select one or more distinct IK methods')
    config={'steps':body.get('steps',240),'speed':body.get('speed_rad_s',1.2),'tolerance':body.get('tolerance_mm',1.),'damping':body.get('damping',.04)}
    results=[(pose5.solve(start,target,body.get('orientation_xy_deg'),method,orientation_tolerance_deg=body.get('orientation_tolerance_deg',1.),target_axis=body.get('target_axis'),**config) if task_mode=='pose5' else solve(start,target,method,**config)) for method in requested]
    results=[retime_result(result,config['speed']) for result in results]
    for result in results:result.setdefault('task',task_mode)
    return {'results':results,'start_deg':np.rad2deg(start).tolist(),'target_mm':(target*1000).tolist(),
            'base_target_mm':base.tolist(),'eef_offset_mm':offset.tolist(),'eef_rotation':rotation.tolist(),
            'simulation_only':True,'task':task_mode,'dt':.05}
