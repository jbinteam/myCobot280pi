"""Candidate-limited cone-region planning for the measured-feedback controller.

Every incumbent includes a checked timed curve with the requested q/v/a join.
A prefix can end before the goal, so selection uses complete geometric route
length, never the short prefix's length as a bound on reaching that goal.
"""
from dataclasses import asdict
from time import perf_counter
import numpy as np
from ..kinematics import ARM
from ..camera_model import CameraModel, configured_camera
from .. import pose5, continuous, profile_tracking
from ..tracking_geometry import region_metrics, annotate_gaze
from . import service
from .endpoint_selection import DeadlineEnv, SelectionBudget
from .streaming_ik import EndpointSource, CandidateBudget, choose_streaming_route
from .visibility import VisibilityEnv
from .planner import RRTConfig
from .smoothing import prepare_tracking


def timed_route(env, path, velocity, acceleration, speed, continuous_motion, shape):
    if continuous_motion:
        if shape is None:
            return continuous.prepare(env, path, velocity, acceleration, speed)
        try:
            return profile_tracking.prepare(env, path, velocity, acceleration, speed, velocity_shape=shape)
        except ValueError:
            if np.linalg.norm(velocity) >= 1e-12 or np.linalg.norm(acceleration) >= 1e-12:
                raise
            result = profile_tracking.prepare_tracking(env, path, speed, velocity_shape=shape,
                                                       duration_limit=continuous.DURATION_LIMIT)
            if len(path)>1 and np.linalg.norm(result[0][-1]-path[0])<=1e-12:
                raise ValueError('No checked shared-profile startup progress')
            return result
    if shape is None:
        return prepare_tracking(env, path, speed=speed)
    return profile_tracking.prepare_tracking(env, path, speed, velocity_shape=shape)


def plan_region(collision, start, cursor, region, seed, poses, speed, velocity, acceleration,
                continuous_motion, shape, tolerance, angle_tolerance, route_planner,
                candidate_validator=None, camera=None, orientation_mode='point_at_object', cancel_check=None):
    camera = configured_camera() if camera is None else camera
    if not isinstance(camera, CameraModel):
        raise ValueError('Region visibility requires a checked camera model')
    started = perf_counter(); budget = SelectionBudget(); candidate_budget = CandidateBudget()
    accepted = lambda q: region_metrics(q, region, tolerance, angle_tolerance)['accepted']
    refined_count = 0
    def refine(q, job, check):
        nonlocal refined_count
        target, axis = job['target'], job['axis']
        error = service.errors(q, target, axis)
        if (not accepted(q) or error['error_mm']>min(1.,tolerance)
                or error['orientation_error_deg']>min(1.,angle_tolerance)):
            refined_count += 1
            solved = pose5.solve(q, target, target_axis=axis, method='bounded_dls', steps=240,
                speed=max(.1,speed), damping=.04, tolerance=min(.25,tolerance/2),
                orientation_tolerance_deg=min(.25,angle_tolerance/2), cancel_check=check,
                goal_constraint=lambda x: collision.valid(x) and accepted(x))
            q = np.deg2rad(solved['frames'][-1]['q'])
        error = service.errors(q, target, axis)
        return q if (accepted(q) and collision.valid(q) and error['error_mm']<=min(1.,tolerance)
                     and error['orientation_error_deg']<=min(1.,angle_tolerance)) else None

    routes = []
    with VisibilityEnv(cursor/1000, camera=camera, cancel_check=cancel_check) as view:
        view.set_obstacle(collision.obstacle); view.reset(start)
        if view.metadata()!=collision.metadata():
            raise RuntimeError('Visibility scene differs from the collision scene')
        visibility_model = view.visibility_metadata()
        if not view.valid(start):
            return {'status':'object_not_visible','reason':'Measured start fails modeled object visibility; pose held',
                    'visibility_model':visibility_model,'visibility_start':view.inspect(start)}
        def metrics(q):
            if orientation_mode == 'point_at_object':
                return region_metrics(q,region,tolerance,angle_tolerance)
            result = region_metrics(q,region,tolerance,180.)
            projection = view.inspect(q)
            result['accepted'] = result['tracking_accepted'] = bool(result['accepted'] and view.valid(q))
            result.update(orientation_mode='camera_fov',camera_projection=projection)
            return result
        accepted = lambda q: metrics(q)['accepted']
        solver = None
        if orientation_mode == 'camera_fov':
            from .camera_endpoint_ik import search_camera_endpoint
            solver = lambda target,axis,**settings: search_camera_endpoint(view,target,axis,**settings)
        source = EndpointSource(view if solver else collision,start,region,seed,poses=poses,
            position_tolerance_mm=tolerance,orientation_tolerance_deg=angle_tolerance,
            refine=refine if solver is None else None,solver=solver,acceptance=accepted,cancel_check=cancel_check)
        def evaluate(goal, cap, deadline):
            env = view if deadline is None else DeadlineEnv(view,deadline)
            q = np.asarray(goal['q_rad'])
            if cap:
                path, diagnostics = route_planner(env,start,q,seed=seed,
                    config=RRTConfig(iterations=cap,max_step_rad=.02))
            else:
                path = np.stack([start,q]) if env.edge_valid(start,q,.02) else None
                diagnostics = {'status':'direct_optimal' if path is not None else 'shared_iteration_budget_exhausted',
                               'iterations':0}
            diagnostics = {**diagnostics,'iterations':diagnostics.get('iterations',0)}
            row = {'goal_index':goal['index'],'accepted':False,'planner':diagnostics,
                   'status':diagnostics['status']}
            routes.append(row)
            trajectory = None
            if path is not None:
                try:
                    if not env.path_valid(path,.005):
                        raise ValueError('Full route fails dense scene/visibility checks')
                    states,times,smoothing = timed_route(env,path,velocity,acceleration,speed,continuous_motion,shape)
                    if not np.array_equal(states[0],start) or not env.path_valid(states,.005):
                        raise ValueError('Timed curve fails start/scene/visibility validation')
                    audit = view.audit(states,.005)
                    if not audit['passed']:
                        raise ValueError('Timed curve loses modeled object visibility')
                    if candidate_validator is not None:
                        candidate_validator(states, times, smoothing, deadline, orientation_mode)
                    row.update(accepted=True,visibility=audit,
                        joint_travel_rad=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()),
                        executed_prefix_travel_rad=float(np.linalg.norm(np.diff(states,axis=0),axis=1).sum()))
                    trajectory = (states,times,smoothing,q,path)
                except ValueError as exc:
                    row['rejection'] = str(exc)
            return {'route':row,'trajectory':trajectory}

        if accepted(start):
            # Holding an accepted pose needs no endpoint sampling. Moving joins
            # still require a checked braking curve; they may fail and hold.
            chosen = evaluate({'index':-1,'q_rad':start.tolist()},budget.first_goal_iterations,None)
            chosen = chosen if chosen['route']['accepted'] else None
            stats = {'stop_reason':'already_in_region','configuration':asdict(budget),
                     'candidate_configuration':asdict(candidate_budget)}
        else:
            chosen, stats = choose_streaming_route(start,source,evaluate,budget,candidate_budget=candidate_budget)
            stats.pop('first_trajectory')
        report = source.report()
        diagnostics = {'endpoint_selection':stats,'endpoint_source':report,
            'endpoint_selection_cost':'full_geometric_route_joint_travel',
            'visibility_model':visibility_model,'region_route_attempts':routes,
            'region_candidates_tested':len({a['point_index'] for a in report['attempts']}),
            'region_candidate_jobs_tested':report['jobs_started'],
            'region_candidates_filtered':report['points_filtered'],
            'ik_elapsed_ms':1000*report['ik_seconds'],'ik_diagnostics':dict(getattr(source.env,'ik_diagnostics',{})),
            'cursor_refinement':bool(refined_count), 'cursor_refinement_attempts':refined_count}
        if chosen is None:
            return {**diagnostics,'status':'no_path' if routes else 'ik_failed',
                    'reason':'No visible timed continuation within the bounded candidate search; pose held'}
        states,times,smoothing,goal,path = chosen['trajectory']
        target = ARM.fk(goal); axis = pose5.unit_axis(cursor-target*1000)
        final = metrics(states[-1])
        frames = continuous.frames_for(states,times,target,axis)
        gaze = annotate_gaze(frames,cursor,region)
        if gaze is not None and orientation_mode=='camera_fov':
            gaze['free_rotation']='orientation within configured camera FOV'
        return {**diagnostics,'status':'target_reached' if final['accepted'] else 'chunk_ready',
            'reason':'Checked candidate-limited IK/RRT route with modeled object visibility',
            'playable':True,'converged':final['accepted'],'replan_required':not final['accepted'],
            'goal_deg':np.rad2deg(goal).tolist(),'target_mm':(target*1000).tolist(),'target_axis':axis.tolist(),
            'goal_cursor_acceptance':metrics(goal),
            'cursor_acceptance':final,'ik_outcome':'accepted_goals','frames':frames,'smoothing':smoothing,'dt':.02,
            'rrt_diagnostics':chosen['route']['planner'],'planning_outcome':chosen['route']['planner']['status'],
            'visibility':chosen['route']['visibility'],'modeled_visibility_checked':True,
            'planned_route_rad':path.tolist(),'selected_route_joint_travel_rad':chosen['route']['joint_travel_rad'],
            'max_velocity_rad_s':smoothing['max_velocity_rad_s'],
            'max_acceleration_rad_s2':smoothing['max_acceleration_rad_s2'],
            'path_elapsed_ms':1000*(perf_counter()-started-report['ik_seconds']),
            'gaze_tracking':gaze,
            **{f'final_{k}':v for k,v in service.errors(states[-1],target,axis).items()}}
