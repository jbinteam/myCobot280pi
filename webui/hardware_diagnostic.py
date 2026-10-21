"""Small, collision-checked single-joint command comparison; no planning moves."""
import numpy as np
from simulation.kinematics import ARM
from simulation.pose5 import xy_from_axis
from webui.fixed_profile import selected_speed
from webui.hardware_diffusion import verified_status, COLLISION_STEP_RAD


def prepare(start_deg, body):
    if set(body)-{'joint','target_deg','variant','speed'}:
        raise ValueError('Joint diagnostic accepts joint, target, variant and speed only')
    joint=body.get('joint',4)
    if isinstance(joint,bool) or not isinstance(joint,int) or not 1<=joint<=6:
        raise ValueError('Diagnostic joint must be 1–6')
    variant=body.get('variant','single_target')
    if variant not in ('single_target','small_waypoints'):
        raise ValueError('Choose single_target or small_waypoints')
    speed=selected_speed(body,70)
    target=body.get('target_deg')
    if isinstance(target,bool) or not isinstance(target,(int,float)) or not np.isfinite(target):
        raise ValueError('Provide a finite diagnostic joint target')
    start=np.asarray(start_deg,float);end=start.copy();end[joint-1]=round(float(target),2)
    distance=abs(end[joint-1]-start[joint-1])
    if not .01<=distance<=3.:
        raise ValueError('Joint diagnostic displacement must be 0.01–3 degrees')
    status=verified_status()
    from simulation.diffusion.env import ArmEnv
    with ArmEnv() as env:
        env.set_obstacle(None)
        if env.metadata()!=status['environment']:
            raise RuntimeError('Diagnostic collision scene differs from the reviewed table scene')
        if not env.path_valid(np.deg2rad([start,end]),COLLISION_STEP_RAD):
            raise ValueError('Diagnostic path fails self/table/joint-limit validation')
    count=1 if variant=='single_target' else int(np.ceil(distance/.02-1e-9))
    points=np.round(np.linspace(start,end,count+1)[1:],2)
    # Quantized packets must strictly advance the selected joint. Each trial
    # uses the same SDK setting; small commands are deliberately timed at .3 s.
    points=[q.tolist() for q in points if abs(q[joint-1]-start[joint-1])>=.005]
    if len({q[joint-1] for q in points})!=len(points):
        raise ValueError('Diagnostic commands collapse at SDK angle resolution')
    transform=ARM.transform(np.deg2rad(end));xyz=(transform[:3,3]*1000).tolist();axis=transform[:3,2].tolist()
    frames=[{'q':start.tolist(),'time':0.}]
    for index,q in enumerate(points,1):
        frames.append({'q':q,'time':index*.3})
    diagnostic={'joint':joint,'variant':variant,'start_deg':start.tolist(),'end_deg':end.tolist(),
        'sdk_speed':speed,'command_interval_s':.3,
        'maximum_step_deg':float(np.max(np.abs(np.diff(np.asarray([start,*points]),axis=0)))),
        'scope':'discrete position-command comparison; not a continuous velocity or acceleration guarantee'}
    result={'method':'joint_diagnostic','task':'pose5','converged':True,'frames':frames,
        'target_mm':xyz,'target_axis':axis,'orientation_xy_deg':xy_from_axis(np.asarray(axis)),
        'final_error_mm':0.,'final_orientation_error_deg':0.,'planned_sdk_speed':speed,
        'joint_velocity_allowance_deg_s':150.*speed/100.,'diagnostic':diagnostic,
        'smoothing':{'duration_s':len(points)*.3},'checkpoint_sha256':status['checkpoint_sha256']}
    return {'result':result,'target_mm':xyz,'eef_rotation':ARM.transform(np.deg2rad(start))[:3,:3].tolist(),
        'waypoints':points,'stop_indices':{len(points)-1},'environment':status['environment'],
        'waypoint_intervals_s':[.3]*len(points)}
