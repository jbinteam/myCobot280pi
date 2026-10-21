"""Optional research prototype: bounded Cartesian gradients on neural routes.

This controller is separate from policy training and pure neural evaluation.
It does not compute an IK endpoint. Its only variables are the denoised joint
references, whose early route and total displacement are constrained. Corrected
routes must pass the same full-path and C2 motion checks before they are returned.
No production service imports this module.
"""
from dataclasses import dataclass
import numpy as np
from ..kinematics import ARM
from ..continuous import prepare


@dataclass(frozen=True)
class GuidanceConfig:
    iterations: int=16
    step_rad: float=.015
    max_correction_rad: float=.08
    axis_weight: float=.15
    smoothness_weight: float=.05
    protected_fraction: float=.25

    def __post_init__(self):
        if (isinstance(self.iterations,bool) or not isinstance(self.iterations,int)
                or not 1<=self.iterations<=64):raise ValueError('Guidance iterations must be 1..64')
        values=[self.step_rad,self.max_correction_rad,self.axis_weight,
                self.smoothness_weight,self.protected_fraction]
        if (not np.isfinite(values).all() or not 0<self.step_rad<=.03
                or not 0<self.max_correction_rad<=.15 or not 0<self.axis_weight<=1
                or not 0<=self.smoothness_weight<=1 or not 0<=self.protected_fraction<1):
            raise ValueError('Invalid bounded guidance configuration')


def pose_loss_gradient(q,target,axis,axis_weight=.15):
    """URDF FK gradient for XYZ and directed flange-Z; no pseudoinverse/IK."""
    transform,jacobian=ARM.spatial_jacobian(q)
    position=(transform[:3,3]-target)/ARM.scale
    direction=transform[:3,2]-axis
    direction_jacobian=np.cross(jacobian[3:].T,transform[:3,2]).T
    loss=.5*(np.dot(position,position)+axis_weight*np.dot(direction,direction))
    gradient=jacobian[:3].T@position/ARM.scale+axis_weight*direction_jacobian.T@direction
    return float(loss),gradient


def guide_chunk(env,current,chunk,target,axis,velocity=None,acceleration=None,speed=1.,
                config=GuidanceConfig()):
    """Refine an already checked neural chunk, then validate a C2 continuation.

    The first quarter of references stays unchanged; remaining corrections grow
    with a quintic taper and are capped relative to the original neural route.
    Every accepted gradient update rechecks the complete route, joint limits and
    discrete reference speed. A final C2 check enforces continuous speed and
    acceleration, including incoming motion. Failed guidance returns the neural
    route unchanged; the caller must still independently validate its execution.
    """
    current=np.asarray(current,float);original=np.asarray(chunk,float)
    target=np.asarray(target,float);axis=np.asarray(axis,float)
    velocity=np.zeros(6) if velocity is None else np.asarray(velocity,float)
    acceleration=np.zeros(6) if acceleration is None else np.asarray(acceleration,float)
    if (current.shape!=(6,) or original.ndim!=2 or original.shape[1:]!=(6,)
            or not 2<=len(original)<=16 or target.shape!=(3,) or axis.shape!=(3,)
            or not np.isfinite(np.r_[current,original.ravel(),target,axis]).all()
            or np.linalg.norm(axis)<1e-9):raise ValueError('Expected finite checked neural references and Cartesian pose')
    if not np.isfinite(speed) or not 0<speed<=1:raise ValueError('Invalid guidance speed')
    axis=axis/np.linalg.norm(axis)

    def safe(route):
        full=np.vstack((current,route))
        return (np.all(full>=ARM.lower) and np.all(full<=ARM.upper)
            and np.max(np.linalg.norm(np.diff(full,axis=0),axis=1))<=speed*.1+1e-9
            and env.path_valid(full))

    if not safe(original):raise ValueError('Guidance requires an independently checked complete neural route')
    progress=np.clip((np.linspace(0,1,len(original))-config.protected_fraction)/(1-config.protected_fraction),0,1)
    taper=10*progress**3-15*progress**4+6*progress**5
    weights=.05+.95*taper
    route=original.copy();accepted=0;collision_rejections=0;speed_rejections=0

    def objective(points,with_gradient=False):
        pairs=[pose_loss_gradient(q,target,axis,config.axis_weight) for q in points]
        loss=np.dot(weights,[pair[0] for pair in pairs])/len(points)
        offset=points-original
        curvature=np.diff(offset,n=2,axis=0)
        loss+=.5*config.smoothness_weight*np.square(curvature).sum()/len(points)
        if not with_gradient:return float(loss)
        gradient=np.asarray([pair[1] for pair in pairs])*weights[:,None]/len(points)
        smooth=np.zeros_like(offset)
        smooth[:-2]+=curvature;smooth[1:-1]-=2*curvature;smooth[2:]+=curvature
        return float(loss),gradient+config.smoothness_weight*smooth/len(points)

    before=objective(route)
    for _ in range(config.iterations):
        loss,gradient=objective(route,True)
        norms=np.linalg.norm(gradient,axis=1)
        delta=-gradient/np.maximum(norms[:,None],1e-12)*config.step_rad*taper[:,None]
        # Neighbor smoothing applies to corrections, preserving neural corners.
        delta[1:-1]=.25*delta[:-2]+.5*delta[1:-1]+.25*delta[2:]
        delta[taper==0]=0
        changed=False
        for factor in (1.,.5,.25,.125,.0625):
            offset=route+factor*delta-original
            limits=config.max_correction_rad*taper
            offset*=np.minimum(1.,limits/np.maximum(np.linalg.norm(offset,axis=1),1e-12))[:,None]
            candidate=original+offset
            if objective(candidate)>=loss-1e-12:continue
            full=np.vstack((current,candidate))
            if np.max(np.linalg.norm(np.diff(full,axis=0),axis=1))>speed*.1+1e-9:
                speed_rejections+=1;continue
            if not safe(candidate):collision_rejections+=1;continue
            route=candidate;accepted+=1;changed=True;break
        if not changed:break
    metadata={'controller':'bounded_cartesian_trajectory_gradient_prototype','production_integrated':False,
        'ik_calls':0,'accepted_updates':accepted,'collision_rejections':collision_rejections,
        'reference_speed_rejections':speed_rejections,'loss_before':before,'loss_after':objective(route),
        'max_correction_rad':float(np.linalg.norm(route-original,axis=1).max()),
        'protected_references':int(np.sum(taper==0)),'c2_validated':False}
    try:
        _,_,motion=prepare(env,np.vstack((current,route)),velocity,acceleration,speed)
    except ValueError as exc:
        metadata.update(applied=False,reason='guided_c2_rejected',c2_reason=str(exc))
        return original.copy(),metadata
    metadata.update(applied=bool(accepted),c2_validated=True,max_velocity_rad_s=motion['max_velocity_rad_s'],
        max_acceleration_rad_s2=motion['max_acceleration_rad_s2'],duration_s=motion['duration_s'])
    return route,metadata
