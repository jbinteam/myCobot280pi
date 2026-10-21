"""Seven nn_ik ideas plus the matched classical null-space baseline."""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from .kinematics import ARM

ARTIFACTS=Path(__file__).parent/'artifacts'
METHODS=[
 ('pinv','Pseudoinverse IK','Classical resolved-rate position IK'),
 ('dls','Damped least squares','Classical DLS with joint-limit clipping'),
 ('direct','Direct NN IK','Target XYZ → one bounded joint pose; no classical refinement'),
 ('conditional','Conditional NN IK','Current pose + position error → bounded joint increment'),
 ('differential','Learned differential IK','Current pose + position error → learned joint velocity'),
 ('null_classical','Classical null-space','DLS primary + exact projection of posture policy'),
 ('null_neural','Neural null-space','DLS primary + exact projection of learned posture policy'),
 ('seed_dls','NN seed + DLS','Four previous-pose-conditioned seeds + bounded seed motion + DLS'),
]


class Models:
    def __init__(self,path=ARTIFACTS/'models.json'):
        self.data=json.loads(path.read_text())
        if self.data['urdf_sha256']!=ARM.fingerprint:raise ValueError('Neural model URDF fingerprint mismatch; retrain')
        limits=self.data.get('limits_rad',{})
        if not np.array_equal(limits.get('min'),ARM.lower) or not np.array_equal(limits.get('max'),ARM.upper):
            raise ValueError('Neural model joint limits mismatch; retrain before using these weights')
        self.layers={name:[(np.array(layer['weight']),np.array(layer['bias'])) for layer in m['layers']] for name,m in self.data['models'].items()}
    def predict(self,name,features):
        value=np.asarray(features,float)
        layers=self.layers[name]
        for index,(weight,bias) in enumerate(layers):
            value=value@weight.T+bias
            if index<len(layers)-1:value=np.tanh(value)
        return value


MODELS=None

def models():
    global MODELS
    if MODELS is None:MODELS=Models()
    return MODELS


def capped(vector,limit):
    norm=np.linalg.norm(vector)
    return vector*min(1.,limit/max(norm,1e-12))


def dls(j,error,damping):return j.T@np.linalg.solve(j@j.T+damping*damping*np.eye(3),error)


def validate(q,target,method,steps,speed,tolerance,damping):
    q=np.asarray(q,float);target=np.asarray(target,float)
    if q.shape!=(6,) or target.shape!=(3,) or not np.isfinite(q).all() or not np.isfinite(target).all():raise ValueError('Expected six finite joint angles and three finite target coordinates')
    if np.any(q<ARM.lower-1e-7) or np.any(q>ARM.upper+1e-7):raise ValueError('Start pose is outside the verified operating joint limits')
    if method not in dict((m[0],m[1]) for m in METHODS):raise ValueError('Unknown IK method')
    if isinstance(steps,bool) or not isinstance(steps,(int,float)) or not np.isfinite(steps) or int(steps)!=steps or not 1<=steps<=600:raise ValueError('Steps must be an integer from 1 to 600')
    for value,low,high,label in [(speed,.1,3.,'Joint speed'),(tolerance,.1,10.,'Tolerance in mm'),(damping,.001,.3,'Damping')]:
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or not low<=value<=high:raise ValueError(f'{label} must be between {low} and {high}')
    if np.linalg.norm(target)>1.:raise ValueError('Target must be within 1 meter of the base')
    return ARM.clip(q),target


def solve(q_start,target,method='seed_dls',steps=240,speed=1.2,tolerance=1.,damping=.04):
    q,target=validate(q_start,target,method,steps,speed,tolerance,damping)
    dt=.05;kp=3.;start=perf_counter();frames=[];hits=0;leaks=[];inference=[];seed=None
    normalized=lambda x:(x-ARM.center)/ARM.radius
    if method=='direct':seed=ARM.center+ARM.radius*np.tanh(models().predict('direct',target/ARM.scale))
    if method=='seed_dls':
        raw=models().predict('seed',np.r_[target/ARM.scale,normalized(q)]).reshape(4,6)
        candidates=ARM.clip(q+ARM.radius*np.tanh(raw))
        candidates=np.vstack([q,candidates])
        costs=np.array([np.linalg.norm((ARM.fk(c)-target)/ARM.scale)**2+.003*np.linalg.norm(c-q)**2 for c in candidates])
        seed=candidates[np.argmin(costs)]
    seed_active=seed is not None
    for iteration in range(int(steps)+1):
        t,j_actual=ARM.transform(q,True);p=t[:3,3];e=(target-p)/ARM.scale;j=j_actual/ARM.scale
        err=float(np.linalg.norm(target-p)*1000)
        frame={'q':np.rad2deg(q).tolist(),'position':(p*1000).tolist(),'error_mm':err,'time':iteration*dt}
        frames.append(frame)
        if err<=tolerance or iteration==steps:break
        tick=perf_counter()
        if seed_active:
            velocity=(seed-q)/dt
            if np.linalg.norm(seed-q)<1e-6:
                seed_active=False
                if method=='direct':break
                velocity=kp*dls(j,e,damping)
        elif method=='pinv':velocity=kp*np.linalg.pinv(j,rcond=1e-5)@e
        elif method in ('dls','seed_dls'):velocity=kp*dls(j,e,damping)
        elif method in ('conditional','differential'):
            raw=models().predict(method,np.r_[normalized(q),e])
            velocity=(.12/dt if method=='conditional' else 1.2)*np.tanh(raw)
        elif method in ('null_classical','null_neural'):
            z=(-.6*normalized(q) if method=='null_classical' else 1.2*np.tanh(models().predict('null',normalized(q))))
            null=(np.eye(6)-np.linalg.pinv(j,rcond=1e-5)@j)@z
            leaks.append(float(np.linalg.norm(j_actual@null)))
            velocity=kp*dls(j,e,damping)+.3*null
        else:break
        velocity=capped(velocity,speed)
        proposed=q+dt*velocity;q_next=ARM.clip(proposed)
        hits+=int(np.any(np.abs(q_next-proposed)>1e-9))
        inference.append((perf_counter()-tick)*1000)
        if np.linalg.norm(q_next-q)<1e-9 and not seed_active:break
        q=q_next
    # Include the final state when a stalled/finished direct solver stops early.
    if not np.allclose(frames[-1]['q'],np.rad2deg(q),atol=1e-9):
        p=ARM.fk(q);frames.append({'q':np.rad2deg(q).tolist(),'position':(p*1000).tolist(),'error_mm':float(np.linalg.norm(p-target)*1000),'time':len(frames)*dt})
    qs=np.deg2rad(np.array([f['q'] for f in frames]))
    travel=float(np.linalg.norm(np.diff(qs,axis=0),axis=1).sum()) if len(qs)>1 else 0.
    return {'method':method,'frames':frames,'target_mm':(target*1000).tolist(),'converged':frames[-1]['error_mm']<=tolerance,
            'final_error_mm':frames[-1]['error_mm'],'iterations':len(frames)-1,'joint_travel_deg':float(np.rad2deg(travel)),
            'elapsed_ms':(perf_counter()-start)*1000,'mean_step_ms':float(np.mean(inference)) if inference else 0.,
            'limit_hits':hits,'null_residual_max':max(leaks,default=0.),'tolerance_mm':tolerance,
            'stop_reason':'target_reached' if frames[-1]['error_mm']<=tolerance else ('iteration_limit' if len(frames)-1>=steps else 'stalled_or_direct_prediction_finished')}


def cases(seed=9001,count=24):
    rng=np.random.default_rng(seed);out=[]
    for i in range(count):
        q0=ARM.center+ARM.radius*rng.uniform(-.8,.8,6)
        if i%4==0:q0=np.zeros(6)
        if i%4<2:goal=ARM.clip(q0+rng.normal(0,.25,6));kind='nearby'
        else:goal=ARM.center+ARM.radius*rng.uniform(-.8,.8,6);kind='global'
        out.append({'name':f'{kind}-{i+1:02d}','start_deg':np.rad2deg(q0).tolist(),'target_mm':(ARM.fk(goal)*1000).tolist(),'kind':kind})
    return out


def benchmark(output=True):
    models()  # Load/validate once, outside timed rollouts for fair inference timing.
    fixtures=cases();rows=[]
    for method,name,_ in METHODS:
        results=[solve(np.deg2rad(c['start_deg']),np.array(c['target_mm'])/1000,method) for c in fixtures]
        row={'method':method,'name':name,'success':sum(r['converged'] for r in results),'total':len(results),
             'mean_error_mm':float(np.mean([r['final_error_mm'] for r in results])),
             'mean_travel_deg':float(np.mean([r['joint_travel_deg'] for r in results])),
             'mean_elapsed_ms':float(np.mean([r['elapsed_ms'] for r in results])),
             'cases':[{k:r[k] for k in ('converged','final_error_mm','iterations','joint_travel_deg','elapsed_ms')} for r in results]}
        rows.append(row);print(f"{name}: {row['success']}/{row['total']}; mean error {row['mean_error_mm']:.2f} mm",flush=True)
    data={'seed':9001,'cases':fixtures,'rows':rows,'tolerance_mm':1.,'steps':240,'dt':.05,'speed_rad_s':1.2,
          'timing':'CPU rollout wall time; model loading excluded; OPENBLAS_NUM_THREADS=1 recommended',
          'task':'XYZ position only; TCP joint6_flange','training_seed':42,'validation_seed':43,'urdf_sha256':ARM.fingerprint,
          'limits_rad':{'min':ARM.lower.tolist(),'max':ARM.upper.tolist()},'joint_limit_profile':ARM.limit_profile,
          'limitations':'One training seed and 24 point cases. No collision checking or hardware validation. Neural runs are exploratory.'}
    if output:(ARTIFACTS/'benchmark.json').write_text(json.dumps(data,indent=2))
    return data


if __name__=='__main__':benchmark()
