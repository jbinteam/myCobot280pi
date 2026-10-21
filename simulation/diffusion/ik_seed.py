"""Versioned diffusion IK initial guesses; every proposal requires numerical refinement."""
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
import numpy as np
from ..kinematics import ARM
from ..pose5 import unit_axis

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1]/'artifacts/ik_seed_v1/models.json'
_CACHE = {}


def configuration(hidden=256, proposals=8):
    """Training/runtime contract; deployment currently accepts the fixed 256x8 model."""
    return {
        'schema_version': 1, 'model_id': 'mycobot280pi_ik_seed_v1',
        'task_type': 'xyz_directed_tool_z_free_local_z_spin',
        'input_schema': ['target_x_scaled', 'target_y_scaled', 'target_z_scaled',
                         'tool_z_x_base', 'tool_z_y_base', 'tool_z_z_base',
                         *[f'q{i+1}_normalized' for i in range(6)]],
        'architecture': [12, hidden, hidden, proposals * 6], 'activation': 'tanh_hidden',
        'normalization': {'position_scale_m': ARM.scale,
                          'joint_center_rad': ARM.center.tolist(), 'joint_radius_rad': ARM.radius.tolist(),
                          'direction': 'unit_vector_in_base_frame', 'joint_range': [-1, 1]},
        'outputs': {'proposals': proposals, 'joints': 6,
                    'parameterization': 'joint_center + joint_radius * tanh(raw)'},
        'fk': {'urdf_sha256': ARM.fingerprint, 'base_frame': ARM.base_frame,
               'tcp_frame': ARM.tcp, 'tcp_offset_m': [0, 0, 0],
               'position_units': 'meters', 'joint_units': 'radians',
               'tool_axis': 'directed local Z expressed in base'},
        'joint_names': ARM.names, 'limits_rad': {'min': ARM.lower.tolist(), 'max': ARM.upper.tolist()},
    }


def features(q_current, target_m, target_axis):
    q=np.asarray(q_current,float);target=np.asarray(target_m,float)
    if q.shape!=(6,) or target.shape!=(3,) or not np.isfinite(q).all() or not np.isfinite(target).all():
        raise ValueError('IKSeed expects six finite joints and finite target XYZ in meters')
    if np.any(q<ARM.lower) or np.any(q>ARM.upper):
        raise ValueError('IKSeed current joints violate operating limits')
    return np.r_[target/ARM.scale,unit_axis(target_axis),(q-ARM.center)/ARM.radius]


class IKSeedModel:
    def __init__(self,path=DEFAULT_CHECKPOINT):
        self.path=Path(path);content=self.path.read_bytes();self.data=json.loads(content)
        self.sha256=hashlib.sha256(content).hexdigest()
        arch=self.data.get('configuration',{}).get('architecture',[])
        hidden=arch[1] if len(arch)==4 else None
        if hidden != 256 or self.data.get('configuration') != configuration():
            raise ValueError('Incompatible IKSeed checkpoint: task/schema/FK/normalization/limits mismatch; XYZ weights cannot be used')
        training=self.data.get('training',{})
        if training.get('completed') is not True or training.get('kind') not in ('smoke','full') or type(training.get('completed_steps')) is not int or training['completed_steps'] < 1:
            raise ValueError('IKSeed checkpoint has no completed training; untrained networks are not served')
        layers=self.data.get('layers',[]);self.layers=[]
        if len(layers)!=3:raise ValueError('IKSeed requires three trained linear layers')
        for layer,inputs,outputs in zip(layers,[12,hidden,hidden],[hidden,hidden,48]):
            w=np.asarray(layer['weight'],float);b=np.asarray(layer['bias'],float)
            if w.shape!=(outputs,inputs) or b.shape!=(outputs,) or not np.isfinite(w).all() or not np.isfinite(b).all():
                raise ValueError('Invalid IKSeed layer shape or nonfinite trained weights')
            self.layers.append((w,b))

    def predict(self,q_current,target_m,target_axis):
        value=features(q_current,target_m,target_axis)
        for i,(w,b) in enumerate(self.layers):
            value=value@w.T+b
            if i<2:value=np.tanh(value)
        return ARM.center + ARM.radius * np.tanh(value.reshape(8, 6))


def checkpoint_path(path=None):
    return Path(path) if path is not None else Path(os.environ.get('MYCOBOT_IK_SEED_MODEL', DEFAULT_CHECKPOINT))


def load(path=None):
    path=checkpoint_path(path);stat=path.stat();key=(str(path.resolve()),stat.st_mtime_ns,stat.st_size)
    if key not in _CACHE:
        model=IKSeedModel(path)
        _CACHE.clear();_CACHE[key]=model
    return _CACHE[key]


def checkpoint_status(path=None):
    path=checkpoint_path(path)
    try:
        model=load(path)
        training=model.data['training'];collision=training.get('collision_training',{})
        return {'status':'ready','model_id':configuration()['model_id'],'checkpoint':str(path),
                'checkpoint_sha256':model.sha256,
                'training_kind':training['kind'],'completed_steps':training['completed_steps'],
                'fine_tuned':bool(training.get('initialization')),
                'collision_training':{'enabled':bool(collision.get('enabled',False)),
                    'scope':collision.get('scope'),
                    'clearance_mm':collision.get('validity',{}).get('assumptions',{}).get('clearance_mm')},
                'reason':'Smoke-trained only; not full training' if training['kind']=='smoke' else 'Compatible diffusion IK initial-guess checkpoint'}
    except FileNotFoundError:
        return {'status':'missing','model_id':configuration()['model_id'],'reason':'No compatible diffusion IK initial-guess checkpoint; numerical fallback is available'}
    except (ValueError,KeyError,TypeError,OSError,OverflowError,AttributeError) as exc:
        return {'status':'incompatible','model_id':configuration()['model_id'],'reason':str(exc)}


def propose(q_current,target_m,target_axis,path=None):
    """Return bounded initial guesses, status, inference time; never IK solutions."""
    status=checkpoint_status(path)
    if status['status']!='ready':return [],status,0.
    began=perf_counter()
    try:
        guesses=load(path).predict(q_current,target_m,target_axis)
        if guesses.shape != (8, 6) or not np.isfinite(guesses).all():
            raise ValueError('Nonfinite IK initial guesses')
        return guesses,status,(perf_counter()-began)*1000
    except (ValueError,KeyError,TypeError,OSError,OverflowError,AttributeError) as exc:
        return [],dict(status,status='incompatible',reason=str(exc)),(perf_counter()-began)*1000
