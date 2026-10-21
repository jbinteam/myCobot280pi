"""Versioned 12-input, four-proposal tool-Z Seed NN. XYZ weights stay separate."""
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from .kinematics import ARM
from .pose5 import unit_axis

DEFAULT_CHECKPOINT = Path(__file__).parent/'artifacts/seed5_v1/models.json'
_CACHE = {}


def configuration(hidden=128):
    return {
        'schema_version': 1, 'model_id': 'mycobot280pi_seed5_v1',
        'task_type': 'xyz_directed_tool_z_free_local_z_spin',
        'input_schema': ['target_x_scaled', 'target_y_scaled', 'target_z_scaled',
                         'tool_z_x_base', 'tool_z_y_base', 'tool_z_z_base',
                         *[f'q{i+1}_normalized' for i in range(6)]],
        'architecture': [12, hidden, hidden, 24], 'activation': 'tanh_hidden',
        'normalization': {'position_scale_m': ARM.scale,
                          'joint_center_rad': ARM.center.tolist(), 'joint_radius_rad': ARM.radius.tolist(),
                          'direction': 'unit_vector_in_base_frame', 'joint_range': [-1, 1]},
        'outputs': {'proposals': 4, 'joints': 6,
                    'parameterization': 'clip(q_current + joint_radius * tanh(raw), lower, upper)'},
        'fk': {'urdf_sha256': ARM.fingerprint, 'base_frame': ARM.base_frame,
               'tcp_frame': ARM.tcp, 'tcp_offset_m': [0, 0, 0],
               'position_units': 'meters', 'joint_units': 'radians',
               'tool_axis': 'directed local Z expressed in base'},
        'joint_names': ARM.names, 'limits_rad': {'min': ARM.lower.tolist(), 'max': ARM.upper.tolist()},
    }


def features(q_current, target_m, target_axis):
    q=np.asarray(q_current,float);target=np.asarray(target_m,float)
    if q.shape!=(6,) or target.shape!=(3,) or not np.isfinite(q).all() or not np.isfinite(target).all():
        raise ValueError('Seed5 expects six finite joints and finite target XYZ in meters')
    if np.any(q<ARM.lower) or np.any(q>ARM.upper):
        raise ValueError('Seed5 current joints violate operating limits')
    return np.r_[target/ARM.scale,unit_axis(target_axis),(q-ARM.center)/ARM.radius]


class Seed5Model:
    def __init__(self,path=DEFAULT_CHECKPOINT):
        self.path=Path(path);content=self.path.read_bytes();self.data=json.loads(content)
        self.sha256=hashlib.sha256(content).hexdigest()
        arch=self.data.get('configuration',{}).get('architecture',[])
        hidden=arch[1] if len(arch)==4 else None
        if hidden not in (128,256,384) or self.data.get('configuration')!=configuration(hidden):
            raise ValueError('Incompatible Seed5 checkpoint: task/schema/FK/normalization/limits mismatch; XYZ weights cannot be used')
        training=self.data.get('training',{})
        if training.get('kind') not in ('smoke','full') or not isinstance(training.get('completed_steps'),int) or training['completed_steps']<1:
            raise ValueError('Seed5 checkpoint has no completed training; untrained networks are not served')
        layers=self.data.get('layers',[]);self.layers=[]
        if len(layers)!=3:raise ValueError('Seed5 requires three trained linear layers')
        for layer,inputs,outputs in zip(layers,[12,hidden,hidden],[hidden,hidden,24]):
            w=np.asarray(layer['weight'],float);b=np.asarray(layer['bias'],float)
            if w.shape!=(outputs,inputs) or b.shape!=(outputs,) or not np.isfinite(w).all() or not np.isfinite(b).all():
                raise ValueError('Invalid Seed5 layer shape or nonfinite trained weights')
            self.layers.append((w,b))

    def predict(self,q_current,target_m,target_axis):
        value=features(q_current,target_m,target_axis)
        for i,(w,b) in enumerate(self.layers):
            value=value@w.T+b
            if i<2:value=np.tanh(value)
        return ARM.clip(np.asarray(q_current)+ARM.radius*np.tanh(value.reshape(4,6)))


def load(path=DEFAULT_CHECKPOINT):
    path=Path(path);stat=path.stat();key=(str(path.resolve()),stat.st_mtime_ns,stat.st_size)
    if key not in _CACHE:
        model=Seed5Model(path)
        _CACHE.clear();_CACHE[key]=model
    return _CACHE[key]


def checkpoint_status(path=DEFAULT_CHECKPOINT):
    path=Path(path)
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
                'reason':'Smoke-trained only; not full training' if training['kind']=='smoke' else 'Compatible separately trained 5D checkpoint'}
    except FileNotFoundError:
        return {'status':'missing','model_id':configuration()['model_id'],'reason':'No compatible 5D Seed NN checkpoint; numerical fallback is available'}
    except (ValueError,KeyError,TypeError,OSError) as exc:
        return {'status':'incompatible','model_id':configuration()['model_id'],'reason':str(exc)}


def propose(q_current,target_m,target_axis,path=DEFAULT_CHECKPOINT):
    status=checkpoint_status(path)
    if status['status']!='ready':return [],status,0.
    began=perf_counter();guesses=load(path).predict(q_current,target_m,target_axis)
    return guesses,status,(perf_counter()-began)*1000
