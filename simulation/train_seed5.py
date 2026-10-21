"""Train a separate best-of-four 5D Seed NN; never overwrite XYZ checkpoints."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from torch import nn
from .kinematics import ARM
from .prm import planning_space
from .seed5 import configuration, Seed5Model
from .seed5_data import load_dataset
from .train import network


def task_costs(predicted_xyz,predicted_axis,target_xyz,target_axis,position_weight=1.,direction_weight=.72):
    """Dimensionless XYZ/(0.3 m) squared + directed 1-dot loss (NOT sin²)."""
    position=((predicted_xyz-target_xyz[:,None,:])/ARM.scale).square().sum(-1)
    axis=nn.functional.normalize(predicted_axis,dim=-1)
    dot=(axis*target_axis[:,None,:]).sum(-1).clamp(-1,1)
    return position_weight*position+direction_weight*(1-dot)


def collision_repair_costs(candidates, witnesses, space):
    """Penalize every invalid proposal toward its known collision-free witness.

    The exact native checker supplies a detached validity mask, not a geometric
    derivative. The normalized joint-distance repair loss supplies gradients.
    Safe proposals get zero repair loss, preserving alternative IK branches.
    Use serving's joint clipping to remove float32 boundary roundoff.
    """
    if space.validation != 'joint_limits_and_self_collision':
        raise ValueError('Collision repair requires the URDF self-collision checker')
    if candidates.ndim != 3 or not len(candidates) or not candidates.shape[1] or candidates.shape[2] != 6 or witnesses.shape != (len(candidates), 6):
        raise ValueError('Expected batched six-joint proposals and safe witnesses')
    if not torch.isfinite(candidates).all() or not torch.isfinite(witnesses).all():
        raise ValueError('Collision audit requires finite predictions and witnesses')
    detached = ARM.clip(candidates.detach().cpu().numpy().astype(float))
    safe = ARM.clip(witnesses.detach().cpu().numpy().astype(float))
    if not all(space.valid(q) for q in safe):
        raise ValueError('Collision repair witnesses must be collision-free')
    invalid = np.array([not space.valid(q) for q in detached.reshape(-1, 6)]).reshape(detached.shape[:2])
    mask = torch.as_tensor(invalid, device=candidates.device)
    radius = torch.as_tensor(ARM.radius, dtype=candidates.dtype, device=candidates.device)
    distance = ((candidates-witnesses[:, None, :])/radius).square().sum(-1)
    costs = torch.where(mask, distance, torch.zeros_like(distance))
    return costs, {'proposals': int(invalid.size), 'invalid_proposals': int(invalid.sum()),
                   'invalid_fraction': float(invalid.mean())}


def collision_repair_loss(candidates, witnesses, space):
    costs,audit=collision_repair_costs(candidates,witnesses,space)
    # Keep rare unsafe predictions from being diluted by many safe heads.
    return costs.sum()/max(1,audit['invalid_proposals']),audit


def train(dataset_path,output,steps=2400,batch_size=512,seed=42,device='cpu',smoke=False,
          position_weight=1.,direction_weight=.72,travel_weight=.003,tolerance_mm=1.,angle_tolerance_deg=1.,hidden=128,witness_weight=0.,
          collision_weight=.1,collision_every=10,collision_batch_size=64,init_checkpoint=None,learning_rate=.001,
          collision_validation_batch_size=512):
    if steps<1 or batch_size<1 or not all(np.isfinite(v) and v>0 for v in (position_weight,direction_weight,learning_rate)) or not all(np.isfinite(v) and v>=0 for v in (travel_weight,witness_weight,collision_weight)):
        raise ValueError('Positive steps, batch size, learning rate and task weights; nonnegative travel, witness and collision weights required')
    for name,value in (('collision_every',collision_every),('collision_batch_size',collision_batch_size),
                       ('collision_validation_batch_size',collision_validation_batch_size)):
        if isinstance(value,bool) or not isinstance(value,int) or value<1:
            raise ValueError(f'{name} must be a positive integer')
    output=Path(output)
    if output.resolve()==(Path(__file__).parent/'artifacts').resolve():raise ValueError('Do not overwrite the existing XYZ checkpoint directory')
    if hidden not in (128,256,384):raise ValueError('Supported 5D hidden widths: 128, 256, 384')
    if output.joinpath('models.json').exists() and json.loads(output.joinpath('models.json').read_text()).get('configuration')!=configuration(hidden):raise ValueError('Output contains incompatible checkpoints; choose a separate directory')
    data,metadata=load_dataset(dataset_path)
    validity=metadata['validity'];collision='self' if validity['validation']=='joint_limits_and_self_collision' else 'limits'
    clearance=validity.get('assumptions',{}).get('clearance_mm',1.)
    space=planning_space(collision,clearance)
    if space.metadata()!=validity:raise ValueError('Dataset validity metadata changed; regenerate data')
    if collision_weight and space.validation!='joint_limits_and_self_collision':
        raise ValueError('Collision-aware training requires a self-collision-validated dataset')
    for prefix in ('','validation_'):
        if not all(space.valid(q) for key in ('q_current_rad','q_goal_rad') for q in data[prefix+key]):raise ValueError('Dataset fails configured finite/limit/collision criteria')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed);torch.set_num_threads(4)
    if device=='cuda':torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    center=torch.tensor(ARM.center,dtype=torch.float32,device=device)
    radius=torch.tensor(ARM.radius,dtype=torch.float32,device=device)
    low=center-radius;high=center+radius
    def split(prefix):
        return tuple(torch.tensor(data[prefix+key],dtype=torch.float32,device=device) for key in ('q_current_rad','target_xyz_m','target_axis_base','q_goal_rad'))
    training=split('');validation=split('validation_')
    model=network(12,24,hidden).to(device)
    initialization=None
    if init_checkpoint is not None:
        initial=Seed5Model(init_checkpoint)
        if initial.data['configuration']!=configuration(hidden):
            raise ValueError('Initialization architecture must match the training configuration')
        for layer,(weight,bias) in zip((layer for layer in model if isinstance(layer,nn.Linear)),initial.layers):
            with torch.no_grad():
                layer.weight.copy_(torch.as_tensor(weight,dtype=layer.weight.dtype,device=device))
                layer.bias.copy_(torch.as_tensor(bias,dtype=layer.bias.dtype,device=device))
        initialization={'checkpoint':str(init_checkpoint),'sha256':hashlib.sha256(Path(init_checkpoint).read_bytes()).hexdigest(),
                        'training_kind':initial.data['training']['kind'],'completed_steps':initial.data['training']['completed_steps']}
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=max(1,steps//2),gamma=.4)
    def loss(batch,audit=False,audit_size=None):
        q,xyz,axis,witness=batch
        raw=model(torch.cat((xyz/ARM.scale,axis,(q-center)/radius),dim=-1)).reshape(-1,4,6)
        candidates=torch.clamp(q[:,None,:]+radius*torch.tanh(raw),min=low,max=high)
        fk=ARM.torch_transform(candidates)
        costs=task_costs(fk[...,:3,3],fk[...,:3,2],xyz,axis,position_weight,direction_weight)
        costs=costs+travel_weight*(candidates-q[:,None,:]).square().sum(-1)
        task_loss=costs.min(dim=-1).values.mean()
        # Optional separate best-of-four supervision: one head learns the FK
        # witness, without averaging distinct joint solutions or constraining
        # inference acceptance/spin. Task loss still selects any valid solution.
        witness_cost=((candidates-witness[:,None,:])/radius).square().sum(-1)
        value=task_loss+witness_weight*witness_cost.min(dim=-1).values.mean()
        diagnostic=None
        if audit and collision_weight:
            count=min(audit_size or collision_batch_size,len(candidates))
            repair,diagnostic=collision_repair_loss(candidates[:count],witness[:count],space)
            value=value+collision_weight*repair
        return value,diagnostic
    began=perf_counter();best=float('inf');best_state=None;best_step=0;history=[]
    collision_audits={'batches':0,'proposals':0,'invalid_proposals':0}
    for step in range(steps):
        indices=torch.randint(len(training[0]),(batch_size,),device=device)
        batch=tuple(value[indices] for value in training)
        optimizer.zero_grad(set_to_none=True);value,audit=loss(batch,step%collision_every==0)
        if audit is not None:
            collision_audits['batches']+=1
            for name in ('proposals','invalid_proposals'):collision_audits[name]+=audit[name]
        if not torch.isfinite(value):raise RuntimeError('Nonfinite Seed5 training loss')
        value.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step();scheduler.step()
        if step%200==0 or step==steps-1:
            # Always apply the same audited validation objective to candidate
            # selection, independent of whether this optimizer step was audited.
            with torch.no_grad():
                validation_value,validation_audit=loss(validation,True,collision_validation_batch_size)
                val=float(validation_value)
            history.append({'step':step+1,'train_loss':float(value.detach()),'validation_loss':val,
                            'collision_validation_audit':validation_audit})
            if val<best:best=val;best_step=step+1;best_state=copy.deepcopy(model.state_dict())
            print(f'Seed5 {step+1}/{steps}: validation loss {val:.6f}',flush=True)
    model.load_state_dict(best_state);model.cpu();output.mkdir(parents=True,exist_ok=True)
    provenance={'kind':'smoke' if smoke else 'full','completed_steps':steps,'seed':seed,
                'dataset':str(dataset_path),'dataset_sha256':hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest(),
                'dataset_generation':metadata['generation'],'validity':validity,'device':device,'torch_version':str(torch.__version__),
                'batch_size':batch_size,'architecture':[12,hidden,hidden,24],'learning_rate':learning_rate,'position_weight':position_weight,'direction_weight':direction_weight,
                'travel_weight':travel_weight,'position_loss_units':'squared XYZ error / (0.3 m)^2',
                'witness_weight':witness_weight,'witness_loss':'separate best-of-four normalized joint distance to FK witness; no joint-label averaging',
                'direction_loss':'1-dot(unit_actual,unit_target), dimensionless; unique aligned minimum',
                'collision_training':{'enabled':collision_weight>0,'weight':collision_weight,'every_steps':collision_every,
                    'batch_size':collision_batch_size,'validation_batch_size':collision_validation_batch_size,
                    'audits':collision_audits,'validity':validity,
                    'loss':'Mean normalized joint-distance to a known safe FK witness over native-checker-invalid proposals only; zero if all are safe',
                    'gradient':'Detached exact validity mask plus differentiable witness repair; not a signed-distance gradient',
                    'scope':'Endpoint self-collision; no environment or trajectory guarantee'},
                'initialization':initialization,
                'objective':'best-of-four task costs + travel/witness regularizers + audited self-collision repair; no averaging joint labels',
                'seconds':perf_counter()-began,'best_validation_loss':best,'selected_step':best_step,'history':history}
    layers=[{'weight':layer.weight.detach().tolist(),'bias':layer.bias.detach().tolist()} for layer in model if isinstance(layer,nn.Linear)]
    checkpoint={'configuration':configuration(hidden),'training':provenance,'layers':layers}
    (output/'models.json').write_text(json.dumps(checkpoint,separators=(',',':'),allow_nan=False))
    torch.save({'configuration':configuration(hidden),'training':provenance,'state_dict':model.state_dict()},output/'seed5_v1.pt')
    serving=Seed5Model(output/'models.json');errors=[];accepted=0;inference_ms=0.
    invalid_proposals=0;tasks_with_valid_proposal=0;pose_best_invalid=0
    for q,xyz,axis in zip(data['validation_q_current_rad'],data['validation_target_xyz_m'],data['validation_target_axis_base']):
        tick=perf_counter();candidates=serving.predict(q,xyz,axis);inference_ms+=(perf_counter()-tick)*1000
        fk=np.stack([ARM.transform(c) for c in candidates]);pos=np.linalg.norm(fk[:,:3,3]-xyz,axis=1)*1000
        angles=np.rad2deg(np.arctan2(np.linalg.norm(np.cross(fk[:,:3,2],axis),axis=1),np.clip(fk[:,:3,2]@axis,-1,1)))
        valid=np.array([space.valid(c) for c in candidates])
        invalid_proposals+=int((~valid).sum());tasks_with_valid_proposal+=bool(valid.any())
        accepted+=bool(np.any(valid & (pos<=tolerance_mm) & (angles<=angle_tolerance_deg)))
        scores=position_weight*(pos/1000/ARM.scale)**2+direction_weight*(1-np.cos(np.deg2rad(angles)))
        best_candidate=int(np.argmin(scores));errors.append((pos[best_candidate],angles[best_candidate]))
        pose_best_invalid+=not valid[best_candidate]
    metrics={'held_out_tasks':len(errors),'raw_valid_success':accepted,'tolerance_mm':tolerance_mm,
             'angle_tolerance_deg':angle_tolerance_deg,'mean_best_position_error_mm':float(np.mean(np.array(errors)[:,0])),
             'mean_best_direction_error_deg':float(np.mean(np.array(errors)[:,1])),
             'mean_inference_ms':inference_ms/len(errors),'refinement':'not included in raw validation',
             'raw_proposals':len(errors)*4,'raw_invalid_proposals':invalid_proposals,
             'raw_invalid_fraction':invalid_proposals/(len(errors)*4),'tasks_with_valid_proposal':tasks_with_valid_proposal,
             'pose_best_invalid_tasks':int(pose_best_invalid),
             'limitation':'Smoke check is not full training' if smoke else 'One reproducible training seed; benchmark before relying on quality'}
    provenance['validation_metrics']=metrics
    (output/'models.json').write_text(json.dumps(checkpoint,separators=(',',':'),allow_nan=False))
    torch.save({'configuration':configuration(hidden),'training':provenance,'state_dict':model.state_dict()},output/'seed5_v1.pt')
    (output/'training.json').write_text(json.dumps(provenance,indent=2,allow_nan=False))
    print(json.dumps({'checkpoint':str(output/'models.json'),'training_kind':provenance['kind'],'metrics':metrics}),flush=True)
    return provenance


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=Path('simulation/artifacts/seed5_v1/dataset.npz'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--steps',type=int,default=2400);parser.add_argument('--batch-size',type=int,default=512)
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--device',choices=['auto','cpu','cuda'],default='cpu');parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--hidden',type=int,choices=[128,256,384],default=256)
    parser.add_argument('--position-weight',type=float,default=1.);parser.add_argument('--direction-weight',type=float,default=.72);parser.add_argument('--travel-weight',type=float,default=.003)
    parser.add_argument('--witness-weight',type=float,default=0.)
    parser.add_argument('--collision-weight',type=float,default=.1,help='Repair penalty for native-checker-invalid proposals; zero reproduces the legacy objective')
    parser.add_argument('--collision-every',type=int,default=10);parser.add_argument('--collision-batch-size',type=int,default=64)
    parser.add_argument('--collision-validation-batch-size',type=int,default=512)
    parser.add_argument('--init-checkpoint',type=Path,help='Compatible trained NumPy checkpoint for fine-tuning')
    parser.add_argument('--learning-rate',type=float,default=.001)
    args=parser.parse_args()
    if args.output is not None:output=args.output
    elif args.collision_weight:
        output=Path('simulation/artifacts/seed5_v1')/('collision_aware_smoke' if args.smoke else 'candidates/collision-aware')
    else:output=Path('simulation/artifacts/seed5_v1'+('/smoke' if args.smoke else ''))
    train(args.dataset,output,args.steps,args.batch_size,args.seed,args.device,args.smoke,args.position_weight,args.direction_weight,args.travel_weight,
          hidden=args.hidden,witness_weight=args.witness_weight,collision_weight=args.collision_weight,
          collision_every=args.collision_every,collision_batch_size=args.collision_batch_size,
          init_checkpoint=args.init_checkpoint,learning_rate=args.learning_rate,
          collision_validation_batch_size=args.collision_validation_batch_size)


if __name__=='__main__':main()
