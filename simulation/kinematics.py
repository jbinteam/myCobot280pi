"""URDF-derived FK and geometric position Jacobian (meters and radians)."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / 'webui/public/models/mycobot_description/urdf/mycobot_280_pi/mycobot_280_pi.urdf'
SDK_MIN = np.deg2rad([-168, -140, -150, -150, -155, -180])
SDK_MAX = np.deg2rad([168, 140, 150, 150, 160, 180])
LIMIT_PROFILE = json.loads((Path(__file__).parent/'joint_limits.json').read_text())


def rotation(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    x,y,z = axis
    skew = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3) + np.sin(angle)*skew + (1-np.cos(angle))*(skew@skew)


def origin_matrix(node):
    result = np.eye(4)
    if node is None: return result
    result[:3,3] = list(map(float,node.get('xyz','0 0 0').split()))
    r,p,y = list(map(float,node.get('rpy','0 0 0').split()))
    result[:3,:3] = rotation([0,0,1],y) @ rotation([0,1,0],p) @ rotation([1,0,0],r)
    return result


class Arm:
    tcp = 'joint6_flange'
    scale = 0.3
    def __init__(self):
        root = ET.parse(URDF).getroot()
        by_child = {j.find('child').get('link'):j for j in root.findall('joint')}
        nodes=[]; link=self.tcp
        while link in by_child:
            j=by_child[link]; nodes.append(j); link=j.find('parent').get('link')
        self.base_frame=link
        self.link_names=[j.find('child').get('link') for j in reversed(nodes)]
        self.chain=[]; self.names=[]; self._joint_rotations=[]; lower=[]; upper=[]
        for j in reversed(nodes):
            active=j.get('type')=='revolute'
            axis=np.array(list(map(float,j.find('axis').get('xyz').split()))) if active else None
            self.chain.append((origin_matrix(j.find('origin')),axis))
            if active:
                # URDF joint axes are constant. Cache Rodrigues coefficients,
                # retaining the original axis for Jacobian/frame conventions.
                x,y,z=axis/np.linalg.norm(axis)
                skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
                self._joint_rotations.append((skew,skew@skew))
                self.names.append(j.get('name'))
                lower.append(float(j.find('limit').get('lower')))
                upper.append(float(j.find('limit').get('upper')))
        if len(self.names)!=6: raise ValueError('Expected six revolute joints')
        self.hardware_lower=np.deg2rad(LIMIT_PROFILE['hardware_min_deg'])
        self.hardware_upper=np.deg2rad(LIMIT_PROFILE['hardware_max_deg'])
        if self.hardware_lower.shape!=(6,) or self.hardware_upper.shape!=(6,) or not np.isfinite(self.hardware_lower).all() or not np.isfinite(self.hardware_upper).all():
            raise ValueError('Expected six finite hardware joint limits')
        margin=float(LIMIT_PROFILE['operating_margin_deg'])
        if not np.isfinite(margin) or margin<0:raise ValueError('Joint operating margin must be nonnegative')
        self.lower=np.maximum(np.maximum(lower,SDK_MIN),self.hardware_lower)+np.deg2rad(margin)
        self.upper=np.minimum(np.minimum(upper,SDK_MAX),self.hardware_upper)-np.deg2rad(margin)
        # Match the UI/command resolution, rounding inwards at every bound.
        self.lower=np.deg2rad(np.ceil(np.rad2deg(self.lower)*10)/10)
        self.upper=np.deg2rad(np.floor(np.rad2deg(self.upper)*10)/10)
        if np.any(self.lower>=self.upper):raise ValueError('Empty operating joint range')
        self.center=(self.lower+self.upper)/2
        self.radius=(self.upper-self.lower)/2
        self.fingerprint=hashlib.sha256(URDF.read_bytes()).hexdigest()
        self.limit_profile=LIMIT_PROFILE

    def transform(self,q,with_jacobian=False,with_angular=False):
        q=np.asarray(q,float)
        if q.shape!=(6,) or not np.isfinite(q).all(): raise ValueError('Expected six finite radians')
        t=np.eye(4); origins=[]; axes=[]; index=0
        for base,axis in self.chain:
            t=t@base
            if axis is not None:
                if with_jacobian:
                    origins.append(t[:3,3].copy()); axes.append(t[:3,:3]@axis)
                r=np.eye(4);r[:3,:3]=self._joint_rotation(index,q[index]);t=t@r;index+=1
        if with_jacobian:
            axes=np.asarray(axes);origins=np.asarray(origins)
            j=np.cross(axes,t[:3,3]-origins).T
            return t,np.vstack((j,axes.T)) if with_angular else j
        return t

    def _joint_rotation(self,index,angle):
        skew,squared=self._joint_rotations[index]
        return np.eye(3)+np.sin(angle)*skew+(1-np.cos(angle))*squared

    def spatial_jacobian(self,q):return self.transform(q,True,True)
    def link_transforms(self,q):
        """URDF base-to-link transforms for collision geometry; same FK chain."""
        q=np.asarray(q,float)
        if q.shape!=(6,) or not np.isfinite(q).all():raise ValueError('Expected six finite radians')
        result={self.base_frame:np.eye(4)};t=np.eye(4);index=0
        for name,(base,axis) in zip(self.link_names,self.chain):
            t=t@base
            if axis is not None:
                r=np.eye(4);r[:3,:3]=self._joint_rotation(index,q[index]);t=t@r;index+=1
            result[name]=t.copy()
        return result

    def link_transforms_batch(self,q):
        """Vectorized collision FK [N,6] -> per-link [N,4,4] transforms.

        Use the same URDF chain and Rodrigues coefficients as scalar FK;
        vectorization is over poses, not a different kinematic model.
        """
        q=np.asarray(q,float)
        if q.ndim!=2 or q.shape[1]!=6 or not np.isfinite(q).all():
            raise ValueError('Expected finite [N,6] radians')
        t=np.broadcast_to(np.eye(4),(len(q),4,4)).copy()
        result={self.base_frame:t.copy()};index=0
        for name,(base,axis) in zip(self.link_names,self.chain):
            t=t@base
            if axis is not None:
                skew,squared=self._joint_rotations[index]
                angle=q[:,index,None,None]
                r=np.broadcast_to(np.eye(4),(len(q),4,4)).copy()
                r[:,:3,:3]=np.eye(3)+np.sin(angle)*skew+(1-np.cos(angle))*squared
                t=t@r;index+=1
            result[name]=t.copy()
        return result

    def fk(self,q):return self.transform(q)[:3,3]
    def jacobian(self,q):return self.transform(q,True)[1]
    def clip(self,q):return np.clip(q,self.lower,self.upper)

    def torch_fk_jacobian(self,q,with_transform=False):
        import torch
        # Shared URDF chain; no separate hand-coded DH convention.
        t=torch.eye(4,device=q.device,dtype=q.dtype).expand(*q.shape[:-1],4,4)
        origins=[]; axes=[]; index=0
        for base,axis in self.chain:
            t=t@torch.as_tensor(base,device=q.device,dtype=q.dtype)
            if axis is not None:
                origins.append(t[...,:3,3]);axes.append(t[...,:3,:3]@torch.as_tensor(axis,device=q.device,dtype=q.dtype))
                x,y,z=axis/np.linalg.norm(axis)
                skew=torch.tensor([[0,-z,y],[z,0,-x],[-y,x,0]],device=q.device,dtype=q.dtype)
                angle=q[...,index]
                r=torch.eye(3,device=q.device,dtype=q.dtype)+torch.sin(angle)[...,None,None]*skew+(1-torch.cos(angle))[...,None,None]*(skew@skew)
                top=torch.cat((r,torch.zeros(*q.shape[:-1],3,1,device=q.device,dtype=q.dtype)),dim=-1)
                bottom=torch.tensor([0,0,0,1],device=q.device,dtype=q.dtype).expand(*q.shape[:-1],1,4)
                t=t@torch.cat((top,bottom),dim=-2);index+=1
        p=t[...,:3,3]
        j=torch.stack([torch.linalg.cross(a,p-o,dim=-1) for a,o in zip(axes,origins)],dim=-1)
        return (t if with_transform else p),j

    def torch_transform(self,q):
        """Differentiable full FK for tool-axis training, from the shared URDF."""
        return self.torch_fk_jacobian(q,with_transform=True)[0]


ARM=Arm()
