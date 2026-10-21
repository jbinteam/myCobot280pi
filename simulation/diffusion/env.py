"""PyBullet myCobot scene with shared FK, conservative self and box collisions.

Reference execution is kinematic: checked joint interpolation is displayed in
PyBullet. This URDF has no calibrated mass/inertia or actuator dynamics.
"""
import argparse
import fcntl
from importlib.metadata import version
import json
import os
from pathlib import Path
import tempfile
import time
import xml.etree.ElementTree as ET
import numpy as np
import pybullet as p
from ..kinematics import ARM, URDF
from ..pose5 import solve, unit_axis, axis_error
from ..prm import planning_space, geometry_digest
from ..self_collision import collision_mesh
from .common import ARTIFACTS, robot_schema, write_json, file_sha256


def bullet_urdf(cache):
    """Convert COLLADA collision hulls to OBJ; preserve joints and mesh origins."""
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    with (cache/'conversion.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _bullet_urdf_unlocked(cache)


def _bullet_urdf_unlocked(cache):
    folder=Path(cache)/geometry_digest();folder.mkdir(parents=True,exist_ok=True)
    target=folder/'mycobot.urdf';manifest=folder/'manifest.json'
    if target.exists() and manifest.exists():
        identity=json.loads(manifest.read_text())
        if identity.get('converter')!='convex_obj_v1' or identity.get('source_geometry_sha256')!=geometry_digest():
            raise ValueError('Incompatible derived Bullet assets')
        files=identity.get('files',{})
        if 'mycobot.urdf' not in files or len(files)!=9:raise ValueError('Incomplete derived Bullet assets')
        for filename,digest in files.items():
            if Path(filename).name!=filename or not (folder/filename).is_file() or file_sha256(folder/filename)!=digest:
                raise ValueError('Derived Bullet asset changed; regenerate the asset cache')
        return target
    root=ET.parse(URDF).getroot()
    for child in list(root):
        if child.tag.endswith('property'):root.remove(child)
    converted={}
    for mesh in root.iter('mesh'):
        name=mesh.get('filename');scale=mesh.get('scale','1 1 1')
        key=(name,scale)
        if key not in converted:
            prefix='package://mycobot_description/'
            if not name.startswith(prefix):raise ValueError('Unsupported URDF mesh path')
            source=URDF.parents[2]/name[len(prefix):]
            if not source.resolve().is_relative_to(URDF.parents[2].resolve()):
                raise ValueError('URDF mesh escapes package')
            hull,_=collision_mesh(source,list(map(float,scale.split())))
            destination=folder/f'mesh_{len(converted)}.obj'
            hull.export(destination);converted[key]=destination.resolve()
        mesh.set('filename',str(converted[key]));mesh.attrib.pop('scale',None)
    # Supply explicit, uncalibrated inertials to avoid Bullet's silent defaults.
    # Dynamics are disabled; these are solely loader-compatible placeholders.
    for link in root.findall('link'):
        if link.find('inertial') is None:
            inertial=ET.SubElement(link,'inertial')
            ET.SubElement(inertial,'mass',value='0.1')
            ET.SubElement(inertial,'inertia',ixx='0.0001',iyy='0.0001',izz='0.0001',ixy='0',ixz='0',iyz='0')
    temporary=target.with_suffix('.tmp')
    ET.ElementTree(root).write(temporary,encoding='utf-8',xml_declaration=True)
    temporary.replace(target)
    write_json(manifest,{'converter':'convex_obj_v1','source_geometry_sha256':geometry_digest(),
        'files':{path.name:file_sha256(path) for path in [target,*converted.values()]}})
    return target


class ArmEnv:
    """Isolated Bullet client; all joint arguments have shape [6] in radians."""
    def __init__(self, gui=False, clearance_mm=1., cache=ARTIFACTS/'bullet_assets'):
        self.space=planning_space('self',clearance_mm)
        self.clearance=clearance_mm/1000
        self.batch_reset=(os.environ.get('MYCOBOT_BULLET_BATCH_RESET','1')!='0'
                          and callable(getattr(p,'resetJointStatesMultiDof',None)))
        self.client=p.connect(p.GUI if gui else p.DIRECT)
        self.closed=False;self.obstacle=None;self.box=None;self._valid_cache={}
        try:
            p.setGravity(0,0,0,physicsClientId=self.client)
            p.setPhysicsEngineParameter(enableFileCaching=0,physicsClientId=self.client)
            self.robot=p.loadURDF(str(bullet_urdf(cache)),useFixedBase=True,physicsClientId=self.client)
            joints={};links={}
            for i in range(p.getNumJoints(self.robot,physicsClientId=self.client)):
                info=p.getJointInfo(self.robot,i,physicsClientId=self.client)
                joints[info[1].decode()]=i;links[info[12].decode()]=i
            self.joints=[joints[name] for name in ARM.names];self.tcp=links[ARM.tcp]
            if len(self.joints)!=6 or any(p.getJointInfo(self.robot,i,physicsClientId=self.client)[2]!=p.JOINT_REVOLUTE for i in self.joints):
                raise ValueError('Bullet joint order differs from shared myCobot FK')
            p.setJointMotorControlArray(self.robot,self.joints,p.VELOCITY_CONTROL,forces=[0]*6,physicsClientId=self.client)
            base=next(b for b in self.space.collision_checker.bodies if b['link']==ARM.base_frame)
            corners=base['corners']@base['origin'][:3,:3].T+base['origin'][:3,3]
            self.table_top=float(corners[:,2].min()-max(.005,self.clearance*2))
            shape=p.createCollisionShape(p.GEOM_BOX,halfExtents=[.45,.45,.025],physicsClientId=self.client)
            visual=p.createVisualShape(p.GEOM_BOX,halfExtents=[.45,.45,.025],rgbaColor=[.65,.58,.47,1],physicsClientId=self.client)
            self.table=p.createMultiBody(baseMass=0,baseCollisionShapeIndex=shape,baseVisualShapeIndex=visual,
                basePosition=[0,0,self.table_top-.025],physicsClientId=self.client)
            self.set_obstacle([.18,0,self.table_top+.11,.025,.035,.11])
            self.reset(np.deg2rad([0,-40,-130,80,0,50]))
            if gui:p.resetDebugVisualizerCamera(.75,35,-25,[0,0,.12],physicsClientId=self.client)
        except Exception:
            self.close();raise

    def close(self):
        if not self.closed:
            p.disconnect(self.client);self.closed=True

    def __enter__(self):return self
    def __exit__(self,*_):self.close()

    def set_obstacle(self, box):
        """Set one axis-aligned box [XYZ, half-extents] in meters, or None."""
        if box is not None:
            box=np.asarray(box,float)
            if box.shape!=(6,) or not np.isfinite(box).all() or np.any(box[3:]<=0):
                raise ValueError('Box requires six finite values and positive half-extents')
        if self.box is not None:p.removeBody(self.box,physicsClientId=self.client)
        self.obstacle=None if box is None else box.copy();self.box=None;self._valid_cache.clear()
        if box is not None:
            shape=p.createCollisionShape(p.GEOM_BOX,halfExtents=box[3:].tolist(),physicsClientId=self.client)
            visual=p.createVisualShape(p.GEOM_BOX,halfExtents=box[3:].tolist(),rgbaColor=[.85,.28,.15,1],physicsClientId=self.client)
            self.box=p.createMultiBody(baseMass=0,baseCollisionShapeIndex=shape,baseVisualShapeIndex=visual,
                                      basePosition=box[:3].tolist(),physicsClientId=self.client)

    def reset(self,q):
        q=np.asarray(q,float)
        if q.shape!=(6,) or not np.isfinite(q).all() or np.any(q<ARM.lower) or np.any(q>ARM.upper):
            raise ValueError('Expected six finite joints within operating limits')
        self._set_joints(q);self._display_q=q.copy()

    def _set_joints(self,q):
        # All six joints are verified revolute at construction. Submit identical
        # scalar positions and zero velocities in one Bullet command; geometry,
        # sampled poses, and subsequent clearance queries stay unchanged.
        if self.batch_reset:
            p.resetJointStatesMultiDof(self.robot,self.joints,[[float(value)] for value in q],
                                      targetVelocities=[[0.]]*6,physicsClientId=self.client)
        else:
            for i,value in zip(self.joints,q):p.resetJointState(self.robot,i,float(value),physicsClientId=self.client)

    def current(self):
        return np.array([s[0] for s in p.getJointStates(self.robot,self.joints,physicsClientId=self.client)])

    def forward_kinematics(self,q):
        """Return shared URDF flange transform [4,4] in the base frame."""
        return ARM.transform(q)

    def bullet_transform(self,q):
        """Return Bullet's URDF link frame (not center of mass) [4,4]."""
        self.reset(q)
        state=p.getLinkState(self.robot,self.tcp,computeForwardKinematics=True,physicsClientId=self.client)
        result=np.eye(4);result[:3,3]=state[4]
        result[:3,:3]=np.asarray(p.getMatrixFromQuaternion(state[5])).reshape(3,3)
        return result

    def valid(self,q):
        """Operating bounds, native conservative self checks, Bullet scene clearance."""
        q=np.asarray(q,float)
        if q.shape!=(6,):return False
        key=q.tobytes()
        # Only finite six-joint states enter this scene-local, exact-byte cache.
        # A hit already proves finiteness; keep distinct nearby poses distinct.
        cached=self._valid_cache.get(key)
        if cached is not None:return cached
        if not np.isfinite(q).all():return False
        valid=self.space.valid(q)
        if valid:
            self._set_joints(q)
            try:
                for body in (self.table,self.box):
                    if body is not None and p.getClosestPoints(self.robot,body,self.clearance,physicsClientId=self.client):
                        valid=False;break
            finally:self._set_joints(self._display_q)
        if len(self._valid_cache)>200000:self._valid_cache.clear()
        self._valid_cache[key]=bool(valid)
        return bool(valid)

    def valid_many(self,states):
        """Validate identical sampled poses with batched self-collision geometry.

        Preserve the scene-local exact-byte cache and Bullet table/block
        clearance queries. Custom validity predicates retain scalar dispatch.
        """
        states=np.asarray(states,float)
        if states.ndim!=2 or states.shape[1]!=6:
            raise ValueError('Expected [N,6] joint states')
        checker=getattr(self.space,'collision_checker',None)
        if (checker is None or getattr(self.valid,'__func__',None) is not ArmEnv.valid
                or getattr(self.space.validity,'__self__',None) is not checker):
            return np.array([self.valid(q) for q in states],dtype=bool)
        result=np.zeros(len(states),dtype=bool)
        for start in range(0,len(states),256):
            block=states[start:start+256]
            missing={}
            for row,q in enumerate(block):
                if not np.isfinite(q).all():continue
                key=q.tobytes();cached=self._valid_cache.get(key)
                if cached is not None:result[start+row]=cached
                else:missing.setdefault(key,[]).append(row)
            if not missing:continue
            keys=list(missing)
            poses=np.array([block[missing[key][0]] for key in keys])
            accepted=np.all((poses>=self.space.lower)&(poses<=self.space.upper),axis=1)
            indices=np.flatnonzero(accepted)
            if len(indices):accepted[indices]=checker.valid_many(poses[indices])
            # Self-collision queries do not touch Bullet. Restore its displayed
            # execution state even if a subsequent clearance query fails.
            try:
                if self.table is not None or self.box is not None:
                    for index in np.flatnonzero(accepted):
                        self._set_joints(poses[index])
                        for body in (self.table,self.box):
                            if body is not None and p.getClosestPoints(self.robot,body,self.clearance,physicsClientId=self.client):
                                accepted[index]=False;break
            finally:
                if self.table is not None or self.box is not None:self._set_joints(self._display_q)
            for key,valid in zip(keys,accepted):
                if len(self._valid_cache)>200000:self._valid_cache.clear()
                self._valid_cache[key]=bool(valid)
                result[start+np.asarray(missing[key])]=valid
        return result

    def edge_valid(self,a,b,max_step_rad=.04):
        return self._edge_valid(a,b,max_step_rad,endpoints_checked=False)

    def _edge_valid(self,a,b,max_step_rad,*,endpoints_checked):
        """Check the original interpolation grid without repeated endpoints."""
        if not np.isfinite(max_step_rad) or max_step_rad<=0:
            raise ValueError('Interpolation step must be positive')
        a=np.asarray(a,float);b=np.asarray(b,float)
        if a.shape!=(6,) or b.shape!=(6,):
            return all(self.valid(q) for q in self.space.interpolate(a,b,max_step_rad))
        # Match JointSpace.interpolate's piece count exactly. For a short edge
        # its complete grid is just a and b: no array construction is required.
        delta=self.space.delta(a,b)
        pieces=max(1,int(np.ceil(np.max(np.abs(delta))/max_step_rad)))
        if pieces==1:
            return endpoints_checked or (self.valid(a) and self.valid(b))
        points=self.space.interpolate(a,b,max_step_rad)
        if endpoints_checked:points=points[1:-1]
        return bool(np.all(self.valid_many(points)))

    def path_valid(self,states,max_step_rad=.04):
        """Check every reference state [T,6] and its sampled connecting edges."""
        states=np.asarray(states,float)
        if states.ndim!=2 or states.shape[1]!=6 or not len(states) or not np.isfinite(states).all():return False
        if not np.isfinite(max_step_rad) or max_step_rad<=0:
            raise ValueError('Interpolation step must be positive')
        if not np.all(self.valid_many(states)):return False
        if len(states)==1:return True
        # Every endpoint is already checked. Identify edges with interior
        # samples together, using the exact JointSpace.interpolate piece count.
        # Dense C2 references normally have no interiors at all.
        delta=self.space.delta(states[:-1],states[1:])
        long_edges=np.flatnonzero(np.ceil(np.max(np.abs(delta),axis=1)/max_step_rad)>1)
        pending=[];count=0
        for index in long_edges:
            points=self.space.interpolate(states[index],states[index+1],max_step_rad)[1:-1]
            for start in range(0,len(points),256):
                block=points[start:start+256];pending.append(block);count+=len(block)
                if count>=256:
                    if not np.all(self.valid_many(np.concatenate(pending))):return False
                    pending=[];count=0
        return not pending or bool(np.all(self.valid_many(np.concatenate(pending))))

    def execute_step(self,q,target,dt=.1,speed=1.,max_step_rad=.04):
        """Validate and display a speed-bounded reference step; no physical IO."""
        q=np.asarray(q,float);target=np.asarray(target,float)
        if not np.isfinite(dt) or dt<=0 or not np.isfinite(speed) or speed<=0:raise ValueError('Positive dt/speed required')
        if not self.valid(q) or not self.valid(target):return False
        if np.linalg.norm(target-q)>speed*dt+1e-8:return False
        if not self.edge_valid(q,target,max_step_rad):return False
        for point in self.space.interpolate(q,target,max_step_rad):self.reset(point)
        return True

    def inverse_kinematics(self,target_pos,target_axis=None,current=None,seed=42,attempts=12,*,robust=False,learned=None,cancel_check=None):
        """Find a scene-valid 5D IK goal [6]; local flange Z spin remains free.

        robust=True enables a bounded numerical fallback after legacy exhaustion.
        learned=None follows MYCOBOT_IK_SEED (default off); learned starts replace
        random restarts and always pass numerical refinement and endpoint gates.
        Each call replaces ik_diagnostics; exhaustion never proves unreachability.
        Optional cancel_check raises to abandon an unfinished numerical search.
        """
        from .endpoint_ik import search_endpoint
        self.ik_diagnostics={'method':None,'legacy_attempts':0,'fallback_attempts':0,
            'fallback_nfev':0,'best_position_error_mm':None,'best_direction_error_deg':None,
            'collision_rejections':0,'elapsed_ms':0.}
        current=self.current() if current is None else np.asarray(current,float)
        target_pos=np.asarray(target_pos,float)
        if target_pos.shape!=(3,) or not np.isfinite(target_pos).all():
            raise ValueError('IK target requires three finite position values')
        if current.shape!=(6,) or not np.isfinite(current).all() or np.any(current<ARM.lower) or np.any(current>ARM.upper):
            raise ValueError('IK current requires six finite joints within operating limits')
        if isinstance(attempts,bool) or not isinstance(attempts,(int,np.integer)) or attempts<1:
            raise ValueError('IK attempts must be a positive integer')
        axis=ARM.transform(current)[:3,2] if target_axis is None else unit_axis(target_axis)
        return search_endpoint(self,target_pos,axis,current,seed,attempts,solve,robust=robust,learned=learned,
                               **({} if cancel_check is None else {'cancel_check': cancel_check}))

    def metadata(self):
        return {'robot':robot_schema(),'self_collision':self.space.metadata(),
                'scene_checker':'PyBullet convex URDF hulls versus table and one axis-aligned box',
                'clearance_mm':self.clearance*1000,'path_sample_rad':.04,
                'table_top_m':self.table_top,'pybullet_version':version('pybullet'),
                'execution':'kinematic reference interpolation; no calibrated dynamics or physical IO'}

    def render(self,path,width=960,height=720):
        """Save a CPU-rendered PNG of the actual Bullet scene/reference pose."""
        from PIL import Image
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        view=p.computeViewMatrixFromYawPitchRoll([0,0,.17],1.,40,-20,0,2)
        projection=p.computeProjectionMatrixFOV(50,width/height,.01,3.)
        result=p.getCameraImage(width,height,view,projection,renderer=p.ER_TINY_RENDERER,physicsClientId=self.client)
        pixels=np.asarray(result[2],dtype=np.uint8).reshape(height,width,4)
        Image.fromarray(pixels).save(path)


def verify(output,gui=False,seed=42):
    """Phase 1: numerical/Bullet FK parity and checked motion to a reachable goal."""
    rng=np.random.default_rng(seed)
    with ArmEnv(gui=gui) as env:
        errors=[]
        for q in [np.zeros(6),ARM.lower,ARM.upper]+[rng.uniform(ARM.lower,ARM.upper) for _ in range(12)]:
            shared=env.forward_kinematics(q);bullet=env.bullet_transform(q)
            errors.append((float(np.linalg.norm(shared[:3,3]-bullet[:3,3])),
                           axis_error(shared[:3,2],bullet[:3,2])[2]))
        assert max(e[0] for e in errors)<1e-6 and max(e[1] for e in errors)<.001
        start=np.deg2rad([0,-40,-130,80,0,50]);witness=ARM.clip(start+np.array([.05,.03,-.03,.02,.02,0]))
        env.set_obstacle([-.23,0,env.table_top+.08,.02,.02,.08])
        env.reset(start);target=ARM.transform(witness)
        goal=env.inverse_kinematics(target[:3,3],target[:3,2],start)
        assert env.edge_valid(start,goal)
        previous=start
        for q in env.space.interpolate(start,goal,.025)[1:]:
            assert env.execute_step(previous,q);previous=q
            if gui:time.sleep(.1)
        report={'phase':1,'seed':seed,'fk_samples':len(errors),'max_position_difference_m':max(e[0] for e in errors),
                'max_axis_difference_deg':max(e[1] for e in errors),'ik_goal_rad':goal.tolist(),
                'checked_motion':True,'metadata':env.metadata()}
    write_json(output,report);print(json.dumps(report,allow_nan=False));return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gui',action='store_true');parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output',type=Path,default=Path('output/diffusion/phase1.json'))
    args=parser.parse_args();verify(args.output,args.gui,args.seed)


if __name__=='__main__':main()
