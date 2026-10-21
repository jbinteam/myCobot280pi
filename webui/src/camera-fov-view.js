import * as THREE from 'three';
import {LineSegments2} from 'three/addons/lines/LineSegments2.js';
import {LineSegmentsGeometry} from 'three/addons/lines/LineSegmentsGeometry.js';
import {LineMaterial} from 'three/addons/lines/LineMaterial.js';

// Optical coordinates: X right, Y down, Z forward. Horizontal and vertical
// limits are independent; a circular cone would depict a different camera.
export function cameraFOVCorners(model, farM=.5) {
  const near=model.near_m;
  if(!Number.isFinite(farM)||farM<=near)throw new Error('FOV illustration must extend beyond the near plane');
  const tx=Math.tan(THREE.MathUtils.degToRad(model.horizontal_fov_deg/2));
  const ty=Math.tan(THREE.MathUtils.degToRad(model.vertical_fov_deg/2));
  return [near,farM].flatMap(z=>[[-z*tx,-z*ty,z],[z*tx,-z*ty,z],[z*tx,z*ty,z],[-z*tx,z*ty,z]]);
}

export function cameraFOVProjection(point, model) {
  const horizontal=THREE.MathUtils.radToDeg(Math.atan2(point.x,point.z));
  const vertical=THREE.MathUtils.radToDeg(Math.atan2(point.y,point.z));
  const margin=Math.min(model.horizontal_fov_deg/2-Math.abs(horizontal),model.vertical_fov_deg/2-Math.abs(vertical));
  return {horizontal,vertical,margin,inFrame:point.z>model.near_m&&margin>=model.angular_margin_deg};
}

export function createCameraFOVView({toggle,sizeOutput,objectOutput,farM=.5}) {
  const group=new THREE.Group();group.name='wrist-camera-fov';group.visible=false;
  const edges=new LineSegments2(new LineSegmentsGeometry().setPositions([]),
    new LineMaterial({color:'#d48800',linewidth:2.5,depthTest:false,depthWrite:false,toneMapped:false}));
  edges.name='wrist-camera-fov-edges';edges.renderOrder=22;edges.frustumCulled=false;group.add(edges);
  const volume=new THREE.Mesh(new THREE.BufferGeometry(),new THREE.MeshBasicMaterial({
    color:'#e6ac36',transparent:true,opacity:.045,side:THREE.DoubleSide,depthWrite:false}));
  volume.name='wrist-camera-fov-volume';group.add(volume);
  const origin=new THREE.Mesh(new THREE.SphereGeometry(.0035,12,8),new THREE.MeshBasicMaterial({color:'#d48800'}));
  origin.name='wrist-camera-optical-origin';group.add(origin);
  const rayGeometry=new THREE.BufferGeometry();rayGeometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(6),3));
  rayGeometry.setAttribute('lineDistance',new THREE.BufferAttribute(new Float32Array(2),1));
  const ray=new THREE.Line(rayGeometry,new THREE.LineDashedMaterial({color:'#d48800',dashSize:.008,gapSize:.005,depthTest:false}));
  ray.name='wrist-camera-object-ray';ray.visible=false;ray.renderOrder=23;group.add(ray);
  group.traverse(object=>object.userData.isFrameVisualization=true);
  let model=null;
  const visibility=()=>{group.visible=!!model&&toggle.checked;};
  toggle.addEventListener('change',visibility);
  return {group,configure(next){
    model=structuredClone(next);group.userData.camera=model;
    group.position.fromArray(model.eef_to_optical_translation_m);
    group.quaternion.fromArray(model.eef_to_optical_quaternion_xyzw);
    const corners=cameraFOVCorners(model,farM);
    const pairs=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
    edges.geometry.dispose();edges.geometry=new LineSegmentsGeometry().setPositions(pairs.flatMap(([a,b])=>[...corners[a],...corners[b]]));
    volume.geometry.dispose();volume.geometry=new THREE.BufferGeometry();
    volume.geometry.setAttribute('position',new THREE.Float32BufferAttribute(corners.flat(),3));
    volume.geometry.setIndex([0,1,2,0,2,3,4,6,5,4,7,6,0,4,5,0,5,1,1,5,6,1,6,2,2,6,7,2,7,3,3,7,4,3,4,0]);
    volume.geometry.computeVertexNormals();
    sizeOutput.textContent=`${model.horizontal_fov_deg}° H × ${model.vertical_fov_deg}° V · amber · drawn to ${farM*1000} mm`;
    visibility();
  },update(objectWorld=null){
    if(!model)return;
    ray.visible=!!objectWorld;
    if(!objectWorld){objectOutput.textContent='No active object';return;}
    group.updateWorldMatrix(true,false);
    const local=group.worldToLocal(objectWorld.clone());
    const result=cameraFOVProjection(local,model);
    const position=ray.geometry.attributes.position;position.setXYZ(0,0,0,0);position.setXYZ(1,local.x,local.y,local.z);
    position.needsUpdate=true;ray.geometry.computeBoundingSphere();ray.geometry.computeBoundingBox();
    ray.geometry.attributes.lineDistance.setX(1,local.length());ray.geometry.attributes.lineDistance.needsUpdate=true;
    ray.material.color.set(result.inFrame?'#a97700':'#c43838');
    objectOutput.textContent=`${result.inFrame?'Inside image bounds':'Outside image bounds'} · H ${result.horizontal.toFixed(1)}° · V ${result.vertical.toFixed(1)}° · margin ${result.margin.toFixed(1)}°`;
    return result;
  }};
}
