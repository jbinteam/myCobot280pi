import * as THREE from 'three';

export function coneDiskRegion(cursor, axis, settings) {
  const direction=new THREE.Vector3().fromArray(axis).normalize();
  const distance=settings.distance_mm, radius=distance*Math.tan(settings.half_angle_deg*Math.PI/180);
  return {mode:'cone_disk',distance_mm:distance,half_angle_deg:settings.half_angle_deg,
    cursor_mm:cursor.toArray().map(v=>v*1000),axis:direction.toArray(),
    center_mm:cursor.clone().addScaledVector(direction,-distance/1000).toArray().map(v=>v*1000),radius_mm:radius};
}

export function regionMatches(expected, actual) {
  if(!actual||actual.mode!=='cone_disk')return false;
  return ['distance_mm','half_angle_deg','radius_mm'].every(key=>Number.isFinite(actual[key])&&Math.abs(actual[key]-expected[key])<1e-6)
    && ['axis','center_mm','cursor_mm'].every(key=>Array.isArray(actual[key])&&actual[key].length===3
      && actual[key].every((v,i)=>Number.isFinite(v)&&Math.abs(v-expected[key][i])<1e-6));
}

export function regionPositionMetrics(position, region) {
  const axis=new THREE.Vector3().fromArray(region.axis),delta=position.clone().multiplyScalar(1000).sub(new THREE.Vector3().fromArray(region.center_mm));
  const axial=delta.dot(axis),radial=delta.clone().addScaledVector(axis,-axial).length();
  return {axialErrorMm:Math.abs(axial),radialMm:radial};
}

export function createTrackingRegionView() {
  const group=new THREE.Group();group.name='mouse-aim-tracking-region';group.visible=false;
  const cone=new THREE.Mesh(new THREE.ConeGeometry(1,1,48,1,true),new THREE.MeshBasicMaterial({color:'#167fbd',transparent:true,opacity:.10,side:THREE.DoubleSide,depthWrite:false}));
  cone.name='mouse-aim-cone';
  const disk=new THREE.Mesh(new THREE.CircleGeometry(1,64),new THREE.MeshBasicMaterial({color:'#17a787',transparent:true,opacity:.19,side:THREE.DoubleSide,depthWrite:false}));
  disk.name='mouse-aim-acceptance-disk';
  const ring=new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(Array.from({length:64},(_,i)=>new THREE.Vector3(Math.cos(i*2*Math.PI/64),Math.sin(i*2*Math.PI/64),0))),new THREE.LineBasicMaterial({color:'#17a787',depthTest:false}));
  ring.name='mouse-aim-acceptance-ring';
  group.add(cone,disk,ring);
  return {group,show(region){
    const axis=new THREE.Vector3().fromArray(region.axis),center=new THREE.Vector3().fromArray(region.center_mm).multiplyScalar(.001),apex=new THREE.Vector3().fromArray(region.cursor_mm).multiplyScalar(.001);
    group.visible=true;group.userData.region=structuredClone(region);
    cone.position.copy(apex).addScaledVector(axis,-region.distance_mm/2000);
    cone.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),axis);
    cone.scale.set(region.radius_mm/1000,region.distance_mm/1000,region.radius_mm/1000);
    disk.position.copy(center);ring.position.copy(center);
    disk.quaternion.setFromUnitVectors(new THREE.Vector3(0,0,1),axis);ring.quaternion.copy(disk.quaternion);
    disk.scale.setScalar(region.radius_mm/1000);ring.scale.copy(disk.scale);
  },clear(){group.visible=false;delete group.userData.region;}};
}
