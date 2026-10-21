import * as THREE from 'three';

// Spread rounds across a 200° arc by rotating the tested pose and object around
// the URDF's joint-1 axis (g_base Z through XY=0), plus a small local variation.
// The optical mount stays unchanged. Every route still passes the usual checks.
export function randomApproachDemo(demo,startPose,random=Math.random) {
  const origin=new THREE.Vector3().fromArray(startPose.position);
  const center=new THREE.Vector3().fromArray(demo.cursor_mm);
  const direction=center.clone().sub(origin),distance=direction.length();
  direction.normalize();
  const reference=Math.abs(direction.z)<.9?new THREE.Vector3(0,0,1):new THREE.Vector3(1,0,0);
  const right=new THREE.Vector3().crossVectors(direction,reference).normalize();
  const up=new THREE.Vector3().crossVectors(right,direction).normalize();
  const angle=2*Math.PI*random(),radius=Math.sqrt(100+3500*random()); // 10–60 mm local variation
  const yawDeg=200*(random()-.5),yaw=THREE.MathUtils.degToRad(yawDeg);
  const offset=direction.clone().multiplyScalar(distance)
    .addScaledVector(right,radius*Math.cos(angle)).addScaledVector(up,radius*Math.sin(angle))
    .normalize().multiplyScalar(distance);
  const start=demo.start_deg.slice();start[0]+=yawDeg;
  const object=origin.add(offset).applyAxisAngle(new THREE.Vector3(0,0,1),yaw);
  return {...demo,start_deg:start,cursor_mm:object.toArray()};
}
