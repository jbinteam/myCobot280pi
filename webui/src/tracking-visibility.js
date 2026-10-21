import * as THREE from 'three';
import {cameraFOVProjection} from './camera-fov-view.js';

// Fresh FK includes roll, which is needed to check rectangular image bounds.
export function cameraProjectionForPose(pose,objectMm,model) {
  if(!model || !Array.isArray(pose.rotation) || pose.rotation.length!==3 ||
      pose.rotation.some(row=>!Array.isArray(row)||row.length!==3||row.some(v=>!Number.isFinite(v))))
    throw new Error('Camera-FOV arrival requires fresh full EEF rotation');
  const r=pose.rotation,p=pose.position.map(v=>v/1000);
  const eef=new THREE.Matrix4().set(r[0][0],r[0][1],r[0][2],p[0],r[1][0],r[1][1],r[1][2],p[1],
    r[2][0],r[2][1],r[2][2],p[2],0,0,0,1);
  const mount=new THREE.Matrix4().compose(new THREE.Vector3().fromArray(model.eef_to_optical_translation_m),
    new THREE.Quaternion().fromArray(model.eef_to_optical_quaternion_xyzw),new THREE.Vector3(1,1,1));
  const local=new THREE.Vector3().fromArray(objectMm).multiplyScalar(.001).applyMatrix4(eef.multiply(mount).invert());
  return cameraFOVProjection(local,model);
}

// Keep the demo target at its initial camera-centred position when the pointer
// first enters the viewport. Subsequent movement steers it on the same plane.
export function pointerOffset(pointer, target, camera) {
  const projected = target.clone().project(camera);
  return new THREE.Vector2(projected.x-pointer.x, projected.y-pointer.y);
}

export function visibilityHoldMessage(data) {
  if (data.status !== 'object_not_visible') return null;
  const view = data.visibility_start;
  if (view?.occluded) {
    const blocker = {table:'the table',robot:'the robot',block:'the obstacle'}[view.occluding_object] || 'scene geometry';
    return `Camera sightline blocked by ${blocker}. Move the target above/around it, or load the visible tracking demo in simulation.`;
  }
  if (view?.in_view === false) {
    if (view.depth_m <= 0) return 'Target is behind the wrist camera. Move it in front of tool Z, or restart the visible tracking demo.';
    const model = data.visibility_model?.camera;
    const size = model ? ` (${model.horizontal_fov_deg}° × ${model.vertical_fov_deg}°)` : '';
    return `Target is outside the wrist camera view${size}. Reverse the last mouse movement or reload the visible tracking demo; the orbiting 3D view is a different camera.`;
  }
  return 'Target is not visible from the wrist camera. Move it into view or load the visible tracking demo in simulation.';
}
