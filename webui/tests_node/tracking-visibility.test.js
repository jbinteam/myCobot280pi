import {test} from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {pointerOffset,visibilityHoldMessage,cameraProjectionForPose} from '../src/tracking-visibility.js';
import {validatedAimPath} from '../src/mouse-aim-path.js';

test('relaxed playback cannot substitute an unverified camera or hide a mode change',()=>{
  const camera={horizontal_fov_deg:82,vertical_fov_deg:68};
  const context={region:{},orientationMode:'camera_fov',camera};
  const result={orientation_mode:'camera_fov',visibility:{passed:true},modeled_visibility_checked:true,
    visibility_model:{camera:{...camera,vertical_fov_deg:80}},
    goal_cursor_acceptance:{orientation_mode:'camera_fov',camera_projection:{visible:true}}};
  assert.throws(()=>validatedAimPath(result,[],{},1,[],context),/checked matching camera/);
  assert.throws(()=>validatedAimPath({...result,orientation_mode:'point_at_object'},[],{},1,[],context),/orientation mode/);
  assert.throws(()=>validatedAimPath({...result,visibility_model:{camera},visibility:{passed:false}},[],{},1,[],context),/checked matching camera/);
});

test('fresh-pose camera bounds are rectangular and include the optical mount',()=>{
  const model={horizontal_fov_deg:82,vertical_fov_deg:68,angular_margin_deg:0,near_m:.005,
    eef_to_optical_translation_m:[0,0,0],eef_to_optical_quaternion_xyzw:[0,0,0,1]};
  const pose={position:[10,20,30],rotation:[[1,0,0],[0,1,0],[0,0,1]]};
  const object=[10+150*Math.tan(36*Math.PI/180),20+150*Math.tan(30*Math.PI/180),180];
  assert.equal(cameraProjectionForPose(pose,object,model).inFrame,true);
  const mounted={...model,eef_to_optical_quaternion_xyzw:[0,Math.SQRT1_2,0,Math.SQRT1_2]};
  assert.equal(cameraProjectionForPose(pose,[160,20,30],mounted).inFrame,true);
  assert.equal(cameraProjectionForPose(pose,[10,20,180],mounted).inFrame,false);
});

test('entering the viewport preserves the visible target; later pointer movement steers it',()=>{
  const camera=new THREE.PerspectiveCamera(50,1.5,.01,10);
  camera.position.set(.6,-.7,.5);camera.lookAt(0,0,.15);camera.updateMatrixWorld(true);
  const target=new THREE.Vector3(-.29,-.06,.26),pointer=new THREE.Vector2(-.98,.9);
  const offset=pointerOffset(pointer,target,camera);
  const plane=new THREE.Plane().setFromNormalAndCoplanarPoint(camera.getWorldDirection(new THREE.Vector3()),target);
  const ray=new THREE.Raycaster(),point=new THREE.Vector3();
  ray.setFromCamera(pointer.clone().add(offset),camera);ray.ray.intersectPlane(plane,point);
  assert.ok(point.distanceTo(target)<1e-10);
  ray.setFromCamera(pointer.clone().add(offset).add(new THREE.Vector2(.02,0)),camera);
  ray.ray.intersectPlane(plane,point);
  assert.ok(point.distanceTo(target)>.001);
});

test('hold messages distinguish occlusion, outside FOV and behind-camera targets',()=>{
  const message=visibility_start=>visibilityHoldMessage({status:'object_not_visible',visibility_start});
  assert.match(message({occluded:true,occluding_object:'table'}),/blocked by the table/);
  assert.match(message({occluded:true,occluding_object:'robot'}),/blocked by the robot/);
  assert.match(message({in_view:false,depth_m:.1}),/outside the wrist camera view/);
  assert.match(message({in_view:false,depth_m:-.1}),/behind the wrist camera/);
  assert.equal(visibilityHoldMessage({status:'no_path'}),null);
});
