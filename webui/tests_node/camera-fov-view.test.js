import {test} from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {cameraFOVCorners,cameraFOVProjection,createCameraFOVView} from '../src/camera-fov-view.js';

const model={horizontal_fov_deg:82,vertical_fov_deg:68,angular_margin_deg:0,near_m:.005,
  eef_to_optical_translation_m:[.01,0,.02],eef_to_optical_quaternion_xyzw:[0,Math.SQRT1_2,0,Math.SQRT1_2]};

test('rectangular frustum uses independent FOV angles and optical +Z',()=>{
  const corners=cameraFOVCorners(model,.5);
  assert.equal(corners.length,8);
  assert.ok(Math.abs(Math.atan(corners[5][0]/corners[5][2])*180/Math.PI-41)<1e-12);
  assert.ok(Math.abs(Math.atan(-corners[5][1]/corners[5][2])*180/Math.PI-34)<1e-12);
  assert.equal(corners[0][2],.005);assert.equal(corners[4][2],.5);
  assert.equal(cameraFOVProjection(new THREE.Vector3(Math.tan(40*Math.PI/180),0,1),model).inFrame,true);
  assert.equal(cameraFOVProjection(new THREE.Vector3(0,Math.tan(35*Math.PI/180),1),model).inFrame,false);
  assert.equal(cameraFOVProjection(new THREE.Vector3(0,0,-1),model).inFrame,false);
  assert.equal(cameraFOVProjection(new THREE.Vector3(0,0,.004),model).inFrame,false);
});

test('FOV follows flange and optical mount; world object projection and toggle work',()=>{
  const flange=new THREE.Group();flange.position.set(.2,-.1,.3);flange.rotation.z=.6;
  const toggle=new EventTarget();toggle.checked=true;
  const sizeOutput={},objectOutput={};const view=createCameraFOVView({toggle,sizeOutput,objectOutput});
  flange.add(view.group);view.configure(model);
  const local=new THREE.Vector3(.01,.01,.27),world=view.group.localToWorld(local.clone());
  const result=view.update(world);
  assert.equal(result.inFrame,true);assert.match(sizeOutput.textContent,/82° H × 68° V/);
  const ray=view.group.getObjectByName('wrist-camera-object-ray');
  assert.ok(Math.abs(ray.geometry.attributes.position.getZ(1)-.27)<1e-6);
  flange.position.x+=1;assert.equal(view.update(world).inFrame,false);
  toggle.checked=false;toggle.dispatchEvent(new Event('change'));assert.equal(view.group.visible,false);
  view.update(null);assert.equal(ray.visible,false);assert.equal(objectOutput.textContent,'No active object');
});
