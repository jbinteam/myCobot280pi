import {test} from 'node:test';
import assert from 'node:assert/strict';
import {effectiveOrientationMode} from '../src/orientation-policy.js';
import {ApproachStages} from '../src/approach-simulation.js';

const camera={horizontal_fov_deg:82,vertical_fov_deg:68,angular_margin_deg:0,near_m:.005,
  eef_to_optical_translation_m:[0,0,0],eef_to_optical_quaternion_xyzw:[0,0,0,1]};
const policy=(mode,fallback,status='ik_failed')=>({orientation_mode:mode,
  requested_orientation_mode:'strict_then_fov',orientation_selection:{policy:'strict_then_fov',
    selected_mode:mode,fallback_used:fallback,
    strict:{mode:'point_at_object',status:fallback?status:'target_reached',playable:!fallback},
    relaxed:fallback?{mode:'camera_fov',status:'target_reached',playable:true}:null}});
test('hybrid accepts only explicit strict selection or fallback after failed search',()=>{
  assert.equal(effectiveOrientationMode(policy('point_at_object',false),'strict_then_fov'),'point_at_object');
  assert.equal(effectiveOrientationMode(policy('camera_fov',true),'strict_then_fov'),'camera_fov');
  assert.throws(()=>effectiveOrientationMode({orientation_mode:'camera_fov'},'strict_then_fov'),/strict-first/);
  for(const status of ['object_not_visible','invalid_start','target_reached'])
    assert.throws(()=>effectiveOrientationMode(policy('camera_fov',true,status),'strict_then_fov'),/failed strict/);
  const forged=policy('camera_fov',true);forged.orientation_selection.strict.playable=true;
  assert.throws(()=>effectiveOrientationMode(forged,'strict_then_fov'),/failed strict/);
  assert.throws(()=>effectiveOrientationMode(policy('camera_fov',true),'point_at_object'),/orientation mode/);
});
test('hybrid is the default approach policy and fresh arrival follows the selected solver',()=>{
  const stages=new ApproachStages({cursor_mm:[0,0,300],distance_mm:350,half_angle_deg:30},camera);
  assert.equal(stages.orientationMode,'strict_then_fov');
  const pose=(z,angle)=>{const s=Math.sin(angle*Math.PI/180),c=Math.cos(angle*Math.PI/180);
    return {position:[0,0,z],axis:[0,-s,c],rotation:[[1,0,0],[0,c,-s],[0,s,c]]};};
  stages.capture(pose(130,0));
  const strict={...policy('point_at_object',false),converged:true,replan_required:false,visibility:{passed:true}};
  assert.throws(()=>stages.arrived(pose(130,20),strict),/not reached/);
  assert.equal(stages.arrived(pose(130,0),strict),false);
  stages.capture(pose(130,0));
  const relaxed={...policy('camera_fov',true),converged:true,replan_required:false,visibility:{passed:true}};
  assert.throws(()=>stages.arrived(pose(150,35),relaxed),/not reached/);
  assert.throws(()=>stages.arrived(pose(150,20),{...relaxed,visibility:{passed:false}}),/not reached/);
  assert.equal(stages.arrived(pose(150,20),relaxed),true);
  assert.deepEqual(stages.completed,[170,150]);
});
