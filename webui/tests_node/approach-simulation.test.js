import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {ApproachStages,ApproachSimulation} from '../src/approach-simulation.js';

const demo={cursor_mm:[0,0,500],distance_mm:350,half_angle_deg:30,orientation_mode:'point_at_object'};
const feedback=z=>({q:Array(6).fill(0),position:[0,0,z],axis:[0,0,1]});
const done={converged:true,replan_required:false};

test('relaxed arrival requires fresh camera projection and checked visibility, not exact pointing',()=>{
  const camera={horizontal_fov_deg:82,vertical_fov_deg:68,angular_margin_deg:0,near_m:.005,
    eef_to_optical_translation_m:[0,0,0],eef_to_optical_quaternion_xyzw:[0,0,0,1]};
  const pose=(z,angle)=>{const s=Math.sin(angle*Math.PI/180),c=Math.cos(angle*Math.PI/180);
    return {position:[0,0,z],axis:[0,-s,c],rotation:[[1,0,0],[0,c,-s],[0,s,c]]};};
  const stages=new ApproachStages({cursor_mm:[0,0,300],distance_mm:350,half_angle_deg:30,orientation_mode:'camera_fov'},camera);
  const accepted={...done,orientation_mode:'camera_fov',visibility:{passed:true}};
  stages.capture(pose(130,20));assert.equal(stages.distance,170);
  assert.throws(()=>stages.arrived(pose(130,20),{...accepted,visibility:{passed:false}}),/not reached/);
  assert.equal(stages.arrived(pose(130,20),accepted),false);stages.capture(pose(130,20));
  assert.throws(()=>stages.arrived(pose(150,35),accepted),/not reached/);
  assert.throws(()=>stages.arrived({position:[0,0,150],axis:[0,0,1]},accepted),/fresh full EEF rotation/);
  assert.equal(stages.arrived(pose(150,20),accepted),true);assert.deepEqual(stages.completed,[170,150]);
});

test('replacement fixture reaches the actual 150 mm final stage using its independent witness',()=>{
  const fixture=JSON.parse(fs.readFileSync(new URL('../../simulation/diffusion/tests/fixtures/ik_failed_reachable_150.json',import.meta.url)));
  const stages=new ApproachStages({cursor_mm:fixture.request.cursor_mm,distance_mm:350,half_angle_deg:30,orientation_mode:'point_at_object'});
  const initial=fixture.approach.initial_feedback;
  assert.equal(stages.capture(initial).distance_mm,170);
  assert.equal(stages.arrived(initial,done),false);
  const region=stages.capture(initial);assert.equal(region.distance_mm,150);
  region.axis.forEach((v,i)=>assert.ok(Math.abs(v-fixture.request.target_axis[i])<1e-12));
  assert.equal(stages.arrived({q:fixture.witness.q_deg,position:fixture.final_stage_proof.position_mm,
    axis:fixture.final_stage_proof.tool_axis},done),true);
  assert.deepEqual(stages.completed,[170,150]);assert.equal(stages.stageCount,2);
});

test('reachable IK failure fixture follows the corrected 270 to 250 mm stage sequence',()=>{
  const fixture=JSON.parse(fs.readFileSync(new URL('../../simulation/diffusion/tests/fixtures/ik_failed_reachable_inward.json',import.meta.url)));
  const stages=new ApproachStages({cursor_mm:fixture.request.cursor_mm,distance_mm:350,half_angle_deg:30,orientation_mode:'point_at_object'});
  const feedback=fixture.approach.initial_feedback;
  assert.equal(stages.capture(feedback).distance_mm,270);
  assert.equal(stages.arrived(feedback,done),false);
  const region=stages.capture(feedback);
  assert.equal(region.distance_mm,fixture.request.tracking_region.distance_mm);
  assert.deepEqual(stages.completed,[270]);assert.equal(stages.seed,fixture.request.seed);
  region.axis.forEach((v,i)=>assert.ok(Math.abs(v-fixture.request.target_axis[i])<1e-12));
  stages.rejected('ik_failed');
  assert.equal(stages.capture(feedback),region);assert.equal(stages.distance,250);assert.equal(stages.seed,43);
});

test('fixed object, eleven bounded stages and actual arrival gate',()=>{
  const stages=new ApproachStages(demo);
  let pose=feedback(150);
  for(const distance of [350,330,310,290,270,250,230,210,190,170,150]) {
    const region=stages.capture(pose);
    assert.equal(stages.distance,distance);
    assert.deepEqual(region.cursor_mm,demo.cursor_mm);
    // A prefix never advances the distance; a wrong preview pose is rejected.
    assert.equal(stages.arrived(pose,{converged:false,replan_required:true}),false);
    if(distance<350)assert.throws(()=>stages.arrived(pose,done),/not reached/);
    pose=feedback(500-distance);
    assert.equal(stages.arrived(pose,done),distance===150);
  }
  assert.deepEqual(stages.completed,[350,330,310,290,270,250,230,210,190,170,150]);
  assert.deepEqual(stages.objectMm,demo.cursor_mm);
  assert.equal(stages.stageCount,11);
});

test('a closer start captures its own disk and only requests inward stages',()=>{
  const stages=new ApproachStages(demo),pose=feedback(260);
  const region=stages.capture(pose);
  assert.equal(stages.initialDistance,240);assert.equal(stages.distance,240);
  assert.deepEqual(region.center_mm,pose.position);assert.equal(stages.stageCount,6);
  const distances=[];
  for(let distance=240;;distance=Math.max(150,distance-20)) {
    distances.push(stages.capture(feedback(500-distance)).distance_mm);
    if(stages.arrived(feedback(500-distance),done))break;
  }
  assert.deepEqual(distances,[240,220,200,180,160,150]);
});

test('a start inside the finish distance holds its measured disk without an outward stage',()=>{
  const stages=new ApproachStages(demo),pose=feedback(371.0950291918665);
  const region=stages.capture(pose);
  assert.ok(Math.abs(region.distance_mm-128.9049708081335)<1e-6);
  assert.equal(stages.finalDistance,stages.initialDistance);assert.equal(stages.stageCount,1);
  assert.equal(stages.arrived(pose,done),true);
  assert.equal(stages.completed.length,1);
});

test('fresh feedback determines the actual first request, and retries keep that disk',async()=>{
  const poses=[feedback(260),feedback(250)],requests=[];
  const runner=new ApproachSimulation({authorized:()=>true,feedback:()=>poses[Math.min(requests.length,1)],
    plan:body=>{
      requests.push(body);
      if(requests.length===2)runner.stop();
      return {playable:false,status:'ik_failed'};
    },execute:()=>assert.fail('Failed search must hold')});
  await runner.start(demo,{});
  assert.deepEqual(requests.map(r=>r.tracking_region.distance_mm),[240,240]);
  assert.deepEqual(requests.map(r=>r.seed),[42,43]);
});

test('retry preserves the region; pause/resume reanchors it from fresh feedback',()=>{
  const stages=new ApproachStages(demo),old=stages.capture(feedback(150));
  stages.rejected('no_path');assert.equal(stages.seed,43);
  assert.equal(stages.capture({position:[20,0,150]}),old);
  stages.reanchor();assert.equal(stages.seed,42);
  assert.notDeepEqual(stages.capture({position:[20,0,150]}).axis,old.axis);
  assert.equal(stages.distance,350);
});

test('Stop invalidates an in-flight proposal before it can execute',async()=>{
  let release,executions=0;
  const waiting=new Promise(resolve=>release=resolve);
  const runner=new ApproachSimulation({authorized:()=>true,feedback:()=>feedback(150),
    plan:()=>waiting,execute:()=>{executions++;}});
  const task=runner.start(demo,{});
  await new Promise(resolve=>setImmediate(resolve));
  runner.stop();release({playable:true});await task;
  assert.equal(executions,0);assert.equal(runner.phase,'stopped');
});

test('visibility loss pauses at the current stage and hardware mode cannot start',async()=>{
  const runner=new ApproachSimulation({authorized:()=>true,feedback:()=>feedback(150),
    plan:()=>({playable:false,status:'object_not_visible',visibility_start:{in_view:false,depth_m:.2}}),
    execute:()=>assert.fail('Invisible proposal executed')});
  await runner.start(demo,{});
  assert.equal(runner.phase,'paused');assert.equal(runner.paused,true);assert.equal(runner.stages.distance,350);
  assert.equal(runner.stages.region,null);
  const hardware=new ApproachSimulation({authorized:()=>false});
  assert.throws(()=>hardware.start(demo,{}),/disconnected simulation/);
});
