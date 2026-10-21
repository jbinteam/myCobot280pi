// Explicit full-process probe against an already-running simulation-only API.
// Runs the same approach controller and sampler used by the WebUI, without a GUI.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import * as THREE from 'three';
import {ApproachSimulation} from '../src/approach-simulation.js';
import {randomApproachDemo} from '../src/approach-demo.js';
import {readPlanningStream} from '../src/planning-stream.js';

const output=process.argv[2];
if(!output)throw new Error('Supply a new output directory');
fs.mkdirSync(output,{recursive:true});
const tracePath=path.join(output,'approach_trace.json');
if(fs.existsSync(tracePath))throw new Error('Preserve the existing trace; choose another directory');
const base='http://127.0.0.1:8000';
const root=fileURLToPath(new URL('../../',import.meta.url));
const chain=JSON.parse(execFileSync(path.join(root,'.venv/bin/python'),['-c',
  'import json;from simulation.kinematics import ARM;print(json.dumps([{"origin":o.tolist(),"axis":None if a is None else a.tolist()} for o,a in ARM.chain]))'],
  {cwd:root,encoding:'utf8',env:{...process.env,OPENBLAS_NUM_THREADS:'1'}}));
function rotationFor(q) {
  const t=new THREE.Matrix4();let index=0;
  for(const joint of chain) {
    t.multiply(new THREE.Matrix4().set(...joint.origin.flat()));
    if(joint.axis)t.multiply(new THREE.Matrix4().makeRotationAxis(new THREE.Vector3().fromArray(joint.axis).normalize(),q[index++]*Math.PI/180));
  }
  const m=t.elements;return [[m[0],m[4],m[8]],[m[1],m[5],m[9]],[m[2],m[6],m[10]]];
}
const state=await fetch(base+'/api/state').then(r=>r.json());
assert.equal(state.hardware_enabled,false);assert.equal(state.connected,false);
const info=await fetch(base+'/api/sim/info').then(r=>r.json());let demo=info.tracking_demo;
let feedback={q:demo.start_deg.slice(),position:[],axis:[]},latest=null;
const plans=[],events=[],searchEvents=[];
// The first proposal's frames include exact FK at the requested start.
// Obtain this initial FK from the previously verified demo's centered proposal.
const initial=await fetch(base+'/api/sim/mouse-aim',{method:'POST',headers:{
  'Content-Type':'application/json','X-Robot-Token':state.token},body:JSON.stringify({
    tracking_engine:'learned_rrt',start_deg:feedback.q,cursor_mm:demo.cursor_mm,
    tracking_region:{mode:'cone_disk',distance_mm:demo.distance_mm,half_angle_deg:demo.half_angle_deg},
    obstacle_mm:null,continuous_motion:true,speed_rad_s:1})}).then(r=>r.json());
assert.equal(initial.playable,true);
feedback={q:initial.frames[0].q,position:initial.frames[0].position,axis:initial.frames[0].tool_axis};
if(process.argv.includes('--random')){
  const values=[.23,.65,.85];demo=randomApproachDemo(demo,feedback,()=>values.shift());
  // The tested URDF's first joint rotates about g_base Z through XY=0.
  // Its exact FK rotates rigidly with this base heading; later feedback is read
  // from the actual planner's checked endpoint frames as before.
  const yaw=THREE.MathUtils.degToRad(demo.start_deg[0]-feedback.q[0]),z=new THREE.Vector3(0,0,1);
  feedback={q:demo.start_deg.slice(),position:new THREE.Vector3().fromArray(feedback.position).applyAxisAngle(z,yaw).toArray(),
    axis:new THREE.Vector3().fromArray(feedback.axis).applyAxisAngle(z,yaw).toArray()};
  assert.notDeepEqual(demo.cursor_mm,info.tracking_demo.cursor_mm);
}
const streamed=process.argv.includes('--stream');
feedback.rotation=rotationFor(feedback.q);
const runner=new ApproachSimulation({authorized:()=>true,feedback:()=>structuredClone(feedback),
  plan:async(request,signal)=>{
    const response=await fetch(base+(streamed?'/api/sim/approach-plan':'/api/sim/mouse-aim'),{method:'POST',signal,
      headers:{'Content-Type':'application/json','X-Robot-Token':state.token},body:JSON.stringify(request)});
    const result=streamed?await readPlanningStream(response,signal,rows=>searchEvents.push(...rows)):await response.json();
    assert.equal(response.ok,true,result.error);
    plans.push({request:structuredClone(request),result:structuredClone(result)});
    return result;
  },
  onPlan:result=>latest=result,
  execute:async(sampler,signal)=>{
    if(signal.aborted)throw new DOMException('Stopped','AbortError');
    // Replay the same analytic sampler; check its end agrees with the checked frame.
    for(const frame of latest.frames)assert.ok(sampler.sample(frame.time).q.every((q,j)=>Math.abs(q-frame.q[j])<1e-6));
    const end=latest.frames.at(-1);
    feedback={q:sampler.sample(sampler.endTime).q,position:end.position,axis:end.tool_axis};
    feedback.rotation=rotationFor(feedback.q);
  },
  onChange:controller=>{
    const event={phase:controller.phase,distance_mm:controller.stages?.distance,message:controller.message};
    events.push(event);console.log(JSON.stringify(event));
  }});
const timeout=setTimeout(()=>runner.stop(),300000);
try {
  await runner.start(demo,info);
  assert.equal(runner.phase,'holding',runner.message);
  assert.deepEqual(runner.stages.completed,[350,330,310,290,270,250,230,210,190,170,150]);
  assert.ok(plans.every(p=>JSON.stringify(p.request.cursor_mm)===JSON.stringify(demo.cursor_mm)));
  if(streamed){assert.ok(searchEvents.some(e=>e.kind==='begin'));assert.ok(searchEvents.some(e=>e.kind==='end'));}
  const final=await fetch(base+'/api/state').then(r=>r.json());
  assert.equal(final.hardware_enabled,false);assert.equal(final.connected,false);
} finally {
  clearTimeout(timeout);
  fs.writeFileSync(tracePath,JSON.stringify({physical_motion:false,source:'WebUI approach controller + analytic preview sampler',
    camera:info.tracking_camera,demo,streamed,search_events:searchEvents,plans,events,completed_stages:runner.stages.completed,
    final_status:{state:runner.phase==='holding'?'completed':runner.phase,
      samples:[{q:feedback.q}],tracking_region:latest?.tracking_region}},null,2)+'\n');
}
