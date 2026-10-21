import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as THREE from 'three';
import {initApproachSimulation} from '../src/approach-controls.js';
import {ApproachStages} from '../src/approach-simulation.js';

test('completed approach creates a fixed new object; repeat, pause, Stop and hardware guards work',async()=>{
  const elements=new Map();
  class Element extends EventTarget {click(){this.dispatchEvent(new Event('click'));}}
  for(const match of fs.readFileSync(new URL('../index.html',import.meta.url),'utf8').matchAll(/id="([^"]+)"/g))elements.set(match[1],new Element());
  for(const id of ['repeat-approach-demo','show-approach-rrt','show-trajectory-points'])elements.get(id).checked=true;
  const saved={document:globalThis.document,window:globalThis.window,fetch:globalThis.fetch};
  globalThis.document=new EventTarget();document.hidden=false;
  document.getElementById=id=>{assert.ok(elements.has(id),id);return elements.get(id);};
  globalThis.window=new EventTarget();
  const state={hardware_enabled:false,connected:false,token:'unit-test'};
  globalThis.fetch=async url=>{assert.equal(url,'/api/state');return Response.json(state);};
  try {
    const demo={start_deg:Array(6).fill(0),cursor_mm:[0,0,500],distance_mm:350,half_angle_deg:30};
    const info={tracking_demo:demo,classical:{status:'ready'},tracking_camera:{}},scene=new THREE.Scene();let q=demo.start_deg.slice(),startPosition=[0,0,150];
    const controls=initApproachSimulation({getPose:()=>q,getEEFPose:()=>({position:[0,0,150],axis:[0,0,1]}),setPose:values=>q=values.slice(),scene,stopMouseAim:()=>{},frame:()=>{}});
    await new Promise(resolve=>setImmediate(resolve));controls.configure(info);
    const starts=[];
    controls.runner.start=(next)=>{
      controls.runner.stop();starts.push(structuredClone(next));
      controls.runner.stages=new ApproachStages(next,info.tracking_camera);controls.runner.active=true;
      controls.runner.stages.capture({position:startPosition,axis:[0,0,1]});
      controls.runner.change('starting','Starting');
    };
    const complete=()=>{
      controls.runner.active=false;controls.runner.stages.completed=[350,330,310,290,270,250,230,210,190,170,150];
      controls.runner.stages.distance=150;controls.runner.change('holding','Approach complete.');
    };
    elements.get('start-approach-simulation').click();assert.deepEqual(starts[0].cursor_mm,demo.cursor_mm);
    assert.equal(starts[0].orientation_mode,'strict_then_fov');
    const object=scene.getObjectByName('approach-fixed-object');
    const first=object.position.toArray();complete();
    controls.update(performance.now()+1000);assert.deepEqual(object.position.toArray(),first);
    elements.get('pause-approach-simulation').click();controls.update(performance.now()+3000);
    assert.equal(starts.length,1);assert.equal(elements.get('resume-approach-simulation').disabled,false);
    elements.get('resume-approach-simulation').click();controls.update(performance.now()+3000);
    assert.equal(starts.length,2);assert.notDeepEqual(starts[1].cursor_mm,demo.cursor_mm);
    assert.deepEqual(controls.runner.stages.objectMm,starts[1].cursor_mm);
    assert.equal(controls.runner.stages.distance,350);assert.equal(q.length,6);
    assert.match(elements.get('approach-progress').textContent,/Round 2/);
    complete();elements.get('repeat-approach-demo').checked=false;
    elements.get('repeat-approach-demo').dispatchEvent(new Event('change'));
    controls.update(performance.now()+3000);assert.equal(starts.length,2);
    elements.get('repeat-approach-demo').checked=true;elements.get('repeat-approach-demo').dispatchEvent(new Event('change'));
    elements.get('stop-approach-simulation').click();controls.update(performance.now()+3000);
    assert.equal(starts.length,2);assert.equal(object.visible,false);
    elements.get('start-approach-simulation').click();complete();
    const fixture=JSON.parse(fs.readFileSync(new URL('../../simulation/diffusion/tests/fixtures/ik_failed_reachable_inward.json',import.meta.url)));
    const caseDemo={start_deg:fixture.approach.initial_request.start_deg,cursor_mm:fixture.request.cursor_mm,distance_mm:350,half_angle_deg:30,orientation_mode:'point_at_object'};
    startPosition=fixture.approach.initial_feedback.position;
    controls.startCase(caseDemo);
    assert.deepEqual(starts.at(-1),caseDemo);assert.deepEqual(q,caseDemo.start_deg);
    assert.equal(elements.get('repeat-approach-demo').checked,false);
    complete();controls.update(performance.now()+3000);
    assert.equal(starts.length,4);assert.equal(object.visible,true);
    const event=new Event('robot-state');event.detail={...state,hardware_enabled:true};window.dispatchEvent(event);
    controls.update(performance.now()+3000);assert.equal(starts.length,4);
    assert.equal(elements.get('approach-simulation-controls').hidden,true);
    assert.throws(()=>controls.startCase(caseDemo),/disconnected simulation mode/);
  } finally {Object.assign(globalThis,saved);}
});
