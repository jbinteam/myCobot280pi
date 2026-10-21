import {test} from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {randomApproachDemo} from '../src/approach-demo.js';
import {readPlanningStream} from '../src/planning-stream.js';
import {createRRTSearchView} from '../src/rrt-search-view.js';

test('new random rounds spread the base heading while preserving distance and stage settings',()=>{
  const demo={cursor_mm:[0,0,500],start_deg:Array(6).fill(0),distance_mm:350,half_angle_deg:30};
  const start={position:[0,0,150]};
  for(let i=0;i<100;i++){
    const values=[i/100,(99-i)/100,i/100],next=randomApproachDemo(demo,start,()=>values.shift());
    assert.equal(next.distance_mm,350);assert.equal(next.half_angle_deg,30);
    assert.ok(Math.abs(Math.hypot(...next.cursor_mm.map((v,j)=>v-start.position[j]))-350)<1e-9);
    const offset=Math.hypot(...next.cursor_mm.map((v,j)=>v-demo.cursor_mm[j]));
    assert.ok(offset>9&&offset<61);assert.deepEqual(demo.cursor_mm,[0,0,500]);
    assert.ok(Math.abs(next.start_deg[0])<=100);assert.deepEqual(demo.start_deg,Array(6).fill(0));
  }
});

test('planning stream handles split lines and final result, and rejects an incomplete stream',async()=>{
  const messages=[{type:'rrt',events:[{kind:'begin'}]},{type:'result',result:{playable:true}}];
  const encoded=new TextEncoder().encode(messages.map(JSON.stringify).join('\n')+'\n');
  const stream=new ReadableStream({start(controller){for(let i=0;i<encoded.length;i+=7)controller.enqueue(encoded.slice(i,i+7));controller.close();}});
  const events=[];const result=await readPlanningStream(new Response(stream),new AbortController().signal,e=>events.push(...e));
  assert.deepEqual(events,[{kind:'begin'}]);assert.equal(result.playable,true);
  await assert.rejects(readPlanningStream(new Response('{"type":"rrt","events":[]}\n'),new AbortController().signal,()=>{}),/without a checked result/);
  const abort=new AbortController();abort.abort();
  await assert.rejects(readPlanningStream(new Response(JSON.stringify(messages[0])+'\n'),abort.signal,()=>assert.fail('late search event')),{name:'AbortError'});
});

test('search display follows tree rewiring, direct candidates, accepted route and toggle',()=>{
  const scene=new THREE.Scene(),toggle=new EventTarget();toggle.checked=true;
  const messages=[];
  const view=createRRTSearchView(scene,q=>({position:q.slice(0,3)}),toggle,m=>messages.push(m));
  view.events([{kind:'begin',start_deg:[0,0,0]},
    {kind:'add',index:1,parent:0,q_deg:[100,0,0]},
    {kind:'add',index:2,parent:0,q_deg:[0,100,0]},
    {kind:'rewire',index:1,parent:2}]);view.update();
  const tree=scene.getObjectByName('approach-rrt-branches');
  assert.equal(tree.geometry.attributes.instanceStart.count,2);
  assert.ok(Math.abs(tree.geometry.attributes.instanceStart.getY(0)-.1)<1e-6);
  assert.equal(tree.isLineSegments2,true);
  // Fat-line screen resolution follows viewport changes, including resizing.
  tree.onBeforeRender({getViewport:viewport=>viewport.set(0,0,800,600)});
  assert.deepEqual(tree.material.resolution.toArray(),[800,600]);
  tree.onBeforeRender({getViewport:viewport=>viewport.set(0,0,400,300)});
  assert.deepEqual(tree.material.resolution.toArray(),[400,300]);
  assert.equal(scene.getObjectByName('approach-rrt-branches-outline').geometry,tree.geometry);
  view.accepted([{position:[0,0,0]},{position:[100,100,0]}]);
  assert.equal(scene.getObjectByName('approach-accepted-route').geometry.attributes.instanceStart.count,1);
  toggle.checked=false;toggle.dispatchEvent(new Event('change'));
  assert.equal(scene.getObjectByName('approach-rrt-search').visible,false);
  view.events([{kind:'begin',start_deg:[0,0,0]},
    {kind:'end',status:'direct_optimal',path_deg:[[0,0,0],[100,100,0]]}]);view.update();
  assert.equal(tree.geometry.attributes.instanceStart.count,1);assert.match(messages.at(-1),/no RRT\* expansion/);
  view.clear();assert.equal(tree.geometry.attributes.instanceStart.count,0);
});

test('accepted proposal restores its winning search tree after a later candidate search',()=>{
  const scene=new THREE.Scene(),toggle=new EventTarget();toggle.checked=true;
  const view=createRRTSearchView(scene,q=>({position:q.slice(0,3)}),toggle);
  for(const goal of [[100,0,0],[0,100,0]])view.events([
    {kind:'begin',start_deg:[0,0,0],goal_deg:goal},
    {kind:'add',index:1,parent:0,q_deg:goal},
    {kind:'end',status:'rrt_star',nodes:2,rewires:0}]);
  view.accepted([{position:[0,0,0]},{position:[100,0,0]}],[100,0,0]);view.update();
  const positions=scene.getObjectByName('approach-rrt-branches').geometry.attributes.instanceEnd;
  assert.ok(Math.abs(positions.getX(0)-.1)<1e-6);assert.equal(positions.getY(0),0);
});
