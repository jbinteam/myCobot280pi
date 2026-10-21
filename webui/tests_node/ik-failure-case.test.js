import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as THREE from 'three';
import {createServer} from 'vite';

for (const [caseKey,filename] of [
  ['ik-failed-reachable','ik_failed_reachable.json'],
  ['ik-failed-reachable-inward','ik_failed_reachable_inward.json'],
  ['ik-failed-reachable-150','ik_failed_reachable_150.json'],
]) test(`${caseKey}: inspection holds failed plans, previews the saved route, and cancels on mode change`, async () => {
  const vite = await createServer({server:{middlewareMode:true,hmr:false}});
  const saved = Object.fromEntries(['document','window','location','fetch'].map(key => [key,globalThis[key]]));
  try {
    const {initIKFailureCase} = await vite.ssrLoadModule('/src/ik-failure-case.js');
    const fixture = JSON.parse(fs.readFileSync(new URL(`../../simulation/diffusion/tests/fixtures/${filename}`,import.meta.url)));
    const elements = new Map();
    class Element extends EventTarget {click(){this.dispatchEvent(new Event('click'));}}
    for (const match of fs.readFileSync(new URL('../index.html',import.meta.url),'utf8').matchAll(/id="([^"]+)"/g)) elements.set(match[1],new Element());
    globalThis.document = new EventTarget();document.getElementById = id => {assert.ok(elements.has(id),id);return elements.get(id);};
    const classes = new Set();document.body = {classList:{add:key=>classes.add(key),remove:key=>classes.delete(key)}};
    let navigation;
    globalThis.window = new EventTarget();globalThis.location = {search:`?case=${caseKey}`,assign:url=>navigation=url};
    const state = {hardware_enabled:false,connected:false,token:'unit-test'};
    const result = {status:'ik_failed',region_candidates_filtered:fixture.expected_baseline.candidates_filtered,
      region_candidate_jobs_tested:fixture.expected_baseline.jobs_started,region_route_attempts:[],
      checkpoint_sha256:fixture.checkpoint_sha256,environment:fixture.environment,visibility_model:{camera:fixture.camera}};
    let posted, delay = false, signal;
    globalThis.fetch = async (url, options) => {
      if (url === '/api/state') return Response.json(state);
      assert.equal(url,'/api/sim/mouse-aim');posted = JSON.parse(options.body);signal = options.signal;
      if (delay) return new Promise((resolve,reject) => signal.addEventListener('abort',()=>reject(new DOMException('Aborted','AbortError'))));
      return Response.json(result);
    };
    elements.get('case-show-route').checked = true;
    const scene = new THREE.Scene();let pose, stops = 0, framed,approachDemo;
    initIKFailureCase({scene,setPose:q=>pose=q.slice(),getEEFPose:q=>({position:q.slice(0,3),axis:[0,0,1]}),
      currentCamera:{...fixture.camera,horizontal_fov_deg:82,vertical_fov_deg:68},
      startApproach:demo=>approachDemo=demo,
      stopTracking:()=>stops++,frame:(view,points)=>framed=points});
    const tick = () => new Promise(resolve=>setImmediate(resolve));await tick();
    assert.deepEqual(pose,fixture.request.start_deg);assert.equal(stops,1);assert.ok(framed.length>=9);
    assert.equal(scene.getObjectByName('ik-failure-region').visible,true);
    assert.equal(scene.getObjectByName('ik-failure-region').userData.region.distance_mm,fixture.request.tracking_region.distance_mm);
    assert.equal(scene.getObjectByName('ik-failure-saved-route').visible,true);
    assert.equal(scene.getObjectByName('ik-failure-initial-disk').visible,!!fixture.approach);
    assert.equal(elements.get('case-selector').value,caseKey);
    assert.match(elements.get('case-camera').textContent,/current camera: 82° × 68°/);
    if(fixture.camera.horizontal_fov_deg!==82 || fixture.camera.vertical_fov_deg!==68)
      assert.match(elements.get('case-camera').textContent,fixture.camera_revalidation.route_visibility_audit.passed
        ? /visible throughout under the current FOV/ : /loses the object under the current FOV/);
    assert.ok(elements.get('case-status').textContent.includes(`${fixture.expected_baseline.jobs_started} exhausted IK jobs`));
    if (fixture.approach) {
      assert.equal(scene.getObjectByName('ik-failure-initial-disk').userData.region.distance_mm,fixture.approach.measured_initial_distance_mm);
      assert.ok(elements.get('case-title').textContent.includes(`${fixture.approach.measured_initial_distance_mm} → ${fixture.request.tracking_region.distance_mm}`));
      assert.match(elements.get('case-description').textContent,fixture.final_stage_proof
        ? /150 mm finish is independently checked/ : /150 mm finish is outside the robot reach bound/);
      assert.match(elements.get('case-distance-note').textContent,fixture.final_stage_proof ? /173.2 mm/ : /288.1 mm/);
    }
    elements.get('case-witness').click();
    pose.forEach((angle,i)=>assert.ok(Math.abs(angle-fixture.witness.q_deg[i])<1e-10));
    assert.equal(elements.get('case-pose-label').textContent,`Reachable witness: candidate ${fixture.witness.candidate_index+1}`);
    assert.equal(elements.get('case-run-approach').hidden,!fixture.approach);
    if(fixture.approach) {
      elements.get('case-run-approach').click();
      assert.deepEqual(approachDemo,{start_deg:fixture.approach.initial_request.start_deg,
        cursor_mm:fixture.request.cursor_mm,distance_mm:350,half_angle_deg:30,orientation_mode:'strict_then_fov'});
      assert.equal(scene.getObjectByName('ik-failure-region').visible,false);
      assert.equal(scene.getObjectByName('ik-failure-saved-route').visible,false);
      assert.equal(classes.has('inspecting-ik-case'),false);
      elements.get('load-ik-failure-case').click();
      elements.get('case-relax-orientation').checked=false;
      elements.get('case-run-approach').click();assert.equal(approachDemo.orientation_mode,'point_at_object');
      elements.get('load-ik-failure-case').click();
    }
    elements.get('case-reproduce').click();await tick();
    assert.deepEqual(posted,{...fixture.request,tracking_engine:'learned_rrt'});
    assert.deepEqual(pose,fixture.request.start_deg);
    assert.match(elements.get('case-status').textContent,/Fresh result: ik_failed/);
    // Even a future successful result remains a diagnostic, never a motion command.
    result.status = 'success';result.frames = [{q:Array(6).fill(0)}];
    elements.get('case-reproduce').click();await tick();assert.deepEqual(pose,fixture.request.start_deg);
    delay = true;elements.get('case-reproduce').click();
    const changed = new Event('robot-state');changed.detail = {...state,hardware_enabled:true};window.dispatchEvent(changed);await tick();
    assert.equal(signal.aborted,true);assert.equal(elements.get('ik-failure-case').hidden,true);
    assert.equal(scene.getObjectByName('ik-failure-region').visible,false);
    assert.equal(scene.getObjectByName('ik-failure-initial-disk').visible,false);
    assert.equal(scene.getObjectByName('ik-failure-saved-route').visible,false);
    const oldPose = pose.slice();elements.get('case-witness').click();assert.deepEqual(pose,oldPose);
    const otherCase=caseKey==='ik-failed-reachable'?'ik-failed-reachable-inward':'ik-failed-reachable';
    elements.get('case-selector').value=otherCase;elements.get('case-selector').dispatchEvent(new Event('change'));
    assert.equal(navigation,`?case=${otherCase}`);
  } finally {Object.assign(globalThis,saved);await vite.close();}
});
