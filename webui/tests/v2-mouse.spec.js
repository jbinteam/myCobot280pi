import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';

let info, fixture;
test.beforeAll(() => {
  info=JSON.parse(execFileSync('.venv/bin/python',['-c',
    'import json; from simulation.service import info; print(json.dumps(info()))'],
    {cwd:'..',env:{...process.env,OPENBLAS_NUM_THREADS:'1'},encoding:'utf8'}));
  // Real server-checked C2 coefficients; HTTP lifecycle and cursor are mocked.
  fixture=JSON.parse(readFileSync('../output/shared_tracking_profile_20261008/hardware_browser_fixture.json','utf8'));
});

async function setup(page, protocol=1, engine="diffusion_v2", encoderProtocol=1, lateAck=false, feedbackProtocol=1) {
  const calls=[],errors=[]; let plans=0, executions=0;
  const state={hardware_enabled:true,connected:true,token:'fake-v2-mouse',angles:fixture.start_deg,
    motion:null,moving:false,firmware:6.5,limits:info.limits_deg,
    ik:{state:'idle',samples:[],plan:null,tracking_protocol:2,v2_tracking_protocol:protocol,mouse_velocity_profile_protocol:1,mouse_tracking_region_protocol:1,mouse_encoder_profile_protocol:encoderProtocol,mouse_feedback_tracking_protocol:feedbackProtocol,mouse_executor:encoderProtocol===1?(feedbackProtocol===1?'encoder_feedback_tracking':'encoder_profile'):'send_angles',
      mouse_aim_session_id:null,refresh_mode:0}};
  page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/api/**',async route=>{
    const request=route.request(),path=new URL(request.url()).pathname;
    if(path==='/api/sim/info')return route.fulfill({json:info});
    if(request.method()==='POST') {
      const body=request.postDataJSON();calls.push({path,body});
      expect(request.headers()['x-robot-token']).toBe(state.token);
      if(path==='/api/ik/mouse-aim/start') {
        state.ik={...state.ik,mouse_aim_session_id:'session-v2',mouse_engine:engine,velocity_shape:body.velocity_shape,tracking_region_settings:body.tracking_region};
        return route.fulfill({json:{...state,session_id:'session-v2',updated:Date.now()/1000}});
      }
      if(path==='/api/ik/mouse-aim/plan') {
        const data=structuredClone(fixture); plans++;
        data.session_id='session-v2';data.result.cursor_mm=body.cursor_mm;
        const settings=body.tracking_region,axis=body.target_axis;
        const center=body.cursor_mm.map((v,j)=>v-settings.distance_mm*axis[j]);
        data.result.tracking_region={...settings,axis,center_mm:center,cursor_mm:body.cursor_mm,radius_mm:settings.distance_mm*Math.tan(settings.half_angle_deg*Math.PI/180)};
        data.result.target_mm=center;data.result.target_axis=axis;data.result.gap_mm=settings.distance_mm;
        data.plan.id=`curve-${plans}`;data.plan.converged=plans>1;data.plan.execution_mode='encoder_feedback_tracking';data.plan.joint_command='encoder_drag_profile';
        if(engine==='learned_rrt'){
          data.result.method='learned_rrt';data.result.planner='learned_rrt';data.result.tracking_strategy='learned_ik_rrt';
          data.result.checkpoint_sha256=info.classical.checkpoint_sha256;
          data.plan.method='learned_rrt';data.plan.hardware_classical_mouse=true;data.plan.hardware_diffusion=false;
        }
        data.plan.playable=true;data.result.converged=plans>1;data.result.replan_required=plans===1;
        return route.fulfill({json:data});
      }
      if(path==='/api/ik/execute') {
        executions++;state.motion='ik';state.ik={...state.ik,state:'running',execution_id:'curve-1',
          accepted_plan_id:body.plan_id,active_plan_id:body.plan_id,replan_required:executions===1,
          refresh_mode:1,execution_mode:'encoder_feedback_tracking',velocity_limit_deg_s:20,acceleration_limit_deg_s2:171.9,
          feedback_replan_required:false,self_collision_checked:true};
        state.ik.samples=[{time:0,q:fixture.start_deg,reference_velocity_deg_s:[10,-5,0,0,0,0],cursor_direction_error_deg:2},
          {time:.1,q:fixture.start_deg.map((q,j)=>q+(j===0?.5:j===1?-.25:0)),
            command_velocity_deg_s:[5,-2.5,0,0,0,0],reference_velocity_deg_s:[10,-5,0,0,0,0],cursor_direction_error_deg:1}];
      } else if(path==='/api/ik/mouse-aim/heartbeat'){
        if(lateAck&&body.plan_id==='curve-2'){
          state.ik={...state.ik,active_plan_id:'curve-1',accepted_plan_id:'curve-1',
            retarget_rejected:'Late encoder acknowledgement; retaining checked braking tail'};
          return route.fulfill({status:409,json:{error:'Encoder continuation missed its checked join; replan'}});
        }
        return route.fulfill({json:{active:true}});
      }
      else if(path==='/api/ik/mouse-aim/stop'||path==='/api/stop') {
        state.motion=null;state.ik={...state.ik,state:'stopped',mouse_aim_session_id:null,replan_required:false};
      } else throw new Error(`Unexpected write ${path}`);
    }
    if(path!=='/api/state'&&request.method()!=='POST')throw new Error(`Unexpected read ${path}`);
    return route.fulfill({json:{...state,updated:Date.now()/1000}});
  });
  await page.goto('/?view=lab');
  await expect(page.locator('#model-status')).toHaveText('Model ready',{timeout:45000});
  await page.locator('#hardware-planner').selectOption(engine);
  await page.locator('#arm-movement').check();await page.locator('#verify-ik-frame').check();
  return {calls,errors,state};
}

for(const engine of ['diffusion_v2','learned_rrt']){
test(`${engine} requires the table gate, continues a stationary cursor and acknowledges replacement under one owner`,async ({page})=>{
  const {calls,errors}=await setup(page,1,engine);
  await expect(page.locator('#hardware-planner')).toHaveValue(engine);
  await expect(page.locator('#toggle-mouse-aim')).toBeDisabled();
  await page.locator('#verify-diffusion-scene').check();
  await page.locator('#ik-sdk-speed').fill('80');
  await expect(page.locator('#toggle-mouse-aim')).toBeEnabled();
  await page.locator('#toggle-mouse-aim').click();
  await expect.poll(()=>calls.filter(c=>c.path.endsWith('/start')).length).toBe(1);
  await page.locator('.viewport-panel').scrollIntoViewIfNeeded();
  await page.locator('#viewport canvas').scrollIntoViewIfNeeded();
  const point=await page.evaluate(async()=>{
    const {camera,scene}=await import(document.querySelector('script[src*="/src/main.js"]').src);
    const p=scene.getObjectByName('mouse-aim-cursor').position.clone().project(camera);
    const r=document.querySelector('#viewport canvas').getBoundingClientRect();
    return{x:r.left+(p.x+1)*r.width/2+2,y:r.top+(1-p.y)*r.height/2+2};
  });
  await page.mouse.move(point.x,point.y);
  await expect.poll(()=>calls.filter(c=>c.path==='/api/ik/execute').length,{timeout:15000}).toBe(2);
  const start=calls.find(c=>c.path.endsWith('/start')).body;
  expect(start).toMatchObject({tracking_engine:engine,scene_verified:true,speed:80,armed:true,frame_verified:true,velocity_shape:{kind:'smooth_trapezoid',rise_percent:5,fall_percent:10}});
  const plans=calls.filter(c=>c.path.endsWith('/plan'));
  expect(plans).toHaveLength(2);
  plans[0].body.cursor_mm.forEach((q,i)=>expect(q).toBeCloseTo(plans[1].body.cursor_mm[i],8));
  for(const {body} of plans){
    expect(body.tracking_region).toEqual({mode:'cone_disk',distance_mm:100,half_angle_deg:20});
    expect(body.velocity_shape).toEqual({kind:'smooth_trapezoid',rise_percent:5,fall_percent:10});expect(body.session_id).toBe('session-v2');expect(body.obstacle_mm).toBeNull();
    for(const key of ['start_deg','goal_hint_deg','start_velocity_rad_s','start_acceleration_rad_s2','target_velocity_m_s'])expect(body[key]).toBeUndefined();
  }
  expect(calls.filter(c=>c.path==='/api/ik/execute').map(c=>c.body)).toEqual([
    {plan_id:'curve-1',session_id:'session-v2',speed:80,armed:true,frame_verified:true,scene_verified:true},
    {plan_id:'curve-2',session_id:'session-v2',speed:80,armed:true,frame_verified:true,scene_verified:true}]);
  expect(calls.some(c=>c.path.endsWith('/heartbeat')&&c.body.execution_id==='curve-1'&&c.body.plan_id==='curve-2')).toBe(true);
  const targetView=await page.evaluate(async()=>{
    const {scene}=await import(document.querySelector('script[src*="/src/main.js"]').src),geometry=scene.getObjectByName('mouse-trajectory-output-points').geometry;
    return {count:geometry.attributes.position.count,target:Array.from(geometry.attributes.position.array),
      movingMarker:!!scene.getObjectByName('mouse-trajectory-output-current')};
  });
  expect(targetView.count).toBe(1);expect(targetView.movingMarker).toBe(false);
  const renderedRegion=await page.evaluate(async()=>{
    const {scene}=await import(document.querySelector('script[src*="/src/main.js"]').src),group=scene.getObjectByName('mouse-aim-tracking-region'),disk=scene.getObjectByName('mouse-aim-acceptance-disk');
    return {visible:group.visible,region:group.userData.region,diskCenter:disk.position.toArray(),diskRadius:disk.scale.x};
  });
  expect(renderedRegion.visible).toBe(true);expect(renderedRegion.region.distance_mm).toBe(100);
  expect(renderedRegion.diskRadius).toBeCloseTo(.1*Math.tan(Math.PI/9),8);
  renderedRegion.diskCenter.forEach((v,j)=>expect(v).toBeCloseTo(renderedRegion.region.center_mm[j]/1000,8));
  plans[0].body.target_axis.forEach((v,j)=>expect(v).toBeCloseTo(plans[1].body.target_axis[j],8));
  targetView.target.forEach((value,j)=>expect(value).toBeCloseTo((plans.at(-1).body.cursor_mm[j]-100*plans.at(-1).body.target_axis[j])/1000,6));
  await expect(page.locator('#mouse-aim-hardware-feedback')).toContainText(engine==='learned_rrt'?'RRT* feedback-paced gaze tracking':'v2 feedback-paced gaze tracking');
  await expect(page.locator('#hardware-velocity-profiles')).toBeVisible();
  await expect(page.locator('#hardware-reference-velocity-chart polyline')).toHaveCount(6);
  await expect(page.locator('#hardware-measured-velocity-chart polyline')).toHaveCount(6);
  await expect(page.locator('#hardware-measured-velocity-chart')).toContainText('Actual elapsed time (s)');
  await expect(page.locator('#hardware-reference-speed')).toContainText('rad/s');
  await expect(page.locator('#hardware-measured-speed')).toContainText('rad/s');
  await expect(page.locator('#velocity-chart polyline')).toHaveCount(6);
  await expect(page.locator('#mouse-planned-velocity')).toBeVisible();
  await expect(page.locator('#mouse-planned-velocity-chart polyline')).toHaveCount(6);
  await expect(page.locator('#mouse-planned-velocity-chart')).toContainText('Trajectory time (s)');
  await expect(page.locator('#mouse-planned-velocity-chart')).toContainText('Planned velocities (rad/s)');
  await expect(page.locator('#mouse-profile-status')).toContainText('Live feedback can stretch execution time');
  expect(await page.locator('#mouse-planned-velocity-chart polyline').evaluateAll(lines=>lines.map(line=>line.getAttribute('points'))))
    .toEqual(await page.locator('#velocity-chart polyline').evaluateAll(lines=>lines.map(line=>line.getAttribute('points'))));
  // Polling live telemetry must preserve the full accepted reference profile.
  await page.evaluate(()=>window.dispatchEvent(new Event('robot-state-refresh')));
  await expect(page.locator('#mouse-planned-velocity-chart polyline')).toHaveCount(6);
  await page.screenshot({path:`../output/cone_disk_tracking_20261008/${engine}-velocity-charts.png`,fullPage:true});
  await page.locator('.viewport-panel').screenshot({path:`../output/cone_disk_tracking_20261008/${engine}-cone-disk.png`});
  // Hardware display remains at live telemetry, never animated predicted joints.
  const joints=await page.evaluate(async()=>{
    const {robot}=await import(document.querySelector('script[src*="/src/main.js"]').src);return Object.values(robot.joints).filter(j=>j.jointType==='revolute').map(j=>j.angle*180/Math.PI);
  });
  joints.forEach((q,i)=>expect(q).toBeCloseTo(fixture.start_deg[i],6));
  await page.keyboard.press('Escape');
  await expect.poll(()=>calls.filter(c=>c.path.endsWith('/stop')).length).toBeGreaterThan(0);
  await expect(page.locator('#mouse-planned-velocity')).toBeHidden();
  await expect(page.locator('#mouse-planned-velocity-chart polyline')).toHaveCount(0);
  expect(errors).toEqual([]);
});

}

test('older backend cannot silently start the classical engine for a v2 selection',async ({page})=>{
  const {calls}=await setup(page);
  await page.route('**/api/ik/mouse-aim/start',route=>route.fulfill({json:{session_id:'old',
    angles:fixture.start_deg,ik:{tracking_protocol:2,mouse_engine:'classical'}}}));
  await page.locator('#verify-diffusion-scene').check();
  await page.locator('#toggle-mouse-aim').click();
  await expect(page.locator('#mouse-aim-status')).toContainText('Restart the control bridge');
  expect(calls.some(c=>c.path==='/api/ik/execute')).toBe(false);
  await expect(page.locator('#toggle-mouse-aim')).toHaveAttribute('aria-pressed','false');
});

test('an angle-target tracking bridge cannot silently own the encoder selection',async({page})=>{
  const {calls,errors}=await setup(page,1,'learned_rrt',0);
  await page.locator('#verify-diffusion-scene').check();
  await page.locator('#toggle-mouse-aim').click();
  await expect(page.locator('#mouse-aim-status')).toContainText('shared encoder velocity executor');
  await expect(page.locator('#toggle-mouse-aim')).toHaveAttribute('aria-pressed','false');
  expect(calls.some(c=>c.path.endsWith('/stop'))).toBe(true);
  expect(calls.some(c=>c.path==='/api/ik/execute')).toBe(false);expect(errors).toEqual([]);
});

test('a late encoder replacement acknowledgement retries while retaining the running owner',async({page})=>{
  const {calls,errors}=await setup(page,1,'learned_rrt',1,true);
  await page.locator('#verify-diffusion-scene').check();
  await page.locator('#toggle-mouse-aim').click();
  await page.locator('#viewport canvas').scrollIntoViewIfNeeded();
  const point=await page.evaluate(async()=>{
    const {camera,scene}=await import(document.querySelector('script[src*="/src/main.js"]').src);
    const p=scene.getObjectByName('mouse-aim-cursor').position.clone().project(camera);
    const r=document.querySelector('#viewport canvas').getBoundingClientRect();
    return{x:r.left+(p.x+1)*r.width/2+2,y:r.top+(1-p.y)*r.height/2+2};
  });
  await page.mouse.move(point.x,point.y);
  await expect.poll(()=>calls.filter(c=>c.path==='/api/ik/execute').length,{timeout:15000}).toBe(3);
  await expect(page.locator('#toggle-mouse-aim')).toHaveAttribute('aria-pressed','true');
  expect(calls.some(c=>c.path.endsWith('/stop'))).toBe(false);
  expect(calls.filter(c=>c.path.endsWith('/heartbeat')&&c.body.plan_id).every(c=>c.body.execution_id==='curve-1')).toBe(true);
  await page.keyboard.press('Escape');expect(errors).toEqual([]);
});

test('a strict trajectory bridge cannot silently start feedback-paced gaze tracking',async({page})=>{
  const {calls,errors}=await setup(page,1,'learned_rrt',1,false,0);
  await page.locator('#verify-diffusion-scene').check();
  await page.locator('#toggle-mouse-aim').click();
  await expect(page.locator('#mouse-aim-status')).toContainText('feedback-paced gaze tracking');
  await expect(page.locator('#toggle-mouse-aim')).toHaveAttribute('aria-pressed','false');
  expect(calls.some(c=>c.path.endsWith('/stop'))).toBe(true);
  expect(calls.some(c=>c.path==='/api/ik/execute')).toBe(false);expect(errors).toEqual([]);
});

test('slow joint feedback keeps gaze tracking active and moving cursor replaces the same owner',async({page})=>{
  const {calls,errors,state}=await setup(page,1,'learned_rrt');
  await page.locator('#verify-diffusion-scene').check();
  await page.locator('#toggle-mouse-aim').click();
  await page.locator('#viewport canvas').scrollIntoViewIfNeeded();
  const point=await page.evaluate(async()=>{
    const {camera,scene}=await import(document.querySelector('script[src*="/src/main.js"]').src);
    const p=scene.getObjectByName('mouse-aim-cursor').position.clone().project(camera);
    const r=document.querySelector('#viewport canvas').getBoundingClientRect();
    return{x:r.left+(p.x+1)*r.width/2+2,y:r.top+(1-p.y)*r.height/2+2};
  });
  await page.mouse.move(point.x,point.y);
  await expect.poll(()=>calls.filter(c=>c.path==='/api/ik/execute').length,{timeout:15000}).toBe(2);
  Object.assign(state.ik,{tracking_error_deg:9,reference_lag_s:1.2,tracking_paused:true,
    tracking_pause_reason:'Waiting for live joint feedback'});
  await page.evaluate(()=>window.dispatchEvent(new Event('robot-state-refresh')));
  await expect(page.locator('#mouse-aim-hardware-feedback')).toContainText('reference delay 1.20 s');
  await expect(page.locator('#mouse-aim-hardware-feedback')).toContainText('self-collision checked');
  await expect(page.locator('#mouse-aim-status')).toContainText('reference waiting for live feedback');
  await expect(page.locator('#hardware-reference-title')).toHaveText('Feedback-paced reference');
  await expect(page.locator('#hardware-orientation-chart')).toContainText('Tool Z aim error to cursor');
  await expect(page.locator('#toggle-mouse-aim')).toHaveAttribute('aria-pressed','true');
  await page.mouse.move(point.x+12,point.y+4);
  await expect.poll(()=>calls.filter(c=>c.path==='/api/ik/execute').length,{timeout:15000}).toBe(3);
  const plans=calls.filter(c=>c.path.endsWith('/plan'));
  expect(Math.hypot(...plans[2].body.cursor_mm.map((v,i)=>v-plans[1].body.cursor_mm[i]))).toBeGreaterThan(1);
  expect(calls.some(c=>c.path.endsWith('/heartbeat')&&c.body.execution_id==='curve-1'&&c.body.plan_id==='curve-3')).toBe(true);
  // A drift-triggered fresh plan must also work with an unchanged cursor.
  state.ik.feedback_replan_required=true;
  await page.evaluate(()=>window.dispatchEvent(new Event('robot-state-refresh')));
  await expect.poll(()=>calls.filter(c=>c.path==='/api/ik/execute').length,{timeout:15000}).toBe(4);
  const latest=calls.filter(c=>c.path.endsWith('/plan'));
  latest[3].body.cursor_mm.forEach((v,j)=>expect(v).toBeCloseTo(latest[2].body.cursor_mm[j],8));
  // A settled endpoint outside gaze/disk acceptance requests a live correction.
  // Completion of this checked reference must not end the tracking session.
  state.motion=null;state.moving=false;
  Object.assign(state.ik,{state:'completed',replan_required:true,feedback_replan_required:true,
    cursor_acceptance:{accepted:false,direction_error_deg:4}});
  await page.evaluate(()=>window.dispatchEvent(new Event('robot-state-refresh')));
  await expect.poll(()=>calls.filter(c=>c.path==='/api/ik/execute').length,{timeout:15000}).toBe(5);
  const corrected=calls.filter(c=>c.path.endsWith('/plan'));
  corrected[4].body.cursor_mm.forEach((v,j)=>expect(v).toBeCloseTo(corrected[3].body.cursor_mm[j],8));
  await expect(page.locator('#mouse-aim-status')).not.toContainText('On target');
  await expect(page.locator('#toggle-mouse-aim')).toHaveAttribute('aria-pressed','true');
  expect(calls.some(c=>c.path.endsWith('/stop'))).toBe(false);
  await page.keyboard.press('Escape');
  await expect.poll(()=>calls.filter(c=>c.path.endsWith('/stop')).length).toBeGreaterThan(0);
  expect(errors).toEqual([]);
});


test('cone circle is perpendicular to the frozen axis and selected off-center aim points at the cursor',async({page})=>{
  const {errors}=await setup(page);
  const geometry=await page.evaluate(async()=>{
    const THREE=await import('/node_modules/.vite/deps/three.js');
    const {coneDiskRegion,createTrackingRegionView,regionPositionMetrics,regionMatches}=await import('/src/tracking-region.js');
    const axis=new THREE.Vector3(1,2,3).normalize(),cursor=new THREE.Vector3(.2,.1,.3);
    const region=coneDiskRegion(cursor,axis.toArray(),{mode:'cone_disk',distance_mm:100,half_angle_deg:10});
    const view=createTrackingRegionView();view.show(region);
    const disk=view.group.getObjectByName('mouse-aim-acceptance-disk'),cone=view.group.getObjectByName('mouse-aim-cone');
    const normal=new THREE.Vector3(0,0,1).applyQuaternion(disk.quaternion);
    const apex=new THREE.Vector3(0,.5,0).multiply(cone.scale).applyQuaternion(cone.quaternion).add(cone.position);
    const transverse=new THREE.Vector3().crossVectors(axis,new THREE.Vector3(0,1,0)).normalize();
    const center=new THREE.Vector3().fromArray(region.center_mm).multiplyScalar(.001),point=center.clone().addScaledVector(transverse,.012);
    const valid=regionPositionMetrics(point,region),outside=regionPositionMetrics(center.clone().addScaledVector(transverse,.03),region);
    const shifted=regionPositionMetrics(point.clone().addScaledVector(axis,.006),region);
    return {normalDot:normal.dot(axis),apexError:apex.distanceTo(cursor),valid,outside,shifted,radius:region.radius_mm,
      acceptsCanonical:regionMatches(region,structuredClone(region)),acceptsTampered:regionMatches(region,{...region,radius_mm:99})};
  });
  expect(geometry.normalDot).toBeCloseTo(1,10);expect(geometry.apexError).toBeLessThan(1e-12);
  expect(geometry.valid.axialErrorMm).toBeLessThan(1e-10);expect(geometry.valid.radialMm).toBeCloseTo(12,10);
  expect(geometry.outside.radialMm).toBeGreaterThan(geometry.radius);expect(geometry.shifted.axialErrorMm).toBeCloseTo(6,10);
  expect(geometry.acceptsCanonical).toBe(true);expect(geometry.acceptsTampered).toBe(false);expect(errors).toEqual([]);
});

test('an already accepted goal can remain within the disk plane tolerance',async({page})=>{
  await setup(page);
  const checked=await page.evaluate(async({fixture,info})=>{
    const THREE=await import('/node_modules/.vite/deps/three.js');
    const {coneDiskRegion}=await import('/src/tracking-region.js');
    const {validatedAimPath}=await import('/src/mouse-aim-path.js');
    const data=structuredClone(fixture.result),axis=new THREE.Vector3().fromArray(data.target_axis).normalize(),cursor=new THREE.Vector3().fromArray(data.cursor_mm).multiplyScalar(.001);
    const region=coneDiskRegion(cursor,axis.toArray(),{mode:'cone_disk',distance_mm:100,half_angle_deg:10});
    data.tracking_region=region;data.gap_mm=100;
    const context={box:null,checkpoint:info.diffusion.checkpoint_sha256,planner:'local_diffusion',conditioningSchema:info.diffusion.conditioning.schema,
      velocityShape:data.smoothing.reference_shape,region,toleranceMm:1};
    return [.5,1.1].map(delta=>{
      const point=new THREE.Vector3().fromArray(region.center_mm).multiplyScalar(.001).addScaledVector(axis,delta/1000);
      data.target_mm=point.toArray().map(v=>v*1000);data.target_axis=cursor.clone().sub(point).normalize().toArray();
      try{validatedAimPath(data,fixture.start_deg,info.limits_deg,1,data.cursor_mm,context,true);return true;}
      catch(error){return error.message;}
    });
  },{fixture,info});
  expect(checked[0]).toBe(true);expect(checked[1]).toContain('outside the tracking circle');
});
