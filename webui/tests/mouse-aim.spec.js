import { test, expect } from '@playwright/test';
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';

let fixture, worker, pending = [], stderr = '';
const code = `import json, sys, numpy as np
from simulation import service
from simulation.kinematics import ARM
from simulation.pose5 import task
for line in sys.stdin:
    try:
        request=json.loads(line); path=request['path']
        if path=='fixture':
            data=service.info(); start=np.deg2rad(data['default_start_deg'])
            goal=ARM.clip(start+np.array([.03,-.02,.04,.03,-.02,.01]))
            t=ARM.transform(goal); _,j,e,angle=task(start,t[:3,3],t[:3,2])
            value={'info':data,'target':t[:3,3].tolist(),'axis':t[:3,2].tolist(),'j':j.tolist(),'error':e.tolist()}
        else:value=service.mouse_aim_plan(request['body'])
        print(json.dumps({'value':value}, allow_nan=False), flush=True)
    except Exception as exc:print(json.dumps({'error':str(exc)}), flush=True)
`;
function rpc(path, body={}) {
  return new Promise((resolve,reject)=>{ pending.push({resolve,reject}); worker.stdin.write(JSON.stringify({path,body})+'\n'); });
}
test.beforeAll(async ({}, testInfo) => {
  testInfo.setTimeout(240000);
  worker=spawn('.venv/bin/python',['-u','-c',code],{cwd:'..',env:{...process.env,OPENBLAS_NUM_THREADS:'1'}});
  worker.stderr.on('data',data=>{stderr=(stderr+data).slice(-10000);});
  createInterface({input:worker.stdout}).on('line',line=>{
    const request=pending.shift(); if(!request)return;
    const data=JSON.parse(line); data.error ? request.reject(new Error(data.error)) : request.resolve(data.value);
  });
  worker.on('exit',()=>{for(const request of pending.splice(0))request.reject(new Error(stderr||'Simulation worker exited'));});
  fixture=await rpc('fixture');
});
test.afterAll(()=>worker?.kill());
test.setTimeout(120000);

async function setup(page, initialState = {}, planner = body => rpc('aim', body), hardwareHandler = null) {
  const posts = [], errors = [], plans = [];
  let state = { hardware_enabled: false, connected: false, token: 'mouse-aim-test', motion: null, moving: false, ik: { state: 'idle', samples: [] }, ...initialState };
  page.on('pageerror', error => errors.push(error.message));
  await page.route('**/api/**', async route => {
    const request = route.request(), path = new URL(request.url()).pathname;
    if (path === '/api/state') {
      if (hardwareHandler && state.connected) state.updated=Date.now()/1000;
      return route.fulfill({ json: state });
    }
    if (path === '/api/sim/info') return route.fulfill({ json: fixture.info });
    if (request.method() === 'POST') posts.push({path,body:request.postDataJSON()});
    if (hardwareHandler && (path.startsWith('/api/ik/') || path==='/api/stop')) {
      const data=await hardwareHandler(path,request.postDataJSON(),state);
      return route.fulfill({status:data.http_status||200,json:data}).catch(()=>{});
    }
    if (path === '/api/sim/mouse-aim') {
      const data=await planner(request.postDataJSON()); plans.push(data);
      return route.fulfill({status:data.http_status || 200,json:data}).catch(()=>{});
    }
    if (path === '/api/sim/solve') {
      const body = request.postDataJSON();
      return route.fulfill({ json: { start_deg: body.start_deg, results: body.methods.map(method => ({
        method, converged: true, final_error_mm: 0, final_orientation_error_deg: 0, iterations: 0,
        joint_travel_deg: 0, elapsed_ms: 1, stop_reason: 'target_reached',
        frames: [{ q: body.start_deg, position: body.target_mm, error_mm: 0, orientation_error_deg: 0, time: 0 }],
      })) } });
    }
    throw new Error(`Unexpected request: ${request.method()} ${path}`);
  });
  await page.goto('/?view=lab');
  await expect(page.locator('#model-status')).toHaveText('Model ready', { timeout: 45000 });
  await expect(page.locator('#ik-method option').first()).toBeAttached({ timeout: 45000 });
  await expect(page.locator('#solve-ik')).toBeEnabled({ timeout: 45000 });
  if(initialState.hardware_enabled!==true){
    await page.locator('#mouse-aim-continuous').uncheck();
    await page.locator('#mouse-aim-planner').selectOption('classical');
    await page.locator('#ik-method').selectOption('dls');
  }
  return { posts, errors, plans, setState: next => { state = { ...state, ...next }; } };
}

async function actualAim(page) {
  return page.evaluate(async () => {
    const { robot, scene } = await import(document.querySelector('script[src*="/src/main.js"]').src);
    robot.updateMatrixWorld(true);
    const flange = robot.links.joint6_flange, cursor = scene.getObjectByName('mouse-aim-cursor');
    const position = flange.getWorldPosition(robot.position.clone());
    const axis = robot.position.clone().set(0, 0, 1).transformDirection(flange.matrixWorld);
    const direction = cursor.position.clone().sub(position).normalize();
    const goal = scene.getObjectByName('mouse-aim-goal');
    return { distance: position.distanceTo(cursor.position) * 1000,
      angle: Math.acos(Math.max(-1, Math.min(1, axis.dot(direction)))) * 180 / Math.PI,
      desiredGap: goal.position.distanceTo(cursor.position) * 1000,
      q: Object.values(robot.joints).filter(j => j.jointType === 'revolute').map(j => j.angle),
    };
  });
}

test('browser pose controller matches the Python URDF task and converges with bounded motion', async ({ page }) => {
  const { errors } = await setup(page);
  const result = await page.evaluate(async f => {
    const { robot } = await import(document.querySelector('script[src*="/src/main.js"]').src);
    const { PoseServo } = await import('/src/pose-servo.js');
    const names = Object.values(robot.joints).filter(j => j.jointType === 'revolute').map(j => j.name);
    const servo = new PoseServo(robot, names, f.info.limits_deg);
    let q = f.info.default_start_deg.map(v => v * Math.PI / 180);
    const target = robot.position.clone().fromArray(f.target), axis = robot.position.clone().fromArray(f.axis);
    const initial = servo.task(q, target, axis);
    let maxTravel = 0, maxMeritIncrease = 0, inLimits = true, last;
    for (let i = 0; i < 320; i++) {
      const previous = servo.task(q, target, axis);
      last = servo.step(q, target, axis, .05);
      maxTravel = Math.max(maxTravel, last.travel);
      maxMeritIncrease = Math.max(maxMeritIncrease, last.merit - previous.merit);
      q = last.q;
      inLimits &&= q.every((v, j) => Number.isFinite(v) && v >= servo.lower[j] && v <= servo.upper[j]);
    }
    const antiAxis = initial.axis.clone().negate();
    const antipodal = servo.step(f.info.default_start_deg.map(v => v * Math.PI / 180), initial.position, antiAxis, .05);
    return { j: initial.j, error: initial.error, maxTravel, maxMeritIncrease, inLimits,
      positionError: last.positionErrorMm, angle: last.angleDeg, antipodalFinite: antipodal.q.every(Number.isFinite),
      invalidRejected: servo.step([NaN, 0, 0, 0, 0, 0], target, axis, .05) === null };
  }, fixture);
  result.j.forEach((row, i) => row.forEach((value, j) => expect(Math.abs(value - fixture.j[i][j])).toBeLessThan(1e-8)));
  result.error.forEach((value, i) => expect(Math.abs(value - fixture.error[i])).toBeLessThan(1e-8));
  expect(result.maxTravel).toBeLessThanOrEqual(.060000001);
  expect(result.maxMeritIncrease).toBeLessThanOrEqual(1e-12);
  expect(result.inLimits).toBe(true);
  expect(result.positionError).toBeLessThan(1);
  expect(result.angle).toBeLessThan(1);
  expect(result.antipodalFinite).toBe(true);
  expect(result.invalidRejected).toBe(true);
  expect(errors).toEqual([]);
});

test('mouse aim follows local IK from current joints with a self-checked 50 mm gap', async ({ page }, testInfo) => {
  const { posts, plans, errors } = await setup(page);
  await expect(page.locator('#toggle-mouse-aim')).toBeEnabled();
  await page.locator('.viewport-panel').scrollIntoViewIfNeeded();
  await page.locator('#toggle-mouse-aim').click();
  await expect(page.locator('#run-results tbody tr')).toHaveCount(0);
  // Start can resize the viewport; project only after its canvas is visible.
  await page.locator('#viewport canvas').scrollIntoViewIfNeeded();
  await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
  const point=await cursorScreenPoint(page);
  expect(point.y).toBeGreaterThan(0);expect(point.y).toBeLessThan(page.viewportSize().height);
  const before = await actualAim(page), postCount = posts.length;
  await page.mouse.move(point.x, point.y);
  await expect(page.locator('#mouse-aim-status')).toContainText('On target', { timeout: 90000 });
  const layout = await page.evaluate(() => ({
    toolbarBottom: document.querySelector('#mouse-aim-controls').getBoundingClientRect().bottom,
    canvasTop: document.querySelector('#viewport canvas').getBoundingClientRect().top,
  }));
  expect(layout.canvasTop).toBeGreaterThanOrEqual(layout.toolbarBottom);
  const after = await actualAim(page);
  expect(Math.abs(after.distance - 50)).toBeLessThanOrEqual(1);
  expect(after.angle).toBeLessThanOrEqual(1);
  expect(after.desiredGap).toBeCloseTo(50, 7);
  const projectedCursor = await page.evaluate(async () => {
    const {camera,scene}=await import(document.querySelector('script[src*="/src/main.js"]').src);
    const p=scene.getObjectByName('mouse-aim-cursor').position.clone().project(camera);
    const r=document.querySelector('#viewport canvas').getBoundingClientRect();
    return {x:r.left+(p.x+1)*r.width/2,y:r.top+(1-p.y)*r.height/2};
  });
  expect(Math.abs(projectedCursor.x-point.x)).toBeLessThan(1.5);
  expect(Math.abs(projectedCursor.y-point.y)).toBeLessThan(1.5);
  expect(Math.hypot(...after.q.map((v, i) => v - before.q[i]))).toBeGreaterThan(.001);
  expect(posts.length).toBeGreaterThan(postCount);
  expect(posts.every(({path}) => ['/api/sim/solve','/api/sim/mouse-aim'].includes(path))).toBe(true);
  expect(plans[0].tracking_source).toBe('local_ik');
  expect(plans[0].ik.config.nn).toBe('none');
  expect(plans[0].ik.config.refine).toBe(false);
  expect(plans[0].ik.config.numerical_fallback).toBe(false);
  expect(plans[0].ik.config.pose5_method).toBe('dls');
  expect(plans[0].ik.goals[0].source).toBe('current_local_ik');
  expect(plans[0].local_ik.raw_path_vertices).toBeGreaterThan(plans[0].waypoints_deg.length);
  expect(plans[0].ik.config.guided_route_repair).toBe(true);
  expect(plans[0].ik.config.route_repair_samples).toBe(32);
  expect(posts.find(p=>p.path==='/api/sim/mouse-aim').body.guided_route_repair).toBe(true);
  expect(plans[0].ik.config.tolerance_mm).toBe(1);
  expect(plans[0].ik.config.orientation_tolerance_deg).toBe(1);
  expect(plans[0].ik.nn_checkpoint).toBeNull();
  expect(plans[0].validation).toBe('joint_limits_and_self_collision');
  expect(plans[0].cursor_acceptance.accepted).toBe(true);
  await expect(page.locator('#mouse-aim-outcome')).toContainText('local IK from current joints');
  await expect(page.locator('#mouse-aim-engine')).toContainText('Local IK first');
  await page.locator('.viewport-panel').screenshot({ path: testInfo.outputPath('mouse-aim-desktop.png') });

  // The pointer stays projected to a plane facing the new camera view.
  await page.locator('#side-view').click();
  await page.mouse.move(point.x + 5, point.y + 2);
  const facing = await page.evaluate(async () => {
    const { camera, scene } = await import(document.querySelector('script[src*="/src/main.js"]').src);
    return Math.abs(camera.quaternion.dot(scene.getObjectByName('mouse-aim-cursor').quaternion));
  });
  expect(facing).toBeCloseTo(1, 5);
  await page.keyboard.press('Escape');
  await expect(page.locator('#toggle-mouse-aim')).toHaveAttribute('aria-pressed', 'false');
  const stopped = (await actualAim(page)).q;
  await page.mouse.move(point.x + 50, point.y + 20);
  await expect.poll(async () => (await actualAim(page)).q).toEqual(stopped);
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.locator('.viewport-panel').screenshot({ path: testInfo.outputPath('mouse-aim-mobile.png') });
  expect(errors).toEqual([]);
});

test('tracking pauses outside the scene and stops when hardware mode or manual editing takes over', async ({ page }) => {
  const { setState, posts, errors } = await setup(page);
  await page.locator('#toggle-mouse-aim').click();
  // Use the nearby cursor marker: cancellation is the behavior under test.
  // An arbitrary scene-center target can leave a cold fallback query running
  // in the shared Python fixture after its browser request has been aborted.
  const point = await cursorScreenPoint(page);
  await page.mouse.move(point.x, point.y);
  await page.mouse.move(0, 0);
  await expect(page.locator('#mouse-aim-status')).toContainText('Paused');
  await page.locator('#joint-0-number').fill('1');
  await page.locator('#joint-0-number').dispatchEvent('change');
  await expect(page.locator('#toggle-mouse-aim')).toHaveAttribute('aria-pressed', 'false');
  await page.locator('#toggle-mouse-aim').click();
  setState({ hardware_enabled: true });
  await expect(page.locator('#toggle-mouse-aim')).toBeDisabled();
  await expect(page.locator('#toggle-mouse-aim')).toHaveAttribute('aria-pressed', 'false');
  expect(posts.every(({path}) => path.startsWith('/api/sim/'))).toBe(true);
  expect(errors).toEqual([]);
});

test('hardware mouse aim is visible but cannot start before connection and confirmations', async ({ page }) => {
  const { errors } = await setup(page, { hardware_enabled: true });
  await expect(page.locator('#mouse-aim-controls')).toBeVisible();
  await expect(page.locator('#mouse-aim-mode-badge')).toHaveText('LIVE ROBOT CONTROL');
  await expect(page.locator('#toggle-mouse-aim')).toBeDisabled();
  expect(errors).toEqual([]);
});

async function cursorScreenPoint(page, dx=8, dy=5) {
  await page.locator('#viewport canvas').scrollIntoViewIfNeeded();
  return page.evaluate(async ({dx,dy}) => {
    const { camera, scene }=await import(document.querySelector('script[src*="/src/main.js"]').src);
    const marker=scene.getObjectByName('mouse-aim-cursor').position.clone().project(camera);
    const rect=document.querySelector('#viewport canvas').getBoundingClientRect();
    return {x:rect.left+(marker.x+1)*rect.width/2+dx,y:rect.top+(1-marker.y)*rect.height/2+dy};
  },{dx,dy});
}

test('coalesces cursor moves and discards a late plan when stopped', async ({page}) => {
  let release; const hold=new Promise(resolve=>{release=resolve;});
  const {posts,errors}=await setup(page,{},async body=>{const plan=await rpc('aim',body); await hold; return plan;});
  await page.locator('#toggle-mouse-aim').click();
  const point=await cursorScreenPoint(page), before=(await actualAim(page)).q;
  await page.mouse.move(point.x,point.y);
  await expect.poll(()=>posts.filter(p=>p.path==='/api/sim/mouse-aim').length).toBe(1);
  for(let i=0;i<15;i++)await page.mouse.move(point.x+i*.2,point.y+i*.2);
  expect(posts.filter(p=>p.path==='/api/sim/mouse-aim')).toHaveLength(1);
  expect((await actualAim(page)).q).toEqual(before);
  await page.keyboard.press('Escape'); release();
  await expect(page.locator('#toggle-mouse-aim')).toHaveAttribute('aria-pressed','false');
  await page.waitForTimeout(400);
  expect((await actualAim(page)).q).toEqual(before);
  expect(errors).toEqual([]);
});

test('IK and routing failures hold the preview and explain separate outcomes',async({page})=>{
  let routing=false;
  const {posts,errors}=await setup(page,{},async()=>({simulation_only:true,task:'pose5',converged:false,frames:[],
    ik_outcome:routing?'accepted_goals':'ik_solution_not_found',planning_outcome:routing?'start_not_connected':'not_started',
    status:routing?'roadmap_connection_not_found':'ik_solution_not_found',reason:routing?'start_not_connected':'No valid IK goal',
    ik:{goals:[],recovery:{phases:[{used_seeds:32}],added_goals:0}}}));
  await page.locator('#toggle-mouse-aim').click();
  const point=await cursorScreenPoint(page),before=(await actualAim(page)).q;
  await page.mouse.move(point.x,point.y);
  await expect(page.locator('#mouse-aim-status')).toContainText('ik_solution_not_found');
  await expect(page.locator('#mouse-aim-outcome')).toContainText('route: not started');
  await expect(page.locator('#mouse-aim-outcome')).toContainText('expanded search: 32 seeds, 0 added goals');
  routing=true;await page.mouse.move(point.x+3,point.y);
  await expect(page.locator('#mouse-aim-status')).toContainText('roadmap_connection_not_found');
  await expect(page.locator('#mouse-aim-outcome')).toContainText('IK: accepted goals');
  await expect(page.locator('#mouse-aim-outcome')).toContainText('route: start not connected');
  expect((await actualAim(page)).q).toEqual(before);
  expect(posts.filter(p=>p.path==='/api/sim/mouse-aim')).toHaveLength(2);
  expect(errors).toEqual([]);
});

test('busy planner retries the latest cursor without queuing mouse events',async({page})=>{
  let calls=0;
  const {posts,errors}=await setup(page,{},body=>++calls===1
    ? {http_status:409,error:'Mouse aim planner busy'} : rpc('aim',body));
  await page.locator('#toggle-mouse-aim').click();
  const point=await cursorScreenPoint(page,0,0);
  await page.mouse.move(point.x,point.y);
  await expect(page.locator('#mouse-aim-status')).toContainText('previous request');
  await page.mouse.move(point.x+6,point.y+4);
  await expect(page.locator('#mouse-aim-status')).toContainText('On target',{timeout:20000});
  const requests=posts.filter(p=>p.path==='/api/sim/mouse-aim');
  expect(requests).toHaveLength(2);
  expect(requests[1].body.cursor_mm).not.toEqual(requests[0].body.cursor_mm);
  expect(errors).toEqual([]);
});

test('a captured failed target transfers exact starting joints and 5D inputs for comparison',async({page})=>{
  const target=[123.4567890123,45.6789012345,101.23456789],axis=[.3,.4,Math.sqrt(.75)];
  let calls=0,release;
  const hold=new Promise(resolve=>{release=resolve;});
  const {posts,plans,errors}=await setup(page,{},async()=>{
    const first=++calls===1;
    if(!first)await hold;
    return {simulation_only:true,task:'pose5',converged:false,
      status:'ik_solution_not_found',reason:'Fixture search budget exhausted',
      target_mm:first?target:target.map(v=>v+10),target_axis:axis,
      ik_outcome:'ik_solution_not_found',planning_outcome:'not_started',frames:[]};
  });
  await expect(page.locator('#inspect-mouse-aim')).toBeDisabled();
  await page.evaluate(async()=>{
    const {robot}=await import(document.querySelector('script[src*="/src/main.js"]').src);
    Object.values(robot.joints).filter(j=>j.jointType==='revolute')[0].setJointValue(.000731);
  });
  await page.locator('#toggle-mouse-aim').click();
  const point=await cursorScreenPoint(page);await page.mouse.move(point.x,point.y);
  await expect(page.locator('#mouse-aim-status')).toContainText('Fixture search budget exhausted');
  const snapshot=posts.find(p=>p.path==='/api/sim/mouse-aim').body;
  // A real pointer crosses the scene on its way to Compare. This starts another
  // request, then leaving the canvas aborts it. Keep the last completed snapshot.
  try{
    await page.mouse.move(point.x,point.y-20,{steps:6});
    await expect.poll(()=>posts.filter(p=>p.path==='/api/sim/mouse-aim').length).toBe(2);
    await expect(page.locator('#mouse-aim-status')).toContainText('Checking local IK');
    await expect(page.locator('#inspect-mouse-aim')).toBeEnabled();
    await page.locator('#inspect-mouse-aim').hover();
    await expect(page.locator('#mouse-aim-status')).toContainText('Paused');
    await expect(page.locator('#inspect-mouse-aim')).toBeEnabled();
  }finally{release();}
  await expect.poll(()=>plans.length).toBe(2);
  await page.locator('#guided-route-repair').uncheck();
  await page.locator('#inspect-mouse-aim').click();
  await expect(page.locator('#guided-route-repair')).toBeChecked();
  await expect(page.locator('#mouse-aim-repair')).toHaveText('Guided repair: enabled for blocked routes');
  await expect(page.locator('#toggle-mouse-aim')).toHaveAttribute('aria-pressed','false');
  await expect(page.locator('#ik-task')).toHaveValue('pose5');
  await expect(page.locator('#direction-entry')).toHaveValue('axis');
  await expect(page.locator('#ik-feedback')).toContainText('Captured exact starting joints');
  for(const [i,a] of [...'xyz'].entries()){
    expect(Number(await page.locator(`#target-${a}`).inputValue())).toBe(target[i]);
    expect(Number(await page.locator(`#target-u${a}`).inputValue())).toBe(axis[i]);
    await expect(page.locator(`#eef-offset-${a}`)).toHaveValue('0');
  }
  await page.locator('#solve-ik').click();
  await expect.poll(()=>posts.filter(p=>p.path==='/api/sim/solve').length).toBe(2);
  const comparison=posts.at(-1).body;
  expect(comparison.start_deg).toEqual(snapshot.start_deg);
  expect(comparison.target_mm).toEqual(target);
  expect(comparison.task).toBe('pose5');
  comparison.target_axis.forEach((v,i)=>expect(v).toBeCloseTo(axis[i],12));
  await page.locator('#toggle-mouse-aim').click();
  await expect(page.locator('#inspect-mouse-aim')).toBeDisabled();
  expect(errors).toEqual([]);
});

test('trajectory gate rejects unchecked, stale, nonfinite and overspeed frames and keeps PRM corners',async({page})=>{
  await setup(page);
  const result=await page.evaluate(async info=>{
    const {validatedAimPath,sampleAimPath,advanceAimTime}=await import('/src/mouse-aim-path.js');
    const start=info.default_start_deg, cursor=[10,20,30];
    const end=start.slice();end[0]+=1;
    const data={simulation_only:true,task:'pose5',mouse_aim:true,validation:'joint_limits_and_self_collision',converged:true,
      cursor_acceptance:{accepted:true},gap_mm:50,coordinate_frame:'g_base',tool_frame:'joint6_flange',cursor_mm:cursor,
      frames:[{q:start.slice(),time:0},{q:end,time:.05}]};
    const check=d=>{try{validatedAimPath(d,start,info.limits_deg,1.2,cursor);return true;}catch{return false;}};
    const unchecked=structuredClone(data);unchecked.validation='joint_limits_only';
    const stale=structuredClone(data);stale.frames[0].q[0]+=1;
    const nan=structuredClone(data);nan.frames[1].q[0]=NaN;
    const fast=structuredClone(data);fast.frames[1].q[0]+=50;
    const reversed=structuredClone(data);reversed.frames[1].time=-1;
    const wrongCursor=structuredClone(data);wrongCursor.cursor_mm[0]+=1;
    const a=[0,0,0,0,0,0],b=[10,0,0,0,0,0],c=[10,10,0,0,0,0];
    const frames=[{q:a,time:0},{q:b,time:1},{q:c,time:2}];
    return {accepted:check(data),rejected:[unchecked,stale,nan,fast,reversed,wrongCursor].map(d=>!check(d)),
      first:sampleAimPath(frames,.5),corner:sampleAimPath(frames,1),second:sampleAimPath(frames,1.5),
      stopAtCorner:advanceAimTime(frames,.8,.5),afterCorner:advanceAimTime(frames,1,.5)};
  },fixture.info);
  expect(result.accepted).toBe(true);expect(result.rejected).toEqual(Array(6).fill(true));
  expect(result.first).toEqual([5,0,0,0,0,0]);expect(result.corner).toEqual([10,0,0,0,0,0]);
  expect(result.second).toEqual([10,5,0,0,0,0]);
  expect(result.stopAtCorner).toBe(1);expect(result.afterCorner).toBe(1.5);
});

test('mouse aim shares the repair toggle and stops when it changes',async({page})=>{
  const {posts,errors}=await setup(page,{},body=>({
    status:'roadmap_connection_not_found',reason:'Fixture route failure',converged:false,
    ik_outcome:'accepted_goals',planning_outcome:'start_not_connected',frames:[],
    guided_repair:{enabled:body.guided_route_repair,triggered:false,recovered:false},
  }));
  await page.locator('#guided-route-repair').uncheck();
  await expect(page.locator('#mouse-aim-repair')).toHaveText('Guided repair: off');
  await page.locator('#toggle-mouse-aim').click();
  const point=await cursorScreenPoint(page);await page.mouse.move(point.x,point.y);
  await expect(page.locator('#mouse-aim-status')).toContainText('Fixture route failure');
  const body=posts.find(p=>p.path==='/api/sim/mouse-aim').body;
  expect(body.guided_route_repair).toBe(false);
  expect(body.route_repair_samples).toBe(32);
  await page.locator('#guided-route-repair').check();
  await expect(page.locator('#toggle-mouse-aim')).toHaveAttribute('aria-pressed','false');
  await expect(page.locator('#mouse-aim-repair')).toHaveText('Guided repair: enabled for blocked routes');
  expect(posts.every(p=>p.path.startsWith('/api/sim/'))).toBe(true);
  expect(errors).toEqual([]);
});

function hardwarePlanner(options={}) {
  let session=null, sequence=0, lastResult=null, execution=null;
  const handler=async(path,body,state)=>{
    if(path==='/api/ik/mouse-aim/start') {
      session='mouse-session-1';
      Object.assign(state,{motion:null,moving:false,updated:Date.now()/1000,ik:{state:'idle',samples:[],mouse_aim_session_id:session,refresh_mode:1,tracking_protocol:2}});
      const response={...state,session_id:session};if(options.startHold)await options.startHold;return response;
    }
    if(path==='/api/ik/mouse-aim/plan') {
      const {session_id,...settings}=body;
      lastResult=await rpc('aim',{...settings,start_deg:state.angles});
      const response={session_id:session,hardware_mouse_aim:true,simulation_only:false,start_deg:state.angles.slice(),result:lastResult,
        plan:lastResult.converged?{id:`mouse-plan-${++sequence}`,converged:true}:null};
      if(options.planHold)await options.planHold;return response;
    }
    if(path==='/api/ik/execute') {
      execution=execution||body.plan_id;
      Object.assign(state,{motion:'ik',moving:true,updated:Date.now()/1000,ik:{state:'running',execution_id:execution,
        accepted_plan_id:body.plan_id,active_plan_id:body.plan_id,mouse_aim_session_id:session,samples:[],refresh_mode:1,
        velocity_limit_deg_s:6.7,acceleration_limit_deg_s2:60}});
      const response={...state};if(options.executeHold)await options.executeHold;return response;
    }
    if(path==='/api/ik/mouse-aim/heartbeat')return options.heartbeatFailure?{http_status:503,error:'Fixture heartbeat failure'}:{active:true};
    if(path==='/api/ik/mouse-aim/stop'||path==='/api/stop') {
      if(path==='/api/stop'||body.session_id===session){session=null;Object.assign(state,{motion:null,moving:false,updated:Date.now()/1000,ik:{state:'stopped',mouse_aim_session_id:null,samples:[]}});}
      return{...state};
    }
    throw new Error(`Unexpected hardware request ${path}`);
  };
  return{handler,get result(){return lastResult;},get execution(){return execution;},get session(){return session;}};
}
async function setupHardware(page,options={}) {
  const hardware=hardwarePlanner(options);
  const controls=await setup(page,{hardware_enabled:true,connected:true,angles:fixture.info.default_start_deg.slice(),updated:Date.now()/1000},undefined,hardware.handler);
  await page.locator('#hardware-planner').selectOption('classical');
  return{...controls,hardware};
}
async function startHardware(page) {
  await page.locator('#arm-movement').check();await page.locator('#verify-ik-frame').check();
  await page.locator('#ik-sdk-speed').fill('10');await page.locator('#toggle-mouse-aim').click();
}

test('hardware mouse aim requires confirmations, sends server plan IDs, and displays only live joints',async({page},testInfo)=>{
  const {posts,errors,hardware,setState}=await setupHardware(page);
  await expect(page.locator('#toggle-mouse-aim')).toBeDisabled();
  await page.locator('#arm-movement').check();await expect(page.locator('#toggle-mouse-aim')).toBeDisabled();
  await page.locator('#verify-ik-frame').check();await page.locator('#ik-sdk-speed').fill('10');await page.locator('#toggle-mouse-aim').click();
  await expect.poll(()=>posts.filter(p=>p.path==='/api/ik/mouse-aim/start').length).toBe(1);
  expect(posts.filter(p=>p.path==='/api/ik/execute')).toHaveLength(0);await expect(page.locator('#send-pose')).toBeDisabled();
  const point=await cursorScreenPoint(page,3,2);await page.mouse.move(point.x,point.y);
  await expect.poll(()=>posts.filter(p=>p.path==='/api/ik/execute').length,{timeout:90000}).toBe(1);
  const before=(await actualAim(page)).q;await page.waitForTimeout(1200);expect((await actualAim(page)).q).toEqual(before);
  expect(posts.find(p=>p.path==='/api/ik/execute').body).toEqual({plan_id:'mouse-plan-1',speed:10,armed:true,frame_verified:true});
  expect(posts.find(p=>p.path==='/api/ik/mouse-aim/plan').body.start_deg).toBeUndefined();
  expect(posts.find(p=>p.path==='/api/ik/mouse-aim/plan').body.guided_route_repair).toBe(true);
  expect(posts.some(p=>p.path==='/api/sim/mouse-aim'||p.path==='/api/move'||p.path==='/api/jog')).toBe(false);
  expect(posts.some(p=>p.path==='/api/ik/mouse-aim/heartbeat'&&p.body.execution_id==='mouse-plan-1')).toBe(true);
  await expect(page.locator('#mouse-aim-hardware-feedback')).toContainText('refresh mode 1 verified');
  await page.mouse.move(point.x+12,point.y+8);
  await expect.poll(()=>posts.filter(p=>p.path==='/api/ik/execute').length,{timeout:10000}).toBe(2);
  await expect.poll(()=>posts.some(p=>p.path==='/api/ik/mouse-aim/heartbeat'&&p.body.execution_id==='mouse-plan-1'&&p.body.plan_id==='mouse-plan-2')).toBe(true);
  setState({angles:hardware.result.frames.at(-1).q,moving:false,motion:null,ik:{state:'completed',execution_id:hardware.execution,mouse_aim_session_id:hardware.session,samples:[]}});
  expect(posts.filter(p=>p.path==='/api/ik/mouse-aim/plan')).toHaveLength(2);
  await page.mouse.move(1,1);await expect.poll(()=>posts.filter(p=>p.path==='/api/ik/mouse-aim/stop').length).toBe(1);
  await expect(page.locator('#toggle-mouse-aim')).toHaveAttribute('aria-pressed','false');
  await page.screenshot({path:testInfo.outputPath('hardware-mouse-aim-desktop.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});expect(await page.evaluate(()=>document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.screenshot({path:testInfo.outputPath('hardware-mouse-aim-mobile.png'),fullPage:true});expect(errors).toEqual([]);
});

test('stopping before a hardware session start response closes the late session without executing',async({page})=>{
  let release;const hold=new Promise(resolve=>{release=resolve;});const {posts,errors}=await setupHardware(page,{startHold:hold});
  try{await startHardware(page);await expect.poll(()=>posts.filter(p=>p.path==='/api/ik/mouse-aim/start').length).toBe(1);await page.locator('#toggle-mouse-aim').click();}finally{release();}
  await expect.poll(()=>posts.filter(p=>p.path==='/api/ik/mouse-aim/stop').length).toBe(1);
  expect(posts.some(p=>p.path==='/api/ik/execute'||p.path==='/api/ik/mouse-aim/plan')).toBe(false);expect(errors).toEqual([]);
});

test('leaving the canvas cancels a late hardware plan before execution',async({page})=>{
  let release;const hold=new Promise(resolve=>{release=resolve;});const {posts,errors}=await setupHardware(page,{planHold:hold});
  try{await startHardware(page);const point=await cursorScreenPoint(page,3,2);await page.mouse.move(point.x,point.y);
    await expect.poll(()=>posts.filter(p=>p.path==='/api/ik/mouse-aim/plan').length).toBe(1);await page.mouse.move(1,1);
    await expect.poll(()=>posts.filter(p=>p.path==='/api/ik/mouse-aim/stop').length).toBe(1);
  }finally{release();}
  await expect(page.locator('#toggle-mouse-aim')).toHaveAttribute('aria-pressed','false');expect(posts.some(p=>p.path==='/api/ik/execute')).toBe(false);expect(errors).toEqual([]);
});

test('a pending execution response cannot authorize a motor heartbeat after mouse aim stops',async({page})=>{
  let release;const hold=new Promise(resolve=>{release=resolve;});const {posts,errors}=await setupHardware(page,{executeHold:hold});
  try{await startHardware(page);const point=await cursorScreenPoint(page,3,2);await page.mouse.move(point.x,point.y);
    await expect.poll(()=>posts.filter(p=>p.path==='/api/ik/execute').length,{timeout:90000}).toBe(1);await page.waitForTimeout(1200);
    expect(posts.some(p=>p.path==='/api/ik/mouse-aim/heartbeat'&&p.body.execution_id)).toBe(false);await page.mouse.move(1,1);
    await expect.poll(()=>posts.filter(p=>p.path==='/api/ik/mouse-aim/stop').length).toBe(1);
  }finally{release();}
  await expect(page.locator('#toggle-mouse-aim')).toHaveAttribute('aria-pressed','false');
  expect(posts.some(p=>p.path==='/api/ik/mouse-aim/heartbeat'&&p.body.execution_id)).toBe(false);expect(errors).toEqual([]);
});

test('hardware heartbeat failure ends mouse tracking and requests owned stop',async({page})=>{
  const {posts,errors}=await setupHardware(page,{heartbeatFailure:true});await startHardware(page);
  await expect.poll(()=>posts.filter(p=>p.path==='/api/ik/mouse-aim/stop').length,{timeout:10000}).toBe(1);
  await expect(page.locator('#toggle-mouse-aim')).toHaveAttribute('aria-pressed','false');expect(errors).toEqual([]);
});
