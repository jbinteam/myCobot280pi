import {test,expect} from '@playwright/test';
import {spawn} from 'node:child_process';
import {createInterface} from 'node:readline';
import {writeFileSync} from 'node:fs';

let worker,fixture,pending=[],stderr='';
const code=`import json,sys,numpy as np
from simulation import service
from simulation.kinematics import ARM
for line in sys.stdin:
    try:
        request=json.loads(line)
        if request['path']=='info':
            info=service.info(); value={'info':info,'cursors':{}}
            for case in info['diffusion']['cases']:
                value['cursors'][case['id']]=(np.asarray(case['target_mm'])+50*np.asarray(case['target_axis'])).tolist()
        else:value=service.mouse_aim_plan(request['body'])
        print(json.dumps({'value':value},allow_nan=False),flush=True)
    except Exception as exc:print(json.dumps({'error':str(exc)}),flush=True)
`;
function rpc(path,body={}){return new Promise((resolve,reject)=>{pending.push({resolve,reject});worker.stdin.write(JSON.stringify({path,body})+'\n');});}
test.beforeAll(async({},testInfo)=>{
  testInfo.setTimeout(120000);
  worker=spawn('.venv/bin/python',['-u','-c',code],{cwd:'..',env:{...process.env,OPENBLAS_NUM_THREADS:'1'}});
  worker.stderr.on('data',v=>stderr=(stderr+v).slice(-4000));
  createInterface({input:worker.stdout}).on('line',line=>{const p=pending.shift();if(!p)return;const r=JSON.parse(line);r.error?p.reject(new Error(r.error)):p.resolve(r.value);});
  worker.on('exit',()=>{for(const p of pending.splice(0))p.reject(new Error(stderr));});
  fixture=await rpc('info');
});
test.afterAll(()=>worker?.kill());
test.setTimeout(120000);

async function setup(page,{delayFirst=false,delaySecond=false,noBlock=false,continuous=false,engine='local_diffusion',cartesianContract=false}={}){
  const cartesianSchema=cartesianContract===4?4:3;
  const posts=[],results=[],errors=[];let release,state={hardware_enabled:false,connected:false,motion:null,moving:false,token:'test',ik:{state:'idle'}};
  const gate=delayFirst||delaySecond?new Promise(r=>release=r):null;
  page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/api/**',async route=>{
    const request=route.request(),path=new URL(request.url()).pathname;
    if(path==='/api/state')return route.fulfill({json:state});
    if(path==='/api/sim/info')return route.fulfill({json:cartesianContract
      ? {...fixture.info,diffusion:{...fixture.info.diffusion,conditioning:{schema:cartesianSchema}}}:fixture.info});
    if(path==='/api/sim/solve')return route.fulfill({json:{results:[]}});
    if(path!=='/api/sim/mouse-aim')throw new Error(`Unexpected ${path}`);
    const body=request.postDataJSON();posts.push(body);
    const data=await rpc('aim',body);
    if(cartesianContract)Object.assign(data,{goal_deg:null,goal_cursor_acceptance:null,
      motion_conditioning:{schema:cartesianSchema,target_kind:'cartesian_pose5'}});
    results.push(data);
    if(gate&&results.length===(delaySecond?2:1))await gate;
    return route.fulfill({json:data}).catch(()=>{});
  });
  await page.goto('/?view=lab');await expect(page.locator('#model-status')).toHaveText('Model ready',{timeout:45000});
  await expect(page.locator('#mouse-aim-planner option[value="local_diffusion"]')).toBeEnabled();
  await page.locator('#diffusion-case').selectOption('000062');
  if(!continuous)await page.locator('#mouse-aim-continuous').uncheck();
  if(noBlock)await page.locator('#diffusion-obstacle-mode').selectOption('none');
  await page.locator('#ik-method').selectOption('bounded_dls');
  await page.locator('#mouse-aim-planner').selectOption(engine);
  await page.locator('#toggle-mouse-aim').click();
  await page.locator('.viewport-panel').scrollIntoViewIfNeeded();
  await page.evaluate(async()=>{
    const {renderer,scene,camera}=await import(document.querySelector('script[src*="/src/main.js"]').src);
    renderer.setAnimationLoop(null);renderer.render(scene,camera);window.aimTestTime=performance.now();
  });
  return {posts,results,errors,release,setState:next=>state={...state,...next}};
}
async function pointerAtMarker(page,dx=0,dy=0){
  await page.mouse.move(0,0);
  await page.locator('#viewport canvas').scrollIntoViewIfNeeded();
  const p=await page.evaluate(async()=>{
    const {scene,camera}=await import(document.querySelector('script[src*="/src/main.js"]').src);
    const anchor=scene.getObjectByName('mouse-aim-cursor').position.clone(),offset=anchor.clone().set(.004,.002,.001);
    const normal=offset.clone().cross(anchor.clone().set(0,0,1)).normalize();
    camera.position.copy(anchor).addScaledVector(normal,.7);camera.lookAt(anchor);camera.updateMatrixWorld(true);
    const p=anchor.add(offset).project(camera);
    const r=document.querySelector('#viewport canvas').getBoundingClientRect();
    return {x:r.left+(p.x+1)*r.width/2,y:r.top+(1-p.y)*r.height/2};
  });await page.mouse.move(p.x,p.y);return p;
}
async function tick(page,steps=1){await page.evaluate(async steps=>{
  const {mouseAim}=await import(document.querySelector('script[src*="/src/main.js"]').src);
  window.aimTestTime=Math.max(window.aimTestTime,performance.now());
  for(let i=0;i<steps;i++)mouseAim.update(window.aimTestTime+=50);
},steps);}
async function pose(page){return page.evaluate(async()=>{
  const {robot}=await import(document.querySelector('script[src*="/src/main.js"]').src);return Object.values(robot.joints).filter(j=>j.jointType==='revolute').map(j=>j.angle);
});}
async function readyMotion(page){await expect.poll(async()=>page.evaluate(async()=>{
  const {mouseAim}=await import(document.querySelector('script[src*="/src/main.js"]').src);return Number.isFinite(mouseAim.getPreviewMotion().time);
})).toBe(true);}

for(const engine of ['local_diffusion','classical','learned_rrt']){
  test(`${engine} continuous mouse aim reaches a stationary target without repeated chunk stops`,async({page})=>{
    const app=await setup(page,{continuous:true,noBlock:true,engine});await pointerAtMarker(page);await tick(page);
    await expect.poll(()=>app.results.length,{timeout:45000}).toBe(1);
    await readyMotion(page);
    await expect(page.locator('#hardware-velocity-profiles')).toBeVisible();
    await expect(page.locator('#mouse-planned-velocity-chart polyline')).toHaveCount(6);
    await expect(page.locator('#hardware-live-velocity-charts')).toBeHidden();
    await expect(page.locator('#velocity-profiles-mode')).toHaveText('SIMULATION PREVIEW');
    const first=app.results[0];expect(first.converged).toBe(true);expect(first.continuous_motion).toBe(true);
    expect(first.smoothing.method).toBe('collision_checked_velocity_profile_c2');expect(first.smoothing.ends_at_rest).toBe(true);
    await tick(page,200);expect(app.posts).toHaveLength(1);
    const motion=await page.evaluate(async()=>{const {mouseAim}=await import(document.querySelector('script[src*="/src/main.js"]').src);return mouseAim.getPreviewMotion();});
    motion.q.forEach((q,j)=>expect(q).toBeCloseTo(first.frames.at(-1).q[j],7));
    expect(Math.hypot(...motion.velocity_rad_s)).toBe(0);await page.keyboard.press('Escape');
    await expect(page.locator('#hardware-velocity-profiles')).toBeHidden();
    await expect(page.locator('#mouse-planned-velocity-chart polyline')).toHaveCount(0);expect(app.errors).toEqual([]);
  });

  test(`${engine} keeps moving during planning and joins the next target with nonzero velocity`,async({page},testInfo)=>{
    const app=await setup(page,{continuous:true,noBlock:true,engine,delaySecond:true});const p=await pointerAtMarker(page);
    await page.mouse.move(p.x+25,p.y+15);await tick(page);await expect.poll(()=>app.results.length,{timeout:45000}).toBe(1);
    await readyMotion(page);
    const first=app.results[0];expect(first.converged).toBe(true);await tick(page,3);
    const firstProfile=await page.locator('#mouse-planned-velocity-chart polyline').evaluateAll(lines=>lines.map(line=>line.getAttribute('points')));
    await page.mouse.move(p.x+24,p.y+15);await tick(page);await expect.poll(()=>app.results.length,{timeout:45000}).toBe(2);
    const second=app.results[1],request=app.posts[1];
    expect(await page.locator('#mouse-planned-velocity-chart polyline').evaluateAll(lines=>lines.map(line=>line.getAttribute('points')))).toEqual(firstProfile);
    expect(second.converged,JSON.stringify({status:second.status,reason:second.reason,local:second.local_ik})).toBe(true);
    expect(Math.hypot(...request.start_velocity_rad_s)).toBeGreaterThan(.01);
    if(engine==='local_diffusion'){
      expect(request.target_velocity_m_s).toHaveLength(3);
      expect(request.target_velocity_m_s.every(Number.isFinite)).toBe(true);
      expect(Math.hypot(...request.target_velocity_m_s)).toBeLessThanOrEqual(.5000001);
    }
    const before=await pose(page);await tick(page,2);expect(await pose(page)).not.toEqual(before);
    request.start_velocity_rad_s.forEach((v,j)=>expect(second.smoothing.start_velocity_rad_s[j]).toBeCloseTo(v,9));
    request.start_acceleration_rad_s2.forEach((v,j)=>expect(second.smoothing.start_acceleration_rad_s2[j]).toBeCloseTo(v,9));
    app.release();await page.waitForTimeout(100);
    const trace=await page.evaluate(async()=>{
      const {mouseAim}=await import(document.querySelector('script[src*="/src/main.js"]').src);
      window.aimTestTime=Math.max(window.aimTestTime,performance.now());
      // Synchronize the old reference after the route release and screenshot
      // bookkeeping before measuring fixed 10 ms intervals. A wall-clock jump
      // here can legitimately advance a capped 50 ms on its first update.
      mouseAim.update(window.aimTestTime);const values=[mouseAim.getPreviewMotion()];
      for(let i=0;i<150;i++){mouseAim.update(window.aimTestTime+=10);values.push(mouseAim.getPreviewMotion());}
      return values;
    });
    const join=trace.findIndex((m,i)=>i>0&&Number.isFinite(m.time)&&Number.isFinite(trace[i-1].time)&&m.time<trace[i-1].time);
    expect(join).toBeGreaterThan(0);
    expect(await page.locator('#mouse-planned-velocity-chart polyline').evaluateAll(lines=>lines.map(line=>line.getAttribute('points')))).not.toEqual(firstProfile);
    expect(Math.hypot(...trace[join].velocity_rad_s)).toBeGreaterThan(.01);
    for(let i=1;i<trace.length;i++){
      const delta=trace[i].q.map((q,j)=>(q-trace[i-1].q[j])*Math.PI/180);
      expect(Math.hypot(...delta)).toBeLessThanOrEqual(.010001);
      const dv=trace[i].velocity_rad_s.map((v,j)=>(v-trace[i-1].velocity_rad_s[j])/.01);
      expect(Math.hypot(...dv)).toBeLessThanOrEqual(3.00001);
    }
    await page.evaluate(async()=>{const {renderer,scene,camera}=await import(document.querySelector('script[src*="/src/main.js"]').src);renderer.render(scene,camera);});
    await page.locator('.viewport-panel').screenshot({path:testInfo.outputPath('continuous-mouse-join.png')});
    await page.keyboard.press('Escape');expect(app.errors).toEqual([]);
  });
}

test('a late continuous continuation keeps the checked braking endpoint without a jump',async({page})=>{
  const app=await setup(page,{continuous:true,noBlock:true,delaySecond:true});const p=await pointerAtMarker(page);
  await page.mouse.move(p.x+25,p.y+15);await tick(page);await expect.poll(()=>app.results.length,{timeout:45000}).toBe(1);
  await readyMotion(page);
  await tick(page,3);await page.mouse.move(p.x+24,p.y+15);await tick(page);
  await expect.poll(()=>app.results.length,{timeout:45000}).toBe(2);
  expect(app.results[1].converged,JSON.stringify({status:app.results[1].status,reason:app.results[1].reason})).toBe(true);
  expect(Math.hypot(...app.posts[1].start_velocity_rad_s)).toBeGreaterThan(.01);
  await tick(page,300);const endpoint=await pose(page);
  app.results[0].frames.at(-1).q.forEach((q,j)=>expect(endpoint[j]).toBeCloseTo(q*Math.PI/180,8));
  app.release();await expect(page.locator('#mouse-aim-status')).toContainText('after its join');
  expect(await pose(page)).toEqual(endpoint);await page.keyboard.press('Escape');expect(app.errors).toEqual([]);
});

test('hybrid tracking plays checked short prefixes and replans a stationary cursor from current joints',async({page},testInfo)=>{
  const app=await setup(page);await pointerAtMarker(page,3,2);await tick(page);
  // The first real inference includes lazy Torch/checkpoint loading and can
  // exceed Playwright's default five seconds while dataset workers are active.
  await expect.poll(()=>app.results.length,{timeout:45000}).toBeGreaterThan(0);
  const first=app.results[0];expect(first.playable).toBe(true);expect(first.replan_required).toBe(true);
  expect(first.tracking_source).toBe('local_ik');expect(first.frames.at(-1).time).toBeLessThanOrEqual(1.3);
  expect(first.smoothing.starts_and_ends_at_rest).toBe(true);expect(first.max_acceleration_rad_s2).toBeLessThanOrEqual(3+1e-8);
  try {
    for(let i=0;i<30&&!app.results.at(-1)?.converged;i++){
      const previous=app.results.length;
      await tick(page,66);
      // Complete each real asynchronous solve before advancing its playback.
      // A fixed 60 ms delay can run thirty clock steps through only five solves.
      await expect.poll(()=>app.results.length,{timeout:45000}).toBeGreaterThan(previous);
      await page.waitForTimeout(20);
    }
  } finally {
    writeFileSync('../output/cone_disk_tracking_20261008/browser-stationary-chunks.json',JSON.stringify({posts:app.posts,results:app.results},null,2));
  }
  await expect.poll(()=>app.results.at(-1)?.converged).toBe(true);
  await tick(page,66);
  expect(app.posts.length).toBeGreaterThan(1);
  expect(app.posts[1].target_axis).toEqual(app.posts[0].target_axis);
  if([3,4].includes(fixture.info.diffusion.conditioning?.schema))expect(app.posts[1].goal_hint_deg).toBeUndefined();
  else expect(app.posts[1].goal_hint_deg).toEqual(first.goal_deg);
  first.frames.at(-1).q.forEach((v,i)=>expect(app.posts[1].start_deg[i]).toBeCloseTo(v,7));
  expect(app.posts.every(b=>b.tracking_engine==='local_diffusion')).toBe(true);
  await expect(page.locator('#mouse-aim-engine')).toContainText('diffusion fallback');
  expect(await page.evaluate(async()=>{const {scene}=await import(document.querySelector('script[src*="/src/main.js"]').src);return scene.getObjectByName('diffusion-scene').visible;})).toBe(true);
  await page.evaluate(async()=>{const {renderer,scene,camera}=await import(document.querySelector('script[src*="/src/main.js"]').src);renderer.render(scene,camera);});
  await page.locator('.viewport-panel').screenshot({path:testInfo.outputPath('hybrid-mouse-tracking.png')});
  await page.keyboard.press('Escape');await page.setViewportSize({width:390,height:844});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.evaluate(async()=>{const {renderer,scene,camera}=await import(document.querySelector('script[src*="/src/main.js"]').src);await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));renderer.render(scene,camera);});
  await page.locator('.viewport-panel').screenshot({path:testInfo.outputPath('hybrid-mouse-tracking-mobile.png')});
  expect(app.errors).toEqual([]);
});

test('a changed cursor discards an in-flight chunk without moving the pose',async({page})=>{
  const app=await setup(page,{delayFirst:true});const p=await pointerAtMarker(page,3,2);
  const before=await pose(page);await tick(page);await expect.poll(()=>app.results.length,{timeout:45000}).toBe(1);
  await page.mouse.move(p.x+8,p.y+4);await tick(page);app.release();
  await expect(page.locator('#mouse-aim-status')).toContainText('discarded stale');
  expect(await pose(page)).toEqual(before);
  await page.evaluate(()=>window.aimTestTime=Math.max(window.aimTestTime,performance.now()));
  await tick(page,6);await expect.poll(()=>app.results.length).toBe(2);
  expect(app.posts[1].cursor_mm).not.toEqual(app.posts[0].cursor_mm);
  await page.keyboard.press('Escape');expect(app.errors).toEqual([]);
});

for(const schema of [3,4])test(`Cartesian schema ${schema} plays and replans without goal joints or hints`,async({page})=>{
  // Existing checked routes isolate browser behavior from neural accuracy.
  const app=await setup(page,{cartesianContract:schema,noBlock:true});await pointerAtMarker(page);await tick(page);
  await expect.poll(()=>app.results.length,{timeout:45000}).toBeGreaterThan(0);
  expect(app.results[0].playable).toBe(true);expect(app.results[0].goal_deg).toBeNull();
  await readyMotion(page);await tick(page,66);await page.waitForTimeout(100);await tick(page,20);
  await expect.poll(()=>app.posts.length,{timeout:45000}).toBeGreaterThan(1);
  expect(app.posts.every(body=>body.goal_deg===undefined&&body.goal_hint_deg===undefined)).toBe(true);
  await page.keyboard.press('Escape');expect(app.errors).toEqual([]);
});

test('no-block mouse tracking plays a checked local prefix and scene switching stops it',async({page},testInfo)=>{
  const app=await setup(page,{noBlock:true});await pointerAtMarker(page);const before=await pose(page);
  await tick(page);await expect.poll(()=>app.results.length,{timeout:45000}).toBe(1);
  const first=app.results[0];
  expect(app.posts[0].obstacle_mm).toBeNull();expect(first.obstacle_mm).toBeNull();expect(first.scene_mode).toBe('no_block');
  expect(first.playable).toBe(true);expect(first.tracking_source).toBe('local_ik');
  expect(first.collision_checks).toEqual(['joint_limits','self','table']);expect(first.smoothing.collision_rechecked).toBe(true);
  await tick(page,2);await expect(page.locator('#mouse-aim-status')).toContainText('Following validated');
  expect(await pose(page)).not.toEqual(before);await expect(page.locator('#mouse-aim-engine')).toContainText('no block');
  await expect(page.locator('#mouse-aim-planner option:checked')).toContainText('no block');
  const scene=await page.evaluate(async()=>{
    const {scene,renderer,camera}=await import(document.querySelector('script[src*="/src/main.js"]').src);renderer.render(scene,camera);
    return {visible:scene.getObjectByName('diffusion-scene').visible,block:scene.getObjectByName('diffusion-block').visible,
      table:scene.getObjectByName('diffusion-table').visible,points:scene.getObjectByName('mouse-trajectory-output-points').geometry.attributes.position.count,
      target:Array.from(scene.getObjectByName('mouse-trajectory-output-points').geometry.attributes.position.array),
      movingMarker:!!scene.getObjectByName('mouse-trajectory-output-current')};
  });
  expect(scene).toMatchObject({visible:true,block:false,table:true,points:1,movingMarker:false});
  scene.target.forEach((value,j)=>expect(value).toBeCloseTo(first.target_mm[j]/1000,6));
  await page.locator('.viewport-panel').screenshot({path:testInfo.outputPath('no-block-mouse-tracking.png')});
  await page.locator('#diffusion-obstacle-mode').selectOption('block');
  await expect(page.locator('#toggle-mouse-aim')).toHaveAttribute('aria-pressed','false');
  await expect(page.locator('#mouse-aim-planner option:checked')).toContainText('/ block');
  await expect(page.locator('#diffusion-width')).toBeEnabled();expect(app.errors).toEqual([]);
});

test('normal retargeting finishes the smoothed chunk before planning from its endpoint',async({page})=>{
  const app=await setup(page);const p=await pointerAtMarker(page);
  await tick(page);await expect.poll(()=>app.results.length).toBe(1);
  const first=app.results[0];expect(first.playable).toBe(true);
  await tick(page,2);await expect(page.locator('#mouse-aim-status')).toContainText('Following validated');
  await page.mouse.move(p.x+8,p.y+4);await tick(page,2);
  expect(app.posts).toHaveLength(1);
  await tick(page,66);await expect.poll(()=>app.posts.length).toBe(2);
  first.frames.at(-1).q.forEach((v,i)=>expect(app.posts[1].start_deg[i]).toBeCloseTo(v,7));
  expect(app.posts[1].cursor_mm).not.toEqual(app.posts[0].cursor_mm);
  await page.keyboard.press('Escape');expect(app.errors).toEqual([]);
});

test('colliding starts hold, scene edits stop tracking and hardware hides the simulation planner',async({page})=>{
  const app=await setup(page);
  await page.locator('#diffusion-width').fill('500');await expect(page.locator('#toggle-mouse-aim')).toHaveAttribute('aria-pressed','false');
  await page.locator('#diffusion-depth').fill('500');await page.locator('#diffusion-x').fill('0');await page.locator('#diffusion-y').fill('0');await page.locator('#diffusion-height').fill('100');
  await page.locator('#toggle-mouse-aim').click();await pointerAtMarker(page,3,2);const before=await pose(page);await tick(page);
  await expect.poll(()=>app.results.length).toBe(1);await tick(page,6);
  expect(app.results[0].status).toBe('invalid_start');expect(app.results[0].frames).toEqual([]);expect(await pose(page)).toEqual(before);
  await expect(page.locator('#mouse-aim-status')).toContainText('Held');
  await page.evaluate(()=>window.dispatchEvent(new CustomEvent('robot-state',{detail:{hardware_enabled:true,connected:false,motion:null,moving:false}})));
  await expect(page.locator('#mouse-aim-planner-control')).toBeHidden();await expect(page.locator('#mouse-aim-diffusion-note')).toBeHidden();
  await expect(page.locator('#toggle-mouse-aim')).toHaveAttribute('aria-pressed','false');expect(app.errors).toEqual([]);
});

test('tracking gate rejects missing goal checks, wrong scenes, stale checkpoints and oversized chunks',async({page})=>{
  await setup(page);
  const result=await page.evaluate(async info=>{
    const {validatedAimPath}=await import('/src/mouse-aim-path.js');
    const start=info.default_start_deg,cursor=[10,20,30],box=[180,0,75,25,35,110],checkpoint=info.diffusion.checkpoint_sha256;
    const q=start.slice();q[0]+=.5;
    const frames=Array.from({length:11},(_,i)=>{const u=i/10,s=10*u**3-15*u**4+6*u**5;return {q:start.map((v,j)=>v+(q[j]-v)*s),time:i*.02};});
    const data={method:'local_diffusion',simulation_only:true,task:'pose5',mouse_aim:true,validation:'joint_limits_self_table_block',
      tracking_strategy:'local_then_diffusion',tracking_chunk:true,playable:true,converged:false,replan_required:true,
      goal_cursor_acceptance:{accepted:true},cursor_acceptance:{accepted:false},gap_mm:50,coordinate_frame:'g_base',tool_frame:'joint6_flange',
      cursor_mm:cursor,obstacle_mm:box,goal_deg:q.slice(),checkpoint_sha256:checkpoint,speed_rad_s:1,acceleration_rad_s2:3,
      smoothing:{method:'collision_checked_shortcuts_quintic',applied:true,collision_rechecked:true,starts_and_ends_at_rest:true,dt:.02,frames:11,duration_s:.2},frames};
    const check=(d,selectedBox=box,conditioningSchema)=>{try{validatedAimPath(d,start,info.limits_deg,1,cursor,{box:selectedBox,checkpoint,conditioningSchema});return true;}catch{return false;}};
    const mutations=[d=>d.goal_cursor_acceptance.accepted=false,d=>d.obstacle_mm[0]+=1,d=>d.checkpoint_sha256='old',d=>d.frames.at(-1).time=1.4,d=>d.frames[0].q[0]+=1,d=>d.playable=false,d=>d.converged=true,d=>d.frames[1].q[0]+=1];
    const noBlock={...structuredClone(data),obstacle_mm:null,scene_mode:'no_block'};
    const cartesian={...structuredClone(data),goal_deg:null,goal_cursor_acceptance:null,
      motion_conditioning:{schema:3,target_kind:'cartesian_pose5'}};
    const relative={...structuredClone(cartesian),motion_conditioning:{schema:4,target_kind:'cartesian_pose5'}};
    return {accepted:check(data),rejected:mutations.map(m=>{const d=structuredClone(data);m(d);return !check(d);}),
      cartesianAccepted:check(cartesian,box,3),cartesianWrongSchemaRejected:!check(cartesian,box,2),
      cartesianMissingSchemaRejected:!check({...cartesian,motion_conditioning:null},box,3),
      relativeAccepted:check(relative,box,4),relativeMismatchedSchemaRejected:!check(relative,box,3)&&!check(cartesian,box,4),
      noBlockAccepted:check(noBlock,null),sceneRejected:[!check(data,null),!check(noBlock),
        !check({...noBlock,scene_mode:'block'},null),!check({...noBlock,scene_mode:undefined},null)]};
  },fixture.info);
  expect(result.accepted).toBe(true);expect(result.rejected).toEqual(Array(8).fill(true));
  expect(result.cartesianAccepted).toBe(true);expect(result.cartesianWrongSchemaRejected).toBe(true);
  expect(result.cartesianMissingSchemaRejected).toBe(true);
  expect(result.relativeAccepted).toBe(true);expect(result.relativeMismatchedSchemaRejected).toBe(true);
  expect(result.noBlockAccepted).toBe(true);expect(result.sceneRejected).toEqual(Array(4).fill(true));
});
