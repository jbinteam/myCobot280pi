import {test,expect} from '@playwright/test';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
const runPython=promisify(execFile);

async function simulationPage(page,request,{hardware=false,delay=false,decisionLimit=null}={}) {
  // Capture the real preview animation callback so time advancement does not
  // depend on headless Chrome's offscreen WebGL frame throttling.
  await page.addInitScript(()=>{
    const raf=window.requestAnimationFrame.bind(window);
    window.requestAnimationFrame=callback=>{
      if(callback.name==='animate')window.previewAnimationTick=callback;
      return raf(callback);
    };
  });
  const state=await(await request.get('http://127.0.0.1:8000/api/state')).json();
  const info=await(await request.get('http://127.0.0.1:8000/api/sim/info')).json();
  const posts=[],errors=[];let result,release;
  const gate=delay?new Promise(resolve=>release=resolve):null;
  page.on('pageerror',error=>errors.push(error.message));
  await page.route('**/api/**',async route=>{
    const req=route.request(),path=new URL(req.url()).pathname;
    if(path==='/api/state')return route.fulfill({json:{...state,hardware_enabled:hardware,connected:false,moving:false,motion:null}});
    if(path==='/api/sim/info')return route.fulfill({json:info});
    if(req.method()!=='POST'||!path.startsWith('/api/sim/'))throw new Error(`Unexpected request ${path}`);
    posts.push({path,body:req.postDataJSON()});
    const body=req.postDataJSON();
    if(path==='/api/sim/diffusion'&&decisionLimit!==null)body.max_decisions=decisionLimit;
    // A live hardware bridge rejects diffusion simulation requests. Run the
    // same read-only planner separately rather than changing the robot mode.
    let data,status=200;
    if(path==='/api/sim/diffusion'&&state.hardware_enabled){
      const {stdout}=await runPython('.venv/bin/python',['-c',
        'import json,sys; from simulation.diffusion.service import plan; print(json.dumps(plan(json.loads(sys.argv[1])),allow_nan=False))',JSON.stringify(body)],
        {cwd:'..',env:{...process.env,OPENBLAS_NUM_THREADS:'1',OMP_NUM_THREADS:'1'},timeout:100000,maxBuffer:10*1024*1024});
      data=JSON.parse(stdout);
    }else{
      const response=await request.post(`http://127.0.0.1:8000${path}`,{headers:{'X-Robot-Token':state.token},data:body,timeout:100000});
      status=response.status();data=await response.json();
    }
    if(path==='/api/sim/diffusion'){result=data;if(gate)await gate;}
    return route.fulfill({status,json:data});
  });
  await page.goto('/?view=lab');
  if(hardware)await expect(page.locator('#seed5-checkpoint-status')).toContainText('5D Seed NN:');
  else await expect(page.locator('#run-results tbody tr')).toHaveCount(4,{timeout:45000});
  await expect(page.locator('#model-status')).toHaveText('Model ready',{timeout:45000});
  // These planner tests drive the real playback callback below. Avoid a
  // continuous software WebGL redraw while doing API/control assertions.
  await page.evaluate(async()=>{
    const {renderer,scene,camera}=await import('/src/main.js');
    renderer.setAnimationLoop(null);renderer.render(scene,camera);
  });
  return {info,posts,errors,result:()=>result,release};
}

async function sceneState(page) {
  return page.evaluate(async()=>{
    const {scene}=await import('/src/main.js');
    const group=scene.getObjectByName('diffusion-scene'),box=scene.getObjectByName('diffusion-block'),table=scene.getObjectByName('diffusion-table');
    return {visible:group.visible,blockVisible:box.visible,tableVisible:table.visible,
      position:box.position.toArray(),size:box.scale.toArray(),tableTop:table.position.z+.025};
  });
}

test('trained diffusion detour renders only the final target and continuous quintic velocity',async({page,request},testInfo)=>{
  test.setTimeout(150000);
  const app=await simulationPage(page,request);
  await expect(page.locator('#diffusion-checkpoint')).toContainText('Trained DDPM ready');
  await expect(page.locator('#plan-diffusion')).toBeEnabled();
  await page.locator('#diffusion-case').selectOption('000590');
  const box=app.info.diffusion.cases[0].obstacle_mm,scene=await sceneState(page);
  expect(scene.visible).toBe(true);expect(scene.tableTop).toBeCloseTo(-.035,8);
  box.slice(0,3).forEach((v,i)=>expect(scene.position[i]).toBeCloseTo(v/1000,8));
  box.slice(3).forEach((v,i)=>expect(scene.size[i]).toBeCloseTo(v/500,8));
  await page.locator('#plan-diffusion').click();
  await expect(page.locator('#diffusion-result')).toContainText('Validated smoothed diffusion preview',{timeout:90000});
  const result=app.result();
  expect(result.converged).toBe(true);expect(result.validation).toBe('joint_limits_self_table_block');
  expect(result.checkpoint_sha256).toBe(app.info.diffusion.checkpoint_sha256);
  expect(result.rejections.raw_collision).toBeGreaterThan(0);
  expect(result.smoothing.collision_rechecked).toBe(true);expect(result.max_acceleration_rad_s2).toBeLessThanOrEqual(3+1e-8);
  await expect(page.locator('#run-results tbody tr')).toHaveCount(1);
  await expect(page.locator('#run-results tbody')).toContainText('Checked diffusion');
  await page.locator('#ik-scrub').fill(String(result.frames.length-1));
  const pose=JSON.parse(await page.locator('#pose-value').innerText());
  result.frames.at(-1).q.forEach((v,i)=>expect(pose[i]).toBeCloseTo(v,1));
  // Probe inside a real retained segment. New routes may have an early corner;
  // playback deliberately stops a render-time jump at that checked boundary.
  const segment=result.smoothing.segments.reduce((a,b)=>a.duration_s>b.duration_s?a:b);
  const probeTime=segment.start_time_s+Math.min(1.007,segment.duration_s/2)+.003;
  const probeDelta=probeTime-segment.start_time_s;
  await page.locator('#ik-scrub').fill(String(segment.start_frame));await page.locator('#playback-speed').selectOption('1');
  const playback=await page.evaluate(async ({result,probeTime,probeDelta})=>{
    const {createTrajectorySampler}=await import('/src/trajectory-playback.js');
    const {robot,scene}=await import('/src/main.js');
    const button=document.getElementById('play-ik'),raf=window.requestAnimationFrame;
    const now=performance.now.bind(performance);let startedState;
    try {
      performance.now=()=>1000;button.click();startedState=button.textContent;
      window.requestAnimationFrame=()=>0;window.previewAnimationTick(1000+probeDelta*1000);
      button.click();
    } finally {performance.now=now;window.requestAnimationFrame=raf;}
    const q=Object.values(robot.joints).filter(j=>j.jointType==='revolute').map(j=>j.angle*180/Math.PI);
    robot.updateMatrixWorld(true);
    const marker=Array.from(scene.getObjectByName('trajectory-output-points').geometry.attributes.position.array);
    const tcp=Array.from(robot.links.joint6_flange.matrixWorld.elements).slice(12,15);
    return {startedState,index:Number(document.getElementById('ik-scrub').value),q,
      expected:createTrajectorySampler(result.frames,result.smoothing).sample(probeTime).q,marker,tcp,
      points:scene.getObjectByName('trajectory-output-points').geometry.attributes.position.count};
  },{result,probeTime,probeDelta});
  expect(playback.startedState).toBe('Ⅱ Pause');
  const index=playback.index;expect(index).toBe(result.frames.findLastIndex(frame=>frame.time<=probeTime));
  playback.q.forEach((q,j)=>expect(q).toBeCloseTo(playback.expected[j],7));
  expect(playback.q.some((q,j)=>Math.abs(q-result.frames[index].q[j])>1e-4)).toBe(true);
  playback.marker.forEach((v,j)=>expect(v).toBeCloseTo(result.target_mm[j]/1000,6));
  expect(playback.points).toBe(1);
  await expect(page.locator('#ik-time')).toHaveText(`${probeTime.toFixed(2)} s`);
  await expect(page.locator('#velocity-chart svg')).toHaveAttribute('aria-label',/Joint velocities/);
  await expect(page.locator('#velocity-chart polyline')).toHaveCount(6);
  for(let j=1;j<=6;j++)await expect(page.locator('#velocity-chart')).toContainText(`J${j}`);
  await expect(page.locator('#trajectory-speed')).toContainText('rad/s');
  await page.locator('#show-trajectory-points').uncheck();
  expect(await page.evaluate(async()=>{const {scene}=await import('/src/main.js');return scene.getObjectByName('trajectory-output').visible;})).toBe(false);
  await page.locator('#show-trajectory-points').check();
  const resumed=await page.evaluate(async ({result,probeTime})=>{
    const {createTrajectorySampler}=await import('/src/trajectory-playback.js');
    const {robot}=await import('/src/main.js');
    const button=document.getElementById('play-ik'),raf=window.requestAnimationFrame,now=performance.now.bind(performance);
    try{performance.now=()=>3000;window.requestAnimationFrame=()=>0;button.click();window.previewAnimationTick(3017);button.click();}
    finally{performance.now=now;window.requestAnimationFrame=raf;}
    return {q:Object.values(robot.joints).filter(j=>j.jointType==='revolute').map(j=>j.angle*180/Math.PI),
      expected:createTrajectorySampler(result.frames,result.smoothing).sample(createTrajectorySampler(result.frames,result.smoothing).advance(probeTime,.017)).q};
  },{result,probeTime});
  resumed.q.forEach((q,j)=>expect(q).toBeCloseTo(resumed.expected[j],7));
  // Freeze only the test's WebGL redraw loop for stable software-rendered
  // screenshots, after exercising the actual playback callback above.
  await page.evaluate(async()=>{
    const {renderer,scene,camera}=await import('/src/main.js');
    renderer.setAnimationLoop(null);renderer.render(scene,camera);
  });
  await page.screenshot({path:testInfo.outputPath('diffusion-desktop.png'),fullPage:true});
  await page.locator('.viewport-panel').screenshot({path:testInfo.outputPath('diffusion-scene.png')});
  await page.setViewportSize({width:390,height:844});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.evaluate(async()=>{
    const {renderer,scene,camera}=await import('/src/main.js');
    await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
    renderer.render(scene,camera);
  });
  await page.screenshot({path:testInfo.outputPath('diffusion-mobile.png'),fullPage:true});
  await page.locator('.viewport-panel').screenshot({path:testInfo.outputPath('diffusion-mobile-scene.png')});
  const held=await page.locator('#pose-value').innerText();
  await page.locator('#diffusion-width').fill('50');
  await expect(page.locator('#play-ik')).toBeDisabled();await expect(page.locator('#diffusion-result')).toBeEmpty();
  expect(await page.locator('#pose-value').innerText()).toBe(held);
  expect(app.posts.every(p=>p.path.startsWith('/api/sim/'))).toBe(true);expect(app.errors).toEqual([]);
});

test('collision failure holds the pose and switching controllers hides the diffusion scene',async({page,request})=>{
  test.setTimeout(120000);
  const app=await simulationPage(page,request);
  await page.locator('#diffusion-case').selectOption('000062');
  for(const [id,value] of [['x','0'],['y','0'],['width','500'],['depth','500'],['height','100']])await page.locator(`#diffusion-${id}`).fill(value);
  const held=await page.locator('#pose-value').innerText();
  await page.locator('#plan-diffusion').click();
  await expect(page.locator('#diffusion-result')).toContainText('No playable path',{timeout:60000});
  expect(app.result().status).toBe('invalid_start');expect(app.result().frames).toEqual([]);
  await expect(page.locator('#play-ik')).toBeDisabled();
  expect(await page.locator('#pose-value').innerText()).toBe(held);
  await page.locator('#solve-ik').click();
  await expect(page.locator('#solve-ik')).toBeEnabled({timeout:45000});
  expect((await sceneState(page)).visible).toBe(false);expect(app.errors).toEqual([]);
});

test('no-block mode removes only the box and safely holds an exhausted neural rollout',async({page,request},testInfo)=>{
  test.setTimeout(120000);
  const app=await simulationPage(page,request,{decisionLimit:1});
  await page.locator('#diffusion-case').selectOption('000062');
  await page.locator('#diffusion-obstacle-mode').selectOption('none');
  await expect(page.locator('#diffusion-width')).toBeDisabled();
  await expect(page.locator('#diffusion-scene-readout')).toContainText('self-collision and table clearance remain checked');
  if(app.info.diffusion.no_block_note)await expect(page.locator('#diffusion-scene-readout')).toContainText('checkpoint trained with blocks');
  await page.locator('#diffusion-case').selectOption('000590');
  await expect(page.locator('#diffusion-obstacle-mode')).toHaveValue('none');
  await page.locator('#diffusion-case').selectOption('000062');
  await page.locator('#diffusion-budget').selectOption('50');
  const scene=await sceneState(page);
  expect(scene.visible).toBe(true);expect(scene.blockVisible).toBe(false);expect(scene.tableVisible).toBe(true);
  expect(scene.tableTop).toBeCloseTo(-.035,8);
  const held=await page.locator('#pose-value').innerText();
  await page.locator('#plan-diffusion').click();
  await expect(page.locator('#diffusion-result')).toContainText('No playable path',{timeout:90000});
  const result=app.result(),post=app.posts.find(p=>p.path==='/api/sim/diffusion');
  expect(post.body.obstacle_mm).toBeNull();expect(result.obstacle_mm).toBeNull();expect(result.scene_mode).toBe('no_block');
  expect(result.collision_checks).toEqual(['joint_limits','self','table']);
  expect(result.decisions).toBeGreaterThan(0);expect(result.decisions).toBeLessThanOrEqual(50);
  expect(result.converged).toBe(false);expect(result.frames).toEqual([]);
  await expect(page.locator('#play-ik')).toBeDisabled();expect(await page.locator('#pose-value').innerText()).toBe(held);
  await page.evaluate(async()=>{const {renderer,scene,camera}=await import('/src/main.js');renderer.render(scene,camera);});
  await page.screenshot({path:testInfo.outputPath('no-block-preview.png'),fullPage:true});
  await page.locator('#diffusion-obstacle-mode').selectOption('block');
  await expect(page.locator('#diffusion-result')).toBeEmpty();await expect(page.locator('#diffusion-width')).toBeEnabled();
  expect((await sceneState(page)).blockVisible).toBe(true);
  for(const hardware_enabled of [true,false]){
    await page.evaluate(hardware_enabled=>window.dispatchEvent(new CustomEvent('robot-state',{
      detail:{hardware_enabled,connected:false,motion:null,moving:false}})),hardware_enabled);
    if(hardware_enabled){
      await expect(page.locator('#diffusion-obstacle-mode')).toBeDisabled();await expect(page.locator('#diffusion-width')).toBeDisabled();
    } else {
      await expect(page.locator('#diffusion-obstacle-mode')).toBeEnabled();await expect(page.locator('#diffusion-width')).toBeEnabled();
    }
  }
  expect(app.errors).toEqual([]);
});

test('edited inputs discard a delayed diffusion response and hardware mode disables this planner',async({page,request})=>{
  test.setTimeout(120000);
  const app=await simulationPage(page,request,{delay:true});
  await page.locator('#diffusion-case').selectOption('000062');
  const held=await page.locator('#pose-value').innerText();
  await page.locator('#plan-diffusion').click();
  await expect.poll(()=>app.posts.some(p=>p.path==='/api/sim/diffusion')).toBe(true);
  await page.locator('#target-x').fill('100');app.release();
  await expect(page.locator('#plan-diffusion')).toBeEnabled({timeout:60000});
  await expect(page.locator('#play-ik')).toBeDisabled();await expect(page.locator('#diffusion-result')).toBeEmpty();
  expect(await page.locator('#pose-value').innerText()).toBe(held);
  await page.unroute('**/api/**');
  const hardware=await simulationPage(page,request,{hardware:true});
  await expect(page.locator('#plan-diffusion')).toBeDisabled();await expect(page.locator('#diffusion-case')).toBeDisabled();
  await expect(page.locator('#diffusion-obstacle-mode')).toBeDisabled();
  expect((await sceneState(page)).visible).toBe(false);
  expect(hardware.posts).toEqual([]);expect(app.errors).toEqual([]);expect(hardware.errors).toEqual([]);
});
