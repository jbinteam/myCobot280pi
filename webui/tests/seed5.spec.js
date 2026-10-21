import { test, expect } from '@playwright/test';

test('base-frame vector entry validates, normalizes and plans a 5D simulation goal',async({page,request},testInfo)=>{
  test.setTimeout(180000);
  const state=await(await request.get('http://127.0.0.1:8000/api/state')).json();
  const info=await(await request.get('http://127.0.0.1:8000/api/sim/info')).json();
  const posts=[],errors=[];let plan;
  page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/api/**',async route=>{
    const req=route.request(),path=new URL(req.url()).pathname;
    if(path==='/api/state')return route.fulfill({json:{...state,hardware_enabled:false,connected:false,moving:false,motion:null}});
    if(path==='/api/sim/info')return route.fulfill({json:info});
    if(req.method()!=='POST'||!path.startsWith('/api/sim/'))throw new Error(`Unexpected request ${path}`);
    const body=req.postDataJSON();posts.push({path,body});
    const response=await request.post(`http://127.0.0.1:8000${path}`,{headers:{'X-Robot-Token':state.token},data:body});
    const data=await response.json();if(path==='/api/sim/prm')plan=data;
    return route.fulfill({status:response.status(),json:data});
  });
  await page.goto('/?view=lab');
  await expect(page.locator('#run-results tbody tr')).toHaveCount(4,{timeout:45000});
  const collisionTrained=info.seed5_checkpoint.collision_training?.enabled;
  if(collisionTrained){
    await expect(page.locator('#seed5-checkpoint-status')).toContainText('trained with self-collision checks (1 mm clearance)');
    await expect(page.locator('#pose5-proposals option[value="seed5"]')).toHaveText('Collision-trained Seed NN + numerical fallback');
    await expect(page.locator('#mouse-aim-engine')).toContainText('collision-trained Seed NN');
  }else{
    await expect(page.locator('#seed5-checkpoint-status')).not.toContainText('trained with self-collision checks');
  }
  await page.locator('#direction-entry').selectOption('axis');
  await expect(page.locator('#orientation-fields')).toBeHidden();
  await expect(page.locator('#direction-vector-fields')).toBeVisible();
  for(const a of 'xyz')await page.locator(`#target-u${a}`).fill('0');
  const before=posts.length;
  await page.locator('#plan-prm').click();
  await expect(page.locator('#ik-feedback')).toContainText('nonzero');
  expect(posts.length).toBe(before);
  // A known FK task, using a scaled input to exercise vector normalization.
  const goal=await(await request.post('http://127.0.0.1:8000/api/sim/solve',{headers:{'X-Robot-Token':state.token},data:{
    task:'pose5',start_deg:info.default_start_deg,target_mm:info.default_target_mm,
    orientation_xy_deg:info.default_orientation_xy_deg,methods:['bounded_dls']}})).json();
  const axis=goal.results[0].target_axis;
  for(const [i,a] of [...'xyz'].entries())await page.locator(`#target-u${a}`).fill(String(axis[i]*7));
  await expect(page.locator('#target-tool-direction')).toContainText(axis[0].toFixed(4));
  await page.locator('#ik-method').selectOption('bounded_dls');
  await page.locator('#plan-prm').click();
  await expect(page.locator('#prm-status')).toContainText('success',{timeout:120000});
  const body=posts.at(-1).body;
  expect(body.task).toBe('pose5');expect(body.nn).toBe('seed5');
  expect(body.pose5_global_recovery).toBe(true);
  expect(plan.ik.config.pose5_global_recovery).toBe(true);
  expect(plan.ik.nn_checkpoint.checkpoint_sha256).toBe(info.seed5_checkpoint.checkpoint_sha256);
  expect(plan.ik.nn_checkpoint.collision_training.enabled).toBe(Boolean(collisionTrained));
  body.target_axis.forEach((v,i)=>expect(v).toBeCloseTo(axis[i],9));
  expect(plan.final_error_mm).toBeLessThanOrEqual(1);
  expect(plan.final_orientation_error_deg).toBeLessThanOrEqual(1);
  await expect(page.locator('#prm-outcomes')).toContainText('accepted goals');
  await expect(page.locator('#prm-goal-sources')).toContainText('Accepted goals:');
  await page.locator('#ik-scrub').fill(await page.locator('#ik-scrub').getAttribute('max'));
  await expect(page.locator('#flange-tool-axis')).toContainText('[');
  await page.screenshot({path:testInfo.outputPath('seed5-vector-desktop.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.screenshot({path:testInfo.outputPath('seed5-vector-mobile.png'),fullPage:true});
  await page.locator('#direction-preset').selectOption('0,0,-1');
  await expect(page.locator('#target-tool-direction')).toHaveText('[0.0000, 0.0000, -1.0000]');
  await expect(page.locator('#prm-results')).toBeHidden();
  await page.locator('#ik-task').selectOption('position');
  await expect(page.locator('#direction-vector-fields')).toBeHidden();
  await expect(page.locator('#pose5-proposal-field')).toBeHidden();
  expect(posts.every(p=>p.path.startsWith('/api/sim/'))).toBe(true);expect(errors).toEqual([]);
});

test('unit direction utility rejects nonfinite vectors and preserves opposite direction',async({page})=>{
  await page.goto('/?view=lab');
  const result=await page.evaluate(async()=>{
    const {normalizeToolAxis,axisFromTilts,tiltsFromAxis}=await import('/src/tool-direction.js');
    const rejected=[[0,0,0],[Infinity,0,0],[NaN,1,0],[true,0,0]].map(v=>{try{normalizeToolAxis(v);return false;}catch{return true;}});
    return {rejected,large:normalizeToolAxis([1e300,-1e300,0]),opposed:axisFromTilts(tiltsFromAxis([0,0,-1]))};
  });
  expect(result.rejected).toEqual([true,true,true,true]);
  expect(result.large[0]).toBeCloseTo(Math.SQRT1_2,12);
  expect(result.opposed[2]).toBeCloseTo(-1,12);
});
