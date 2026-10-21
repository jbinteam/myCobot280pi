import {test,expect} from '@playwright/test';

test('five targets align the flange Z direction, allow wrist spin, and reject direction-only failures',async({page,request},testInfo)=>{
  const info=(await(await request.get('http://127.0.0.1:8000/api/sim/info')).json());
  const count=info.pose5_methods.length;
  await page.route('**/api/state', async route => {
    const state = await (await route.fetch()).json();
    await route.fulfill({json:{...state,hardware_enabled:false,connected:false,moving:false,motion:null}});
  });
  const errors=[],posts=[];let result;
  page.on('pageerror',e=>errors.push(e.message));
  page.on('request',r=>{if(r.method()==='POST')posts.push(new URL(r.url()).pathname);});
  page.on('response',async r=>{if(new URL(r.url()).pathname==='/api/sim/solve'&&r.ok())result=await r.json();});
  await page.goto('/?view=lab');
  await expect(page.locator('#run-results tbody tr')).toHaveCount(count,{timeout:45000});
  await expect(page.locator('#ik-task')).toHaveValue('pose5');
  await expect(page.locator('#orientation-fields')).toBeVisible();
  await expect(page.locator('#ik-summary')).toContainText('Reached target');
  await expect(page.locator('#orientation-chart')).toContainText('Flange Z direction error (degrees)');
  await expect(page.locator('#benchmark-results tbody tr')).toHaveCount(info.benchmark_pose5.rows.length);
  await page.locator('#ik-scrub').fill(await page.locator('#ik-scrub').getAttribute('max'));
  const capture=async()=>page.evaluate(async()=>{
    const {robot,scene}=await import(document.querySelector('script[src*="/src/main.js"]').src);robot.updateMatrixWorld(true);scene.updateMatrixWorld(true);
    const e=robot.links.joint6_flange.matrixWorld.elements,arrow=scene.getObjectByName('pose5-target-direction');
    return {position:e.slice(12,15),z:e.slice(8,11),targetZ:arrow.matrixWorld.elements.slice(4,7),arrowVisible:arrow.visible};
  });
  const first=await capture();
  expect(first.arrowVisible).toBe(true);
  const angle=Math.acos(Math.min(1,Math.max(-1,first.z.reduce((sum,v,i)=>sum+v*first.targetZ[i],0))))*180/Math.PI;
  expect(angle).toBeLessThanOrEqual(1);
  await page.locator('#joint-5-number').fill('95');await page.locator('#joint-5-number').press('Tab');
  const spun=await capture();
  first.z.forEach((v,i)=>expect(Math.abs(v-spun.z[i])).toBeLessThan(1e-10));
  first.position.forEach((v,i)=>expect(Math.abs(v-spun.position[i])).toBeLessThan(1e-10));
  // Current position within position tolerance, but reversed desired tool direction, one-step budget.
  for(const [i,axis] of ['x','y','z'].entries())await page.locator(`#target-${axis}`).fill(String(spun.position[i]*1000));
  const rx=Math.asin(-spun.z[1])*180/Math.PI;
  const ry=Math.atan2(spun.z[0],spun.z[2])*180/Math.PI;
  await page.locator('#target-rx').fill(String(rx>=0?rx-180:rx+180));await page.locator('#target-ry').fill(String(ry));
  await page.locator('#ik-steps').fill('1');await page.locator('#solve-ik').click();
  await expect(page.locator('#ik-feedback')).toContainText('0/1 methods reached');
  expect(result.results[0].frames[0].error_mm).toBeLessThan(1);expect(result.results[0].final_orientation_error_deg).toBeGreaterThan(170);
  await page.screenshot({path:testInfo.outputPath('pose5.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  expect(posts.every(p=>p==='/api/sim/solve')).toBe(true);expect(errors).toEqual([]);
});

test('hardware planning carries all five targets and direction edits invalidate execution',async({page,request})=>{
  const backend=await(await request.get('http://127.0.0.1:8000/api/state')).json();
  const info=await(await request.get('http://127.0.0.1:8000/api/sim/info')).json();
  const demo=await(await request.post('http://127.0.0.1:8000/api/sim/solve',{headers:{'X-Robot-Token':backend.token},data:{task:'pose5',start_deg:info.default_start_deg,target_mm:info.default_target_mm,orientation_xy_deg:info.default_orientation_xy_deg,methods:['dls']}})).json();
  const start=info.default_start_deg,p0=demo.results[0].frames[0].position,n0=demo.results[0].frames[0].tool_axis;
  const xy=[Math.asin(-n0[1])*180/Math.PI,Math.atan2(n0[0],n0[2])*180/Math.PI];
  const calls=[];let planned;
  let state={hardware_enabled:true,connected:false,token:'pose5-ui',angles:null,updated:Date.now()/1000,motion:null,moving:false,firmware:6.5,limits:info.limits_deg,ik:{state:'idle',samples:[],plan:null,estimated_flange_mm:p0,orientation_xy_deg:xy,flange_z_axis:n0}};
  await page.route('**/api/**',async route=>{
    const r=route.request(),path=new URL(r.url()).pathname;
    if(path==='/api/sim/info'){await route.fulfill({json:info});return;}
    if(r.method()==='POST'){
      const body=r.postDataJSON();calls.push({path,body});
      if(path==='/api/connect')state={...state,connected:true,angles:start};
      if(path==='/api/ik/plan'){
        const solved=await(await request.post('http://127.0.0.1:8000/api/sim/solve',{headers:{'X-Robot-Token':backend.token},data:{...body,start_deg:start,methods:[body.method]}})).json();
        planned=solved.results[0];
        const plan={id:'five-target-plan',task:'pose5',method:body.method,target_mm:body.target_mm,start_deg:start,orientation_xy_deg:body.orientation_xy_deg,target_axis:planned.target_axis,converged:planned.converged,waypoints:2,expires_in:60,final_error_mm:planned.final_error_mm,final_orientation_error_deg:planned.final_orientation_error_deg};
        state.ik={...state.ik,state:'planned',plan};
        await route.fulfill({json:{plan,result:planned,start_deg:start,target_mm:body.target_mm}});return;
      }
      if(path==='/api/ik/execute')state={...state,motion:'ik',moving:true,ik:{...state.ik,plan:null,state:'running',task:'pose5',execution_id:'five-target-plan',completed_waypoints:0,total_waypoints:2,sdk_speed:30}};
      if(path==='/api/ik/heartbeat'){
        const f=planned.frames.at(-1);
        state={...state,angles:f.q,motion:null,moving:false,ik:{...state.ik,state:'completed',completed_waypoints:2,estimated_error_mm:f.error_mm,estimated_orientation_error_deg:f.orientation_error_deg,message:'Joint waypoints completed',samples:[{time:1,error_mm:f.error_mm,orientation_error_deg:f.orientation_error_deg}]}};
        await route.fulfill({json:{active:true}});return;
      }
    }
    await route.fulfill({json:{...state,updated:Date.now()/1000}});
  });
  await page.goto('/?view=lab');
  await expect(page.locator('#ik-method option')).toHaveCount(info.pose5_methods.length,{timeout:45000});
  await page.locator('#connect-robot').click();await page.locator('#use-live-xyz').click();
  expect(Number(await page.locator('#target-rx').inputValue())).toBeCloseTo(xy[0],3);
  expect(Number(await page.locator('#target-ry').inputValue())).toBeCloseTo(xy[1],3);
  await page.locator('#target-x').fill(String(p0[0]+2));await page.locator('#plan-hardware-ik').click();
  await expect(page.locator('#hardware-ik-feedback')).toContainText('Plan ready');
  expect(calls[1].body).toMatchObject({task:'pose5',orientation_tolerance_deg:1});
  expect(calls[1].body.orientation_xy_deg).toHaveLength(2);expect(calls[1].body.start_deg).toBeUndefined();
  await page.locator('#arm-movement').check();await page.locator('#verify-ik-frame').check();
  await expect(page.locator('#execute-hardware-ik')).toBeEnabled();
  await page.locator('#target-rx').fill(String(xy[0]+.2));await page.locator('#target-rx').press('Tab');
  await expect(page.locator('#execute-hardware-ik')).toBeDisabled();
  await page.locator('#plan-hardware-ik').click();await expect(page.locator('#execute-hardware-ik')).toBeEnabled();
  await page.locator('#execute-hardware-ik').click();
  await expect(page.locator('#hardware-ik-progress')).toContainText('completed',{timeout:10000});
  await expect(page.locator('#hardware-ik-progress')).toContainText('Z direction error');
  await expect(page.locator('#hardware-orientation-chart svg')).toHaveCount(1);
  expect(calls.some(c=>['/api/move','/api/jog'].includes(c.path))).toBe(false);
});
