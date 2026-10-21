import {test,expect} from '@playwright/test';
test.beforeEach(async({page})=>{await page.addInitScript(()=>localStorage.setItem('ik-task-mode','position'));});

test('phase 2 plans from live joints and executes only after explicit gates',async({page,request},testInfo)=>{
  const info=await (await request.get('http://127.0.0.1:8000/api/sim/info')).json();
  const calls=[];let beats=0;
  const start=[0,-40,-130,80,0,50],end=[3,-40,-130,80,0,50];
  let state={hardware_enabled:true,connected:false,token:'hardware-test',angles:null,updated:Date.now()/1000,motion:null,moving:false,firmware:6.5,limits:info.limits_deg,ik:{state:'idle',samples:[],estimated_flange_mm:[160,-64,83],plan:null}};
  await page.route('**/api/**',async route=>{
    const r=route.request(),path=new URL(r.url()).pathname;
    if(path==='/api/sim/info'){await route.fulfill({json:info});return;}
    if(r.method()==='POST') {
      const body=r.postDataJSON();calls.push({path,body});expect(r.headers()['x-robot-token']).toBe('hardware-test');
      if(path==='/api/connect')state={...state,connected:true,angles:start};
      if(path==='/api/ik/plan'){
        const plan={id:'live-plan-id',method:body.method,target_mm:body.target_mm,start_deg:start,waypoints:2,expires_in:60,converged:true,final_error_mm:.8};
        const result={method:body.method,converged:true,final_error_mm:.8,iterations:1,joint_travel_deg:3,elapsed_ms:2,stop_reason:'target_reached',frames:[{q:start,position:[160,-64,83],error_mm:5,time:0},{q:end,position:[165,-64,83],error_mm:.8,time:.05}]};
        state.ik={...state.ik,state:'planned',plan};
        await route.fulfill({json:{plan,result,start_deg:start,target_mm:body.target_mm}});return;
      }
      if(path==='/api/ik/execute')state={...state,motion:'ik',moving:true,ik:{...state.ik,plan:null,state:'running',execution_id:'live-plan-id',completed_waypoints:0,total_waypoints:2,sdk_speed:30,estimated_error_mm:5}};
      if(path==='/api/ik/heartbeat'){
        beats++;state.ik.samples.push({time:beats,q:end,position_mm:[165,-64,83],error_mm:.8});
        if(beats>=2)state={...state,angles:end,moving:false,motion:null,ik:{...state.ik,state:'completed',completed_waypoints:2,estimated_error_mm:.8,message:'Joint waypoints completed'}};
        await route.fulfill({json:{active:true}});return;
      }
      if(path==='/api/stop')state={...state,moving:false,motion:null,ik:{...state.ik,state:'stopped',message:'Execution stopped'}};
    }
    await route.fulfill({json:{...state,updated:Date.now()/1000}});
  });
  await page.goto('/?view=lab');
  await expect(page.locator('#plan-hardware-ik')).toBeDisabled();
  await expect(page.locator('#model-status')).toHaveText('Model ready',{timeout:45000});
  await page.locator('#fixed-target-planner').selectOption('classical');
  await page.locator('#ik-task').selectOption('position');
  expect(calls).toEqual([]);
  await page.locator('#connect-robot').click();
  await expect(page.locator('#plan-hardware-ik')).toBeEnabled();
  await page.locator('#use-live-xyz').click();
  await page.locator('#target-x').fill('165');
  await page.locator('#plan-hardware-ik').click();
  await expect(page.locator('#hardware-ik-feedback')).toContainText('Plan ready');
  expect(calls.map(c=>c.path)).toEqual(['/api/connect','/api/ik/plan']);
  expect(calls[1].body).toMatchObject({target_mm:[165,-64,83],method:'dls'});
  expect(calls[1].body.start_deg).toBeUndefined();
  await expect(page.locator('#follow-live')).not.toBeChecked();
  await expect(page.locator('#execute-hardware-ik')).toBeDisabled();
  await page.locator('#arm-movement').check();
  await expect(page.locator('#execute-hardware-ik')).toBeDisabled();
  await page.locator('#verify-ik-frame').check();
  await expect(page.locator('#execute-hardware-ik')).toBeEnabled();
  // Input changes invalidate a prepared plan and require a new live-start solve.
  await page.locator('#target-x').fill('166');await page.locator('#target-x').press('Tab');
  await expect(page.locator('#execute-hardware-ik')).toBeDisabled();
  await page.locator('#plan-hardware-ik').click();
  await expect(page.locator('#execute-hardware-ik')).toBeEnabled();
  await page.locator('#execute-hardware-ik').click();
  await expect(page.locator('#follow-live')).toBeChecked();
  await expect(page.locator('#execute-hardware-ik')).toBeDisabled();
  await expect(page.locator('#hardware-ik-progress')).toContainText('completed',{timeout:10000});
  await expect(page.locator('#hardware-error-chart svg')).toHaveCount(1);
  await expect(page.locator('#hardware-measured-velocity-chart polyline')).toHaveCount(6);
  expect(calls.find(c=>c.path==='/api/ik/execute').body).toEqual({plan_id:'live-plan-id',speed:30,armed:true,frame_verified:true});
  expect(calls.some(c=>['/api/move','/api/jog'].includes(c.path))).toBe(false);
  await page.screenshot({path:testInfo.outputPath('hardware-ik-ui.png'),fullPage:true});
});
