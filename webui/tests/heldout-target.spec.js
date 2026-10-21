import {test,expect} from '@playwright/test';
for(const hardware of [true,false])test(`held-out case loads ${hardware?'target only from live hardware':'saved simulation start and target'}`,async({page,request})=>{
 const info=await(await request.get('http://127.0.0.1:8000/api/sim/info')).json();
 const start=[-79.1,17.66,-28.91,-25.48,1.23,-2.28],calls=[];
 let state={hardware_enabled:hardware,connected:hardware,token:'heldout-test',angles:hardware?start:null,moving:false,motion:null,firmware:6.5,limits:info.limits_deg,ik:{state:'idle',plan:null,samples:[]}};
 await page.route('**/api/**',async route=>{
  const r=route.request(),path=new URL(r.url()).pathname;
  if(path==='/api/sim/info')return route.fulfill({json:info});
  if(r.method()==='POST'){
   calls.push({path,body:r.postDataJSON()});
   if(path==='/api/ik/diffusion/plan'){
    state.ik={...state.ik,state:'planned',plan:null};
    return route.fulfill({json:{plan:null,result:{converged:false,status:'trial_limit',reason:'No collision-free path for this fixture'}}});
   }
   throw new Error(`Unexpected command: ${path}`);
  }
  return route.fulfill({json:{...state,updated:Date.now()/1000}});
 });
 await page.goto('/');
 await expect(page.locator('#model-status')).toHaveText('Model ready',{timeout:45000});
 await expect(page.locator('#ik-case option')).toHaveCount(25);
 await page.locator('#advanced-workbench').evaluate(el=>el.open=true);
 const fixture=info.cases[1];
 await page.locator('#ik-case').selectOption('1');
 for(const [i,axis] of ['x','y','z'].entries())await expect(page.locator(`#target-${axis}`)).toHaveValue(fixture.target_mm[i].toFixed(4));
 for(const [i,axis] of ['rx','ry'].entries())await expect(page.locator(`#target-${axis}`)).toHaveValue(fixture.orientation_xy_deg[i].toFixed(4));
 const pose=JSON.parse(await page.locator('#pose-value').textContent());
 (hardware?start:fixture.start_deg).forEach((q,i)=>expect(Math.abs(pose[i]-q)).toBeLessThan(.051));
 expect(calls).toEqual([]);
 await expect(page.locator('#execute-hardware-ik')).toBeDisabled();
 if(hardware){
  await expect(page.locator('#hardware-ik-feedback')).toContainText('Held-out target nearby-02 loaded');
  await page.locator('#plan-hardware-ik').click();
  await expect(page.locator('#hardware-ik-feedback')).toContainText('No collision-free path');
  expect(calls).toHaveLength(1);
  expect(calls[0].path).toBe('/api/ik/diffusion/plan');
  const body=calls[0].body;
  expect(body).toMatchObject({target_mm:fixture.target_mm.map(q=>Number(q.toFixed(4))),eef_offset_mm:[0,0,0],speed:5,joint_command:'encoder_drag_profile'});
  expect(body.start_deg).toBeUndefined();
  expect(body.goal_deg).toBeUndefined();
 }
});
