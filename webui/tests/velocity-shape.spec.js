import {test,expect} from '@playwright/test';
import {readFileSync} from 'node:fs';
import {execFileSync} from 'node:child_process';
let info,fixture;
test.beforeAll(()=>{
  info=JSON.parse(execFileSync('.venv/bin/python',['-c','import json; from simulation.service import info; print(json.dumps(info()))'],{cwd:'..',encoding:'utf8'}));
  fixture=JSON.parse(readFileSync('../output/velocity_shape_ui_fixture.json','utf8'));
});
test('reference shape preview edits the plan and invalidates execution until replanned',async({page})=>{
  const calls=[],errors=[];
  let state={hardware_enabled:true,connected:true,token:'shape-test',angles:fixture.start_deg,
    moving:false,motion:null,firmware:6.5,limits:info.limits_deg,
    ik:{state:'idle',samples:[],plan:null,estimated_flange_mm:fixture.target_mm,orientation_xy_deg:[0,0]}};
  page.on('pageerror',error=>errors.push(error.message));
  await page.route('**/api/**',async route=>{
    const request=route.request(),path=new URL(request.url()).pathname;
    if(path==='/api/sim/info')return route.fulfill({json:info});
    if(request.method()==='POST') {
      calls.push({path,body:request.postDataJSON()});
      if(path==='/api/ik/diffusion/plan') {
        state.ik={...state.ik,state:'planned',plan:fixture.plan};
        return route.fulfill({json:fixture});
      }
      throw new Error(`Unexpected write ${path}`);
    }
    return route.fulfill({json:{...state,updated:Date.now()/1000}});
  });
  await page.goto('/?view=lab');
  await expect(page.locator('#model-status')).toHaveText('Model ready',{timeout:45000});
  await expect(page.locator('#velocity-shape-settings')).toBeVisible();
  await expect(page.locator('#velocity-shape')).toHaveValue('smooth_trapezoid');
  await expect(page.locator('#velocity-rise')).toHaveValue('5');
  await expect(page.locator('#velocity-fall')).toHaveValue('10');
  await expect(page.locator('#velocity-shape-feedback')).toContainText('85% of each leg at peak speed');
  await page.locator('#velocity-shape').selectOption('quintic');
  const original=await page.locator('#velocity-shape-preview polyline').getAttribute('points');
  await page.locator('#velocity-shape').selectOption('smooth_trapezoid');
  async function setRamp(id,value) {await page.locator(id).evaluate((el,value)=>{el.value=value;el.dispatchEvent(new Event('input',{bubbles:true}));},String(value));}
  await setRamp('#velocity-rise',20);await setRamp('#velocity-fall',40);
  await expect(page.locator('#velocity-rise-value')).toHaveText('20%');
  await expect(page.locator('#velocity-fall-value')).toHaveText('40%');
  await expect(page.locator('#velocity-shape-feedback')).toContainText('40% of each leg at peak speed');
  expect(await page.locator('#velocity-shape-preview polyline').getAttribute('points')).not.toBe(original);
  expect(calls).toEqual([]);
  await page.locator('#arm-movement').check();await page.locator('#verify-ik-frame').check();await page.locator('#verify-diffusion-scene').check();
  await page.locator('#plan-hardware-ik').click();
  await expect(page.locator('#hardware-ik-feedback')).toContainText('Reference shape: 20% acceleration, 40% deceleration.');
  await expect(page.locator('#hardware-reference-velocity-chart polyline')).toHaveCount(6);
  await expect(page.locator('#execute-hardware-ik')).toBeEnabled();
  expect(calls).toHaveLength(1);
  expect(calls[0].body).toMatchObject({speed:70,joint_command:'encoder_drag_profile',velocity_shape:{kind:'smooth_trapezoid',rise_percent:20,fall_percent:40}});
  await page.locator('#velocity-shape-settings').evaluate(el=>el.scrollIntoView({block:'center'}));
  await page.locator('#velocity-shape-settings').screenshot({path:'../output/velocity_shape_ui.png'});
  await setRamp('#velocity-rise',30);
  await expect(page.locator('#execute-hardware-ik')).toBeDisabled();
  await expect(page.locator('#hardware-reference-velocity-chart polyline')).toHaveCount(0);
  await setRamp('#velocity-rise',80);
  await expect(page.locator('#velocity-shape-feedback')).toContainText('total at most 100%');
  await page.locator('#plan-hardware-ik').click();
  await expect(page.locator('#hardware-ik-feedback')).toContainText('total at most 100%');
  expect(calls).toHaveLength(1);
  expect(errors).toEqual([]);
});
