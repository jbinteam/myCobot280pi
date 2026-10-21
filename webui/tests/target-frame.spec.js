import { test, expect } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { addLocalOffset } from '../src/target-frame.js';

test('local offset vectors use the full EEF rotation without translation', () => {
  expect(addLocalOffset([100,200,300], [10,20,30], [[1,0,0],[0,1,0],[0,0,1]])).toEqual([110,220,330]);
  expect(addLocalOffset([100,200,300], [10,20,30], [[0,-1,0],[1,0,0],[0,0,1]])).toEqual([80,210,330]);
});

test('simulation and hardware use the resolved offset once; offset edits invalidate plans', async ({page,request}, testInfo) => {
  const info = await (await request.get('http://127.0.0.1:8000/api/sim/info')).json();
  const calls = [];
  const start = info.default_start_deg;
  const rotation = JSON.parse(execFileSync('.venv/bin/python',['-c',
    'import json; from simulation.kinematics import ARM; import numpy as np; print(json.dumps(ARM.transform(np.deg2rad('+JSON.stringify(start)+'))[:3,:3].tolist()))'],
    {cwd:'..',encoding:'utf8'}));
  const resolved = offset => addLocalOffset([100,200,300],offset,rotation);
  const text = offset => `[${resolved(offset).map(v=>v.toFixed(2)).join(', ')}]`;
  const state = { hardware_enabled:true, connected:true, token:'offset-test', angles:start,
    updated:Date.now()/1000, motion:null, moving:false, limits:info.limits_deg, ik:{state:'idle',samples:[],plan:null} };
  await page.route('**/api/**', async route => {
    const r=route.request(), path=new URL(r.url()).pathname;
    if(path==='/api/sim/info')return route.fulfill({json:info});
    if(r.method()==='POST') {
      const body=r.postDataJSON();calls.push({path,body});
      const target = addLocalOffset(body.target_mm,body.eef_offset_mm,rotation);
      const result={task:body.task,orientation_xy_deg:body.orientation_xy_deg,method:'dls',converged:true,final_error_mm:0,
        final_orientation_error_deg:0,iterations:0,joint_travel_deg:0,elapsed_ms:1,stop_reason:'target_reached',
        frames:[{q:start,position:target,error_mm:0,orientation_error_deg:0,time:0}]};
      if(path==='/api/sim/solve')return route.fulfill({json:{start_deg:start,results:[result]}});
      if(path==='/api/ik/plan') {
        const plan={id:'offset-plan',converged:true,waypoints:1,expires_in:60,final_error_mm:0};
        state.ik.plan=plan;
        return route.fulfill({json:{plan,result,start_deg:start,target_mm:target}});
      }
      throw new Error(`Unexpected action ${path}`);
    }
    return route.fulfill({json:{...state,updated:Date.now()/1000}});
  });
  await page.goto('/?view=lab');
  await expect(page.locator('#solve-ik')).toBeEnabled({timeout:45000});
  await page.locator('#ik-task').selectOption('pose5');
  for(const [id,value] of Object.entries({'target-x':100,'target-y':200,'target-z':300,'target-rx':90,'target-ry':90,'eef-offset-x':10,'eef-offset-y':20,'eef-offset-z':30})) {
    await page.locator(`#${id}`).fill(String(value));
  }
  await expect(page.locator('#resolved-target')).toHaveText(text([10,20,30]));
  // Desired target tilt must not replace the starting EEF rotation.
  await page.locator('#target-rx').fill('0');
  await page.locator('#target-ry').fill('0');
  await expect(page.locator('#resolved-target')).toHaveText(text([10,20,30]));
  await page.locator('#solve-ik').click();
  await expect(page.locator('#run-results tbody tr')).toHaveCount(1);
  await page.locator('#plan-hardware-ik').click();
  await expect(page.locator('#hardware-ik-feedback')).toContainText('Plan ready');
  expect(calls.map(c=>c.path)).toEqual(['/api/sim/solve','/api/ik/plan']);
  for(const {body} of calls) {
    expect(body.target_mm).toEqual([100,200,300]);
    expect(body.eef_offset_mm).toEqual([10,20,30]);
  }
  await expect(page.locator('#target-x')).toHaveValue('100');
  await expect(page.locator('#eef-offset-z')).toHaveValue('30');
  await page.locator('#arm-movement').check();await page.locator('#verify-ik-frame').check();
  await expect(page.locator('#execute-hardware-ik')).toBeEnabled();
  await page.locator('#eef-offset-z').fill('31');
  await expect(page.locator('#execute-hardware-ik')).toBeDisabled();
  await expect(page.locator('#resolved-target')).toHaveText(text([10,20,31]));
  await page.screenshot({path:testInfo.outputPath('eef-local-offset.png'),fullPage:true});
  await page.locator('#ik-task').selectOption('position');
  await expect(page.locator('#resolved-target')).toHaveText(text([10,20,31]));
  await page.locator('#eef-offset-x').fill('');
  await page.locator('#plan-hardware-ik').click();
  await expect(page.locator('#hardware-ik-feedback')).toContainText('Enter finite');
  expect(calls).toHaveLength(2);
  await page.setViewportSize({width:390,height:844});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
});
