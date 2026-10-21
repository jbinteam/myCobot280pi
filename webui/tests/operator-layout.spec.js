import { test, expect } from '@playwright/test';
import { execFileSync } from 'node:child_process';

let info;
test.beforeAll(()=>{
  info=JSON.parse(execFileSync('.venv/bin/python',['-c',
    'import json; from simulation.service import info; print(json.dumps(info()))'],
    {cwd:'..',encoding:'utf8',env:{...process.env,OPENBLAS_NUM_THREADS:'1'}}));
});
async function setup(page, hardware=false){
  const writes=[],errors=[];
  const state={hardware_enabled:hardware,connected:false,token:'operator-ui-fake',motion:null,moving:false,
    firmware:6.5,angles:null,limits:info.limits_deg,ik:{state:'idle',samples:[],plan:null,tracking_protocol:2,v2_tracking_protocol:1}};
  page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/api/**',async route=>{
    const request=route.request(),path=new URL(request.url()).pathname;
    if(path==='/api/sim/info')return route.fulfill({json:info});
    if(request.method()==='POST'){
      writes.push({path,body:request.postDataJSON()});
      if(path==='/api/connect'){state.connected=true;state.angles=info.default_start_deg;}
      else throw new Error(`Unexpected write in setup-only test: ${path}`);
    }
    if(path!=='/api/state'&&path!=='/api/connect')throw new Error(`Unexpected endpoint ${path}`);
    return route.fulfill({json:{...state,updated:Date.now()/1000}});
  });
  await page.goto('/');
  await expect(page.locator('#model-status')).toHaveText('Model ready',{timeout:45000});
  await expect(page.locator('#recommended-model-note')).toContainText('Recommended for fixed targets');
  return {writes,errors};
}

test('default workspace focuses on mouse tracking with learned IK + RRT* and keeps experiments closed',async({page},testInfo)=>{
  const {writes,errors}=await setup(page);
  await expect(page.locator('#recommended-model')).toHaveText('Learned IK + classical RRT*');
  await expect(page.locator('#mouse-aim-planner')).toHaveValue('learned_rrt');
  await expect(page.locator('#mouse-aim-continuous')).toBeChecked();
  await expect(page.locator('#ik-method')).toHaveValue('bounded_dls');
  await expect(page.locator('#diffusion-obstacle-mode')).toHaveValue('none');
  await expect(page.locator('#advanced-workbench')).toHaveJSProperty('open',false);
  await expect(page.locator('#simulation-lab')).toBeHidden();
  await expect(page.locator('#hardware-ik')).toBeHidden();
  await expect(page.locator('#simulation-setup')).toBeVisible();
  await expect(page.locator('#mode-badge')).toContainText('Simulation only');
  await expect(page.locator('.robot-toolbar')).toBeHidden();
  await expect(page.locator('.robot-actions')).toBeHidden();
  await expect(page.locator('#fixed-target-controls')).toBeHidden();
  await expect(page.locator('.subtitle')).toContainText('No robot connection needed');
  await expect(page.locator('#tracking-details')).toHaveJSProperty('open',false);
  await expect(page.locator('#toggle-mouse-aim')).toBeEnabled();
  expect(writes).toEqual([]);
  const bounds=await page.locator('#viewport').boundingBox();
  expect(bounds.width).toBeGreaterThan(600);expect(bounds.height).toBeGreaterThan(450);
  await page.screenshot({path:testInfo.outputPath('operator-desktop.png'),fullPage:true});
  await page.locator('#open-advanced').focus();await page.keyboard.press('Enter');
  await expect(page.locator('#advanced-workbench')).toHaveJSProperty('open',true);
  await expect(page.locator('#simulation-lab')).toBeVisible();
  await page.locator('#tracking-details summary').click();
  await expect(page.locator('#mouse-aim-planner')).toBeVisible();
  await page.locator('#mouse-aim-planner').selectOption('classical');
  await expect(page.locator('#recommended-model')).toHaveText('Learned IK + classical RRT*');
  await page.locator('#fixed-target-planner').selectOption('diffusion_v2');
  await expect(page.locator('#recommended-model')).toHaveText('Learned IK + Diffusion v2');
  await page.locator('#fixed-target-planner').selectOption('learned_rrt');
  await page.locator('#mouse-aim-planner').selectOption('local_diffusion');
  await expect(page.locator('#recommended-model')).toHaveText('Learned IK + classical RRT*');
  expect(writes).toEqual([]);expect(errors).toEqual([]);
});

test('hardware setup is grouped, confirmations remain explicit and mobile has no overflow',async({page},testInfo)=>{
  const {writes,errors}=await setup(page,true);
  await expect(page.locator('#hardware-planner')).toHaveValue('learned_rrt');
  await expect(page.locator('#simulation-setup')).toBeHidden();
  await expect(page.locator('#ik-sdk-speed')).toHaveValue('70');
  await expect(page.locator('#arm-movement')).not.toBeChecked();
  await expect(page.locator('#verify-ik-frame')).not.toBeChecked();
  await expect(page.locator('#verify-diffusion-scene')).not.toBeChecked();
  await expect(page.locator('#toggle-mouse-aim')).toBeDisabled();
  expect(writes).toEqual([]);
  await page.locator('#connect-robot').click();
  await page.locator('#verify-ik-frame').check();
  await page.locator('#verify-diffusion-scene').check();
  await expect(page.locator('#toggle-mouse-aim')).toBeDisabled();
  await page.locator('#arm-movement').check();
  await expect(page.locator('#toggle-mouse-aim')).toBeEnabled();
  await expect(page.locator('#setup-status')).toContainText('Ready.');
  await expect(page.locator('#stop-robot')).toBeVisible();
  await page.screenshot({path:testInfo.outputPath('operator-hardware-desktop.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await expect(page.locator('#hardware-ik')).toBeVisible();
  await page.screenshot({path:testInfo.outputPath('operator-hardware-mobile.png'),fullPage:true});
  expect(writes.map(w=>w.path)).toEqual(['/api/connect']);
  expect(errors).toEqual([]);
});
