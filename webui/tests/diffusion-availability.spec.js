import { test, expect } from '@playwright/test';
import { execFileSync } from 'node:child_process';

let info;
test.beforeAll(() => {
  info=JSON.parse(execFileSync('.venv/bin/python',['-c',
    'import json; from simulation.service import info; print(json.dumps(info()))'],
    {cwd:'..',encoding:'utf8',env:{...process.env,OPENBLAS_NUM_THREADS:'1'}}));
});

async function setup(page,{hardware=false,unavailable=false}={}) {
  const posts=[],errors=[];
  const modelInfo=unavailable?{...info,diffusion:{...info.diffusion,status:'unavailable',reason:'Missing test checkpoint'}}:info;
  const state={hardware_enabled:hardware,mode_selectable:true,connected:hardware,angles:hardware?info.default_start_deg:null,
    motion:null,moving:false,firmware:6.5,token:'diffusion-availability-test',limits:info.limits_deg,
    ik:{state:'idle',samples:[],plan:null,tracking_protocol:2,v2_tracking_protocol:1}};
  page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/api/**',async route=>{
    const req=route.request(),path=new URL(req.url()).pathname;
    if(path==='/api/state')return route.fulfill({json:{...state,updated:Date.now()/1000}});
    if(path==='/api/sim/info')return route.fulfill({json:modelInfo});
    if(path!=='/api/sim/diffusion')throw new Error(`Unexpected request: ${path}`);
    posts.push({path,body:req.postDataJSON()});
    return route.fulfill({json:{converged:false,frames:[],scene_mode:'no_block',obstacle_mm:null,
      status:'no_path',reason:'No checked path in this test'}});
  });
  await page.goto('/');
  await expect(page.locator('#model-status')).toHaveText('Model ready',{timeout:45000});
  await expect(page.locator('#diffusion-checkpoint')).toContainText(unavailable?'Diffusion unavailable':'Trained DDPM ready');
  await page.locator('#open-advanced').click();
  return {posts,errors};
}

test('hardware mode explains the disabled simulation button and links to real-arm planning',async({page},testInfo)=>{
  const app=await setup(page,{hardware:true});
  await expect(page.locator('#plan-diffusion')).toBeDisabled();
  await expect(page.locator('#diffusion-availability')).toContainText('This button is for simulation only');
  await expect(page.locator('#diffusion-availability')).toContainText('Plan from live joints');
  await expect(page.locator('#diffusion-hardware-help')).toBeVisible();
  await page.locator('#diffusion-hardware-help').click();
  await expect(page.locator('#fixed-target-controls')).toBeVisible();
  await expect(page.locator('#plan-hardware-ik')).toBeEnabled();
  await expect(page.locator('#hardware-planner')).toHaveValue('diffusion_v2');
  await expect(page.locator('#execute-hardware-ik')).toBeDisabled();
  await page.screenshot({path:testInfo.outputPath('hardware-diffusion-guidance.png'),fullPage:true});
  expect(app.posts).toEqual([]);expect(app.errors).toEqual([]);
});

test('simulation button is clickable and sends a simulation-only diffusion request',async({page})=>{
  const app=await setup(page);
  await expect(page.locator('#plan-diffusion')).toBeEnabled();
  await expect(page.locator('#diffusion-availability')).toContainText('Ready.');
  await expect(page.locator('#diffusion-hardware-help')).toBeHidden();
  await page.locator('#plan-diffusion').click();
  await expect(page.locator('#diffusion-result')).toContainText('No playable path');
  expect(app.posts).toHaveLength(1);
  expect(app.posts[0]).toMatchObject({path:'/api/sim/diffusion',body:{obstacle_mm:null}});
  await expect(page.locator('#plan-diffusion')).toBeEnabled();
  expect(app.errors).toEqual([]);
});

test('tracking and XYZ task explain their gates and planning re-enables when corrected',async({page})=>{
  const app=await setup(page);
  await page.locator('#toggle-mouse-aim').click();
  await expect(page.locator('#plan-diffusion')).toBeDisabled();
  await expect(page.locator('#diffusion-availability')).toContainText('Stop mouse aim');
  await page.locator('#toggle-mouse-aim').click();
  await expect(page.locator('#plan-diffusion')).toBeEnabled();
  await page.locator('#ik-task').selectOption('position');
  await expect(page.locator('#plan-diffusion')).toBeDisabled();
  await expect(page.locator('#diffusion-availability')).toContainText('Select 5D pose');
  await page.locator('#ik-task').selectOption('pose5');
  await expect(page.locator('#plan-diffusion')).toBeEnabled();
  expect(app.posts).toEqual([]);expect(app.errors).toEqual([]);
});

test('unavailable checkpoint explains why diffusion planning cannot start',async({page})=>{
  const app=await setup(page,{unavailable:true});
  await expect(page.locator('#plan-diffusion')).toBeDisabled();
  await expect(page.locator('#diffusion-availability')).toContainText('Missing test checkpoint');
  expect(app.posts).toEqual([]);expect(app.errors).toEqual([]);
});
