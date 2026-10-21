import {test,expect} from '@playwright/test';
import {readFileSync,existsSync} from 'node:fs';

// An offline response from the production planner with fake transport. The
// fixture uses the failing target; it contains no SDK motion or browser token.
const fixturePath=new URL('../../output/v2_original_method_plan_fixture.json',import.meta.url);
for(const omitTravel of [false,true])test(`restored v2 plan renders ${omitTravel?'legacy missing travel metric':'production planner response'}`,async({page,request})=>{
  test.skip(!existsSync(fixturePath),'Generate the offline restored-planner fixture with tmp/validate_v2_rollback.py');
  const data=JSON.parse(readFileSync(fixturePath,'utf8'));
  expect(Number.isFinite(data.result.joint_travel_deg)).toBe(true);
  if(omitTravel)delete data.result.joint_travel_deg;
  const info=await (await request.get('http://127.0.0.1:8000/api/sim/info')).json();
  const errors=[],posts=[];page.on('pageerror',error=>errors.push(error.message));
  const state={hardware_enabled:true,connected:true,token:'offline-waypoint-test',angles:data.start_deg,
    motion:null,moving:false,firmware:6.5,limits:info.limits_deg,
    ik:{state:'idle',samples:[],plan:null,estimated_flange_mm:data.result.frames[0].position,
        orientation_xy_deg:data.result.orientation_xy_deg}};
  await page.route('**/api/**',async route=>{
    const r=route.request(),path=new URL(r.url()).pathname;
    if(path==='/api/sim/info'){await route.fulfill({json:info});return;}
    if(r.method()==='POST'){
      posts.push(path);
      if(path==='/api/ik/diffusion/plan'){
        data.plan.expires_in=600;state.ik={...state.ik,state:'planned',plan:data.plan};
        await route.fulfill({json:data});return;
      }
      throw new Error(`Unexpected command ${path}`);
    }
    await route.fulfill({json:{...state,updated:Date.now()/1000}});
  });
  await page.goto('/?view=lab');
  await expect(page.locator('#model-status')).toHaveText('Model ready',{timeout:45000});
  await page.locator('#hardware-joint-command').selectOption(data.plan.joint_command);
  await page.locator('#plan-hardware-ik').click();
  await expect(page.locator('#hardware-ik-feedback')).toContainText('Plan ready');
  await expect(page.locator('#run-results tbody tr')).toHaveCount(1);
  await expect(page.locator('#hardware-reference-velocity-chart polyline')).toHaveCount(6);
  await expect(page.locator('#execute-hardware-ik')).toBeDisabled();
  expect(errors).toEqual([]);expect(posts).toEqual(['/api/ik/diffusion/plan']);
});
