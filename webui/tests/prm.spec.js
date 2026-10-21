import { test, expect } from '@playwright/test';
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { readFileSync } from 'node:fs';

// Persistent, isolated simulation backend: no hardware bridge or robot requests.
let worker, pending=[], stderr='';
const code=`import json, sys, numpy as np
from pathlib import Path
from unittest.mock import patch
from simulation import service, planning
from simulation.kinematics import ARM
from simulation.prm import Roadmap, planning_space
fixture=json.loads(Path('simulation/fixtures/self_collision_path.json').read_text())
goal=np.deg2rad(fixture['goal_deg']); pose=ARM.transform(goal)
def repair_fixture(body):
    road=Roadmap(planning_space('self',1.))
    def generate(start,target,roadmap,config,**kwargs):
        accepted=planning.checked_goal(goal,target,road.space,config,kwargs.get('target_axis'))
        if accepted is None: raise ValueError('Fixture endpoint does not match UI target')
        return {'goals':[{**accepted,'source':'validated_fixture'}],'attempts':[],
                'ik_ms':0.,'config':__import__('dataclasses').asdict(config)}
    with patch.object(planning,'generate_goals',side_effect=generate):
        # A known joint endpoint isolates route repair from new IK branches.
        return service.plan(dict(body,pose5_global_recovery=False),road)
for line in sys.stdin:
    try:
        req=json.loads(line)
        value={'/api/sim/info':service.info, '/api/sim/solve':lambda:service.run(req['body']), '/api/sim/prm':lambda:service.plan(req['body']),
               '/repair-fixture':lambda:repair_fixture(req['body']),
               '/repair-input':lambda:{'start_deg':fixture['start_deg'],'target_mm':(pose[:3,3]*1000).tolist(),'target_axis':pose[:3,2].tolist()},
               '/prewarm':lambda:service.load_roadmap(service.ARTIFACTS/'roadmap.json', __import__('simulation.prm',fromlist=['planning_space']).planning_space('self',1.)).stats}[req['path']]()
        print(json.dumps({'value':value}, allow_nan=False), flush=True)
    except Exception as exc:
        print(json.dumps({'error':str(exc)}), flush=True)
`;
function rpc(path,body={}) {
  return new Promise((resolve,reject)=>{
    pending.push({resolve,reject});worker.stdin.write(JSON.stringify({path,body})+'\n');
  });
}
test.beforeAll(async({},testInfo)=>{
  testInfo.setTimeout(240000);
  worker=spawn('.venv/bin/python',['-u','-c',code],{cwd:'..',env:{...process.env,OPENBLAS_NUM_THREADS:'1'}});
  worker.stderr.on('data',data=>{stderr=(stderr+data).slice(-10000);});
  createInterface({input:worker.stdout}).on('line',line=>{
    const request=pending.shift();
    if(!request)return;
    const data=JSON.parse(line);data.error ? request.reject(new Error(data.error)) : request.resolve(data.value);
  });
  worker.on('exit',()=>{for(const request of pending.splice(0))request.reject(new Error(stderr||'Simulation backend exited'));});
  // Revalidate the real cached graph before rendering software WebGL; this
  // expensive cold-cache work is independent of the UI behaviors below.
  await rpc('/prewarm');
});
test.afterAll(()=>{worker.kill();});
test.beforeEach(async({page})=>{await page.addInitScript(()=>localStorage.setItem('ik-task-mode','position'));});
test.setTimeout(180000);

async function setup(page, holdPlan=null, planner=body=>rpc('/api/sim/prm',body)) {
  const posts=[], plans=[], errors=[];
  page.on('pageerror',error=>errors.push(error.message));
  await page.route('**/api/**',async route=>{
    const r=route.request(), path=new URL(r.url()).pathname;
    if(path==='/api/state')return route.fulfill({json:{hardware_enabled:false,connected:false,token:'prm-test',motion:null,moving:false,ik:{state:'idle',samples:[]}}});
    if(r.method()==='POST')posts.push({path,body:r.postDataJSON()});
    if(!['/api/sim/info','/api/sim/solve','/api/sim/prm'].includes(path))throw new Error(`Unexpected endpoint ${path}`);
    const data=path==='/api/sim/prm' ? await planner(r.postDataJSON()) : await rpc(path,r.method()==='POST' ? r.postDataJSON() : {});
    if(path==='/api/sim/prm') { plans.push(data);if(holdPlan)await holdPlan; }
    await route.fulfill({json:data});
  });
  await page.goto('/?view=lab');
  await expect(page.locator('#run-results tbody tr')).toHaveCount(8,{timeout:45000});
  return {posts,plans,errors};
}

test('PRM uses the real self-checker and previews joint/FK waypoints without hardware actions',async({page},testInfo)=>{
  const {posts,plans,errors}=await setup(page);
  await page.locator('#plan-prm').click();
  await expect(page.locator('#prm-status')).toContainText('success',{timeout:120000});
  expect(plans[0].validation).toBe('joint_limits_and_self_collision');
  expect(plans[0].ik.config.nn).toBe('seed');
  expect(posts.at(-1).body.collision).toBe('self');
  expect(posts.at(-1).body.guided_route_repair).toBe(true);
  expect(plans[0].ik.config.guided_route_repair).toBe(true);
  expect(plans[0].guided_repair.triggered).toBe(false);
  await expect(page.locator('#guided-route-repair')).toBeChecked();
  await expect(page.locator('#prm-repair')).toHaveText('Guided repair: not needed');
  await expect(page.locator('#ik-summary')).toContainText('Validated plan');
  await expect(page.locator('#prm-waypoints tbody tr')).toHaveCount(plans[0].waypoints_deg.length);
  await expect(page.locator('#prm-stats')).toContainText('IK');
  await expect(page.locator('#prm-attempts')).toContainText('accepted');
  await page.locator('#ik-scrub').fill(await page.locator('#ik-scrub').getAttribute('max'));
  const fk=await page.evaluate(async()=>{
    const {robot}=await import(document.querySelector('script[src*="/src/main.js"]').src);
    robot.updateMatrixWorld(true);return robot.links.joint6_flange.matrixWorld.elements.slice(12,15).map(v=>v*1000);
  });
  fk.forEach((v,i)=>expect(Math.abs(v-plans[0].frames.at(-1).position[i])).toBeLessThan(.001));
  // A short valid path can finish at 4x speed before Playwright's next poll.
  // Capture the real click handler's state before the animation frame runs.
  const play=page.locator('#play-ik');
  await play.evaluate(button=>button.addEventListener('click',()=>{
    button.dataset.stateAfterClick=button.textContent;
  },{once:true}));
  await play.click();
  await expect(play).toHaveAttribute('data-state-after-click','Ⅱ Pause');
  await play.evaluate(button=>{if(button.textContent==='Ⅱ Pause')button.click();});
  await page.screenshot({path:testInfo.outputPath('prm-desktop.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.screenshot({path:testInfo.outputPath('prm-mobile.png'),fullPage:true});
  await page.locator('#target-x').fill('101');
  await expect(page.locator('#prm-results')).toBeHidden();
  await expect(page.locator('#play-ik')).toBeDisabled();
  expect(posts.every(r=>r.path.startsWith('/api/sim/'))).toBe(true);
  expect(errors).toEqual([]);
});

test('colliding start and unsuccessful IK explain failure and never enable playback',async({page})=>{
  const {plans,errors}=await setup(page);
  const fixture=JSON.parse(readFileSync('../simulation/fixtures/self_collision_path.json','utf8'));
  for(const [i,q] of fixture.mid_deg.entries()){
    await page.locator(`#joint-${i}-number`).fill(q.toFixed(1));
    await page.locator(`#joint-${i}-number`).dispatchEvent('change');
  }
  await page.locator('#plan-prm').click();
  await expect(page.locator('#prm-status')).toContainText('invalid start',{timeout:90000});
  await expect(page.locator('#prm-status')).toContainText('g_base / joint6');
  await expect(page.locator('#play-ik')).toBeDisabled();
  await expect(page.locator('#run-results tbody tr')).toHaveCount(0);
  await page.locator('#rest-pose').click();
  await page.locator('#target-x').fill('600');
  await page.locator('#target-y').fill('0');
  await page.locator('#target-z').fill('0');
  await page.locator('#plan-prm').click();
  await expect(page.locator('#prm-status')).toContainText('ik solution not found');
  await expect(page.locator('#ik-feedback')).toContainText('does not prove target unreachability');
  await expect(page.locator('#play-ik')).toBeDisabled();
  expect(plans.map(p=>p.status)).toEqual(['invalid_start','ik_solution_not_found']);
  expect(errors).toEqual([]);
});

test('5D PRM preview uses numerical IK and displays orientation error',async({page})=>{
  const {plans,errors}=await setup(page);
  await page.locator('#ik-task').selectOption('pose5');
  await page.locator('#pose5-proposals').selectOption('none');
  await page.locator('#ik-method').selectOption('null_classical');
  await page.locator('#plan-prm').click();
  await expect(page.locator('#prm-status')).toContainText('success',{timeout:120000});
  expect(plans[0].ik.config.nn).toBe('none');
  expect(plans[0].ik.config.pose5_method).toBe('null_classical');
  expect(plans[0].ik.attempts.some(a=>a.refinement_method==='null_classical')).toBe(true);
  await page.locator('#ik-scrub').fill(await page.locator('#ik-scrub').getAttribute('max'));
  await expect(page.locator('#frame-stats')).toContainText('Z direction');
  await expect(page.locator('#orientation-chart')).toBeVisible();
  expect(plans[0].final_orientation_error_deg).toBeLessThanOrEqual(1);
  expect(errors).toEqual([]);
});

test('editing a target during an in-flight PRM query discards its late response',async({page})=>{
  let release;const hold=new Promise(resolve=>{release=resolve;});
  const {plans,errors}=await setup(page,hold);
  await page.locator('#plan-prm').click();
  await expect.poll(()=>plans.length,{timeout:90000}).toBe(1);
  await page.locator('#target-x').fill('101');
  release();
  await expect(page.locator('#plan-prm')).toBeEnabled();
  await expect(page.locator('#prm-results')).toBeHidden();
  await expect(page.locator('#run-results tbody tr')).toHaveCount(0);
  await expect(page.locator('#play-ik')).toBeDisabled();
  expect(errors).toEqual([]);
});

test('guided repair recovers a real blocked route and the checkbox restores original failure',async({page},testInfo)=>{
  const input=await rpc('/repair-input');
  const {posts,plans,errors}=await setup(page,null,body=>rpc('/repair-fixture',body));
  await page.locator('#ik-task').selectOption('pose5');
  for(const [i,q] of input.start_deg.entries()){
    await page.locator(`#joint-${i}-number`).fill(String(q));
    await page.locator(`#joint-${i}-number`).dispatchEvent('change');
  }
  await page.locator('#direction-entry').selectOption('axis');
  for(const [i,a] of [...'xyz'].entries()){
    await page.locator(`#target-${a}`).fill(String(input.target_mm[i]));
    await page.locator(`#target-u${a}`).fill(String(input.target_axis[i]));
  }
  await page.locator('#plan-prm').click();
  await expect(page.locator('#prm-repair')).toContainText('route recovered',{timeout:90000});
  expect(plans[0].converged).toBe(true);
  expect(plans[0].guided_repair.recovered).toBe(true);
  expect(plans[0].guided_repair.new_nodes).toBe(32);
  expect(plans[0].validation).toBe('joint_limits_and_self_collision');
  await expect(page.locator('#play-ik')).toBeEnabled();
  await page.screenshot({path:testInfo.outputPath('guided-repair-desktop.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.screenshot({path:testInfo.outputPath('guided-repair-mobile.png'),fullPage:true});
  await page.locator('#guided-route-repair').uncheck();
  await expect(page.locator('#prm-results')).toBeHidden();
  await expect(page.locator('#play-ik')).toBeDisabled();
  await page.locator('#plan-prm').click();
  await expect(page.locator('#prm-status')).toContainText('start not connected');
  await expect(page.locator('#prm-repair')).toHaveText('Guided repair: off');
  expect(plans[1].frames).toEqual([]);
  expect(posts.filter(p=>p.path==='/api/sim/prm').map(p=>p.body.guided_route_repair)).toEqual([true,false]);
  expect(posts.every(p=>p.path.startsWith('/api/sim/'))).toBe(true);
  expect(errors).toEqual([]);
});

test('PRM preview invalidates an older hardware execution plan',async({page})=>{
  const {posts,errors}=await setup(page);
  const info=await rpc('/api/sim/info');
  const start=info.default_start_deg;
  const state={hardware_enabled:true,connected:true,token:'fake-hardware',angles:start,motion:null,moving:false,ik:{state:'idle',samples:[],plan:null}};
  await page.route('**/api/state',route=>route.fulfill({json:{...state,updated:Date.now()/1000}}));
  await page.route('**/api/ik/plan',route=>{
    const body=route.request().postDataJSON();posts.push({path:'/api/ik/plan',body});
    const plan={id:'old-plan',converged:true,waypoints:1,expires_in:60,final_error_mm:0};
    state.ik.plan=plan;
    return route.fulfill({json:{plan,start_deg:start,result:{task:'position',method:'dls',converged:true,final_error_mm:0,iterations:0,joint_travel_deg:0,elapsed_ms:1,stop_reason:'target_reached',frames:[{q:start,position:body.target_mm,error_mm:0,time:0}]}}});
  });
  await page.route('**/api/sim/prm',route=>{
    posts.push({path:'/api/sim/prm',body:route.request().postDataJSON()});
    return route.fulfill({json:{status:'ik_solution_not_found',reason:'No valid goal',simulation_only:true,validation:'joint_limits_and_self_collision',converged:false,frames:[],waypoints_deg:[],ik_ms:1,planning_ms:0}});
  });
  await expect(page.locator('#hardware-ik')).toBeVisible();
  await page.locator('#plan-hardware-ik').click();
  await expect(page.locator('#hardware-ik-feedback')).toContainText('Plan ready');
  await page.locator('#arm-movement').check();
  await page.locator('#verify-ik-frame').check();
  await expect(page.locator('#execute-hardware-ik')).toBeEnabled();
  await page.locator('#plan-prm').click();
  await expect(page.locator('#prm-status')).toContainText('ik solution not found');
  await expect(page.locator('#execute-hardware-ik')).toBeDisabled();
  expect(posts.some(p=>p.path==='/api/ik/execute')).toBe(false);
  expect(errors).toEqual([]);
});

test('changing repair while planning discards the late preview',async({page})=>{
  let release;const hold=new Promise(resolve=>{release=resolve;});
  const {plans,errors}=await setup(page,hold);
  try{
    await page.locator('#plan-prm').click();
    await expect.poll(()=>plans.length,{timeout:90000}).toBe(1);
    await page.locator('#guided-route-repair').uncheck();
  }finally{release();}
  await expect(page.locator('#plan-prm')).toBeEnabled();
  await expect(page.locator('#prm-results')).toBeHidden();
  await expect(page.locator('#play-ik')).toBeDisabled();
  expect(errors).toEqual([]);
});
