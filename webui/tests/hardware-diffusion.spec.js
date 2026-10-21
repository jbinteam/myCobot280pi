import {test,expect} from '@playwright/test';

for(const {converged,encoder,failure,classical} of [{converged:true,encoder:false,classical:true},{converged:true,encoder:false},{converged:false,encoder:false},{converged:false,encoder:false,failure:"ik_failed"},{converged:true,encoder:'encoder_drag'},{converged:true,encoder:'encoder_drag_stream'},{converged:true,encoder:'encoder_drag_staged'},{converged:true,encoder:'encoder_drag_profile'}])test(`fixed ${classical?"learned RRT* ":"v2 "}${encoder?encoder+' ':''}hardware plan ${converged?'requires scene review and sends plan ID only':failure||'failure leaves execution disabled'}`,async({page,request})=>{
  const info=await (await request.get('http://127.0.0.1:8000/api/sim/info')).json();
  await page.addInitScript(()=>localStorage.setItem('ik-task-mode','position'));
  const profiled=encoder==='encoder_drag_profile',staged=encoder==='encoder_drag_staged',stream=['encoder_drag_stream','encoder_drag_staged'].includes(encoder);
  const start=[0,-40,-130,80,0,50],end=[.8,-40,-130,80,0,50],calls=[];
  let state={hardware_enabled:true,connected:false,token:'v2-hardware-test',angles:null,
    motion:null,moving:false,firmware:6.5,limits:info.limits_deg,
    ik:{state:'idle',samples:[],plan:null,estimated_flange_mm:[160,-64,83],orientation_xy_deg:[0,0]}};
  await page.route('**/api/**',async route=>{
    const r=route.request(),path=new URL(r.url()).pathname;
    if(path==='/api/sim/info'){await route.fulfill({json:info});return;}
    if(r.method()==='POST'){
      const body=r.postDataJSON();calls.push({path,body});
      expect(r.headers()['x-robot-token']).toBe('v2-hardware-test');
      if(path==='/api/connect')state={...state,connected:true,angles:start};
      if(path==='/api/ik/small-target'){
        const larger=body.preset==='j3_plus_6';
        await route.fulfill({json:{staged:larger,recommended_joint_command:larger?'encoder_drag_staged':null,preset:body.preset,start_deg:start,target_mm:[161.234,-64.987,84.321],orientation_xy_deg:[12.3456,-45.6789],eef_offset_mm:[0,0,0],maximum_speed_percent:5,requires_planning:true,motion_sent:false}});return;
      }
      if(path==='/api/ik/diffusion/plan'){
        const plan=converged?{id:'v2-live-plan',method:classical?'learned_rrt':'diffusion',target_mm:body.target_mm,
          start_deg:start,waypoints:1,expires_in:60,converged:true,hardware_diffusion:true,
          execution_mode:profiled?'encoder_profile':stream?'encoder_stream':'path_waypoints',path_arrival_deg:3,arrival_deg:1.5,max_live_command_deg:4,max_step_deg:2,
          encoder_drag:encoder?{staged,stage_count:staged?3:1}:undefined,max_sdk_speed:encoder&&!profiled?5:80,joint_command:body.joint_command||'send_angles',table_top_mm:-35,final_error_mm:.8,final_orientation_error_deg:.1}:null;
        const result={method:classical?'learned_rrt':'diffusion',task:'pose5',converged,status:converged?'success':failure||'no_safe_chunk',
          ik_diagnostics:failure?{best_position_error_mm:23.65,best_direction_error_deg:1.86}:undefined,
          reason:'No safe path',target_mm:body.target_mm,target_axis:body.target_axis,
          planning_timing_ms:{server_total:1234.5,encoder_profile_prepare:54,
            v2_prepare:{trajectory_generation:900,hardware_route_validation:100,velocity_retiming:120}},
          orientation_xy_deg:[0,0],final_error_mm:converged?.8:10,final_orientation_error_deg:.1,
          iterations:1,joint_travel_deg:.8,elapsed_ms:20,stop_reason:'checked_smoothed_diffusion_rollout',
          frames:converged?[{q:start,position:[160,-64,83],error_mm:2,time:0},
            {q:end,position:[162,-64,83],error_mm:.8,time:.2}]:[]};
        state.ik={...state.ik,state:'planned',plan};
        await route.fulfill({json:{plan,result,start_deg:start,target_mm:body.target_mm,hardware_diffusion:true}});return;
      }
      if(path==='/api/ik/execute')state={...state,motion:'ik',moving:true,
        ik:{...state.ik,plan:null,state:'running',execution_id:'v2-live-plan',total_waypoints:1,completed_waypoints:0,sdk_speed:5,
          joint_command:encoder||'send_angles',execution_mode:profiled?'encoder_profile':stream?'encoder_stream':'path_waypoints',
          encoder_drag:{staged},completed_stages:1,total_stages:3,tracking_error_deg:.12,reference_phase_s:.4,actual_send_interval_s:.24,sent_encoder_speeds:encoder?[2,3,1,1,1,1]:null}};
      if(path==='/api/ik/heartbeat'){await route.fulfill({json:{active:true}});return;}
    }
    await route.fulfill({json:{...state,updated:Date.now()/1000}});
  });
  await page.goto('/?view=lab');
  await expect(page.locator('#model-status')).toHaveText('Model ready',{timeout:45000});
  await expect(page.locator('#hardware-planner')).toHaveValue('learned_rrt');
  await expect(page.locator('#fixed-target-planner')).toHaveValue('learned_rrt');
  if(!classical)await page.locator('#fixed-target-planner').selectOption('diffusion_v2');
  await expect(page.locator('#hardware-joint-command')).toHaveValue('encoder_drag_profile');
  await expect(page.locator('#ik-task')).toHaveValue('pose5');
  await expect(page.locator('#ik-task')).toBeDisabled();
  await expect(page.locator('#ik-sdk-speed')).toHaveValue('70');
  await expect(page.locator('#small-target-preset')).toBeDisabled();
  await page.locator('#connect-robot').click();
  await page.locator('#hardware-joint-command').selectOption(encoder||'send_angles');
  // This fixture uses original quintic references; adjustable ramps have their
  // own velocity-shape regression. Keep current product defaults unchanged.
  if(profiled)await page.locator('#velocity-shape').selectOption('quintic');
  if(encoder){
    await expect(page.locator('#encoder-drag-note')).toBeVisible();
    await expect(page.locator('#ik-sdk-speed')).toHaveAttribute('max',profiled?'80':'5');
  }
  if(encoder==='encoder_drag_stream'){
    await page.locator('#eef-offset-x').fill('5');
    await page.locator('#ik-sdk-speed').fill('6');
    await page.locator('#direction-entry').selectOption('axis');
    await page.locator('#small-target-preset').selectOption('j4_minus_2');
    await expect(page.locator('#hardware-ik-feedback')).toContainText('Small target loaded from live joints');
    await expect(page.locator('#target-x')).toHaveValue('161.234');
    await expect(page.locator('#target-y')).toHaveValue('-64.987');
    await expect(page.locator('#target-z')).toHaveValue('84.321');
    await expect(page.locator('#target-rx')).toHaveValue('12.3456');
    await expect(page.locator('#target-ry')).toHaveValue('-45.6789');
    await expect(page.locator('#eef-offset-x')).toHaveValue('0');
    await expect(page.locator('#direction-entry')).toHaveValue('tilts');
    await expect(page.locator('#ik-sdk-speed')).toHaveValue('5');
    await expect(page.locator('#small-target-preset')).toHaveValue('j4_minus_2');
    await expect(page.locator('#execute-hardware-ik')).toBeDisabled();
    expect(calls.find(c=>c.path==='/api/ik/small-target').body).toEqual({preset:'j4_minus_2'});
    expect(calls.some(c=>['/api/ik/diffusion/plan','/api/ik/execute','/api/move','/api/jog'].includes(c.path))).toBe(false);
    await page.locator('#target-x').fill('170');
    await expect(page.locator('#small-target-preset')).toHaveValue('');
  }
  if(staged){
    await page.locator('#hardware-joint-command').selectOption('send_angles');
    await page.locator('#small-target-preset').selectOption('j3_plus_6');
    await expect(page.locator('#hardware-joint-command')).toHaveValue('encoder_drag_staged');
    await expect(page.locator('#hardware-ik-feedback')).toContainText('Larger target loaded');
    await expect(page.locator('#encoder-drag-note')).toContainText('no total-excursion cap');
    await expect(page.locator('#execute-hardware-ik')).toBeDisabled();
    expect(calls.some(c=>['/api/ik/diffusion/plan','/api/ik/execute','/api/move','/api/jog'].includes(c.path))).toBe(false);
    await page.locator('#small-target-preset').selectOption('j4_minus_2');
    await expect(page.locator('#hardware-joint-command')).toHaveValue('encoder_drag_stream');
    await page.locator('#small-target-preset').selectOption('j3_plus_6');
  }
  await page.locator('#use-live-xyz').click();
  await page.locator('#target-x').fill('162');
  const selectedSpeed=profiled?80:5;
  await page.locator('#ik-sdk-speed').fill(String(selectedSpeed));
  await page.locator('#plan-hardware-ik').click();
  await expect(page.locator('#hardware-ik-feedback')).toContainText(converged?'Plan ready':'No executable fixed-target path');
  if(converged){
    await expect(page.locator('#hardware-ik-feedback')).toContainText('Plan processing 1.23 s.');
    await expect(page.locator('#hardware-ik-feedback')).toContainText('generation 0.90 s · route validation 0.10 s · retiming 0.12 s · encoder preparation 0.05 s');
    await expect(page.locator('#hardware-reference-title')).toHaveText('Planned trajectory velocities');
    await expect(page.locator('#hardware-reference-velocity-chart polyline')).toHaveCount(6);
    await expect(page.locator('#hardware-reference-velocity-chart')).toContainText('Trajectory time (s)');
  }
  if(converged)await expect(page.locator('#hardware-ik-feedback')).toContainText(profiled?'Checked velocity profile: instantaneous signed derivatives':staged?'Staged encoder path':stream?'Continuous encoder samples replace targets while moving':'Joint steps ≤2°. Interior tracking allows 3°; corners and the endpoint stop within 1.5°');
  const sent=calls.find(c=>c.path==='/api/ik/diffusion/plan').body;
  if(classical)expect(sent.planner).toBe('learned_rrt');
  else expect(sent.planner).toBeUndefined();
  expect(sent).toMatchObject({target_mm:[162,-64,83],eef_offset_mm:[0,0,0],target_axis:[0,0,1]});
  expect(sent.speed).toBe(selectedSpeed);
  expect(Object.keys(sent).sort()).toEqual(['eef_offset_mm','joint_command','speed','target_axis','target_mm',...(classical?['planner']:[])].sort());
  expect(sent.joint_command).toBe(encoder||'send_angles');
  await expect(page.locator('#plan-diffusion')).toBeDisabled();
  await page.locator('#arm-movement').check();
  await page.locator('#verify-ik-frame').check();
  await expect(page.locator('#execute-hardware-ik')).toBeDisabled();
  await page.locator('#verify-diffusion-scene').check();
  if(!converged){
    await expect(page.locator('#execute-hardware-ik')).toBeDisabled();
    if(failure){
      await expect(page.locator('#hardware-ik-feedback')).toContainText('23.65 mm position, 1.86° direction');
      await expect(page.locator('#hardware-ik-feedback')).toContainText('does not prove the target is unreachable');
    }
    return;
  }
  await expect(page.locator('#execute-hardware-ik')).toBeEnabled();
  if(encoder==='encoder_drag_stream'){
    const plannedCount=calls.filter(c=>c.path==='/api/ik/diffusion/plan').length;
    await page.locator('#small-target-preset').selectOption('j4_plus_2');
    await expect(page.locator('#hardware-ik-feedback')).toContainText('Small target loaded');
    await expect(page.locator('#execute-hardware-ik')).toBeDisabled();
    expect(calls.filter(c=>c.path==='/api/ik/diffusion/plan').length).toBe(plannedCount);
    await page.locator('#plan-hardware-ik').click();
    await expect(page.locator('#execute-hardware-ik')).toBeEnabled();
  }
  await page.locator('#ik-sdk-speed').fill('81');
  await expect(page.locator('#execute-hardware-ik')).toBeDisabled();
  await page.locator('#plan-hardware-ik').click();
  await expect(page.locator('#hardware-ik-feedback')).toContainText(encoder&&!profiled?'1 to 5':'1 to 80');
  expect(calls.some(c=>c.path==='/api/ik/execute')).toBe(false);
  await page.locator('#ik-sdk-speed').fill('5');
  await page.locator('#target-x').fill('163');
  await expect(page.locator('#execute-hardware-ik')).toBeDisabled();
  await expect(page.locator('#hardware-reference-velocity-chart svg')).toHaveCount(0);
  await page.locator('#plan-hardware-ik').click();
  await expect(page.locator('#execute-hardware-ik')).toBeEnabled();
  await page.locator('#execute-hardware-ik').click();
  await expect(page.locator('#hardware-joint-command')).toBeDisabled();
  await expect(page.locator('#small-target-preset')).toBeDisabled();
  if(stream||profiled){
    if(staged)await expect(page.locator('#hardware-ik-progress')).toContainText('stages 1/3');
    await expect(page.locator('#hardware-ik-progress')).toContainText('tracking gap 0.12°');
    await expect(page.locator('#hardware-ik-progress')).toContainText('send interval 0.24 s');
  }
  expect(calls.find(c=>c.path==='/api/ik/execute').body).toEqual({plan_id:'v2-live-plan',speed:5,
    armed:true,frame_verified:true,scene_verified:true});
  expect(calls.some(c=>['/api/move','/api/jog','/api/ik/plan'].includes(c.path))).toBe(false);
  await expect(page.locator('#follow-live')).toBeChecked();
});
