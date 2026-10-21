import {readShapeEditor,renderShapeEditor} from './reference-shape.js';
import { offsetInputIds, resetLocalOffset } from './target-frame.js';
import { renderErrorPlot, renderJointVelocityPlot } from './ik-plot.js';
import { hardwareVelocityProfiles } from './hardware-velocity.js';
import { createTrajectorySampler } from './trajectory-playback.js';
import { readToolAxis, readReferenceTilts, syncAxisFromTilts, directionInputIds } from './tool-direction.js';
const $ = id => document.getElementById(id);

export function initHardwareIK({ onPlan, beforeExecute, pausePreview, getTargetFrame }) {
  let state={}, token='', plan=null, busy=false, ownedExecution=null, heartbeatBusy=false;
  let mouseTracking=false;
  let mousePreview=false;
  let reviewedProfile=[];
  let applyingPreset=false;
  const diffusion=()=>['diffusion_v2','learned_rrt'].includes($('fixed-target-planner').value);
  const classical=()=>$('fixed-target-planner').value==='learned_rrt';
  const plannerName=()=>classical()?'learned IK + classical RRT*':'diffusion v2';
  const encoderTrial=()=>encoder()&&$('hardware-joint-command').value!=='encoder_drag_profile';
  const encoder=()=>diffusion()&&$('hardware-joint-command').value.startsWith('encoder_drag');
  const feedback=(text,error=false)=>{ $('hardware-ik-feedback').textContent=text; $('hardware-ik-feedback').classList.toggle('error',error); };
  function invalidate() { plan=null; reviewedProfile=[]; liveChart(state.ik||{}); update(); }
  function update() {
    $('hardware-ik').hidden=state.hardware_enabled!==true;
    const live=state.hardware_enabled===true;
    $('hardware-velocity-profiles').hidden=!live&&!mousePreview;
    $('velocity-profiles-mode').textContent=live?'LIVE MOTION':'SIMULATION PREVIEW';
    $('mouse-planned-velocity').hidden=!mousePreview;
    for(const id of ['hardware-velocity-speeds','hardware-live-velocity-charts','hardware-velocity-note'])$(id).hidden=!live;
    const running=!!(state.motion||state.moving);
    $('hardware-planner').disabled=running||busy||mouseTracking;
    $('fixed-target-planner').disabled=running||busy||mouseTracking;
    $('hardware-joint-command').disabled=running||busy||mouseTracking||!diffusion();
    $('small-target-preset').disabled=!state.connected||running||busy||mouseTracking||!diffusion();
    const shapeEnabled=diffusion()&&$('hardware-joint-command').value==='encoder_drag_profile';
    $('velocity-shape-settings').hidden=false;
    for(const id of ['velocity-shape','velocity-rise','velocity-fall'])$(id).disabled=running||busy||mouseTracking;
    $('encoder-drag-note').hidden=!encoder();
    $('encoder-drag-note').textContent=$('hardware-joint-command').value==='encoder_drag_profile'
      ?'The selected planner generates the checked trajectory. Its analytic derivative sets the six joint speed words on a monotonic trajectory clock, with encoder targets on the route to bound travel. No added 2° stops or total-excursion cap. Tracking lag stops execution; speed conversion is nominal. Arm speed is adjustable from 1–80%; trajectory limits can reduce the peak.'
      :$('hardware-joint-command').value==='encoder_drag_staged'
      ?'Staged encoder control: no total-excursion cap; targets must respect joint limits and collision checks. Speed stays at 5% or less. The checked path is divided into stages of at most 2°. Each stage stops, settles and verifies live encoders before continuing. Plan and preview the full path.'
      :'Encoder trials stay within 3° of the starting joints at 5% or less. Continuous mode samples the checked trajectory, pauses for tracking lag and stops at corners and the endpoint. Live encoders are verified; speed conversion is nominal.';
    $('hardware-diffusion-note').hidden=!diffusion();
    $('hardware-scene-confirmation').hidden=!(diffusion()||['diffusion_v2','learned_rrt'].includes($('hardware-planner').value));
    $('verify-diffusion-scene').disabled=running||busy;
    $('ik-sdk-speed').max=String(plan?.max_sdk_speed||(encoderTrial()?5:80));
    for(const id of ['ik-task','target-x','target-y','target-z','target-rx','target-ry','ik-orientation-tolerance','ik-method','ik-case','ik-steps','ik-speed','ik-tolerance','ik-damping','ik-sdk-speed','verify-ik-frame',...offsetInputIds,...directionInputIds,'pose5-proposals'])$(id).disabled=running||busy;
    if(diffusion()&&state.hardware_enabled===true){
      if($('ik-task').value!=='pose5'){$('ik-task').value='pose5';$('ik-task').dispatchEvent(new Event('change'));}
      $('ik-task').disabled=true;
    }
    $('plan-hardware-ik').disabled=!state.connected||running||busy||mouseTracking;
    $('use-live-xyz').disabled=!state.connected||running||busy;
    const expired=plan && Date.now()>plan.expires;
    $('execute-hardware-ik').disabled=!state.connected||running||busy||mouseTracking||!plan||expired||!plan.converged||!$('arm-movement').checked||!$('verify-ik-frame').checked||(plan.hardware_diffusion&&!$('verify-diffusion-scene').checked);
  }
  async function request(path,body) {
    const response=await fetch(`/api/${path}`,{method:'POST',headers:{'Content-Type':'application/json','X-Robot-Token':token},body:JSON.stringify(body),signal:AbortSignal.timeout(path==='ik/diffusion/plan'?180000:12000)});
    const data=await response.json(); if(!response.ok)throw new Error(data.error||'Hardware IK request failed'); return data;
  }
  function liveChart(ik) {
    const samples=ik.samples||[];
    renderErrorPlot($('hardware-error-chart'),[{frames:samples,color:'#327559'}],{field:'error_mm',label:'URDF position error (mm)',timeLabel:'Actual elapsed time (s)',minimum:5,height:190});
    const gaze=ik.execution_mode==='encoder_feedback_tracking'&&samples.some(s=>Number.isFinite(s.cursor_direction_error_deg));
    const oriented=samples.some(s=>Number.isFinite(gaze?s.cursor_direction_error_deg:s.orientation_error_deg));
    $('hardware-orientation-chart').hidden=!oriented;
    if(oriented)renderErrorPlot($('hardware-orientation-chart'),[{frames:samples,color:'#ac682e'}],{field:gaze?'cursor_direction_error_deg':'orientation_error_deg',label:gaze?'Tool Z aim error to cursor (degrees)':'Flange Z direction error (degrees)',timeLabel:'Actual elapsed time (s)',minimum:3,height:190});
    else $('hardware-orientation-chart').replaceChildren();
    const {reference,measured}=hardwareVelocityProfiles(samples);
    const planned=reviewedProfile.length>0&&(!reference.length||ik.state==='planned');
    $('hardware-reference-title').textContent=planned?'Planned trajectory velocities':ik.execution_mode==='encoder_feedback_tracking'?'Feedback-paced reference':'Commanded reference';
    renderJointVelocityPlot($('hardware-reference-velocity-chart'),planned?reviewedProfile:reference,
      {label:planned?'Planned velocities (rad/s)':'Reference velocities (rad/s)',
        timeLabel:planned?'Trajectory time (s)':'Actual elapsed time (s)'});
    renderJointVelocityPlot($('hardware-measured-velocity-chart'),measured,
      {label:'Measured velocities (rad/s)',timeLabel:'Actual elapsed time (s)'});
    $('hardware-reference-speed').textContent=!planned&&reference.length?`${reference.at(-1).speed_rad_s.toFixed(3)} rad/s`:'—';
    $('hardware-measured-speed').textContent=measured.length?`${measured.at(-1).speed_rad_s.toFixed(3)} rad/s`:'—';
  }
  function apply(next) {
    const disconnected=state.connected&&!next.connected;
    state=next;token=next.token||token;
    if(disconnected){plan=null;reviewedProfile=[];ownedExecution=null;$('small-target-preset').value='';$('verify-ik-frame').checked=$('verify-diffusion-scene').checked=false;}
    const ik=next.ik||{};
    if(plan && ik.plan && ik.plan.id!==plan.id)plan=null;
    if(plan && !ik.plan && Object.hasOwn(next,'ik'))plan=null;
    $('live-flange').textContent=ik.estimated_flange_mm ? `[${ik.estimated_flange_mm.map(v=>v.toFixed(1)).join(', ')}] mm` : 'Connect to read live joints';
    $('live-flange-direction').textContent=ik.orientation_xy_deg ? `tilt X ${ik.orientation_xy_deg[0].toFixed(1)}° / Y ${ik.orientation_xy_deg[1].toFixed(1)}° · local Z spin free` : '';
    const directionError=Number.isFinite(ik.estimated_orientation_error_deg) ? ` · Z direction error ${ik.estimated_orientation_error_deg.toFixed(2)}°` : '';
    if(ik.state==='running') {
      pausePreview();
      const speedText=ik.joint_command?.startsWith('encoder_drag')?`profile allowance ${ik.sdk_speed}%`:`SDK speed ${ik.sdk_speed}`;
      const stageText=ik.encoder_drag?.staged?` · stages ${ik.completed_stages??0}/${ik.total_stages??0}`:'';
      const trackingText=['encoder_stream','encoder_profile'].includes(ik.execution_mode)?` · tracking gap ${ik.tracking_error_deg?.toFixed(2)??'—'}° · reference ${ik.reference_phase_s?.toFixed(1)??'—'} s · send interval ${ik.actual_send_interval_s?.toFixed(2)??'—'} s${stageText}${ik.tracking_paused?' · waiting for feedback':''}`:'';
      const encoderText=ik.sent_encoder_speeds?` · J1–J6 speed words [${ik.sent_encoder_speeds.join(', ')}] step/s`:'';
      $('hardware-ik-progress').textContent=`${ik.completed_waypoints}/${ik.total_waypoints} waypoints · estimated FK error ${ik.estimated_error_mm?.toFixed(2)??'—'} mm${directionError} · ${speedText}${encoderText}${trackingText}`;
    } else if(['completed','failed','stopped','disconnected'].includes(ik.state)) {
      if(ownedExecution)feedback(ik.message,ik.state==='failed');
      ownedExecution=null;
      $('hardware-ik-progress').textContent=`${ik.state}${ik.encoder_drag?.staged?` · stages ${ik.completed_stages??0}/${ik.total_stages??0}`:''} · ${ik.completed_waypoints??0}/${ik.total_waypoints??0} waypoints · estimated FK error ${ik.estimated_error_mm?.toFixed(2)??'—'} mm${directionError}`;
    }
    liveChart(ik);update();
  }
  renderShapeEditor();
  for(const id of ['velocity-shape','velocity-rise','velocity-fall']) {
    const changed=()=>{renderShapeEditor();invalidate();};
    $(id).addEventListener('input',changed);$(id).addEventListener('change',changed);
  }
  window.addEventListener('robot-state',event=>apply(event.detail));
  window.addEventListener('mouse-aim-change',event=>{
    mousePreview=event.detail.active;
    mouseTracking=mousePreview&&event.detail.hardware;
    if(mouseTracking){plan=null;reviewedProfile=[];}
    update();
  });
  window.addEventListener('mouse-aim-profile',({detail})=>{
    renderJointVelocityPlot($('mouse-planned-velocity-chart'),detail.points,
      {label:'Planned velocities (rad/s)',timeLabel:'Trajectory time (s)'});
    $('mouse-profile-status').textContent=detail.points.length
      ?`Latest accepted tracking path · ${detail.points.at(-1).time.toFixed(2)} s reference duration. ${detail.feedbackPaced?'Live feedback can stretch execution time. ':''}Updates when the next checked path takes over.`
      :'Move in the scene to generate a checked velocity profile.';
  });
  window.addEventListener('preview-edited',invalidate);
  window.addEventListener('heldout-case-loaded',event=>{
    if(!event.detail.hardware)return;
    $('small-target-preset').value='';
    invalidate();
    feedback(`Held-out target ${event.detail.name} loaded. Click Plan from live joints and review the route.${encoder()?' Joint limits, collision checks and the selected driver’s speed/tracking guards still apply.':''}`);
  });
  for(const id of ['ik-task','target-x','target-y','target-z','target-rx','target-ry','ik-orientation-tolerance','ik-method','ik-case','ik-steps','ik-speed','ik-sdk-speed','ik-tolerance','ik-damping',...offsetInputIds,...directionInputIds]) {
    const edited=()=>{if(!applyingPreset)$('small-target-preset').value='';invalidate();};
    $(id).addEventListener('change',edited);
    $(id).addEventListener('input',edited);
  }
  for(const id of ['solve-ik','compare-ik','plan-prm']) $(id).addEventListener('click',invalidate);
  $('verify-ik-frame').addEventListener('change',update);$('arm-movement').addEventListener('change',update);
  $('verify-diffusion-scene').addEventListener('change',update);
  $('fixed-target-planner').addEventListener('change',()=>{
    $('small-target-preset').value='';
    if(!diffusion())$('hardware-joint-command').value='send_angles';
    invalidate();$('ik-sdk-speed').value=diffusion()?(encoderTrial()?'5':'70'):'30';
    if(diffusion()){$('ik-task').value='pose5';$('ik-task').dispatchEvent(new Event('change'));}
    feedback(diffusion()?`${plannerName()} selected. Plan and preview the trajectory from live joints.`:'Legacy IK selected.');
  });
  $('hardware-joint-command').addEventListener('change',()=>{
    if(encoderTrial())$('ik-sdk-speed').value='5';
    else if(diffusion()&&$('hardware-joint-command').value==='encoder_drag_profile')$('ik-sdk-speed').value='70';
    invalidate();
    feedback($('hardware-joint-command').value==='encoder_drag_profile'
      ?'Velocity profile selected. Plan from live joints, preview the trajectory and its signed joint velocities, then execute the timed encoder/speed stream.'
      :$('hardware-joint-command').value==='encoder_drag_staged'
      ?'Staged encoder mode selected. Total excursion is unlimited within joint limits and collision checks, at 5% speed or less. Every 2° stage settles and verifies live encoders before continuing.'
      :encoder()
      ?'Encoder trial selected. Use a target within 3° of the starting joints, at 5% or less. Planning checks live encoders without moving the arm.'
      :'Joint angle targets selected. Plan again before execution.');
  });
  $('use-live-xyz').onclick=()=>{
    if(!state.ik?.estimated_flange_mm)return;
    $('small-target-preset').value='';
    resetLocalOffset();
    ['x','y','z'].forEach((axis,i)=>{ $(`target-${axis}`).value=state.ik.estimated_flange_mm[i].toFixed(4);$(`target-${axis}`).dispatchEvent(new Event('input'));});
    if(state.ik.orientation_xy_deg)['rx','ry'].forEach((axis,i)=>{$(`target-${axis}`).value=state.ik.orientation_xy_deg[i].toFixed(4);$(`target-${axis}`).dispatchEvent(new Event('input'));});
    if(state.ik.orientation_xy_deg){syncAxisFromTilts();$('direction-entry').dispatchEvent(new Event('change'));}
    feedback('Live flange position and tilt copied. Adjust the target, then plan from live joints.');
  };
  $('small-target-preset').addEventListener('change',async()=>{
    const preset=$('small-target-preset').value;
    invalidate();
    if(!preset)return;
    applyingPreset=true;
    busy=true;update();feedback('Reading live joints to fill the target preset; no motion is sent…');
    try {
      const data=await request('ik/small-target',{preset});
      if(!state.connected||state.motion||state.moving||mouseTracking)throw new Error('Robot session changed; choose the target again when the arm is connected and idle');
      resetLocalOffset();
      if(data.recommended_joint_command&&$('hardware-joint-command').value!=='encoder_drag_profile')$('hardware-joint-command').value=data.recommended_joint_command;
      else if($('hardware-joint-command').value==='encoder_drag_staged')$('hardware-joint-command').value='encoder_drag_stream';
      $('direction-entry').value='tilts';
      ['x','y','z'].forEach((axis,i)=>{$(`target-${axis}`).value=data.target_mm[i].toFixed(3);$(`target-${axis}`).dispatchEvent(new Event('input'));});
      ['rx','ry'].forEach((axis,i)=>{$(`target-${axis}`).value=data.orientation_xy_deg[i].toFixed(4);$(`target-${axis}`).dispatchEvent(new Event('input'));});
      syncAxisFromTilts();$('direction-entry').dispatchEvent(new Event('change'));
      const speed=Number($('ik-sdk-speed').value);
      if(!Number.isInteger(speed)||speed<1||speed>data.maximum_speed_percent)$('ik-sdk-speed').value=String(data.maximum_speed_percent);
      $('small-target-preset').value=preset;
      invalidate();
      feedback($('hardware-joint-command').value==='encoder_drag_profile'?'Target loaded from live joints. Velocity-profile control remains selected; plan and preview before execution.'
        :data.staged?'Larger target loaded from live joints. Staged encoder mode is selected; EEF offsets are zero and speed is at most 5%. Plan and preview every stage before execution.'
        :'Small target loaded from live joints. EEF offsets are zero; speed is at most 5%. Plan and preview the complete path before execution.');
    }catch(error){$('small-target-preset').value='';feedback(error.message,true);}
    finally{applyingPreset=false;busy=false;update();}
  });
  $('plan-hardware-ik').onclick=async()=>{
    busy=true;plan=null;reviewedProfile=[];liveChart({});update();feedback(diffusion()?`Reading live joints and planning with ${plannerName()}; no motion is sent…`:'Reading live start and solving IK; no motion is sent…');
    try {
      const { base, offset, target: xyz }=getTargetFrame();
      if(!xyz.every(Number.isFinite))throw new Error('Enter finite target and EEF local offset values');
      const velocityShape=diffusion()&&$('hardware-joint-command').value==='encoder_drag_profile'?readShapeEditor():{kind:'quintic'};
      const task=$('ik-task').value;
      const speed=Number($('ik-sdk-speed').value);
      const speedLimit=diffusion()&&encoderTrial()?5:80;
      if(!Number.isInteger(speed)||speed<1||speed>speedLimit)throw new Error(`Arm speed must be an integer from 1 to ${speedLimit}%`);
      if(task==='pose5')readToolAxis();
      const orientation=task==='pose5'?readReferenceTilts():null;
      const data=diffusion()
        ?await request('ik/diffusion/plan',{...(classical()?{planner:'learned_rrt'}:{}),target_mm:base,eef_offset_mm:offset,target_axis:readToolAxis(),speed,
          joint_command:$('hardware-joint-command').value,...(velocityShape.kind!=='quintic'?{velocity_shape:velocityShape}:{})})
        :await request('ik/plan',{task,speed,...(task==='pose5'?{orientation_xy_deg:orientation,orientation_tolerance_deg:Number($('ik-orientation-tolerance').value)}:{}),target_mm:base,eef_offset_mm:offset,method:$('ik-method').value,steps:Number($('ik-steps').value),speed_rad_s:Number($('ik-speed').value),tolerance_mm:Number($('ik-tolerance').value),damping:Number($('ik-damping').value)});
      if(!data.plan){
        const result=data.result,diagnostic=result.ik_diagnostics;
        const best=diagnostic&&Number.isFinite(diagnostic.best_position_error_mm)&&Number.isFinite(diagnostic.best_direction_error_deg)
          ?` Best candidate errors: ${diagnostic.best_position_error_mm.toFixed(2)} mm position, ${diagnostic.best_direction_error_deg.toFixed(2)}° direction.`:'';
        const guidance=result.status==='ik_failed'
          ?`${best} The search did not find a valid position-and-direction match; this does not prove the target is unreachable. Adjust position or tool direction and plan again.`
          :' Adjust the target and plan again.';
        feedback(`No executable fixed-target path: ${result.reason||result.status}.${guidance}`,true);return;
      }
      plan={...data.plan,expires:Date.now()+data.plan.expires_in*1000};
      onPlan(data);
      reviewedProfile=createTrajectorySampler(data.result.frames,data.result.smoothing).velocityPoints;
      liveChart({});
      const shapeNote=velocityShape.kind==='quintic'?'':` Reference shape: ${velocityShape.rise_percent}% acceleration, ${velocityShape.fall_percent}% deceleration.`;
      const allowance=data.result.joint_velocity_allowance_deg_s;
      const scalingNote=Number.isFinite(allowance)?` Normalized profile × ${allowance.toFixed(1)}°/s joint allowance (${speed}%); acceleration and flange limits can reduce the peak.`:'';
      const processingMs=data.result.planning_timing_ms?.server_total;
      const timing=data.result.planning_timing_ms;
      const stages=[['generation',timing?.v2_prepare?.trajectory_generation],
        ['route validation',timing?.v2_prepare?.hardware_route_validation],
        ['retiming',timing?.v2_prepare?.velocity_retiming],
        ['encoder preparation',timing?.encoder_profile_prepare]]
        .filter(([,ms])=>Number.isFinite(ms)&&ms>=0)
        .map(([label,ms])=>`${label} ${(ms/1000).toFixed(2)} s`);
      const processingNote=Number.isFinite(processingMs)
        ?` Plan processing ${(processingMs/1000).toFixed(2)} s.${stages.length?` Breakdown: ${stages.join(' · ')}. Total also includes live reads and plan checks.`:''}`:'';
      const trackingNote=data.plan.execution_mode==='path_waypoints' ? ` Joint steps ≤${data.plan.max_step_deg}°. Interior tracking allows ${data.plan.path_arrival_deg}°; corners and the endpoint stop within ${data.plan.arrival_deg}°.` : '';
      const commandNote=data.plan.execution_mode==='encoder_profile'
        ?' Checked velocity profile: instantaneous signed derivatives generate individual speed words, sampled against the original trajectory clock. Encoder targets bound travel on the route. No added 2° stages or arrival waits between samples. Tracking lag sends Stop; speed units remain nominal.'
        :data.plan.encoder_drag?.staged
        ?` Staged encoder path: ${data.plan.encoder_drag.stage_count} stopped stages of at most 2°, with no total-excursion cap. Each stage settles for one second and verifies live encoders. Reference speed is capped at 1°/s; feedback delays stretch execution time.`
        :data.plan.execution_mode==='encoder_stream'
        ?' Continuous encoder samples replace targets while moving, at most 5 sends/s. Tracking lead stays within 1°; corners and the endpoint stop within 0.75° with one second of settled observation. Feedback delays stretch execution time.'
        :data.plan.joint_command==='encoder_drag'?' Encoder targets and individual joint speeds will be sent together. Speed conversion is nominal; this is a bounded trial.':'';
      feedback(data.plan.converged ? `Plan ready: ${data.plan.waypoints} waypoints, predicted error ${data.plan.final_error_mm.toFixed(2)} mm${Number.isFinite(data.plan.final_orientation_error_deg) ? ` / ${data.plan.final_orientation_error_deg.toFixed(2)}°` : ''}.${trackingNote}${scalingNote}${shapeNote}${commandNote}${processingNote} Review window: ${Math.floor(data.plan.expires_in)} s. Preview the entire path; enable movement and verify the frame to execute.` : `IK did not converge (${data.plan.final_error_mm.toFixed(2)} mm${Number.isFinite(data.plan.final_orientation_error_deg) ? ` / ${data.plan.final_orientation_error_deg.toFixed(2)}°` : ''}). Execution is disabled.`,!data.plan.converged);
    } catch(error){plan=null;feedback(error.message,true);}
    finally{busy=false;update();}
  };
  $('execute-hardware-ik').onclick=async()=>{
    if(!plan)return;busy=true;update();
    try {
      const speed=Number($('ik-sdk-speed').value);
      const limit=plan.max_sdk_speed||80;
      if(!Number.isInteger(speed)||speed<1||speed>limit)throw new Error(`IK SDK speed must be an integer from 1 to ${limit}`);
      const id=plan.id;beforeExecute();
      const next=await request('ik/execute',{plan_id:id,speed,armed:$('arm-movement').checked,frame_verified:$('verify-ik-frame').checked,...(plan.hardware_diffusion?{scene_verified:$('verify-diffusion-scene').checked}:{})});
      ownedExecution=id;plan=null;$('small-target-preset').value='';apply(next);
      window.dispatchEvent(new CustomEvent('robot-state-refresh'));
      feedback(next.ik?.execution_mode==='encoder_profile'?'Tracking the planned joint velocity profile. Stop robot or Escape cancels execution.':'Executing planned joint waypoints. Stop robot or Escape cancels the remaining path.');
    }catch(error){feedback(error.message,true);}
    finally{busy=false;update();}
  };
  setInterval(async()=>{
    update();
    if(!ownedExecution||heartbeatBusy)return;
    heartbeatBusy=true;
    try{await request('ik/heartbeat',{execution_id:ownedExecution});}
    catch(error){
      try {
        const next=await (await fetch('/api/state')).json();
        if(['completed','failed','stopped','disconnected'].includes(next.ik?.state)){apply(next);return;}
      } catch { /* Stop the heartbeat so the server lease expires. */ }
      ownedExecution=null;feedback(`Execution heartbeat failed: ${error.message}. Check live state before planning again.`,true);
    }
    finally{heartbeatBusy=false;}
  },1000);
  fetch('/api/state').then(r=>r.json()).then(apply).catch(()=>update());
}
