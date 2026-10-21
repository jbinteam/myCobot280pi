import * as THREE from 'three';
import { PoseServo, pointingError } from './pose-servo.js';
import { directionInputIds } from './tool-direction.js';
import { validatedAimPath } from './mouse-aim-path.js';
import { routeRepairStatus } from './route-repair-status.js';
import { HardwareMouseAim } from './mouse-aim-hardware.js';
import {readShapeEditor} from './reference-shape.js';
import { readDiffusionScene } from './diffusion-controls.js';
import { createTrajectorySampler } from './trajectory-playback.js';
import { trajectoryView } from './trajectory-view.js';
import { renderJointVelocityPlot } from './ik-plot.js';
import {coneDiskRegion,regionPositionMetrics,createTrackingRegionView} from './tracking-region.js';
import {pointerOffset,visibilityHoldMessage} from './tracking-visibility.js';

const GAP_M = .05;
const $ = id => document.getElementById(id);

export function initMouseAim({ robot, jointNames, getPose, setPose, scene, camera, canvas, controls }) {
  const panel = $('mouse-aim-controls'), toggle = $('toggle-mouse-aim');
  const outputView=trajectoryView(scene,'mouse-trajectory-output');
  const resizeViewport = () => {
    const top = panel.offsetTop + panel.offsetHeight + 12;
    canvas.parentElement.style.top = panel.hidden ? '' : `${top}px`;
    // Preserve room for the robot below the expanded planning readouts on
    // narrow screens instead of covering most of the scene with the panel.
    panel.closest('.viewport-panel').style.minHeight = panel.hidden ? '' : `${Math.max(740, top + 560)}px`;
  };
  new ResizeObserver(resizeViewport).observe(panel);
  let state = {}, fk = null, limits = null, enabled = false, inside = false, dragging = false;
  let lastTime = null, lastReadout = 0, generation = 0, inFlight = false, controller = null;
  let playback = null, pendingPlayback = null, lastRequested = null, nextPlanAt = 0, phase = 'idle', failure = '';
  let planningLead = .6;
  let lastTargetMotion = null;
  let capturedTarget = null;
  let starting = false;
  let demoPointerMode = false, demoStartTarget = null;
  const demoPointerOffset = new THREE.Vector2();
  let stoppedMessage = false;
  let simulationInfo = null, requestSerial = 0, chunkFailures = 0, cursorGoal = null, goalHint = null, acceptedRegion = null;
  const hardware = new HardwareMouseAim({ getState: () => state,
    onState(next) {
      state = { ...state, ...next };
      if (Array.isArray(next.angles)) setPose(next.angles, true);
      window.dispatchEvent(new CustomEvent('robot-state', { detail: state }));
    },
    onFault: message => stop(`Robot mouse aim stopped — ${message}`),
    onComplete(next) {
      phase = 'complete'; nextPlanAt = performance.now() + 200;
      if(next.replan_required){lastRequested=null;nextPlanAt=performance.now();}
    },
  });
  const raycaster = new THREE.Raycaster(), pointer = new THREE.Vector2();
  const plane = new THREE.Plane(), anchor = new THREE.Vector3(), cursorPoint = new THREE.Vector3();
  const lastDirection = new THREE.Vector3(0, 0, 1);
  const overlay = new THREE.Group();
  overlay.name = 'mouse-aim-overlay'; overlay.visible = false;
  scene.add(overlay);
  const material = color => new THREE.MeshBasicMaterial({ color, depthTest: false, depthWrite: false });
  const cursor = new THREE.Group(); cursor.name = 'mouse-aim-cursor';
  cursor.add(new THREE.Mesh(new THREE.RingGeometry(.006, .008, 32), material('#167fbd')));
  cursor.add(new THREE.Mesh(new THREE.SphereGeometry(.0025, 12, 8), material('#ffffff')));
  const goal = new THREE.Mesh(new THREE.SphereGeometry(.003, 16, 12), material('#7860be'));
  goal.name = 'mouse-aim-goal';
  const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]), new THREE.LineDashedMaterial({ color: '#167fbd', dashSize: .004, gapSize: .003, depthTest: false, depthWrite: false }));
  line.name = 'mouse-aim-gap'; overlay.add(cursor, goal, line);
  const regionView=createTrackingRegionView();overlay.add(regionView.group);
  overlay.traverse(object => { object.renderOrder = 30; object.userData.isFrameVisualization = true; });

  const hardwareMode = () => state.hardware_enabled === true;
  const learnedRRT = () => (hardwareMode() ? $('hardware-planner') : $('mouse-aim-planner')).value==='learned_rrt';
  const hardwareV2 = () => hardwareMode() && ['diffusion_v2','learned_rrt'].includes($('hardware-planner').value);
  const hybrid = () => hardwareMode() ? hardwareV2() : ['local_diffusion','learned_rrt'].includes($('mouse-aim-planner').value);
  const continuous = () => !hardwareMode() && $('mouse-aim-continuous').checked;
  const regionSettings=()=>({mode:'cone_disk',distance_mm:Number($('tracking-distance').value),half_angle_deg:Number($('tracking-half-angle').value)});
  const gapM=()=>hybrid()?regionSettings().distance_mm/1000:GAP_M;
  const notifyMode = active => window.dispatchEvent(new CustomEvent('mouse-aim-change', {
    detail: {active,hardware:hardwareMode(),engine:learnedRRT()?'learned_rrt':hybrid()?'local_diffusion':'classical'} }));
  const showProfile = points => window.dispatchEvent(new CustomEvent('mouse-aim-profile',{detail:{points,feedbackPaced:hardwareV2()}}));
  function engineReadout() {
    const model = simulationInfo?.seed5_checkpoint;
    $('tracking-region-settings').hidden=!hybrid();
    $('mouse-aim-strategy').textContent=hybrid()?'Aim tool Z at the cursor · reach inside the green disk':'Point in the scene · keep a 50 mm gap';
    const obstacleLabel=hardwareV2()||$('diffusion-obstacle-mode').value==='none'?' · no block':' / block';
    $('mouse-aim-planner').querySelector('[value="learned_rrt"]').textContent=`Learned IK + classical RRT* · self / table${obstacleLabel}`;
    $('mouse-aim-planner').querySelector('[value="local_diffusion"]').textContent=`Local IK → diffusion · self / table${obstacleLabel}`;
    $('mouse-aim-planner-control').hidden = hardwareMode();
    $('mouse-aim-continuous-control').hidden=hardwareMode();
    $('mouse-aim-continuous').disabled=hardwareMode();
    $('mouse-aim-diffusion-note').hidden = !hybrid() || hardwareMode() || learnedRRT();
    $('mouse-aim-engine').textContent = learnedRRT()
      ? `Learned-assisted numerical IK → checked direct / RRT* · self / table${obstacleLabel} · ${hardwareMode()?'feedback-paced gaze tracking':'checked tracking curves'}`
      : hybrid()
      ? `Local IK first → checked diffusion${hardwareV2()?' v2':''} fallback · self / table${obstacleLabel} · ${hardwareV2()?'feedback-paced gaze tracking':'smoothed chunks'}`
      : model?.status === 'ready'
        ? `Local IK first · self-checks · ${model.collision_training?.enabled ? 'collision-trained Seed NN' : 'Seed NN'} / PRM fallback`
        : `Local IK first · self-checks · numerical / PRM fallback (checkpoint ${model?.status || 'unavailable'})`;
    if(!enabled)$('mouse-aim-repair').textContent = learnedRRT() ? 'Checked direct / RRT*: inactive' : hybrid() ? 'Diffusion fallback: inactive' : $('guided-route-repair').checked ? 'Guided repair: enabled for blocked routes' : 'Guided repair: off';
  }
  const allowed = () => !!fk && (hardwareMode()
    ? state.connected && Array.isArray(state.angles) && state.angles.length === 6 && state.angles.every(Number.isFinite)
      && Number.isFinite(state.updated) && Date.now()/1000-state.updated < 3
      && $('arm-movement').checked && $('verify-ik-frame').checked
      && (!hardwareV2()||$('verify-diffusion-scene').checked)
      // Polling may expose the new server session before Start's response.
      // Keep Stop available while waiting; update() prevents planning/motion.
      && (!state.ik?.mouse_aim_session_id || state.ik.mouse_aim_session_id === hardware.sessionId || starting)
      && (!(state.motion || state.moving) || (state.motion === 'ik' && state.ik?.mouse_aim_session_id === hardware.sessionId && !!hardware.sessionId))
    : state.hardware_enabled === false && !state.connected && !state.motion && !state.moving);
  const currentPose = () => fk.pose((hardwareMode() && Array.isArray(state.angles) ? state.angles : getPose()).map(THREE.MathUtils.degToRad));
  const tracking = () => enabled && inside && !dragging && allowed() && !document.hidden;
  function status(text, limited = false) {
    $('mouse-aim-status').textContent = text;
    panel.classList.toggle('aim-limited', limited);
  }
  function invalidate() {
    generation++; controller?.abort(); playback = null; pendingPlayback=null; lastRequested = null;
    outputView.clear();regionView.clear();
    showProfile([]);
    if(!hardwareMode())$('trajectory-speed').textContent='0.000 rad/s';
    lastTime = null; phase = 'idle'; failure = ''; nextPlanAt = 0;
    chunkFailures = 0;cursorGoal = null;goalHint = null;acceptedRegion=null;lastTargetMotion=null;
    $('mouse-aim-outcome').textContent = 'IK: inactive · route: no active playback';
    $('mouse-aim-repair').textContent = learnedRRT() ? 'Checked direct / RRT*: inactive' : hybrid() ? 'Diffusion fallback: inactive' : $('guided-route-repair').checked ? 'Guided repair: enabled for blocked routes' : 'Guided repair: off';
  }
  function stop(message = 'Move over the scene to plan a self-checked mouse-aim path.') {
    if (!enabled) return;
    invalidate(); enabled = false; inside = false; starting = false;
    stoppedMessage = true;
    void hardware.stop();
    overlay.visible = false; panel.classList.remove('aim-active');
    toggle.textContent = hardwareMode() ? 'Start robot mouse aim' : 'Start mouse aim'; toggle.setAttribute('aria-pressed', 'false');
    canvas.classList.remove('mouse-aiming'); status(message);
    notifyMode(false);
  }
  function pause(message) {
    if (hardwareMode()) { stop('Robot mouse aim stopped — pointer left the scene or tracking paused.'); return; }
    invalidate(); inside = false;
    if (enabled) status(message);
  }
  function availability() {
    $('visible-tracking-demo-controls').hidden = hardwareMode();
    $('load-visible-tracking-demo').disabled = !allowed() || hardwareMode() || !simulationInfo?.tracking_demo || simulationInfo?.classical?.status!=='ready';
    if (!allowed()) stop(hardwareMode() ? 'Robot mouse aim stopped — connection, movement permission or live state changed.' : 'Mouse aim is available only with hardware disabled.');
    panel.hidden = !fk || (!hardwareMode() && !allowed());
    toggle.disabled = !enabled && !allowed(); resizeViewport();
    panel.setAttribute('aria-label', hardwareMode() ? 'Mouse aim robot control' : 'Mouse aim simulation');
    $('mouse-aim-mode-badge').textContent = hardwareMode() ? 'LIVE ROBOT CONTROL' : 'SIMULATION ONLY';
    const feedback = $('mouse-aim-hardware-feedback'); feedback.hidden = !hardwareMode();
    const mode = state.ik?.refresh_mode;
    const modeText = mode === 1 ? 'refresh mode 1 verified' : mode === 0 ? 'mode 0 read · switches to 1 after execution ACK' : 'refresh mode unverified';
    const lag=Number.isFinite(state.ik?.reference_lag_s)?` · reference delay ${state.ik.reference_lag_s.toFixed(2)} s`:'';
    const checks=state.ik?.self_collision_checked===true?' · self-collision checked':'';
    feedback.textContent = `${hardwareV2()?(learnedRRT()?'RRT* feedback-paced gaze tracking':'v2 feedback-paced gaze tracking'):'Velocity feedback'} · ${modeText}${Number.isFinite(state.ik?.velocity_limit_deg_s) ? ` · reference ≤${state.ik.velocity_limit_deg_s.toFixed(1)}°/s · acceleration ≤${state.ik.acceleration_limit_deg_s2.toFixed(1)}°/s²` : ''}${lag}${checks}`;
    if (!enabled) toggle.textContent = hardwareMode() ? 'Start robot mouse aim' : 'Start mouse aim';
    if (hardwareMode() && !enabled && !allowed()) status(!state.connected ? 'Connect the robot to use mouse aim.' : 'Enable movement and verify the flange frame; start only while the arm is stopped.');
    else if(!enabled && allowed() && !stoppedMessage)status(hardwareMode()?'Setup complete. Start tracking, then move slowly inside the scene.':'Start tracking, then point in the scene. This moves the preview only.');
    $('inspect-mouse-aim').disabled = !capturedTarget || (hardwareMode() && !!(state.motion || state.moving));
    engineReadout();
  }
  function configure(info) {
    limits = info?.limits_deg;
    if (!limits || limits.min?.length !== 6 || limits.max?.length !== 6) return;
    fk = new PoseServo(robot, jointNames, limits); // FK only; no browser IK motion.
    simulationInfo = info;
    $('mouse-aim-planner').querySelector('[value="local_diffusion"]').disabled = info.diffusion?.status!=='ready';
    $('mouse-aim-planner').querySelector('[value="learned_rrt"]').disabled = info.classical?.status!=='ready';
    $('hardware-planner').querySelector('[value="learned_rrt"]').disabled = info.classical?.status!=='ready';
    $('mouse-aim-planner').value=info.classical?.status==='ready'?'learned_rrt':info.diffusion?.status==='ready'?'local_diffusion':'classical';
    window.dispatchEvent(new CustomEvent('tracking-model-ready',{detail:info}));
    availability();
  }
  async function start() {
    if (!allowed()) return;
    invalidate();
    stoppedMessage = false;
    capturedTarget = null; $('inspect-mouse-aim').disabled = true;
    const mode = hardwareMode(), epoch = generation;
    let pose = currentPose();
    anchor.copy(pose.position).addScaledVector(pose.axis, gapM());
    if (!mode && demoStartTarget) {anchor.copy(demoStartTarget);demoStartTarget=null;}
    demoPointerOffset.set(0,0);
    cursorPoint.copy(anchor); lastDirection.copy(pose.axis);
    enabled = true; inside = false; overlay.visible = true; panel.classList.add('aim-active');
    toggle.textContent = mode ? 'Stop robot mouse aim' : 'Stop mouse aim'; toggle.setAttribute('aria-pressed', 'true');
    canvas.classList.add('mouse-aiming');
    $('mouse-aim-outcome').textContent = 'IK: waiting · route: waiting';
    $('mouse-aim-goal-errors').textContent = '—';
    status(demoPointerMode&&!mode?'Move gently in the scene to steer the target from camera centre. Esc stops.':'Move over the scene. Plans are computed before preview motion; Esc stops.');
    notifyMode(true);
    if (mode) {
      starting = true; status('Reading live joints and starting robot mouse aim…');
      $('follow-live').checked = true; $('follow-live').dispatchEvent(new Event('change'));
      try {
        const next = await hardware.start(hardwareV2()?{tracking_engine:learnedRRT()?'learned_rrt':'diffusion_v2',scene_verified:$('verify-diffusion-scene').checked,
          speed:Number($('ik-sdk-speed').value),speed_rad_s:Math.min(1,Number($('ik-speed').value)),velocity_shape:readShapeEditor(),tracking_region:regionSettings()}:{});
        if (!next || epoch !== generation || !enabled) return;
        pose = currentPose(); anchor.copy(pose.position).addScaledVector(pose.axis, gapM());
        cursorPoint.copy(anchor); lastDirection.copy(pose.axis);
        status('Move over the scene to aim at the cursor with live feedback. Leaving the scene or Esc stops.');
      } catch (error) { if (epoch === generation) stop(`Robot mouse aim could not start: ${error.message}`); }
      finally { if (epoch === generation) starting = false; }
    }
    draw(pose);
  }
  toggle.addEventListener('click', () => enabled ? stop() : start());
  $('load-visible-tracking-demo').addEventListener('click', () => {
    if (hardwareMode() || !allowed() || !simulationInfo?.tracking_demo) return;
    stop();
    const demo = simulationInfo.tracking_demo;
    setPose(demo.start_deg);
    $('mouse-aim-planner').value='learned_rrt';
    $('tracking-distance').value=String(demo.distance_mm);
    $('tracking-half-angle').value=String(demo.half_angle_deg);
    $('diffusion-obstacle-mode').value='none';
    $('diffusion-obstacle-mode').dispatchEvent(new Event('change'));
    $('mouse-aim-continuous').checked=true;
    demoPointerMode=true;
    demoStartTarget=new THREE.Vector3().fromArray(demo.cursor_mm).multiplyScalar(.001);
    engineReadout();
    status('Visible tracking demo loaded. Click Start mouse aim, then move gently inside the scene.');
  });
  window.addEventListener('control-mode-changing', () => {demoPointerMode=false;demoStartTarget=null;stop('Mouse tracking stopped — control mode changed.');});
  $('mouse-aim-planner').addEventListener('change', () => { demoPointerMode=false;demoStartTarget=null;stop('Mouse aim stopped — planner changed.'); engineReadout(); });
  $('mouse-aim-continuous').addEventListener('change',()=>stop('Mouse aim stopped — motion mode changed.'));
  $('guided-route-repair').addEventListener('change', () => {
    stop('Mouse aim stopped — route repair setting changed.');
    $('mouse-aim-repair').textContent = $('guided-route-repair').checked ? 'Guided repair: enabled for blocked routes' : 'Guided repair: off';
  });
  $('inspect-mouse-aim').addEventListener('click', () => {
    if (!allowed() || !capturedTarget) return;
    const snapshot = capturedTarget;
    stop('Mouse aim stopped — captured 5D target loaded into experiments.');
    window.dispatchEvent(new CustomEvent('mouse-aim-inspect', { detail: snapshot }));
    $('ik-task').scrollIntoView({ behavior: 'smooth', block: 'center' });
  });
  window.addEventListener('robot-state', event => {
    const changedMode = state.hardware_enabled !== event.detail.hardware_enabled;
    state = { ...state, ...event.detail };
    if (changedMode && enabled) stop('Mouse aim stopped — control mode changed.');
    hardware.observe(state); availability();
  });
  for (const id of ['arm-movement', 'verify-ik-frame', 'verify-diffusion-scene']) $(id).addEventListener('change', availability);
  $('hardware-planner').addEventListener('change',()=>{stop('Mouse aim stopped — hardware planner changed.');availability();});
  fetch('/api/state').then(response => {
    if (!response.ok) throw new Error('State unavailable');
    return response.json();
  }).then(next => { state = next; availability(); }).catch(() => { state = {}; availability(); });

  function projectPointer() {
    camera.updateMatrixWorld(true);
    plane.setFromNormalAndCoplanarPoint(camera.getWorldDirection(new THREE.Vector3()), anchor);
    raycaster.setFromCamera(demoPointerMode&&!hardwareMode()?pointer.clone().add(demoPointerOffset):pointer, camera);
    return raycaster.ray.intersectPlane(plane, cursorPoint);
  }
  canvas.addEventListener('pointermove', event => {
    if (!enabled || event.pointerType === 'touch' || event.buttons || dragging) return;
    const rect = canvas.getBoundingClientRect();
    pointer.set(2 * (event.clientX - rect.left) / rect.width - 1, 1 - 2 * (event.clientY - rect.top) / rect.height);
    if (!inside && demoPointerMode && !hardwareMode()) {
      camera.updateMatrixWorld(true);
      demoPointerOffset.copy(pointerOffset(pointer,cursorPoint,camera));
    }
    inside = true; projectPointer();
  });
  canvas.addEventListener('pointerleave', () => pause(capturedTarget
    ? 'Paused — compare the last target or return to the scene.'
    : 'Paused — move back over the scene to continue.'));
  controls.addEventListener('start', () => { dragging = true; pause('Paused while orbiting the camera.'); });
  controls.addEventListener('end', () => { dragging = false; });
  window.addEventListener('blur', () => pause('Paused — return to the scene to continue.'));
  document.addEventListener('visibilitychange', () => { if (document.hidden) pause('Paused while the page is hidden.'); });
  document.addEventListener('keydown', event => { if (event.key === 'Escape') stop('Mouse aim stopped.'); });
  window.addEventListener('preview-edited', () => {demoStartTarget=null;stop('Mouse aim stopped — preview pose edited.');});
  for (const id of ['solve-ik', 'compare-ik', 'plan-prm', 'plan-classical', 'plan-diffusion', 'play-ik', 'zero-pose', 'rest-pose', 'send-pose', 'jog-negative', 'jog-positive', 'stop-robot', 'disconnect-robot', 'plan-hardware-ik', 'execute-hardware-ik', 'use-live-pose', 'use-live-xyz']) {
    $(id).addEventListener('click', () => stop(), { capture: true });
  }
  for (const id of ['ik-scrub', 'ik-task', 'ik-method', 'ik-case', 'target-x', 'target-y', 'target-z', 'target-rx', 'target-ry', 'eef-offset-x', 'eef-offset-y', 'eef-offset-z', ...directionInputIds, 'pose5-proposals', 'ik-speed', 'ik-sdk-speed', 'ik-steps', 'ik-tolerance', 'ik-orientation-tolerance', 'ik-damping', 'diffusion-case', 'diffusion-obstacle-mode', 'diffusion-x', 'diffusion-y', 'diffusion-width', 'diffusion-depth', 'diffusion-height', 'diffusion-seed', 'diffusion-samples', 'diffusion-denoise', 'diffusion-budget', 'velocity-shape', 'velocity-rise', 'velocity-fall', 'tracking-distance', 'tracking-half-angle']) {
    $(id).addEventListener('input', () => {demoStartTarget=null;stop();}); $(id).addEventListener('change', () => {demoStartTarget=null;stop();});
  }
  $('diffusion-obstacle-mode').addEventListener('change',engineReadout);

  function desired(pose) {
    const direction = cursorPoint.clone().sub(pose.position);
    if(hybrid()) {
      if(!cursorGoal || cursorGoal.cursor.distanceTo(cursorPoint)>.0005) {
        if(direction.lengthSq()>1e-10)lastDirection.copy(direction.normalize());
        cursorGoal={cursor:cursorPoint.clone(),axis:lastDirection.toArray(),decisions:0};chunkFailures=0;
      }
      lastDirection.fromArray(cursorGoal.axis);
    } else if(direction.lengthSq()>1e-10)lastDirection.copy(direction.normalize());
    return {axis:lastDirection.clone(),position:cursorPoint.clone().addScaledVector(lastDirection,-gapM())};
  }
  function draw(pose) {
    const target = desired(pose);
    cursor.position.copy(cursorPoint); cursor.quaternion.copy(camera.quaternion);
    goal.position.copy(target.position);
    let regionMetrics=null;
    if(hybrid()) {
      const region=acceptedRegion&&cursorGoal.cursor.distanceTo(new THREE.Vector3().fromArray(acceptedRegion.cursor_mm).multiplyScalar(.001))<=.0005
        ?acceptedRegion:coneDiskRegion(cursorGoal.cursor,cursorGoal.axis,regionSettings());
      regionView.show(region);goal.position.fromArray(region.center_mm).multiplyScalar(.001);
      regionMetrics=regionPositionMetrics(pose.position,region);
      $('tracking-region-size').textContent=`Radius ${region.radius_mm.toFixed(1)} mm · axis held for each cursor target`;
    } else regionView.clear();
    const vertices = line.geometry.attributes.position;
    vertices.setXYZ(0, ...pose.position.toArray()); vertices.setXYZ(1, ...cursorPoint.toArray());
    vertices.needsUpdate = true; line.geometry.computeBoundingSphere(); line.computeLineDistances();
    const aim=cursorPoint.clone().sub(pose.position);
    if(aim.lengthSq()>1e-10)aim.normalize();else aim.copy(target.axis);
    $('mouse-aim-axis').textContent = `[${aim.toArray().map(v => v.toFixed(3)).join(', ')}]`;
    $('mouse-aim-region-errors').textContent=regionMetrics?`Plane ${regionMetrics.axialErrorMm.toFixed(1)} mm · radial ${regionMetrics.radialMm.toFixed(1)} mm`: '—';
    return {distanceMm:pose.position.distanceTo(cursorPoint)*1000,
      angleDeg:THREE.MathUtils.radToDeg(pointingError(pose.axis,aim).angle),...regionMetrics};
  }
  function showPlayback(next) {
    playback=next;phase='following';
    if(next.region)acceptedRegion=next.region;
    if(next.sampler){
      outputView.show(next.frames,next.targetMm);
      renderJointVelocityPlot($('velocity-chart'),next.sampler.velocityPoints);
      showProfile(next.sampler.velocityPoints);
    }
  }
  const sameMotion=(a,b)=>['q','velocity_rad_s','acceleration_rad_s2'].every(key=>
    Array.isArray(a[key])&&Array.isArray(b[key])&&a[key].every((v,i)=>Math.abs(v-b[key][i])<1e-6));
  async function requestPlan(handoff=null) {
    const epoch = generation, start = handoff?handoff.motion.q.slice():getPose().slice(), snapshot = cursorPoint.clone();
    const useRRT = learnedRRT(), useHybrid = hybrid(), useContinuous=continuous(), speed = Math.min(Number($('ik-speed').value),useHybrid||useContinuous?1:Infinity), abort = new AbortController();
    inFlight = true; controller = abort; phase = 'planning';
    if(!useContinuous){playback = null;outputView.clear();lastTime=null;}
    if((useHybrid||useContinuous)&&!playback){$('velocity-chart').replaceChildren();$('trajectory-speed').textContent='0.000 rad/s';}
    lastRequested = snapshot;
    $('mouse-aim-outcome').textContent = 'IK: computing · route: waiting';
    // Retain the last completed snapshot while the next request is pending.
    // Moving toward Compare can start a request that pointerleave then aborts.
    // Only a completed, current-generation response replaces the saved target.
    status(playback&&useContinuous?'Following checked motion while planning the next target.'
      : hardwareMode() && hardware.running ? 'Checking the latest cursor from live joint feedback.' : useRRT?'Checking learned IK + classical RRT*… preview held.':'Checking local IK, with planner fallback… preview held.');
    try {
      const request = { start_deg: start, cursor_mm: snapshot.toArray().map(v => v * 1000),
        pose5_method: $('ik-task').value === 'pose5' ? $('ik-method').value : 'dls',
        steps: Number($('ik-steps').value), speed_rad_s: speed, tolerance_mm: Number($('ik-tolerance').value),
        orientation_tolerance_deg: Number($('ik-orientation-tolerance').value), damping: Number($('ik-damping').value) };
      if(!hardwareMode()||hardwareV2())request.velocity_shape=readShapeEditor();
      request.guided_route_repair = $('guided-route-repair').checked;
      request.route_repair_samples = 32;
      if(useContinuous){
        const motion=handoff?.motion||{velocity_rad_s:Array(6).fill(0),acceleration_rad_s2:Array(6).fill(0)};
        Object.assign(request,{continuous_motion:true,start_velocity_rad_s:motion.velocity_rad_s,
          start_acceleration_rad_s2:motion.acceleration_rad_s2});
      }
      let trackingScene = null;
      if(useHybrid||useContinuous){
        if(!cursorGoal || cursorGoal.cursor.distanceTo(snapshot)>.0005) {
          const direction=snapshot.clone().sub(currentPose().position).normalize();
          cursorGoal={cursor:snapshot.clone(),axis:direction.toArray(),decisions:0};chunkFailures=0;
        }
        request.target_axis=cursorGoal.axis.slice();
        if(useHybrid)request.tracking_region=regionSettings();
      }
      if (useHybrid) {
        const deployment = useRRT?simulationInfo.classical:simulationInfo.diffusion;
        const box = hardwareV2()?null:readDiffusionScene(deployment.environment.table_top_m);
        if(++cursorGoal.decisions>Number($('diffusion-budget').value))throw new Error('Tracking decision budget reached; move the cursor to try a new target');
        trackingScene = {box,checkpoint:deployment.checkpoint_sha256,
          planner:useRRT?'learned_rrt':'local_diffusion',
          conditioningSchema:useRRT?null:deployment.conditioning?.schema,velocityShape:request.velocity_shape,
          region:coneDiskRegion(snapshot,request.target_axis,request.tracking_region),serverOwnsRegionAxis:hardwareV2(),toleranceMm:Number($('ik-tolerance').value)};
        Object.assign(request,{tracking_engine:useRRT?'learned_rrt':'local_diffusion',obstacle_mm:box,target_axis:cursorGoal.axis.slice(),
          seed:(Number($('diffusion-seed').value)+requestSerial++) % 2147483648,
          samples:Number($('diffusion-samples').value),inference_steps:Number($('diffusion-denoise').value)});
        const target=snapshot.clone().addScaledVector(new THREE.Vector3().fromArray(request.target_axis),-gapM());
        const now=performance.now(), interval=lastTargetMotion?(now-lastTargetMotion.time)/1000:0;
        const targetVelocity=lastTargetMotion&&interval>=.02&&interval<=2
          ?target.clone().sub(lastTargetMotion.position).multiplyScalar(1/interval):new THREE.Vector3();
        if(targetVelocity.length()>.5)targetVelocity.setLength(.5);
        if(!hardwareMode())request.target_velocity_m_s=targetVelocity.toArray();
        lastTargetMotion={position:target,time:now};
        if(!useRRT&&!hardwareMode()&&goalHint && ![3,4].includes(simulationInfo.diffusion.conditioning?.schema))request.goal_hint_deg=goalHint.slice();
      }
      let envelope, data, response;
      if (hardwareMode()) {
        const { start_deg, cursor_mm, ...settings } = request;
        envelope = await hardware.plan(cursor_mm, settings, abort.signal);
        data = envelope.result; request.start_deg = envelope.start_deg;
      } else {
        response = await fetch('/api/sim/mouse-aim', { method: 'POST', signal: abort.signal,
          headers: { 'Content-Type': 'application/json', 'X-Robot-Token': state.token }, body: JSON.stringify(request) });
        data = await response.json();
      }
      if (epoch !== generation || !tracking()) return;
      if (useHybrid && !hardwareMode() && !useContinuous && cursorPoint.distanceTo(snapshot)>.0005) {
        lastRequested=null;nextPlanAt=performance.now()+50;phase='busy';
        status('Cursor changed — discarded stale chunk; checking the latest target.');return;
      }
      if (response?.status === 409) {
        lastRequested = null; nextPlanAt = performance.now() + 750; phase = 'busy';
        status('Planner finishing a previous request… latest cursor will retry.'); return;
      }
      if (response && !response.ok) throw new Error(data.error || `Planner HTTP ${response.status}`);
      if ([data.target_mm, data.target_axis].every(value => Array.isArray(value) && value.length === 3 && value.every(Number.isFinite))) {
        capturedTarget = { ...request, target_mm: data.target_mm.slice(), target_axis: data.target_axis.slice() };
        $('inspect-mouse-aim').disabled = false;
      }
      const sourceName = source => source.startsWith('recovery_nn_') ? 'Seed NN + expanded recovery'
        : source.startsWith('nn_') ? (source.endsWith('_refined') ? 'Seed NN + refinement' : 'Seed NN')
        : source.startsWith('recovery_') ? 'expanded numerical recovery'
        : source === 'current_state' ? 'already at target'
        : source === 'current_local_ik' ? 'local IK from current joints' : 'numerical fallback';
      const sources = [...new Set((data.ik?.goals || []).map(goal => sourceName(goal.source)))];
      const checkpoint = data.ik?.nn_checkpoint;
      $('mouse-aim-repair').textContent = useRRT
        ? `Classical route: ${(data.rrt_diagnostics?.status||'not found').replaceAll('_',' ')} · learned-assisted IK + refinement`
        : useHybrid
        ? `Diffusion fallback: ${data.tracking_source==='diffusion_fallback'?'used':'not used'} · ${Object.values(data.rejections||{}).reduce((n,v)=>n+v,0)}/${data.proposed_chunks||0} proposals rejected`
        : routeRepairStatus(data);
      const recovery = data.ik?.recovery;
      const recoveryLabel = recovery ? ` · expanded search: ${recovery.phases.reduce((n, phase) => n + phase.used_seeds, 0)} seeds, ${recovery.added_goals} added goals` : '';
      const label = value => (value || 'not started').replaceAll('_', ' ');
      $('mouse-aim-outcome').textContent = `IK: ${label(data.ik_outcome || data.status)} · route: ${label(data.planning_outcome)} · ${[3,4].includes(data.motion_conditioning?.schema)?'Cartesian target + direction':useRRT?'learned-assisted numerical IK':sources.join(', ') || 'no accepted goals'}${recoveryLabel}${checkpoint && checkpoint.status !== 'ready' ? ` · checkpoint ${checkpoint.status}` : ''}`;
      $('mouse-aim-error-label').textContent = useHybrid?'Chunk end error':data.converged ? 'Planned goal error' : 'Best candidate error';
      $('mouse-aim-goal-errors').textContent = Number.isFinite(data.final_error_mm)
        ? `${data.final_error_mm.toFixed(2)} mm · ${Number.isFinite(data.final_orientation_error_deg) ? data.final_orientation_error_deg.toFixed(2) : '—'}° · compute ${Math.round(data.server_compute_ms ?? ((data.ik_ms || 0) + (data.planning_ms || 0)))} ms`
        : '—';
      if (!(useHybrid?data.playable:data.converged)) {
        if(useHybrid&&data.status==='no_safe_chunk'&&++chunkFailures<5){lastRequested=null;nextPlanAt=performance.now()+500;}
        phase = 'failed'; failure = visibilityHoldMessage(data) || `${data.status}: ${data.reason || data.stop_reason || 'no validated route'}`;
        status(playback&&useContinuous?`New target rejected — continuing checked motion. ${failure}`
          :hardwareMode() && hardware.running ? `New target rejected — continuing the last checked route. ${failure}` : `Held — ${failure}`, true); return;
      }
      if(useContinuous&&data.continuous_motion!==true)throw new Error('Planner did not return continuous motion');
      const frames = validatedAimPath(data, envelope ? envelope.start_deg : useContinuous?start:getPose(), limits, speed, snapshot.toArray().map(v => v * 1000), trackingScene, hardwareV2());
      if(useHybrid)goalHint=Array.isArray(data.goal_deg)?data.goal_deg.slice():null;
      const sampler=createTrajectorySampler(frames,data.smoothing);
      if (envelope) {
        if (cursorPoint.distanceTo(snapshot) > .01) {
          lastRequested = null; nextPlanAt = performance.now() + 100; phase = 'busy'; return;
        }
        await hardware.execute(envelope.plan, Number($('ik-sdk-speed').value));
        if (epoch !== generation || !tracking()) return;
        acceptedRegion=data.tracking_region||null;
        outputView.show(frames,data.target_mm);
        renderJointVelocityPlot($('velocity-chart'),sampler.velocityPoints);
        showProfile(sampler.velocityPoints);
        nextPlanAt = performance.now() + 150;
        phase = hardware.running ? 'following' : 'complete'; return;
      }
      const next={frames,sampler,targetMm:data.target_mm,region:data.tracking_region,elapsed:0,replan:useHybrid&&data.replan_required};
      if(useContinuous){
        const expected=handoff?.motion||{q:start,velocity_rad_s:Array(6).fill(0),acceleration_rad_s2:Array(6).fill(0)};
        if(!sameMotion(sampler.sample(0),expected))throw new Error('Planner changed the predicted join motion');
        planningLead=Math.max(.35,Math.min(1.5,(data.server_compute_ms||100)/1000*1.8+.2));
        if(handoff&&playback===handoff.owner&&playback.elapsed<=handoff.time+1e-9){
          pendingPlayback={owner:playback,time:handoff.time,next};phase='following';
        } else if(!playback&&sameMotion(sampler.sample(0),{q:getPose(),velocity_rad_s:Array(6).fill(0),acceleration_rad_s2:Array(6).fill(0)})) {
          showPlayback(next);
        } else {
          lastRequested=null;nextPlanAt=performance.now()+50;phase=playback?'following':'busy';
          status('Continuation arrived after its join — keeping the checked route and replanning.');return;
        }
      } else {showPlayback(next);lastTime=null;}
      chunkFailures=0;
      // Allow some validated movement before replanning a newer cursor. The
      // request queue is always one snapshot plus the latest pointer position.
      nextPlanAt = performance.now() + 200;
    } catch (error) {
      if (epoch !== generation || error.name === 'AbortError') return;
      if (hardwareMode()) {
        if (error.status === 409) { lastRequested = null; nextPlanAt = performance.now() + 750; phase = 'busy'; return; }
        stop(`Robot mouse aim stopped — ${error.message}`); return;
      }
      phase = 'failed'; failure = error.message; status(playback&&useContinuous?`Continuing checked motion — ${failure}`:`Held — ${failure}`, true);
    } finally {
      inFlight = false; if (controller === abort) controller = null;
    }
  }
  function update(time) {
    if (!enabled || starting) return;
    if (hardwareMode() && !allowed()) { stop('Robot mouse aim stopped — live state or permissions changed.'); return; }
    let pose = currentPose();
    const dt = lastTime === null ? 0 : Math.min(.05, Math.max(0, (time - lastTime) / 1000));
    lastTime = tracking() ? time : null;
    if (tracking()) {
      projectPointer();
      if (hardwareMode() && state.ik?.retarget_rejected && phase !== 'busy') {
        lastRequested = null; nextPlanAt = time + 150; phase = 'busy';
      }
      const changed = !lastRequested || cursorPoint.distanceTo(lastRequested) > (hardwareMode() ? .001 : .0005);
      const needsContinuation=(continuous()&&playback?.replan)||(hardwareV2()&&(state.ik?.replan_required||state.ik?.feedback_replan_required));
      if (!inFlight && !pendingPlayback && !hardware.pendingExecute && time >= nextPlanAt && (changed||needsContinuation)
          && (continuous()||!hybrid()||!playback)) {
        const handoff=continuous()&&playback?{owner:playback,time:Math.min(playback.sampler.endTime,playback.elapsed+planningLead)}:null;
        if(handoff)handoff.motion=playback.sampler.sample(handoff.time);
        requestPlan(handoff);
      }
      if (!hardwareMode() && (!inFlight||continuous()) && playback && dt > 0) {
        const proposed=playback.sampler.advance(playback.elapsed,dt);
        const splice=pendingPlayback?.owner===playback?pendingPlayback:null;
        playback.elapsed = splice?Math.min(proposed,splice.time):proposed;
        setPose(playback.sampler.sample(playback.elapsed).q);
        if(splice&&playback.elapsed>=splice.time-1e-9){
          if(!sameMotion(playback.sampler.sample(splice.time),splice.next.sampler.sample(0))){
            pendingPlayback=null;phase='failed';failure='Continuation join changed; following checked braking tail';
          } else {
            pendingPlayback=null;showPlayback(splice.next);
            playback.elapsed=playback.sampler.advance(0,Math.max(0,proposed-splice.time));
            setPose(playback.sampler.sample(playback.elapsed).q);
          }
        }
        pose = fk.pose(getPose().map(THREE.MathUtils.degToRad));
        if(playback.sampler){
          $('trajectory-speed').textContent=`${Math.hypot(...playback.sampler.sample(playback.elapsed).velocity_rad_s).toFixed(3)} rad/s`;
        }
        if (playback.elapsed >= playback.frames.at(-1).time) {
          const replan = playback.replan;playback = null; phase = 'complete';
          if(replan){lastRequested=null;nextPlanAt=time;}
        }
      }
    }
    const metrics = draw(pose);
    if (time - lastReadout > 100) {
      $('mouse-aim-distance').textContent = `${metrics.distanceMm.toFixed(1)} mm`;
      $('mouse-aim-direction').textContent = `${metrics.angleDeg.toFixed(2)}°`;
      $('mouse-aim-tolerances').textContent = `±${$('ik-tolerance').value} mm · ${$('ik-orientation-tolerance').value}°${hardwareMode() ? ' · live acceptance ±5 mm · 3°' : ''}`;
      const tolerance=Number($('ik-tolerance').value);
      const reached=(hybrid()?metrics.axialErrorMm<=tolerance&&metrics.radialMm<=regionView.group.userData.region.radius_mm+1e-6:Math.abs(metrics.distanceMm-50)<=tolerance) && metrics.angleDeg<=Number($('ik-orientation-tolerance').value);
      if (tracking()) {
        if (phase === 'failed') status(playback&&continuous()?`New target rejected — continuing checked motion. ${failure}`:hardwareMode() && hardware.running ? `New target rejected — continuing the last checked route. ${failure}` : `Held — ${failure}`, true);
        else if (phase === 'planning') status(playback&&continuous()?'Following checked motion while planning the next target.':hardwareMode() && hardware.running ? 'Checking the latest mouse target from live joint feedback.' : learnedRRT()?'Checking learned IK + classical RRT*… preview held.':'Checking local IK, with planner fallback… preview held.');
        else if (phase === 'following') status(hardwareMode() ? hardwareV2()?state.ik?.tracking_paused?'Gaze tracking active — reference waiting for live feedback; cursor retargeting remains available.':'Gaze tracking active — self-checked commands paced by live feedback.':'Velocity tracking on a checked route — displaying live telemetry.' : continuous()?'Following validated continuous motion — planning ahead without chunk stops.':'Following validated joint path — direction constrained at the goal.');
        else if (phase === 'complete') status(hardwareMode() ? state.ik?.replan_required?'Checked prefix completed — planning the next cursor continuation.':'Robot path completed — move cursor for the next live-start plan.' : reached ? hybrid()?'On target — inside the acceptance disk with tool Z aimed at the cursor.':'On target — validated 50 mm gap and tool direction.' : 'Validated endpoint reached; move cursor to request a new target.', !reached && !hardwareMode());
      }
      lastReadout = time;
    }
  }
  return { configure, update, stop,
    getPreviewMotion:()=>playback?.sampler?playback.sampler.sample(playback.elapsed)
      :{q:getPose(),velocity_rad_s:Array(6).fill(0),acceleration_rad_s2:Array(6).fill(0)} };
}
