import * as THREE from 'three';
import { renderErrorPlot, renderJointVelocityPlot } from './ik-plot.js';
import { readTargetFrame, offsetInputIds, resetLocalOffset } from './target-frame.js';
import { initWorkspace } from './workspace-controls.js';
import { initHardwareIK } from './hardware-ik-controls.js';
import { initToolDirection, readToolAxis, readReferenceTilts, syncAxisFromTilts, directionInputIds } from './tool-direction.js';
import { routeRepairStatus } from './route-repair-status.js';
import { initDiffusion } from './diffusion-controls.js';
import { createTrajectorySampler } from './trajectory-playback.js';
import { trajectoryView } from './trajectory-view.js';
const $ = id => document.getElementById(id);
const COLORS = ['#386baf','#30795c','#b75283','#8a652c','#795ac0','#478385','#b45d37','#193e7b'];
const PRM = { id:'nn_prm', name:'Self-checked PRM', description:'Four bounded Seed NN proposals are checked independently, then refined with numerical fallback when needed. 5D uses a separate checkpoint. Accepted goals connect directly or through the reusable joint-space roadmap. Direction is constrained at the goal only. Preview only.' };
const CLASSICAL = {id:'learned_rrt',name:'Learned IK + classical RRT*',description:'Learned virtual starts refined by numerical IK, followed by checked direct/RRT* paths and collision-checked quintic timing.'};
const DIFFUSION = {id:'diffusion',name:'Checked diffusion',description:'Trained conditional DDPM with 16 joint references, checked receding-horizon motion and independent self/table/block checks. Final playback uses collision-checked quintic timing with speed and acceleration limits.'};

export async function initSimulation({ getPose, getEEFRotation, getEEFPose, setPose, scene, frame }) {
  let info, results = [], active, playing = false, stamp = 0, playTime = 0, initial = null, busy = false;
  let livePose = null, revision = 0, mouseAiming = false;
  let robotState = {}, diffusionUI;
  let sampler=null;
  const outputView=trajectoryView(scene,'trajectory-output');
  $('guided-route-repair').checked = localStorage.getItem('guided-route-repair') !== 'off';
  const direction = initToolDirection(() => getEEFRotation(getPose()).map(row => row[2]));
  initWorkspace({ scene, frame, getStartPose: () => initial || getPose() });
  const targetFrame = () => readTargetFrame(getEEFRotation(active ? initial : livePose || initial || getPose()));
  const marker = new THREE.Mesh(new THREE.SphereGeometry(.006, 20, 16), new THREE.MeshBasicMaterial({ color: '#e58328', depthTest: false, depthWrite: false }));
  const ring = new THREE.Mesh(new THREE.SphereGeometry(.012, 16, 12), new THREE.MeshBasicMaterial({ color: '#e58328', wireframe: true, transparent: true, opacity: .45, depthTest: false, depthWrite: false }));
  marker.renderOrder = ring.renderOrder = 10;
  marker.add(ring); marker.visible = false; scene.add(marker);
  const targetDirection=new THREE.ArrowHelper(new THREE.Vector3(0,0,1),new THREE.Vector3(),.065,0xe58328,.011,.008);
  for(const part of [targetDirection.line,targetDirection.cone]) {part.material.depthTest=false;part.material.depthWrite=false;part.renderOrder=11;}
  targetDirection.name='pose5-target-direction';
  targetDirection.visible=false;scene.add(targetDirection);
  const targetAxes = new THREE.AxesHelper(.045);
  targetAxes.name = 'target-eef-local-frame';
  targetAxes.material.depthTest = false;
  targetAxes.renderOrder = 11;
  scene.add(targetAxes);
  const taskMode=()=> $('ik-task').value;
  const methods=()=> taskMode()==='pose5' ? info.pose5_methods : info.methods;
  const methodInfo=id=>id===CLASSICAL.id ? CLASSICAL : id===PRM.id ? PRM : id===DIFFUSION.id ? DIFFUSION : methods().find(m=>m.id===id);
  const methodName=id=>methodInfo(id)?.name||id;
  const methodColor=id=>id===CLASSICAL.id ? '#30795c' : id===PRM.id ? '#176978' : id===DIFFUSION.id ? '#a25424' : COLORS[methods().findIndex(m=>m.id===id)];
  const orientation=readReferenceTilts;
  function feedback(message, error = false) { $('ik-feedback').textContent = message; $('ik-feedback').classList.toggle('error', error); }
  function pause() { playing = false; $('play-ik').textContent = '▶ Play'; }
  function clearResults() {
    revision++;
    diffusionUI?.invalidate();
    pause(); results = []; active = null;sampler=null;outputView.clear();
    $('prm-results').hidden = true;
    $('prm-waypoints').querySelector('tbody').replaceChildren();
    $('prm-attempts').replaceChildren();
    $('play-ik').disabled = $('ik-scrub').disabled = true;
    $('orientation-chart').replaceChildren();
    $('velocity-chart').replaceChildren();$('trajectory-speed').textContent='—';
    $('error-chart').replaceChildren(); $('chart-legend').replaceChildren();
    $('run-results').querySelector('tbody').replaceChildren();
    $('ik-summary').textContent = 'Inputs changed · run a controller';
    $('frame-stats').textContent = '';
  }
  diffusionUI = initDiffusion({scene,clearResults,loadCase(c) {
    initial=c.start_deg.slice();setPose(initial);resetLocalOffset();$('ik-case').value='';
    $('ik-task').value='pose5';$('direction-entry').value='axis';$('direction-preset').value='';
    ['x','y','z'].forEach((axis,i)=>{$(`target-${axis}`).value=String(c.target_mm[i]);$(`target-u${axis}`).value=String(c.target_axis[i]);});
    populateMethods();renderBenchmark();direction.refresh();updateMarker();
    frame('perspective',[target().map(v=>v/1000)]);
    feedback(`Loaded ${c.name}. Plan checked diffusion to generate a new policy rollout.`);
  }});
  window.addEventListener('preview-edited', () => {
    clearResults(); initial = getPose(); updateMarker(); $('ik-case').value = '';
    feedback('Starting pose changed. Run a controller to compute a new trajectory.');
  });
  window.addEventListener('control-mode-changing', clearResults);
  function target() { return targetFrame().target; }
  function updateMarker() {
    if (mouseAiming) {
      marker.visible = targetAxes.visible = targetDirection.visible = false;
      return;
    }
    const { target: xyz, rotation } = targetFrame();
    $('resolved-target').textContent = xyz.every(Number.isFinite) ? `[${xyz.map(v => v.toFixed(2)).join(', ')}]` : 'Enter finite target, offset and tilt values';
    targetAxes.visible = xyz.every(Number.isFinite);
    if (targetAxes.visible) {
      targetAxes.position.set(...xyz.map(v => v / 1000));
      const m = new THREE.Matrix4().set(rotation[0][0],rotation[0][1],rotation[0][2],0, rotation[1][0],rotation[1][1],rotation[1][2],0, rotation[2][0],rotation[2][1],rotation[2][2],0, 0,0,0,1);
      targetAxes.quaternion.setFromRotationMatrix(m);
    }
    marker.visible = xyz.every(Number.isFinite);
    if (marker.visible) marker.position.set(...xyz.map(v => v / 1000));
    let axis; try { axis=readToolAxis(); } catch { axis=null; }
    targetDirection.visible=marker.visible&&taskMode()==='pose5'&&!!axis;
    if(targetDirection.visible){
      targetDirection.position.copy(marker.position);
      targetDirection.setDirection(new THREE.Vector3(...axis));
    }
  }
  window.addEventListener('mouse-aim-change', event => {
    mouseAiming = event.detail.active;
    diffusionUI.availability(robotState,busy,mouseAiming);
    diffusionUI.setMode(mouseAiming && event.detail.engine==='local_diffusion');
    clearResults(); initial = getPose(); updateMarker();
    if(mouseAiming&&event.detail.hardware&&event.detail.engine==='local_diffusion')diffusionUI.showHardwareScene();
  });
  for (const id of [...['x','y','z','rx','ry'].map(axis => `target-${axis}`), ...offsetInputIds]) $(id).addEventListener('input', () => { clearResults(); updateMarker(); $('ik-case').value = ''; feedback('Target changed. Run a controller to compute a new trajectory.'); });
  for (const id of [...directionInputIds,'pose5-proposals']) $(id).addEventListener(id.startsWith('target-')?'input':'change', () => { clearResults(); updateMarker(); });
  function putFrame(index) {
    if (!active) return;
    index = Math.min(active.frames.length - 1, Math.max(0, index));
    const current = active.frames[index];
    playTime=current.time;
    setPose(sampler.sample(current.time).q);
    $('ik-scrub').value = index;
    $('ik-time').textContent = `${current.time.toFixed(2)} s`;
    $('frame-stats').textContent = `TCP [${current.position.map(v => v.toFixed(1)).join(', ')}] mm · error ${current.error_mm.toFixed(2)} mm${Number.isFinite(current.orientation_error_deg) ? ` · Z direction ${current.orientation_error_deg.toFixed(2)}°` : ''} · frame ${index}/${active.frames.length - 1}`;
    $('trajectory-speed').textContent=`${Math.hypot(...sampler.sample(current.time).velocity_rad_s).toFixed(3)} rad/s`;
  }
  function putTime(time) {
    const sample=sampler.sample(time);setPose(sample.q);
    const pose=getEEFPose(sample.q),goal=target(),error=Math.hypot(...pose.position.map((v,i)=>v-goal[i]));
    const axis=active.target_axis,angle=axis?Math.acos(Math.max(-1,Math.min(1,pose.axis.reduce((sum,v,i)=>sum+v*axis[i],0))))*180/Math.PI:null;
    $('ik-scrub').value=sample.index;$('ik-time').textContent=`${sample.time.toFixed(2)} s`;
    $('frame-stats').textContent=`TCP [${pose.position.map(v=>v.toFixed(1)).join(', ')}] mm · error ${error.toFixed(2)} mm${angle===null?'':` · Z direction ${angle.toFixed(2)}°`} · continuous reference · frame ${sample.index}/${active.frames.length-1}`;
    $('trajectory-speed').textContent=sample.velocity_rad_s?`${Math.hypot(...sample.velocity_rad_s).toFixed(3)} rad/s`:'—';
  }
  function activate(id) {
    pause(); active = results.find(r => r.method === id) || results[0];
    if (!active) return;
    sampler=createTrajectorySampler(active.frames,active.smoothing);
    outputView.show(active.frames,active.target_mm||target());
    renderJointVelocityPlot($('velocity-chart'),sampler.velocityPoints);
    if (![PRM.id,DIFFUSION.id,CLASSICAL.id].includes(active.method)) $('ik-method').value = active.method;
    $('ik-description').textContent = methodInfo(active.method).description;
    $('ik-scrub').max = active.frames.length - 1;
    $('ik-scrub').disabled = $('play-ik').disabled = active.frames.length < 2;
    putFrame(0);
    frame('perspective', [target().map(v => v / 1000), ...active.frames.map(f => f.position.map(v => v / 1000))]);
    $('ik-summary').textContent = `${[PRM.id,DIFFUSION.id,CLASSICAL.id].includes(active.method) ? 'Validated plan' : active.converged ? 'Reached target' : 'Did not reach tolerance'} · ${active.final_error_mm.toFixed(2)} mm${Number.isFinite(active.final_orientation_error_deg) ? ` / ${active.final_orientation_error_deg.toFixed(2)}°` : ''} · ${active.iterations} steps · ${active.stop_reason.replaceAll('_',' ')}`;
    for (const row of $('run-results').querySelectorAll('tbody tr')) row.classList.toggle('selected-result', row.dataset.method === active.method);
  }
  function chart() {
    const curves=results.map(r=>({frames:r.frames,color:methodColor(r.method)}));
    renderErrorPlot($('error-chart'),curves,{field:'error_mm',label:'Position error (mm)'});
    $('orientation-chart').hidden=taskMode()!=='pose5';
    if(taskMode()==='pose5')renderErrorPlot($('orientation-chart'),curves,{field:'orientation_error_deg',label:'Flange Z direction error (degrees)'});
    $('chart-legend').replaceChildren(...results.map(r => {
      const button = document.createElement('button');
      button.textContent = methodName(r.method);
      button.style.borderLeft = `4px solid ${methodColor(r.method)}`;
      button.onclick = () => activate(r.method); return button;
    }));
  }
  function runTable() {
    const tbody = $('run-results').querySelector('tbody'); tbody.replaceChildren();
    for (const r of results) {
      const row = document.createElement('tr'); row.dataset.method = r.method;
      const button = document.createElement('button'); button.textContent = methodName(r.method);
      button.onclick = () => activate(r.method);
      const cell = document.createElement('td'); cell.append(button); row.append(cell);
      let column=0;
      for (const value of [r.converged ? 'Yes' : 'No', r.final_error_mm.toFixed(2), r.final_orientation_error_deg?.toFixed(2)??'—', Number.isFinite(r.joint_travel_deg)?r.joint_travel_deg.toFixed(1):'—', r.iterations, Number.isFinite(r.elapsed_ms)?r.elapsed_ms.toFixed(1):'—']) {
        const td = document.createElement('td'); td.textContent = value; if(column++===2){td.className='orientation-column';td.hidden=taskMode()!=='pose5';} row.append(td);
      }
      tbody.append(row);
    }
  }
  function populateMethods() {
    const previous=$('ik-method').value;
    $('ik-method').replaceChildren(...methods().map(m=>{const option=document.createElement('option');option.value=m.id;option.textContent=m.name;return option;}));
    $('ik-method').value=methods().some(m=>m.id===previous) ? previous
      : taskMode()==='pose5'&&methods().some(m=>m.id==='bounded_dls')?'bounded_dls':'dls';
    direction.refresh(); $('orientation-tolerance-field').hidden=taskMode()!=='pose5';
    $('orientation-chart').hidden=taskMode()!=='pose5';
    $('compare-ik').textContent=taskMode()==='pose5' ? `Compare all ${methods().length}` : 'Compare all eight';
    $('ik-task-description').textContent=taskMode()==='pose5'
      ? 'Base XYZ (mm) + directed unit tool Z in the base frame (two independent constraints). Reference tilts or vector entry; local tool-Z spin is free. Orange arrow = target direction. PRM constrains direction at the goal only.'
      : 'XYZ at joint6_flange in the URDF base frame. Orientation is free. Orange sphere = target; colored trail = simulated TCP.';
    document.querySelectorAll('.orientation-column').forEach(cell=>{cell.hidden=taskMode()!=='pose5';});
    localStorage.setItem('ik-task-mode',taskMode());
    updateMarker();
  }
  function renderBenchmark() {
    const benchmark=taskMode()==='pose5' ? info.benchmark_pose5 : info.benchmark;
    const tbody=$('benchmark-results').querySelector('tbody');tbody.replaceChildren();
    if(!benchmark){$('benchmark-description').textContent='No saved benchmark for this task';return;}
    $('benchmark-description').textContent=`${taskMode()==='pose5' ? 'Five-constraint poses: success requires XYZ and flange Z direction' : 'Position-only targets'}. 12 nearby + 12 global reachable FK targets; seed ${benchmark.seed}. Tolerance ${benchmark.tolerance_mm} mm${taskMode()==='pose5' ? ` and ${benchmark.orientation_tolerance_deg}°` : ''}; ${benchmark.steps} × ${benchmark.dt} s; joint speed norm ≤ ${benchmark.speed_rad_s} rad/s. Local solvers may fail on reachable poses. Compute times measure CPU rollout.`;
    for(const r of benchmark.rows){
      const row=document.createElement('tr');
      [r.name,`${r.success}/${r.total}`,r.mean_error_mm.toFixed(2),r.mean_orientation_error_deg?.toFixed(2)??'—',r.mean_travel_deg.toFixed(1),r.mean_elapsed_ms.toFixed(1)].forEach((value,i)=>{const td=document.createElement('td');td.textContent=value;if(i===3){td.className='orientation-column';td.hidden=taskMode()!=='pose5';}row.append(td);});tbody.append(row);
    }
  }
  $('ik-task').onchange=()=>{if(!info)return;clearResults();populateMethods();renderBenchmark();feedback('Task changed. Run a controller or plan from live joints.');};
  window.addEventListener('mouse-aim-inspect', event => {
    if (!info) return;
    const snapshot = event.detail;
    clearResults(); initial = snapshot.start_deg.slice(); setPose(initial);
    $('ik-task').value = 'pose5'; $('ik-case').value = ''; resetLocalOffset();
    ['x','y','z'].forEach((axis,i) => { $(`target-${axis}`).value = String(snapshot.target_mm[i]); $(`target-u${axis}`).value = String(snapshot.target_axis[i]); });
    $('direction-entry').value = 'axis'; $('direction-preset').value = '';
    for (const [id,key] of [['ik-speed','speed_rad_s'],['ik-steps','steps'],['ik-tolerance','tolerance_mm'],['ik-orientation-tolerance','orientation_tolerance_deg'],['ik-damping','damping']]) $(id).value = String(snapshot[key]);
    if (typeof snapshot.guided_route_repair === 'boolean') {
      $('guided-route-repair').checked = snapshot.guided_route_repair;
      $('guided-route-repair').dispatchEvent(new Event('change'));
    }
    populateMethods(); $('ik-method').value = snapshot.pose5_method;
    $('ik-description').textContent = methodInfo(snapshot.pose5_method).description;
    renderBenchmark(); direction.refresh(); updateMarker();
    feedback('Captured exact starting joints and 5D target. Run or compare numerical IK; Plan checks self-collision and routing.');
  });
  function showPlan(data) {
    $('prm-results').hidden = false;
    $('prm-repair').textContent = routeRepairStatus(data);
    const label = data.status.replaceAll('_',' ');
    const pairs = data.self_collision?.conflicts.map(c=>`${c.links.join(' / ')} (${c.intersecting ? 'intersection' : `${c.distance_mm.toFixed(2)} mm gap`})`) || [];
    $('prm-status').textContent = `${label} · ${data.reason?.replaceAll('_',' ') || ''}${pairs.length ? ` · ${pairs.join('; ')}` : ''}`;
    $('prm-status').classList.toggle('error', !data.converged);
    const recovery = data.ik?.recovery;
    $('prm-stats').textContent = `IK ${(data.ik_ms||0).toFixed(1)} ms · planning ${(data.planning_ms||0).toFixed(1)} ms · ${(data.waypoints_deg||[]).length} waypoints${Number.isFinite(data.path_length_rad) ? ` · ${data.path_length_rad.toFixed(3)} rad travel` : ''} · ${data.ik?.goals.length||0} valid goal candidates${recovery ? ` · expanded search: ${recovery.phases.reduce((n,p)=>n+p.used_seeds,0)} seeds, ${recovery.added_goals} added goals` : ''}`;
    $('prm-outcomes').textContent=`IK: ${(data.ik_outcome||data.status).replaceAll('_',' ')} · planning: ${(data.planning_outcome||'not started').replaceAll('_',' ')}${Number.isFinite(data.final_error_mm)?` · position ${data.final_error_mm.toFixed(3)} mm`:''}${Number.isFinite(data.final_orientation_error_deg)?` · direction ${data.final_orientation_error_deg.toFixed(3)}°`:''}${data.achieved_axis?` · achieved tool Z [${data.achieved_axis.map(v=>v.toFixed(4)).join(', ')}]`:''}`;
    const sources=[...new Set((data.ik?.goals||[]).map(g=>g.source.replaceAll('_',' ')))];
    $('prm-goal-sources').textContent=`Accepted goals: ${sources.join(', ')||'none'}${data.ik?.nn_checkpoint?` · 5D checkpoint ${data.ik.nn_checkpoint.status}${data.ik.nn_checkpoint.reason?`: ${data.ik.nn_checkpoint.reason}`:''}`:''}`;
    const tbody = $('prm-waypoints').querySelector('tbody'); tbody.replaceChildren();
    (data.waypoints_deg||[]).forEach((q,i)=>{
      const t=data.fk_transforms_m[i], row=document.createElement('tr');
      for(const value of [i, `[${q.map(v=>v.toFixed(2)).join(', ')}]`, `[${t.slice(0,3).map(r=>(r[3]*1000).toFixed(2)).join(', ')}]`, `[${t.slice(0,3).map(r=>r[2].toFixed(3)).join(', ')}]`]) {
        const cell=document.createElement('td');cell.textContent=value;row.append(cell);
      }
      tbody.append(row);
    });
    $('prm-attempts').replaceChildren(...(data.ik?.attempts||[]).map(a=>{
      const item=document.createElement('li');
      const conflicts=a.configuration_diagnostic?.self_collision?.conflicts.map(c=>c.links.join(' / '))||[];
      const source=a.source.replace(/^(current|roadmap|random)_dls$/, '$1 seed').replaceAll('_',' ');
      item.textContent=`${source}: ${a.rollout_status?.replaceAll('_',' ') || (a.accepted ? 'accepted' : 'rejected')}${a.refinement_method ? `; ${a.refinement_method.replaceAll('_',' ')}` : ''}${Number.isFinite(a.error_mm) ? `; FK error ${a.error_mm.toFixed(2)} mm` : ''}${Number.isFinite(a.orientation_error_deg) ? `; direction error ${a.orientation_error_deg.toFixed(2)}°` : ''}${a.stop_reason ? `; ${a.stop_reason.replaceAll('_',' ')}` : ''}${a.configuration_diagnostic ? `; ${a.configuration_diagnostic.reason}` : ''}${conflicts.length ? `; ${conflicts.join(', ')}` : ''}`;
      return item;
    }));
  }
  async function run(all = false, prm = false) {
    if (busy) return; pause(); busy = true;
    if([DIFFUSION.id,CLASSICAL.id].includes(active?.method))clearResults();
    diffusionUI.setMode(false);diffusionUI.availability(robotState,true,mouseAiming);
    if (prm) clearResults();
    const requestRevision = revision;
    $('solve-ik').disabled = $('compare-ik').disabled = $('plan-prm').disabled = true;
    feedback(prm ? 'Checking IK and planning a self-checked path… First query may take a minute or more to validate the saved roadmap.' : 'Computing simulated rollouts…');
    try {
      if(taskMode()==='pose5')readToolAxis();
      if (!target().every(Number.isFinite)) throw new Error('Enter finite target and EEF local offset values');
      const stateResponse = await fetch('/api/state');
      if (!stateResponse.ok) throw new Error('Simulation bridge unavailable');
      const state = await stateResponse.json();
      if (requestRevision !== revision) return;
      livePose = state.connected ? state.angles : null;
      let start = initial || getPose();
      if (state.connected) {
        if (state.motion || state.moving) throw new Error('Wait for the robot to stop before starting the solver');
        if (!Array.isArray(state.angles) || state.angles.length !== 6 || !state.angles.every(Number.isFinite)
            || !Number.isFinite(state.updated) || Date.now()/1000-state.updated > 5) {
          throw new Error('Live joint telemetry is unavailable or stale. Reconnect before starting the solver');
        }
        start = [...state.angles];
        $('follow-live').checked=false; $('follow-live').dispatchEvent(new Event('change'));
      }
      const response = await fetch(prm ? '/api/sim/prm' : '/api/sim/solve', { method:'POST', headers:{'Content-Type':'application/json','X-Robot-Token':state.token},
        body:JSON.stringify({ task:taskMode(), ...(taskMode()==='pose5'?{orientation_xy_deg:orientation(),...($('direction-entry').value==='axis'?{target_axis:readToolAxis()}: {})}:{}), orientation_tolerance_deg:Number($('ik-orientation-tolerance').value), start_deg:start, target_mm:targetFrame().base, eef_offset_mm:targetFrame().offset, methods:all ? methods().map(m => m.id) : [$('ik-method').value],
          ...(prm ? {nn:taskMode()==='pose5'?$('pose5-proposals').value:'seed',collision:'self', self_clearance_mm:1, shortcut:true, guided_route_repair:$('guided-route-repair').checked, route_repair_samples:32, pose5_method:taskMode()==='pose5' ? $('ik-method').value : 'bounded_dls', ...(taskMode()==='pose5'?{pose5_global_recovery:true}:{})} : {}),
          steps:Number($('ik-steps').value), speed_rad_s:Number($('ik-speed').value), tolerance_mm:Number($('ik-tolerance').value), damping:Number($('ik-damping').value) }) });
      const data = await response.json();
      if (requestRevision !== revision) return;
      if (!response.ok) throw new Error(data.error || 'Simulation failed');
      if (prm) {
        if (data.converged && data.validation !== 'joint_limits_and_self_collision') throw new Error('The server did not return self-collision validation. Restart the bridge with the updated code.');
        showPlan(data);
        if (!data.converged || !data.frames?.length) {
          $('ik-summary').textContent = `Plan failed · ${data.status.replaceAll('_',' ')}`;
          feedback($('prm-status').textContent + '. No validated preview path. This does not prove target unreachability.', true);
          return;
        }
        initial = data.start_deg;
        results = [{...data, iterations:data.frames.length-1, elapsed_ms:data.ik_ms+data.planning_ms}];
        updateMarker(); chart(); runTable(); activate(PRM.id);
        feedback(`Validated self-checked plan from ${data.start_source || (state.connected ? 'live joints' : 'simulation pose')}. Play or scrub the preview; inspect joint waypoints below. External obstacles are not checked.`);
        return;
      }
      $('prm-results').hidden = true;
      initial = data.start_deg; updateMarker(); results = data.results; chart(); runTable(); activate($('ik-method').value);
      feedback(`${results.filter(r => r.converged).length}/${results.length} methods reached the target. Start: ${state.connected ? 'live robot joints' : 'simulation pose'} [${initial.join(', ')}]°. Select a result and play or scrub its motion. All results use the same start, target and budget.`);
    } catch (error) { if (requestRevision === revision) feedback(error.message, true); }
    finally { busy = false; $('solve-ik').disabled = $('compare-ik').disabled = $('plan-prm').disabled = false;diffusionUI.availability(robotState,false,mouseAiming); }
  }
  async function runDiffusion(classical=false) {
    const label=classical?"learned IK + RRT*":"diffusion";
    if(busy)return;
    clearResults();busy=true;const requestRevision=revision;
    $('solve-ik').disabled=$('compare-ik').disabled=$('plan-prm').disabled=true;
    diffusionUI.setMode(true);diffusionUI.availability(robotState,true,mouseAiming);
    feedback(classical?'Refining learned IK starts and planning a checked classical path…':'Generating a checked diffusion rollout… First request loads the trained policy; budget up to 90 seconds.');
    try {
      if(taskMode()!=='pose5')throw new Error('Select the 5D pose task for diffusion planning');
      const stateResponse=await fetch('/api/state');
      if(!stateResponse.ok)throw new Error('Simulation bridge unavailable');
      const state=await stateResponse.json();
      if(state.hardware_enabled||state.connected||state.motion||state.moving||mouseAiming)throw new Error('Diffusion requires disconnected simulation mode with mouse aim off');
      if(requestRevision!==revision)return;
      initial=getPose().slice();updateMarker();
      const body=diffusionUI.body(initial,target(),readToolAxis());
      if(classical)delete body.goal_deg;
      const response=await fetch(classical?'/api/sim/classical':'/api/sim/diffusion',{method:'POST',headers:{'Content-Type':'application/json','X-Robot-Token':state.token},
        body:JSON.stringify(body)});
      const data=await response.json();
      if(requestRevision!==revision)return;
      if(!response.ok)throw new Error(data.error||`${label} planning failed`);
      if(data.method!==(classical?'learned_rrt':'diffusion'))throw new Error('Server returned a different fixed-target planner');
      const sceneMatches=body.obstacle_mm===null
        ? data.obstacle_mm===null&&data.scene_mode==='no_block'
        : data.scene_mode==='block'&&Array.isArray(data.obstacle_mm)&&data.obstacle_mm.length===6
          &&data.obstacle_mm.every((v,i)=>Number.isFinite(v)&&Math.abs(v-body.obstacle_mm[i])<1e-6);
      if(!sceneMatches)throw new Error('Server returned a different diffusion obstacle scene');
      diffusionUI.showResult(data);
      if(!data.converged||!data.frames?.length){$('ik-summary').textContent=`${label} failed · ${data.status.replaceAll('_',' ')}`;feedback(`No validated ${label} path. The preview pose is held; change the target, block or sampling seed and try again.`,true);return;}
      if(data.validation!=='joint_limits_self_table_block'||data.simulation_only!==true
        || data.smoothing?.method!=='collision_checked_shortcuts_quintic'||!data.smoothing.collision_rechecked)throw new Error('Server did not return a checked, smoothed simulation-only diffusion path');
      initial=data.start_deg;results=[data];chart();runTable();activate(classical?CLASSICAL.id:DIFFUSION.id);
      feedback(`Validated ${label} path. Play or scrub to preview the self- and table-checked path${body.obstacle_mm===null?' with no block':', including block clearance'}.`);
    } catch(error){if(requestRevision===revision)feedback(error.message,true);}
    finally{busy=false;$('solve-ik').disabled=$('compare-ik').disabled=$('plan-prm').disabled=false;diffusionUI.availability(robotState,false,mouseAiming);}
  }
  $('plan-diffusion').onclick=()=>runDiffusion(false);
  $('plan-classical').onclick=()=>runDiffusion(true);
  $('solve-ik').onclick = () => run(); $('compare-ik').onclick = () => run(true);
  $('plan-prm').onclick = () => run(false, true);
  $('guided-route-repair').addEventListener('change', () => {
    localStorage.setItem('guided-route-repair', $('guided-route-repair').checked ? 'on' : 'off');
    clearResults();
    feedback('Route repair setting changed. Plan again from the captured start.');
  });
  for (const id of ['ik-speed','ik-steps','ik-tolerance','ik-orientation-tolerance','ik-damping']) $(id).addEventListener('input', clearResults);
  $('ik-method').onchange = () => {
    if (busy || [PRM.id,DIFFUSION.id,CLASSICAL.id].includes(active?.method)) clearResults();
    $('ik-description').textContent = methods().find(m => m.id === $('ik-method').value).description;
    if (results.some(r => r.method === $('ik-method').value)) activate($('ik-method').value);
    else { pause(); active = null; $('play-ik').disabled = $('ik-scrub').disabled = true; feedback('Run the selected method to generate its trajectory.'); }
  };
  $('ik-case').onchange = () => {
    clearResults(); const c = info.cases[Number($('ik-case').value)];
    if ($('ik-case').value === '' || !c) return;
    resetLocalOffset();
    const hardware=robotState.hardware_enabled===true;
    initial = (hardware ? livePose || getPose() : c.start_deg).slice(); setPose(initial);
    $('direction-entry').value='tilts';
    ['x','y','z'].forEach((axis,i) => { $(`target-${axis}`).value = c.target_mm[i].toFixed(4); });
    ['rx','ry'].forEach((axis,i)=>{$(`target-${axis}`).value=c.orientation_xy_deg[i].toFixed(4);});
    syncAxisFromTilts(); direction.refresh();
    updateMarker();
    feedback(hardware?`Loaded held-out target ${c.name}. Hardware planning starts from live joints; the saved case start is used only in simulation.`:`Loaded ${c.name} with its held-out start and target.`);
    window.dispatchEvent(new CustomEvent('heldout-case-loaded',{detail:{name:c.name,hardware}}));
    frame('perspective', [target().map(v => v / 1000)]);
  };
  $('ik-scrub').oninput = () => { pause(); putFrame(Number($('ik-scrub').value)); };
  $('play-ik').onclick = () => {
    if (playing) { pause(); return; }
    if (!active) return;
    let index = Number($('ik-scrub').value);
    if (index >= active.frames.length-1) { index = 0; putFrame(0); }
    stamp = performance.now(); playing = true; $('play-ik').textContent = 'Ⅱ Pause';
  };
  function animate(now) {
    if (playing && active) {
      const dt=Math.max(0,(now-stamp)/1000)*Number($('playback-speed').value);stamp=now;
      playTime=sampler.advance(playTime,dt);putTime(playTime);
      if (playTime >= active.frames.at(-1).time) pause();
    }
    requestAnimationFrame(animate);
  }
  requestAnimationFrame(animate);
  initHardwareIK({
    getTargetFrame:targetFrame,
    onPlan(data) {
      diffusionUI.setMode(false);
      clearResults();
      $('follow-live').checked=false; $('follow-live').dispatchEvent(new Event('change'));
      $('ik-task').value=data.result.task||'position'; populateMethods(); renderBenchmark();
      initial=data.start_deg; results=[data.result];
      // Preserve base target and local offset: data.target_mm already includes the offset.
      if(data.result.orientation_xy_deg)['rx','ry'].forEach((axis,i)=>{$(`target-${axis}`).value=data.result.orientation_xy_deg[i].toFixed(4);});
      syncAxisFromTilts(); direction.refresh();
      if(data.hardware_diffusion)diffusionUI.showHardwareScene();
      livePose = data.start_deg; updateMarker(); chart(); runTable(); activate(data.result.method);
      feedback('Hardware plan preview from live joints. Play or scrub to inspect the full path before execution.');
    },
    beforeExecute() { pause(); $('follow-live').checked=true; $('follow-live').dispatchEvent(new Event('change')); },
    pausePreview:pause,
  });
  window.addEventListener('robot-state', event => {
    robotState=event.detail;diffusionUI.availability(robotState,busy,mouseAiming);
    const moving=!!(event.detail.motion||event.detail.moving);
    livePose = event.detail.connected ? event.detail.angles : null;
    if (!active && !busy) updateMarker();
    if(event.detail.connected && $('follow-live').checked && !active && !busy)initial=event.detail.angles;
    if(moving)pause();
    if(info){$('solve-ik').disabled=$('compare-ik').disabled=$('plan-prm').disabled=moving||busy;}
    $('play-ik').disabled=$('ik-scrub').disabled=moving||!active||active.frames.length<2;
  });
  try {
    const state = await (await fetch('/api/state')).json();
    robotState=state;
    const hardwareMode=state.hardware_enabled === true;
    livePose = state.connected ? state.angles : null;
    if(hardwareMode){
      $('simulation-phase').textContent='SIMULATION PREVIEW';
      document.querySelector('.subtitle').textContent='Connect, confirm the setup, then guide the arm in the scene.';
    }
    const response = await fetch('/api/sim/info'); info = await response.json();
    if (!response.ok || !Array.isArray(info.methods)) throw new Error(info.error || 'Start the simulation server to load IK models');
    diffusionUI.initialize(info.diffusion,state,info.classical);
    if(hardwareMode&&$('hardware-planner').value==='diffusion_v2')diffusionUI.showHardwareScene();
    const stored=localStorage.getItem('ik-task-mode');
    $('ik-task').value=hardwareMode && $('hardware-planner').value==='diffusion_v2'
      ? 'pose5' : new URLSearchParams(location.search).get('view')==='lab'&&['position','pose5'].includes(stored)
        ? stored : 'pose5';
    populateMethods();
    info.cases.forEach((c,i) => { const option=document.createElement('option'); option.value=i; option.textContent=c.name; $('ik-case').append(option); });
    initial = hardwareMode ? getPose() : info.default_start_deg;
    if (!hardwareMode) setPose(initial); frame();
    ['x','y','z'].forEach((axis,i) => { $(`target-${axis}`).value = info.default_target_mm[i].toFixed(4); });
    ['rx','ry'].forEach((axis,i)=>{$(`target-${axis}`).value=info.default_orientation_xy_deg[i].toFixed(4);}); syncAxisFromTilts(); direction.refresh(); updateMarker();
    const checkpoint=info.seed5_checkpoint;
    const collisionTrained=checkpoint?.status==='ready' && checkpoint.collision_training?.enabled;
    const collisionLabel=collisionTrained?` · trained with self-collision checks${Number.isFinite(checkpoint.collision_training.clearance_mm)?` (${checkpoint.collision_training.clearance_mm} mm clearance)`:''}`:'';
    $('seed5-checkpoint-status').textContent=checkpoint?`5D Seed NN: ${checkpoint.status}${checkpoint.training_kind?` (${checkpoint.training_kind}, ${checkpoint.completed_steps} ${checkpoint.fine_tuned?'fine-tuning ':''}steps)`:''}${collisionLabel}${checkpoint.reason?` · ${checkpoint.reason}`:''}${checkpoint.status!=='ready'?' · numerical fallback available':''}`:'5D checkpoint status unavailable; restart the bridge';
    $('pose5-proposals').querySelector('option[value="seed5"]').textContent=collisionTrained?'Collision-trained Seed NN + numerical fallback':'Seed NN + numerical fallback';
    // Pose editor respects the same URDF/SDK intersection as every solver.
    info.limits_deg.min.forEach((min,i) => {
      const max=info.limits_deg.max[i];
      for (const suffix of ['slider','number']) { const input=$(`joint-${i}-${suffix}`); input.min=min; input.max=max; }
      const labels=$(`joint-${i}-slider`).closest('.joint-control').querySelectorAll('.joint-limits span');
      labels[0].textContent=`${min}°`;labels[1].textContent=`${max}°`;
    });
    renderBenchmark();
    if (info.training) $('training-description').textContent = `Preserved XYZ networks: ${info.training.train_samples.toLocaleString()} training + ${info.training.validation_samples.toLocaleString()} validation samples, seeds ${info.training.training_seed}/${info.training.validation_seed}. 5D Seed NN uses a separate versioned 12-input checkpoint with four bounded proposals; PRM can refine them and use numerical fallback. Classical controller comparisons remain available.`;
    $('solve-ik').disabled = $('compare-ik').disabled = $('plan-prm').disabled = false;
    if (hardwareMode) {
      $('ik-method').value=taskMode()==='pose5'?'bounded_dls':'dls'; feedback('Simulation preview is available. Use Plan from live joints above to prepare hardware execution.');
    } else if(new URLSearchParams(location.search).get('view')==='lab')await run(true);
    else feedback('Ready for mouse tracking. Open Advanced controls to compare solvers or set a target.');
    return info;
  } catch (error) { feedback(error.message, true);window.dispatchEvent(new Event('tracking-model-error')); }
}
