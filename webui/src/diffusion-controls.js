import * as THREE from 'three';
const $ = id => document.getElementById(id);
const boxInputs = ['diffusion-x','diffusion-y','diffusion-width','diffusion-depth','diffusion-height'];

export function readDiffusionScene(tableTop=-.035) {
  if($('diffusion-obstacle-mode').value==='none')return null;
  const [x,y,width,depth,height] = boxInputs.map(id => Number($(id).value));
  if (!boxInputs.every(id => $(id).value.trim()) || ![x,y,width,depth,height].every(Number.isFinite)
      || Math.abs(x)>400 || Math.abs(y)>400 || [width,depth,height].some(v=>v<10||v>500)) {
    throw new Error('Enter block X/Y within ±400 mm and sizes between 10 and 500 mm');
  }
  return [x,y,tableTop*1000+height/2,width/2,depth/2,height/2];
}

export function initDiffusion({ scene, clearResults, loadCase }) {
  let classicalInfo=null;
  let info, goal = null, tableTop = -.035, allowed = true, hardwareScene = false;
  let availabilityState = {}, planningBusy = false, trackingActive = false;
  const cartesian = () => [3,4].includes(info?.conditioning?.schema);
  const group = new THREE.Group(); group.name = 'diffusion-scene'; group.visible = false;
  const table = new THREE.Mesh(new THREE.BoxGeometry(.9,.9,.05),new THREE.MeshLambertMaterial({color:'#a99a80'}));
  table.name = 'diffusion-table'; table.receiveShadow = true; group.add(table);
  const block = new THREE.Mesh(new THREE.BoxGeometry(1,1,1),new THREE.MeshLambertMaterial({color:'#df8036'}));
  block.name = 'diffusion-block'; block.castShadow = block.receiveShadow = true;
  const outline = new THREE.LineSegments(new THREE.EdgesGeometry(block.geometry),new THREE.LineBasicMaterial({color:'#88491e'}));
  block.add(outline); group.add(block); scene.add(group);
  function geometry() {
    return readDiffusionScene(tableTop);
  }
  function updateScene() {
    try {
      const box = hardwareScene?null:geometry();block.visible=box!==null;
      if(box){
        block.position.set(...box.slice(0,3).map(v=>v/1000));block.scale.set(...box.slice(3).map(v=>v/500));
        $('diffusion-scene-readout').textContent = `Block center Z ${box[2].toFixed(1)} mm · table top ${(tableTop*1000).toFixed(1)} mm · axis-aligned in g_base`;
      } else $('diffusion-scene-readout').textContent=`No block · self-collision and table clearance remain checked · table top ${(tableTop*1000).toFixed(1)} mm${info?.no_block_note?'. Experimental: checkpoint trained with blocks.':'.'}`;
    } catch(error) { block.visible = false; $('diffusion-scene-readout').textContent = error.message; }
    table.position.z = tableTop-.025;
    group.visible = hardwareScene || (allowed && $('diffusion-scene-enabled').checked);
    for(const id of boxInputs)$(id).disabled=!allowed||$('diffusion-obstacle-mode').value==='none';
  }
  function setMode(enabled) { hardwareScene=false;$('diffusion-scene-enabled').checked = enabled && allowed; updateScene(); }
  function invalidate() { hardwareScene=false;updateScene();$('diffusion-result').textContent = ''; $('diffusion-result').classList.remove('error'); }
  $('diffusion-scene-enabled').onchange = () => { clearResults(); updateScene(); };
  $('diffusion-obstacle-mode').addEventListener('change',()=>{clearResults();updateScene();});
  for (const id of [...boxInputs,'diffusion-seed','diffusion-samples','diffusion-denoise','diffusion-budget']) {
    $(id).addEventListener('input', () => { clearResults(); updateScene(); });
    if ($(id).tagName==='SELECT') $(id).addEventListener('change', () => { clearResults(); updateScene(); });
  }
  // Keep the case's exact joint goal only while its Cartesian goal is untouched.
  for (const id of ['target-x','target-y','target-z','target-rx','target-ry','target-ux','target-uy','target-uz','eef-offset-x','eef-offset-y','eef-offset-z','direction-entry','direction-preset','ik-task','ik-case']) {
    $(id).addEventListener($(id).tagName==='SELECT'?'change':'input', () => { goal=null; $('diffusion-case').value=''; });
  }
  $('diffusion-case').onchange = () => {
    clearResults();
    const c = info?.cases.find(c=>c.id===$('diffusion-case').value);
    goal = c?.goal_deg?.slice() || null;
    if (!c) return;
    const box = c.obstacle_mm;
    $('diffusion-obstacle-mode').value=box?'block':'none';
    if(box)[box[0],box[1],box[3]*2,box[4]*2,box[5]*2].forEach((v,i)=>$(boxInputs[i]).value=String(v));
    setMode(true); loadCase(c);
  };
  function availability(state, busy=false, mouseAiming=false) {
    availabilityState=state;planningBusy=busy;trackingActive=mouseAiming;
    allowed = state.hardware_enabled === false && !state.connected && !state.motion && !state.moving;
    const pose5=$('ik-task').value==='pose5';
    const disabled = !allowed || busy || mouseAiming || info?.status!=='ready' || !pose5;
    for(const id of ['plan-diffusion','diffusion-case','diffusion-scene-enabled']) $(id).disabled=disabled;
    $('plan-classical').disabled=!allowed||busy||mouseAiming||!pose5||classicalInfo?.status!=='ready';
    $('diffusion-obstacle-mode').disabled=!allowed||busy||info?.status!=='ready';
    $('diffusion-hardware-help').hidden=state.hardware_enabled!==true;
    $('diffusion-availability').textContent=state.hardware_enabled===true
      ? 'This button is for simulation only. Choose Simulation only in Mode for a preview. For the real arm, use the fixed-target planner and Plan from live joints.'
      : state.hardware_enabled!==false ? 'Waiting for the server to confirm simulation mode…'
      : state.connected||state.motion||state.moving ? 'Disconnect the arm and wait for movement to stop before planning a simulation preview.'
      : !info ? 'Loading the diffusion checkpoint…'
      : info.status!=='ready' ? `Diffusion checkpoint unavailable: ${info.reason||'restart the bridge to check the model'}.`
      : mouseAiming ? 'Stop mouse aim before planning a fixed-target diffusion preview.'
      : busy ? 'Another planner is running. Wait for it to finish.'
      : !pose5 ? 'Select 5D pose in Target mode to use the diffusion planner.'
      : 'Ready. Learned IK + RRT* is recommended for fixed targets; diffusion is available as an alternative. Use Play after a valid path is found.';
    if (!allowed) { $('diffusion-scene-enabled').checked=false;updateScene(); }
    else updateScene();
  }
  $('ik-task').addEventListener('change',()=>availability(availabilityState,planningBusy,trackingActive));
  return {
    setMode, invalidate, availability,
    showHardwareScene() { hardwareScene=true;updateScene(); },
    initialize(data,state,classical) {
      classicalInfo=classical;
      info = data;
      if(info?.environment) tableTop=info.environment.table_top_m;
      $('diffusion-checkpoint').textContent = info?.status==='ready'
        ? `Trained DDPM ready · ${(info.dataset_episodes||1000).toLocaleString()} demonstration episodes${cartesian()?' · Cartesian target + direction · no goal IK required':info.observation_dim===34?' · motion conditioned':''} · held-out ${info.evaluation.successes}/${info.evaluation.episodes} reached · checkpoint ${info.checkpoint_sha256.slice(0,12)}`
        : `Diffusion unavailable · ${info?.reason || 'Restart the updated simulation bridge'}`;
      for(const c of info?.cases || []) { const option=document.createElement('option');option.value=c.id;option.textContent=c.name;$('diffusion-case').append(option); }
      availability(state); updateScene();
    },
    body(start,target,axis) {
      if(!allowed)throw new Error('Diffusion requires disconnected simulation mode');
      return {start_deg:start,target_mm:target,target_axis:axis,obstacle_mm:geometry(),
        ...(goal && !cartesian() ? {goal_deg:goal}:{}),seed:Number($('diffusion-seed').value),
        samples:Number($('diffusion-samples').value),inference_steps:Number($('diffusion-denoise').value),
        max_decisions:Number($('diffusion-budget').value)};
    },
    showResult(data) {
      $('diffusion-result').classList.toggle('error',!data.converged);
      const rejects=Object.values(data.rejections || {}).reduce((n,v)=>n+v,0);
      $('diffusion-result').textContent = data.converged
        ? data.method==='learned_rrt'
          ? `Validated learned IK + RRT* preview · ${data.rrt_diagnostics?.status.replaceAll('_',' ')} · IK ${(data.ik_elapsed_ms/1000).toFixed(2)} s · path ${(data.path_elapsed_ms/1000).toFixed(2)} s · checked quintic timing`
          : `Validated smoothed diffusion preview · ${data.decisions} decisions · ${rejects}/${data.proposed_chunks} chunks rejected · ${[3,4].includes(data.motion_conditioning?.schema)?'Cartesian target + direction':`${data.endpoint_capture_steps} final goal capture step`} · max speed ${data.max_velocity_rad_s.toFixed(2)} / 1.00 rad/s · acceleration ${data.max_acceleration_rad_s2.toFixed(2)} / 3.00 rad/s² · checked quintic timing`
        : `No playable path · ${data.reason || data.status.replaceAll('_',' ')}${data.decisions ? ` · ${data.decisions} decisions · ${rejects} chunks rejected` : ''}`;
    },
  };
}
