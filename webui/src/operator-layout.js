const $ = id => document.getElementById(id);

// Presentation state only: confirmations and motor authorization stay with
// the existing controllers. Opening a view never changes their values.
export function initOperatorLayout() {
  const advanced=$('advanced-workbench');
  let controlState={}, modelInfo=null;
  function updateModel(){
    if(!modelInfo)return;
    const fixed=$('fixed-target-planner').value;
    const ready=fixed==='learned_rrt'?modelInfo.classical?.status==='ready':modelInfo.diffusion?.status==='ready';
    $('recommended-model').textContent=fixed==='learned_rrt'?(ready?'Learned IK + classical RRT*':'Learned IK + RRT* unavailable')
      :fixed==='diffusion_v2'?(ready?'Learned IK + Diffusion v2':'Diffusion v2 unavailable'):'Legacy IK methods';
    $('recommended-model-note').textContent=fixed==='learned_rrt'
      ?(ready?'Recommended for fixed targets · checked direct paths and RRT* detours. Mouse tracking uses its separate planner.':'Open Advanced controls to inspect the model status.')
      :fixed==='diffusion_v2'?'Diffusion fixed-target alternative · checked trajectory proposals.':'Legacy fixed-target solver selection.';
    document.querySelector('.recommended-badge').textContent=fixed==='learned_rrt'?(ready?'Recommended':'Unavailable'):'Alternative';
  }
  const showLab=()=>{
    advanced.open=true;
    for(const details of document.querySelectorAll('details'))details.open=true;
  };
  if(new URLSearchParams(location.search).get('view')==='lab')showLab();
  $('open-advanced').addEventListener('click',()=>{advanced.open=true;});
  $('diffusion-hardware-help').addEventListener('click',()=>{
    advanced.open=true;
    requestAnimationFrame(()=>$('fixed-target-controls').scrollIntoView({behavior:'smooth',block:'start'}));
  });
  window.addEventListener('mouse-aim-inspect',()=>{
    showLab();
    advanced.scrollIntoView({behavior:'smooth',block:'start'});
  });
  window.addEventListener('robot-state',({detail:state})=>{
    controlState=state;updateModel();
    const hardware=state.hardware_enabled===true;
    const simulation=state.hardware_enabled===false;
    document.body.classList.toggle('robot-connected',!!state.connected);
    document.body.classList.toggle('simulation-only',simulation);
    $('simulation-setup').hidden=!simulation;
    document.querySelector('.subtitle').textContent=hardware
      ?'Connect, confirm the setup, then guide the arm in the scene.'
      :simulation?'Start tracking and guide the simulated arm in the scene. No robot connection needed.'
      :'Checking control mode…';
    $('fixed-target-controls').hidden=!hardware;
    $('tracking-context').textContent=hardware
      ?'The model shows live joint readings. Escape stops the arm.'
      :'Simulation moves the preview only.';
    document.querySelector('.quick-guide li:first-child p').textContent=hardware
      ?'Connect and confirm the frame and clear path.'
      :'The preview is ready when the model loads.';
    document.querySelector('.quick-guide li:first-child strong').textContent=hardware
      ?'Complete setup':'Wait for the model';
    const ready=state.connected&&$('arm-movement').checked&&$('verify-ik-frame').checked
      &&(!['diffusion_v2','learned_rrt'].includes($('hardware-planner').value)||$('verify-diffusion-scene').checked);
    $('setup-status').textContent=!state.connected?'Connect to the arm to complete setup.'
      :state.motion||state.moving?'Arm moving. Escape or Stop ends movement.'
      :ready?'Ready. Start mouse tracking above the scene.'
      :'Confirm the frame and clear path, then enable movement.';
    $('setup-status').classList.toggle('ready',!!ready&&!state.motion&&!state.moving);
  });
  for(const id of ['arm-movement','verify-ik-frame','verify-diffusion-scene','hardware-planner'])$(id).addEventListener('change',()=>{
    updateModel();
    window.dispatchEvent(new Event('robot-state-refresh'));
  });
  $('mouse-aim-planner').addEventListener('change',updateModel);
  $('fixed-target-planner').addEventListener('change',updateModel);
  window.addEventListener('tracking-model-ready',({detail:info})=>{
    modelInfo=info;updateModel();
  });
  window.addEventListener('tracking-model-error',()=>{
    $('recommended-model').textContent='Tracking unavailable';
    $('recommended-model-note').textContent='Open Advanced controls to inspect the model loading error.';
  });
}
