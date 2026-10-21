import * as THREE from 'three';
import {ApproachSimulation} from './approach-simulation.js';
import {createTrackingRegionView} from './tracking-region.js';
import {trajectoryView} from './trajectory-view.js';
import {randomApproachDemo} from './approach-demo.js';
import {readPlanningStream} from './planning-stream.js';
import {createRRTSearchView} from './rrt-search-view.js';

const $=id=>document.getElementById(id);

export function initApproachSimulation({getPose,getEEFPose,setPose,scene,stopMouseAim,frame}) {
  $('approach-relax-orientation').checked=true;
  let state={},info=null,motion=null,lastTime=null,nextRoundAt=null,betweenRoundsPaused=false,round=0;
  const regionView=createTrackingRegionView();regionView.group.name='approach-stage-region';scene.add(regionView.group);
  const routeView=trajectoryView(scene,'approach-trajectory');
  const object=new THREE.Mesh(new THREE.SphereGeometry(.006,20,14),new THREE.MeshBasicMaterial({color:'#e99129'}));
  object.name='approach-fixed-object';object.visible=false;scene.add(object);
  const searchView=createRRTSearchView(scene,getEEFPose,$('show-approach-rrt'),
    message=>{$('approach-search-status').textContent=message;});
  const authorized=()=>state.hardware_enabled===false&&!state.connected&&!state.motion&&!state.moving;
  function updateControls() {
    $('approach-simulation-controls').hidden=state.hardware_enabled!==false;
    $('start-approach-simulation').disabled=!authorized()||!info?.tracking_demo||info?.classical?.status!=='ready'||runner.active;
    $('pause-approach-simulation').disabled=!runner.active&&nextRoundAt===null;
    $('resume-approach-simulation').disabled=!(runner.paused||betweenRoundsPaused)||!authorized();
    $('stop-approach-simulation').disabled=!(runner.active||runner.paused||runner.phase==='holding');
  }
  const runner=new ApproachSimulation({authorized,
    feedback:()=>{const q=getPose();return {q,...getEEFPose(q)};},
    plan:async(body,signal)=>{
      searchView.clear();$('approach-search-status').textContent='Finding an endpoint, then checking the RRT* route…';
      // An aborted request can still be releasing the shared server planner.
      while(!signal.aborted){
        const response=await fetch('/api/sim/approach-plan',{method:'POST',signal,
          headers:{'Content-Type':'application/json','X-Robot-Token':state.token},body:JSON.stringify(body)});
        if(response.status===409){await response.json();await new Promise(resolve=>setTimeout(resolve,250));continue;}
        return readPlanningStream(response,signal,events=>{if(!signal.aborted)searchView.events(events);});
      }
      throw new DOMException('Approach interrupted','AbortError');
    },
    execute:(sampler,signal)=>new Promise((resolve,reject)=>{
      motion={sampler,elapsed:0,resolve,reject};lastTime=null;
      signal.addEventListener('abort',()=>{
        if(motion?.sampler===sampler)motion=null;
        reject(new DOMException('Approach interrupted','AbortError'));
      },{once:true});
      if(signal.aborted){motion=null;reject(new DOMException('Approach interrupted','AbortError'));}
    }),
    onPlan:(result)=>{
      regionView.show(result.tracking_region);routeView.show(result.frames,result.target_mm);searchView.accepted(result.frames,result.goal_deg);
      const diagnostics=result.rrt_diagnostics;
      $('approach-search-status').textContent=diagnostics.status==='direct_optimal'
        ?'Accepted direct route · no RRT* expansion needed'
        :`Accepted RRT* route · ${diagnostics.nodes} nodes · ${diagnostics.rewires} rewires`;
      if(result.orientation_selection)$('approach-search-status').textContent+=result.orientation_selection.fallback_used
        ?` · camera-FOV fallback after strict ${result.orientation_selection.strict.status}`:' · strict pointing';
    },
    onChange:next=>{
      $('approach-status').textContent=next.message;
      const stages=next.stages;
      if(next.phase==='planning'&&stages?.region)regionView.show(stages.region);
      $('approach-progress').textContent=stages
        ?stages.initialDistance===null?'Capturing current EEF-to-object distance…'
          :`Round ${round} · ${stages.completed.length}/${stages.stageCount} stages reached · start ${stages.initialDistance.toFixed(1)} mm · current ${stages.distance.toFixed(1)} mm · final ${stages.finalDistance.toFixed(1)} mm`
        :'Current distance → 150 mm · 20 mm stages';
      if(next.phase==='holding'&&$('repeat-approach-demo').checked) {
        nextRoundAt=performance.now()+2000;
        $('approach-status').textContent='Approach complete. A new random object starts in 2 seconds.';
      }
      if(next.phase==='stopped'){
        nextRoundAt=null;betweenRoundsPaused=false;regionView.clear();routeView.clear();searchView.clear();object.visible=false;
        $('approach-object-distance').textContent='—';$('approach-search-status').textContent='Magenta: live search branches · blue: accepted route';
      }
      updateControls();
    }});
  function startRound(demo) {
    if(!authorized()||!info?.tracking_demo||runner.active)return;
    stopMouseAim();
    setPose(demo.start_deg);round++;
    void runner.start({...demo,orientation_mode:demo.orientation_mode??($('approach-relax-orientation').checked?'strict_then_fov':'point_at_object')},info);
    object.position.fromArray(demo.cursor_mm).multiplyScalar(.001);object.visible=true;
    $('approach-object-position').textContent=demo.cursor_mm.map(v=>v.toFixed(1)).join(', ');
    frame('perspective',[object.position.toArray()]);
  }
  $('start-approach-simulation').addEventListener('click',()=>{
    if(!info?.tracking_demo)return;
    round=0;startRound(info.tracking_demo);
  });
  function pause(message) {
    if(nextRoundAt!==null){
      nextRoundAt=null;betweenRoundsPaused=true;
      $('approach-status').textContent='Approach complete. Next random object paused; Resume to continue.';
      updateControls();
    } else runner.pause(message);
  }
  $('pause-approach-simulation').addEventListener('click',()=>pause());
  $('resume-approach-simulation').addEventListener('click',()=>{
    if(betweenRoundsPaused&&authorized()){
      betweenRoundsPaused=false;nextRoundAt=performance.now()+2000;
      $('approach-status').textContent='New random object starts in 2 seconds.';updateControls();
    } else void runner.resume();
  });
  $('stop-approach-simulation').addEventListener('click',()=>runner.stop());
  $('approach-relax-orientation').addEventListener('change',()=>runner.stop());
  $('repeat-approach-demo').addEventListener('change',()=>{
    betweenRoundsPaused=false;
    if(!$('repeat-approach-demo').checked){nextRoundAt=null;if(runner.phase==='holding')$('approach-status').textContent=runner.message;}
    else if(runner.phase==='holding'&&authorized()){
      nextRoundAt=performance.now()+2000;$('approach-status').textContent='New random object starts in 2 seconds.';
    }
    updateControls();
  });
  window.addEventListener('robot-state',({detail})=>{state=detail;if(!authorized()&&(runner.active||runner.paused||nextRoundAt!==null))runner.stop();updateControls();});
  window.addEventListener('control-mode-changing',()=>runner.stop());
  window.addEventListener('preview-edited',()=>runner.stop());
  window.addEventListener('mouse-aim-change',({detail})=>{if(detail.active)runner.stop();});
  for(const id of ['load-visible-tracking-demo','zero-pose','rest-pose','solve-ik','compare-ik','plan-prm','plan-classical','plan-diffusion','play-ik'])
    $(id).addEventListener('click',()=>runner.stop(),{capture:true});
  window.addEventListener('blur',()=>pause('Approach paused while the window is unfocused. Resume to continue.'));
  document.addEventListener('visibilitychange',()=>{if(document.hidden)pause('Approach paused while the page is hidden.');});
  document.addEventListener('keydown',event=>{if(event.key==='Escape')runner.stop();});
  fetch('/api/state').then(r=>r.json()).then(next=>{if(!state.token)state=next;updateControls();})
    .catch(()=>{$('approach-status').textContent='Cannot read robot state. Reload after the server is available.';});
  return {runner,startCase(demo){
    if(!authorized()||info?.classical?.status!=='ready')
      throw new Error('Approach case requires a ready planner in disconnected simulation mode');
    runner.stop();round=0;
    $('approach-relax-orientation').checked=demo.orientation_mode!=='point_at_object';
    // A saved case is one fixed-object round, rather than a random demo loop.
    $('repeat-approach-demo').checked=false;
    startRound(demo);
  },configure(next){info=next;updateControls();},update(time){
    searchView.update();
    if(nextRoundAt!==null&&time>=nextRoundAt){
      nextRoundAt=null;
      if(authorized()&&$('repeat-approach-demo').checked&&!document.hidden)
        startRound(randomApproachDemo(info.tracking_demo,getEEFPose(info.tracking_demo.start_deg)));
    }
    if(!motion||!runner.active){lastTime=null;return;}
    const dt=lastTime===null?0:Math.min(.05,Math.max(0,(time-lastTime)/1000));lastTime=time;
    motion.elapsed=motion.sampler.advance(motion.elapsed,dt);
    setPose(motion.sampler.sample(motion.elapsed).q);
    const pose=getEEFPose(getPose());
    $('approach-object-distance').textContent=`${(object.position.distanceTo(new THREE.Vector3().fromArray(pose.position).multiplyScalar(.001))*1000).toFixed(1)} mm`;
    if(motion.elapsed>=motion.sampler.endTime){const done=motion.resolve;motion=null;lastTime=null;done();}
  }};
}
