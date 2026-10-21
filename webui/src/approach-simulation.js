import * as THREE from 'three';
import {coneDiskRegion,regionPositionMetrics} from './tracking-region.js';
import {validatedAimPath} from './mouse-aim-path.js';
import {createTrajectorySampler} from './trajectory-playback.js';
import {visibilityHoldMessage,cameraProjectionForPose} from './tracking-visibility.js';
import {effectiveOrientationMode} from './orientation-policy.js';

const position = feedback => new THREE.Vector3().fromArray(feedback.position).multiplyScalar(.001);

// One frozen region per distance stage. Preview completion supplies fresh FK;
// partial checked prefixes and exhausted searches never shrink the stage.
export class ApproachStages {
  constructor(demo,camera=null) {
    if(!Array.isArray(demo.cursor_mm)||demo.cursor_mm.length!==3||!demo.cursor_mm.every(Number.isFinite)||
        !Number.isFinite(demo.distance_mm)||demo.distance_mm<150||demo.distance_mm>500||
        !Number.isFinite(demo.half_angle_deg)||demo.half_angle_deg<1||demo.half_angle_deg>45)
      throw new Error('Invalid approaching simulation setup');
    this.objectMm=Object.freeze(demo.cursor_mm.slice());
    this.orientationMode=demo.orientation_mode??'strict_then_fov';
    if(!['point_at_object','camera_fov','strict_then_fov'].includes(this.orientationMode))throw new Error('Unknown approach orientation mode');
    if(this.orientationMode!=='point_at_object'&&!camera)throw new Error('Camera-FOV approach requires the configured camera');
    this.camera=camera;
    // The nominal demo distance describes its setup, not the first goal.
    // Capture the actual separation once so a closer start is never sent
    // outward to the nominal 350 mm disk. Retries retain the captured stage.
    this.distance=null;this.initialDistance=null;
    this.halfAngle=demo.half_angle_deg;this.finalDistance=150;this.step=20;
    this.region=null;this.seed=42;this.completed=[];
  }
  capture(feedback) {
    if(!this.region) {
      const object=new THREE.Vector3().fromArray(this.objectMm).multiplyScalar(.001);
      const delta=object.clone().sub(position(feedback));
      if(this.initialDistance===null) {
        const distance=delta.length()*1000;
        if(!Number.isFinite(distance)||distance<10||distance>500)
          throw new Error('Initial object distance must be 10–500 mm for cone-disk planning');
        // Remove only FK floating-point noise; preserve measured millimeters.
        this.distance=Math.round(distance*1e6)/1e6;
        this.initialDistance=this.distance;
        this.finalDistance=Math.min(this.finalDistance,this.distance);
      }
      const axis=delta.normalize();
      this.region=coneDiskRegion(object,axis.toArray(),{distance_mm:this.distance,half_angle_deg:this.halfAngle});
    }
    return this.region;
  }
  get stageCount() {
    return this.initialDistance===null?null:Math.ceil((this.initialDistance-this.finalDistance)/this.step)+1;
  }
  arrived(feedback,result) {
    if(!result.converged || result.replan_required)return false;
    const region=this.capture(feedback),metrics=regionPositionMetrics(position(feedback),region);
    const aim=new THREE.Vector3().fromArray(this.objectMm).multiplyScalar(.001).sub(position(feedback)).normalize();
    const angle=THREE.MathUtils.radToDeg(Math.acos(THREE.MathUtils.clamp(
      aim.dot(new THREE.Vector3().fromArray(feedback.axis).normalize()),-1,1)));
    const orientationMode=effectiveOrientationMode(result,this.orientationMode);
    const orientationAccepted=orientationMode==='camera_fov'
      ? result.orientation_mode==='camera_fov'&&result.visibility?.passed===true
        &&cameraProjectionForPose(feedback,this.objectMm,this.camera).inFrame : angle<=3;
    if(metrics.axialErrorMm>10 || metrics.radialMm>region.radius_mm || !orientationAccepted)
      throw new Error(`Preview has not reached the current stage within 10 mm / ${orientationMode==='camera_fov'?'camera FOV':'3°'}`);
    this.completed.push(this.distance);this.seed=42;
    if(this.distance===this.finalDistance)return true;
    this.distance=Math.max(this.finalDistance,this.distance-this.step);
    this.region=null;
    return false;
  }
  rejected(status) {
    if(['no_path','ik_failed'].includes(status))this.seed=(this.seed+1)%2147483648;
  }
  reanchor() {this.region=null;this.seed=42;}
}

export class ApproachSimulation {
  constructor({feedback,plan,execute,authorized,onChange=()=>{},onPlan=()=>{}}) {
    Object.assign(this,{feedback,plan,execute,authorized,onChange,onPlan});
    this.phase='idle';this.active=false;this.paused=false;this.generation=0;this.abort=null;
  }
  change(phase,message) {this.phase=phase;this.message=message;this.onChange(this);}
  start(demo,info) {
    if(!this.authorized())throw new Error('Approach preview requires disconnected simulation mode');
    this.stop();this.stages=new ApproachStages(demo,info?.tracking_camera);this.info=info;
    this.speed=Math.min(1,THREE.MathUtils.degToRad(20*70/30));
    this.shape={kind:'smooth_trapezoid',rise_percent:5,fall_percent:10};
    this.active=true;this.paused=false;
    this.change('starting','Starting the approaching simulation…');
    this.task=this.run(++this.generation);
    return this.task;
  }
  pause(message='Approach paused. Resume from the current preview pose.') {
    if(!this.active)return;
    this.active=false;this.paused=true;++this.generation;this.abort?.abort();
    this.stages.reanchor();this.change('paused',message);
  }
  resume() {
    if(!this.paused || !this.authorized())return;
    this.active=true;this.paused=false;this.change('starting','Resuming from the current preview pose…');
    this.task=this.run(++this.generation);return this.task;
  }
  stop() {
    this.active=false;this.paused=false;++this.generation;this.abort?.abort();
    this.change('stopped','Approach stopped.');
  }
  current(epoch) {return this.active && epoch===this.generation && this.authorized();}
  async run(epoch) {
    try {
      while(this.current(epoch)) {
        const feedback=await this.feedback();
        if(!this.current(epoch))return;
        const region=this.stages.capture(feedback);
        const request={tracking_engine:'learned_rrt',start_deg:feedback.q.slice(),
          cursor_mm:this.stages.objectMm.slice(),target_axis:region.axis.slice(),
          tracking_region:{mode:'cone_disk',distance_mm:this.stages.distance,half_angle_deg:this.stages.halfAngle},
          obstacle_mm:null,seed:this.stages.seed,continuous_motion:true,speed_rad_s:this.speed,
          tolerance_mm:1,orientation_tolerance_deg:1,velocity_shape:this.shape,
          orientation_mode:this.stages.orientationMode,
          start_velocity_rad_s:Array(6).fill(0),start_acceleration_rad_s2:Array(6).fill(0)};
        this.abort=new AbortController();
        this.change('planning',`Checking the ${this.stages.distance.toFixed(1)} mm stage…`);
        const result=await this.plan(request,this.abort.signal);
        if(!this.current(epoch))return;
        if(!result.playable) {
          if(result.status==='object_not_visible') {
            this.pause(visibilityHoldMessage(result));return;
          }
          if(!['no_path','ik_failed'].includes(result.status))throw new Error(result.reason||result.status);
          this.stages.rejected(result.status);
          this.change('held',`Holding at ${this.stages.distance.toFixed(1)} mm; trying another bounded search.`);
          // Yield between attempts; Stop and Pause remain responsive.
          await new Promise(resolve=>setTimeout(resolve,200));continue;
        }
        if(result.visibility?.passed!==true || result.modeled_visibility_checked!==true ||
            JSON.stringify(result.visibility_model?.camera)!==JSON.stringify(this.info.tracking_camera))
          throw new Error('Approach planner did not verify the configured camera visibility');
        const frames=validatedAimPath(result,request.start_deg,this.info.limits_deg,this.speed,
          request.cursor_mm,{box:null,checkpoint:this.info.classical.checkpoint_sha256,
            planner:'learned_rrt',region,toleranceMm:1,velocityShape:this.shape,
            orientationMode:this.stages.orientationMode,camera:this.stages.camera});
        const sampler=createTrajectorySampler(frames,result.smoothing);
        this.onPlan(result,request);
        this.change('moving',`Approaching the ${this.stages.distance.toFixed(1)} mm stage…`);
        await this.execute(sampler,this.abort.signal);
        if(!this.current(epoch))return;
        const measured=await this.feedback();
        if(!this.current(epoch))return;
        this.stages.seed=42;
        if(this.stages.arrived(measured,result)) {
          this.active=false;this.change('holding',`Approach complete. Holding the ${this.stages.finalDistance.toFixed(1)} mm stage.`);return;
        }
      }
      if(epoch===this.generation)this.stop();
    } catch(error) {
      if(epoch!==this.generation || error.name==='AbortError')return;
      this.pause(`Held — ${error.message}`);
    }
  }
}
