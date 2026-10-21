import * as THREE from 'three';
import {coneDiskRegion,regionMatches,regionPositionMetrics} from './tracking-region.js';
import {profilePieces,PROFILE_METHOD} from './profile-playback.js';
import {effectiveOrientationMode} from './orientation-policy.js';
// Only backend-validated joint trajectories may move the mouse-aim preview.
export function validatedAimPath(result, start, limits, speed, cursor, trackingScene=null, hardwareCurve=false) {
  const continuous=result.continuous_motion===true;
  const orientationMode=effectiveOrientationMode(result,trackingScene?.orientationMode??'point_at_object');
  if(orientationMode==='camera_fov'&&(!trackingScene?.region||!trackingScene.camera||
      result.visibility?.passed!==true||result.modeled_visibility_checked!==true||
      JSON.stringify(result.visibility_model?.camera)!==JSON.stringify(trackingScene.camera)||
      result.goal_cursor_acceptance?.orientation_mode!=='camera_fov'||
      result.goal_cursor_acceptance?.camera_projection?.visible!==true))
    throw new Error('Relaxed orientation requires a checked matching camera and visible endpoint');
  if(trackingScene?.region) {
    const expected=trackingScene.region,actual=result.tracking_region;
    // Hardware freezes the cone from its live EEF; verify that authoritative
    // axis and center agree geometrically instead of imposing a browser axis.
    const canonical=actual&&Array.isArray(actual.axis)&&actual.axis.length===3&&actual.axis.every(Number.isFinite)
      && Math.abs(Math.hypot(...actual.axis)-1)<1e-6
      ?coneDiskRegion(new THREE.Vector3().fromArray(cursor).multiplyScalar(.001),actual.axis,expected):null;
    if(!canonical||!regionMatches(trackingScene.serverOwnsRegionAxis?canonical:expected,actual))
      throw new Error('Planner changed the acceptable tracking region');
    if(!Array.isArray(result.target_mm)||result.target_mm.length!==3||!result.target_mm.every(Number.isFinite)
      ||!Array.isArray(result.target_axis)||result.target_axis.length!==3||!result.target_axis.every(Number.isFinite))
      throw new Error('Planner returned no selected tracking goal');
    const selected=new THREE.Vector3().fromArray(result.target_mm).multiplyScalar(.001),metrics=regionPositionMetrics(selected,actual);
    const aim=new THREE.Vector3().fromArray(cursor).multiplyScalar(.001).sub(selected).normalize();
    if(metrics.axialErrorMm>(trackingScene.toleranceMm??1)+1e-6||metrics.radialMm>actual.radius_mm+1e-5
      ||result.target_axis.some((v,j)=>Math.abs(v-aim.getComponent(j))>1e-6))
      throw new Error('Planner selected a goal outside the tracking circle or changed its aim');
  }

  const smoothingAccepted=continuous
    ? ['collision_checked_bezier_c2',PROFILE_METHOD].includes(result.smoothing?.method)&&result.smoothing.ends_at_rest===true
    : ['collision_checked_shortcuts_quintic',PROFILE_METHOD].includes(result.smoothing?.method)&&result.smoothing.starts_and_ends_at_rest===true;
  const goalAccepted=[3,4].includes(trackingScene?.conditioningSchema)
    ? result.motion_conditioning?.schema===trackingScene.conditioningSchema&&result.motion_conditioning.target_kind==='cartesian_pose5'
    : result.goal_cursor_acceptance?.accepted===true
      && Array.isArray(result.goal_deg)&&result.goal_deg.length===6
      && result.goal_deg.every((q,j)=>Number.isFinite(q)&&q>=limits.min[j]-1e-7&&q<=limits.max[j]+1e-7);
  const accepted = trackingScene ? result.validation === 'joint_limits_self_table_block'
    && (trackingScene.planner==='learned_rrt'
      ? result.method==='learned_rrt'&&result.tracking_strategy==='learned_ik_rrt'
      : result.method==='local_diffusion'&&result.tracking_strategy==='local_then_diffusion') && result.tracking_chunk === true
    && result.playable === true && typeof result.converged === 'boolean'
    && result.replan_required === !result.converged && goalAccepted
    && (!result.converged || result.cursor_acceptance?.accepted === true)
    && Number.isFinite(result.speed_rad_s) && result.speed_rad_s>0
    && smoothingAccepted && result.smoothing.applied===true
    && result.smoothing.collision_rechecked===true
    && result.acceleration_rad_s2===3
    && result.checkpoint_sha256 === trackingScene.checkpoint
    && (trackingScene.box===null ? result.obstacle_mm===null && result.scene_mode==='no_block'
      : (result.scene_mode===undefined||result.scene_mode==='block') && Array.isArray(result.obstacle_mm) && result.obstacle_mm.length===6
        && result.obstacle_mm.every((v,i)=>Number.isFinite(v)&&Math.abs(v-trackingScene.box[i])<1e-6))
    : result.validation === 'joint_limits_and_self_collision' && result.converged && result.cursor_acceptance?.accepted;
  if(trackingScene?.velocityShape&&(result.smoothing?.method!==PROFILE_METHOD||JSON.stringify(result.smoothing.reference_shape)!==JSON.stringify(trackingScene.velocityShape)))throw new Error('Planner changed the selected tracking velocity profile');
  const modeAccepted=hardwareCurve
    ?result.simulation_only===false&&result.hardware_v2_mouse===true&&continuous&&!!trackingScene
    :result.simulation_only===true;
  if (!modeAccepted || result.task !== 'pose5' || !result.mouse_aim ||
      !accepted || result.gap_mm !== (trackingScene?.region?.distance_mm??50) ||
      result.coordinate_frame !== 'g_base' || result.tool_frame !== 'joint6_flange') {
    throw new Error('Planner did not return an accepted, self-checked 5D mouse-aim path');
  }
  if (!Array.isArray(result.cursor_mm) || result.cursor_mm.length !== 3 ||
      result.cursor_mm.some((v, i) => !Number.isFinite(v) || Math.abs(v - cursor[i]) > 1e-6)) {
    throw new Error('Planner returned a different cursor target');
  }
  const frames = result.frames;
  if (!Array.isArray(frames) || !frames.length) throw new Error('Planner returned no trajectory');
  if (continuous && (!smoothingAccepted||!result.smoothing.applied||!result.smoothing.collision_rechecked
      ||result.acceleration_rad_s2!==3||!Array.isArray(result.smoothing.segments)))
    throw new Error('Planner returned unchecked continuous mouse curves');
  if(continuous&&result.smoothing.method==='collision_checked_bezier_c2'&&result.smoothing.segments.some(segment=>
      !Array.isArray(segment.controls_deg)||segment.controls_deg.some(row=>!Array.isArray(row)||row.length!==6||row.some((v,j)=>
        !Number.isFinite(v)||v<limits.min[j]-1e-7||v>limits.max[j]+1e-7))))
    throw new Error('Planner returned unchecked continuous mouse curves');
  if(result.smoothing?.method===PROFILE_METHOD)for(const segment of result.smoothing.segments)for(const piece of profilePieces(segment,result.smoothing.reference_shape))if(piece.controls_deg.some(row=>row.some((v,j)=>!Number.isFinite(v)||v<limits.min[j]-1e-7||v>limits.max[j]+1e-7)))throw new Error('Shared velocity profile exceeds joint limits');
  if ((trackingScene||continuous) && (frames.length>(continuous?601:66) || frames.at(-1).time>(continuous?12:1.3)+1e-9 || result.speed_rad_s>Math.min(speed,1)+1e-9)) {
    throw new Error('Planner exceeded the tracking chunk or speed budget');
  }
  if((trackingScene||continuous) && (result.smoothing.dt!==.02 || result.smoothing.frames!==frames.length
      || Math.abs(result.smoothing.duration_s-frames.at(-1).time)>1e-9))throw new Error('Planner returned inconsistent smoothing metadata');
  for (let i = 0; i < frames.length; i++) {
    const frame = frames[i];
    if (!Array.isArray(frame.q) || frame.q.length !== 6 || !Number.isFinite(frame.time) ||
        frame.q.some((q, j) => !Number.isFinite(q) || q < limits.min[j] - 1e-7 || q > limits.max[j] + 1e-7)) {
      throw new Error('Planner returned an invalid trajectory frame');
    }
    if (!i) {
      if (frame.time !== 0 || frame.q.some((q, j) => Math.abs(q - start[j]) > 1e-6)) {
        throw new Error('Preview changed while planning; discard this path');
      }
    } else {
      const previous = frames[i - 1], dt = frame.time - previous.time;
      if((trackingScene||continuous)&&Math.abs(dt-.02)>1e-9)throw new Error('Planner returned an invalid smoothing interval');
      const travel = Math.hypot(...frame.q.map((q, j) => (q - previous.q[j]) * Math.PI / 180));
      if (dt <= 0 || travel > speed * dt + 1e-7) throw new Error('Planner returned an invalid trajectory speed');
    }
  }
  if(trackingScene && !continuous && frames.length>1) {
    const velocities=[];const intervals=[];
    for(let i=1;i<frames.length;i++){
      const dt=frames[i].time-frames[i-1].time;intervals.push(dt);
      velocities.push(frames[i].q.map((q,j)=>(q-frames[i-1].q[j])*Math.PI/180/dt));
    }
    const padded=[Array(6).fill(0),...velocities,Array(6).fill(0)];
    for(let i=1;i<padded.length;i++){
      const dt=i===1?intervals[0]:i===padded.length-1?intervals.at(-1):(intervals[i-2]+intervals[i-1])/2;
      if(Math.hypot(...padded[i].map((v,j)=>(v-padded[i-1][j])/dt))>3+1e-6)throw new Error('Planner returned an invalid trajectory acceleration');
    }
  }
  return frames;
}

// Interpolate adjacent checked joint segments, never shortcut a PRM corner.
export function sampleAimPath(frames, time) {
  if (time <= 0) return frames[0].q.slice();
  const next = frames.findIndex(frame => frame.time >= time);
  if (next < 0) return frames.at(-1).q.slice();
  const a = frames[next - 1], b = frames[next], t = (time - a.time) / (b.time - a.time);
  return a.q.map((q, j) => q + t * (b.q[j] - q));
}

// Stop at every checked frame before entering its next segment, even when a
// slow animation frame spans a PRM corner. Dropping surplus dt slows playback.
export function advanceAimTime(frames, current, dt) {
  const next = frames.find(frame => frame.time > current);
  return next ? Math.min(next.time, current + Math.max(0, dt)) : frames.at(-1).time;
}
