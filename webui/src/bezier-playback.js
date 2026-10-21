// Reproduce the backend's collision-checked C2 curves and convex derivative bounds.
import {positionFromVelocity,velocityPoint} from './joint-velocity.js';
const radians=Math.PI/180;
const zero=()=>Array(6).fill(0);
const close=(a,b,tol=1e-6)=>Array.isArray(a)&&Array.isArray(b)&&a.length===b.length&&a.every((v,i)=>Number.isFinite(v)&&Math.abs(v-b[i])<=tol);
const difference=rows=>rows.slice(1).map((row,i)=>row.map((v,j)=>v-rows[i][j]));
export function bezierValue(rows,u) {
  rows=rows.map(row=>row.slice());
  while(rows.length>1)rows=rows.slice(1).map((row,i)=>row.map((v,j)=>(1-u)*rows[i][j]+u*v));
  return rows[0];
}
export function bezierBound(rows,depth=4) {
  if(!depth)return Math.max(...rows.map(row=>Math.hypot(...row)));
  const levels=[rows];
  while(levels.at(-1).length>1){const last=levels.at(-1);levels.push(last.slice(1).map((row,i)=>row.map((v,j)=>(v+last[i][j])/2)));}
  return Math.max(bezierBound(levels.map(row=>row[0]),depth-1),bezierBound(levels.slice().reverse().map(row=>row.at(-1)),depth-1));
}
function derivatives(segment) {
  const velocity=difference(segment.controls_deg).map(row=>row.map(v=>5*v*radians/segment.duration_s));
  const acceleration=difference(velocity).map(row=>row.map(v=>4*v/segment.duration_s));
  return {velocity,acceleration};
}
function evaluate(segment,time) {
  const u=Math.max(0,Math.min(1,(time-segment.start_time_s)/segment.duration_s));
  const {velocity,acceleration}=derivatives(segment);
  return {q:positionFromVelocity(segment.controls_deg[0],elapsed=>bezierValue(velocity,elapsed/segment.duration_s),u*segment.duration_s),
    velocity_rad_s:bezierValue(velocity,u),acceleration_rad_s2:bezierValue(acceleration,u)};
}
export function createBezierSampler(frames,smoothing,{frameIndex,advanceToBoundary}) {
  const segments=smoothing.segments,endTime=frames.at(-1).time;
  if(smoothing.method!=='collision_checked_bezier_c2'||!smoothing.applied||!smoothing.collision_rechecked||!smoothing.ends_at_rest
    ||!Array.isArray(segments)||frames[0].time!==0||!Number.isFinite(smoothing.speed_limit_rad_s)
    ||smoothing.speed_limit_rad_s<=0||smoothing.speed_limit_rad_s>1||smoothing.acceleration_limit_rad_s2!==3
    ||!close(smoothing.start_velocity_rad_s,smoothing.start_velocity_rad_s)||smoothing.start_velocity_rad_s.length!==6
    ||!close(smoothing.start_acceleration_rad_s2,smoothing.start_acceleration_rad_s2)||smoothing.start_acceleration_rad_s2.length!==6)
    throw new Error('Missing checked continuous mouse timing');
  let expectedStart=0,previousVelocity=smoothing.start_velocity_rad_s,previousAcceleration=smoothing.start_acceleration_rad_s2;
  for(const segment of segments) {
    const {start_frame:start,end_frame:end,start_time_s:time,duration_s:duration,controls_deg:controls}=segment;
    if(!Number.isInteger(start)||!Number.isInteger(end)||start!==expectedStart||end<=start||end>=frames.length
      ||!Number.isFinite(time)||!Number.isFinite(duration)||duration<=0||!Array.isArray(controls)||controls.length!==6
      ||controls.some(row=>!Array.isArray(row)||row.length!==6||!row.every(Number.isFinite))
      ||Math.abs(time-frames[start].time)>1e-9||Math.abs(time+duration-frames[end].time)>1e-9
      ||!close(controls[0],frames[start].q)||!close(controls.at(-1),frames[end].q))throw new Error('Invalid continuous mouse curve');
    const {velocity,acceleration}=derivatives(segment);
    if(bezierBound(velocity)>smoothing.speed_limit_rad_s+1e-8||bezierBound(acceleration)>3+1e-8)
      throw new Error('Continuous mouse curve exceeds motion limits');
    if(!close(velocity[0],previousVelocity,1e-8)||!close(acceleration[0],previousAcceleration,1e-8)
      ||!close(velocity.at(-1),zero(),1e-8)||!close(acceleration.at(-1),zero(),1e-8))throw new Error('Discontinuous mouse trajectory join');
    for(let i=start;i<=end;i++)if(!close(evaluate(segment,frames[i].time).q,frames[i].q))throw new Error('Mouse curve differs from checked references');
    expectedStart=end;previousVelocity=velocity.at(-1);previousAcceleration=acceleration.at(-1);
  }
  if((segments.length&&expectedStart!==frames.length-1)||(!segments.length&&
    (frames.some(frame=>!close(frame.q,frames[0].q))||!close(previousVelocity,zero(),1e-8)||!close(previousAcceleration,zero(),1e-8))))
    throw new Error('Incomplete continuous mouse timing');
  const boundaries=segments.length?segments.map(s=>s.start_time_s+s.duration_s):[endTime];
  return {smooth:true,endTime,
    sample(time){time=Math.max(0,Math.min(endTime,time));const segment=segments.find(s=>time<=s.start_time_s+s.duration_s+1e-12)||segments.at(-1);
      return {...(segment?evaluate(segment,time):{q:frames[0].q.slice(),velocity_rad_s:zero(),acceleration_rad_s2:zero()}),time,index:frameIndex(frames,time)};},
    advance:(current,dt)=>advanceToBoundary(boundaries,current,dt),
    velocityPoints:segments.flatMap(segment=>Array.from({length:Math.ceil(segment.duration_s/.02)+1},(_,i)=>{
      const time=segment.start_time_s+Math.min(segment.duration_s,i*.02);return velocityPoint(time,evaluate(segment,time).velocity_rad_s);
    })).concat(segments.length?[]:[velocityPoint(endTime,zero())])};
}
