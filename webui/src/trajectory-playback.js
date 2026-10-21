// Evaluate the backend's checked straight segments with their exact quintic timing.
// Arbitrary splines between output frames could cut through a checked corner.
import {shapeSettings,shapePeaks,shapeValue} from './reference-shape.js';
import {createProfileSampler,PROFILE_METHOD} from './profile-playback.js';
import {createBezierSampler} from './bezier-playback.js';
import {positionFromVelocity,velocityPoint} from './joint-velocity.js';
const radians=Math.PI/180;
const close=(a,b,tolerance=1e-6)=>a.length===b.length&&a.every((v,i)=>Math.abs(v-b[i])<=tolerance);
const zero=()=>Array(6).fill(0);
export function frameIndex(frames,time) {
  let low=0,high=frames.length;
  while(low<high){const mid=(low+high)>>>1;if(frames[mid].time<=time)low=mid+1;else high=mid;}
  return Math.max(0,low-1);
}
export function linearSample(frames,time) {
  if(time<=frames[0].time)return frames[0].q.slice();
  const i=frameIndex(frames,time),a=frames[i],b=frames[i+1];
  if(!b)return a.q.slice();
  const velocity=a.q.map((q,j)=>(b.q[j]-q)*radians/(b.time-a.time));
  return positionFromVelocity(a.q,()=>velocity,time-a.time);
}
export function advanceToBoundary(boundaries,current,dt) {
  const next=boundaries.find(time=>time>current+1e-12);
  return next===undefined?boundaries.at(-1):Math.min(next,current+Math.max(0,dt));
}
function evaluate(segment,time,shape) {
  const u=Math.max(0,Math.min(1,(time-segment.start_time_s)/segment.duration_s));
  if(shape?.kind==='smooth_trapezoid') {
    const value=shapeValue(u,shape),delta=segment.end_deg.map((q,j)=>q-segment.start_deg[j]);
    return {q:delta.map((d,j)=>segment.start_deg[j]+value.progress*d),
      velocity_rad_s:delta.map(d=>d*radians*value.rate/segment.duration_s),
      acceleration_rad_s2:delta.map(d=>d*radians*value.acceleration/segment.duration_s**2)};
  }
  const rate=30*u**2*(1-u)**2/segment.duration_s;
  const acceleration=60*u*(1-u)*(1-2*u)/segment.duration_s**2;
  const delta=segment.end_deg.map((q,j)=>q-segment.start_deg[j]);
  const velocityAt=elapsed=>{
    const phase=elapsed/segment.duration_s;
    return delta.map(d=>d*radians*30*phase**2*(1-phase)**2/segment.duration_s);
  };
  return {q:positionFromVelocity(segment.start_deg,velocityAt,u*segment.duration_s),
    velocity_rad_s:delta.map(d=>d*radians*rate),acceleration_rad_s2:delta.map(d=>d*radians*acceleration)};
}
export function createTrajectorySampler(frames,smoothing=null) {
  if(!Array.isArray(frames)||!frames.length||frames.some((f,i)=>
    !Number.isFinite(f.time)||!Array.isArray(f.q)||f.q.length!==6||!f.q.every(Number.isFinite)
    ||(i&&f.time<=frames[i-1].time)))throw new Error('Invalid playback references');
  const endTime=frames.at(-1).time;
  if(smoothing?.method===PROFILE_METHOD)return createProfileSampler(frames,smoothing,{frameIndex,advanceToBoundary});
  if(smoothing?.method==='collision_checked_bezier_c2')return createBezierSampler(frames,smoothing,{frameIndex,advanceToBoundary});
  if(!smoothing) {
    const velocities=frames.slice(1).map((b,i)=>b.q.map((q,j)=>(q-frames[i].q[j])*radians/(b.time-frames[i].time)));
    return {smooth:false,endTime,
      sample(time) {
        time=Math.max(frames[0].time,Math.min(endTime,time));
        const index=frameIndex(frames,time);
        // Velocity jumps at raw waypoint joins; acceleration is zero within
        // each constant-velocity interval, not a C2 motion guarantee.
        return {q:linearSample(frames,time),time,index,velocity_rad_s:velocities[index]?.slice()||zero(),acceleration_rad_s2:zero()};
      },
      advance:(current,dt)=>advanceToBoundary(frames.map(f=>f.time),current,dt),
      velocityPoints:velocities.length?velocities.flatMap((v,i)=>[
        velocityPoint(frames[i].time,v),velocityPoint(frames[i+1].time,v)
      ]).concat(velocityPoint(endTime,zero())):[velocityPoint(endTime,zero())]};
  }
  const segments=smoothing.segments;
  const shaped=smoothing.method==='specification_checked_velocity_shape';
  const specificationTiming=shaped||smoothing.method==='specification_checked_quintic';
  const referenceShape=shapeSettings(smoothing.reference_shape),peaks=shapePeaks(referenceShape);
  if(shaped&&(referenceShape.kind!=='smooth_trapezoid'||smoothing.speed_limit_kind!=='per_joint')
    ||!shaped&&referenceShape.kind!=='quintic')throw new Error('Invalid reference shape timing');
  const perJoint=specificationTiming&&smoothing.speed_limit_kind==='per_joint';
  if((!specificationTiming&&smoothing.method!=='collision_checked_shortcuts_quintic')
    ||(specificationTiming?!smoothing.specification_checked:!smoothing.collision_rechecked)
    ||!smoothing.starts_and_ends_at_rest||!Array.isArray(segments)||frames[0].time!==0
    ||!Number.isFinite(smoothing.speed_limit_rad_s)||smoothing.speed_limit_rad_s<=0||smoothing.speed_limit_rad_s>(perJoint?150*radians*.8:specificationTiming?2.6:1)+1e-12
    ||!Number.isFinite(smoothing.acceleration_limit_rad_s2)||smoothing.acceleration_limit_rad_s2<=0||smoothing.acceleration_limit_rad_s2>(specificationTiming?200*radians:3))
    throw new Error('Missing checked continuous timing');
  if(perJoint&&(!Number.isInteger(smoothing.speed_percent)||smoothing.speed_percent<1||smoothing.speed_percent>80
    ||Math.abs(smoothing.speed_limit_rad_s-150*radians*smoothing.speed_percent/100)>1e-12
    ||Math.abs(smoothing.joint_velocity_allowance_deg_s-150*smoothing.speed_percent/100)>1e-9))
    throw new Error('Invalid fixed-target speed allowance');
  let expectedStart=0;
  for(const segment of segments) {
    const {start_frame:start,end_frame:end,start_time_s:time,duration_s:duration,start_deg:a,end_deg:b}=segment;
    if(!Number.isInteger(start)||!Number.isInteger(end)||start!==expectedStart||end<=start||end>=frames.length
      ||!Number.isFinite(time)||!Number.isFinite(duration)||duration<=0
      ||!Array.isArray(a)||!Array.isArray(b)||a.length!==6||b.length!==6||!a.every(Number.isFinite)||!b.every(Number.isFinite)
      ||Math.abs(time-frames[start].time)>1e-9||Math.abs(time+duration-frames[end].time)>1e-9
      ||!close(a,frames[start].q)||!close(b,frames[end].q))throw new Error('Invalid continuous segment');
    const displacement=b.map((q,j)=>(q-a[j])*radians);
    const distance=Math.hypot(...displacement),velocityDistance=perJoint?Math.max(...displacement.map(Math.abs)):distance;
    if(peaks.rate*velocityDistance/duration>smoothing.speed_limit_rad_s+1e-8
      ||peaks.acceleration*distance/duration**2>smoothing.acceleration_limit_rad_s2+1e-8)
      throw new Error('Continuous timing exceeds motion limits');
    for(let i=start;i<=end;i++)if(!close(evaluate(segment,frames[i].time,referenceShape).q,frames[i].q))
      throw new Error('Continuous timing differs from checked references');
    expectedStart=end;
  }
  if((segments.length&&expectedStart!==frames.length-1)
    ||(!segments.length&&frames.some(f=>!close(f.q,frames[0].q))))throw new Error('Incomplete continuous timing');
  const boundaries=segments.length?segments.map(s=>s.start_time_s+s.duration_s):[endTime];
  return {smooth:true,endTime,
    sample(time) {
      time=Math.max(0,Math.min(endTime,time));
      const segment=segments.find(s=>time<=s.start_time_s+s.duration_s+1e-12)||segments.at(-1);
      const value=segment?evaluate(segment,time,referenceShape):{q:frames[0].q.slice(),velocity_rad_s:zero(),acceleration_rad_s2:zero()};
      return {...value,time,index:frameIndex(frames,time)};
    },
    // Visit retained corners even after a delayed render; each corner is at rest.
    advance:(current,dt)=>advanceToBoundary(boundaries,current,dt),
    velocityPoints:segments.flatMap(s=>{
      const steps=Math.max(2,Math.ceil(s.duration_s/.02));
      return Array.from({length:steps+1},(_,i)=>{
        const time=s.start_time_s+s.duration_s*i/steps;
        return velocityPoint(time,evaluate(s,time,referenceShape).velocity_rad_s);
      });
    }).concat(segments.length?[]:[velocityPoint(endTime,zero())])};
}
