// Shared fixed-target time law, with C2 boundary terms for a moving cursor join.
import {shapeSettings,shapeValue} from './reference-shape.js';
import {bezierBound} from './bezier-playback.js';
import {velocityPoint} from './joint-velocity.js';
const rad=Math.PI/180, zero=()=>Array(6).fill(0);
const close=(a,b,tol=1e-7)=>Array.isArray(a)&&Array.isArray(b)&&a.length===b.length&&a.every((v,i)=>Number.isFinite(v)&&Math.abs(v-b[i])<=tol);
export const PROFILE_METHOD='collision_checked_velocity_profile_c2';
export function profileMotion(segment,time,shape){
  const T=segment.duration_s,u=Math.max(0,Math.min(1,(time-segment.start_time_s)/T));
  const value=shapeValue(u,shape),delta=segment.end_deg.map((v,j)=>(v-segment.start_deg[j])*rad);
  const h=u-6*u**3+8*u**4-3*u**5,dh=1-18*u**2+32*u**3-15*u**4,ddh=-36*u+96*u**2-60*u**3;
  const k=.5*(u**2-3*u**3+3*u**4-u**5),dk=u-4.5*u**2+6*u**3-2.5*u**4,ddk=1-9*u+18*u**2-10*u**3;
  const v=segment.start_velocity_rad_s,a=segment.start_acceleration_rad_s2;
  return {q:delta.map((d,j)=>segment.start_deg[j]+(d*value.progress+v[j]*T*h+a[j]*T*T*k)/rad),
    velocity_rad_s:delta.map((d,j)=>d*value.rate/T+v[j]*dh+a[j]*T*dk),
    acceleration_rad_s2:delta.map((d,j)=>d*value.acceleration/(T*T)+v[j]*ddh/T+a[j]*ddk)};
}
const choose=[1,5,10,10,5,1];
function controlsFor(samples){
  const rows=samples.map((sample,i)=>{const u=i/5;return [...choose.map((c,j)=>c*u**j*(1-u)**(5-j)),...sample];});
  for(let i=0;i<6;i++){
    let pivot=i;for(let j=i+1;j<6;j++)if(Math.abs(rows[j][i])>Math.abs(rows[pivot][i]))pivot=j;
    [rows[i],rows[pivot]]=[rows[pivot],rows[i]];
    const divisor=rows[i][i];rows[i]=rows[i].map(x=>x/divisor);
    for(let j=0;j<6;j++)if(j!==i){const f=rows[j][i];rows[j]=rows[j].map((x,k)=>x-f*rows[i][k]);}
  }
  return rows.map(row=>row.slice(6));
}
export function profilePieces(segment,shape){
  const r=shape.rise_percent/100,f=shape.fall_percent/100;
  const intervals=shape.kind==='quintic'?[[0,1]]:[[0,r],...(1-f-r>1e-12?[[r,1-f]]:[]),[1-f,1]];
  return intervals.map(([lo,hi])=>({duration_s:(hi-lo)*segment.duration_s,
    controls_deg:controlsFor(Array.from({length:6},(_,i)=>profileMotion(segment,segment.start_time_s+segment.duration_s*(lo+(hi-lo)*i/5),shape).q))}));
}
export function createProfileSampler(frames,smoothing,{frameIndex,advanceToBoundary}){
  const shape=shapeSettings(smoothing.reference_shape),segments=smoothing.segments,endTime=frames.at(-1).time;
  if(smoothing.method!==PROFILE_METHOD||!smoothing.applied||!smoothing.collision_rechecked||!smoothing.ends_at_rest
    ||!smoothing.specification_checked||!Array.isArray(segments)||frames[0].time!==0
    ||!Number.isFinite(smoothing.speed_limit_rad_s)||smoothing.speed_limit_rad_s<=0||smoothing.speed_limit_rad_s>1
    ||smoothing.acceleration_limit_rad_s2!==3)throw new Error('Missing checked shared velocity profile');
  let start=0,v=smoothing.start_velocity_rad_s,a=smoothing.start_acceleration_rad_s2;
  for(const s of segments){
    if(!Number.isInteger(s.start_frame)||!Number.isInteger(s.end_frame)||s.start_frame!==start||s.end_frame<=start||s.end_frame>=frames.length
      ||!Number.isFinite(s.start_time_s)||!Number.isFinite(s.duration_s)||s.duration_s<=0
      ||Math.abs(s.start_time_s-frames[start].time)>1e-8||Math.abs(s.start_time_s+s.duration_s-frames[s.end_frame].time)>1e-8
      ||!close(s.start_deg,frames[start].q)||!close(s.end_deg,frames[s.end_frame].q)
      ||!close(s.start_velocity_rad_s,v,1e-8)||!close(s.start_acceleration_rad_s2,a,1e-8))throw new Error('Invalid shared-profile segment or join');
    for(const p of profilePieces(s,shape)){
      const velocity=p.controls_deg.slice(1).map((row,i)=>row.map((q,j)=>5*(q-p.controls_deg[i][j])*rad/p.duration_s));
      const acceleration=velocity.slice(1).map((row,i)=>row.map((q,j)=>4*(q-velocity[i][j])/p.duration_s));
      if(bezierBound(velocity)>smoothing.speed_limit_rad_s+1e-7||bezierBound(acceleration)>3+1e-7)throw new Error('Shared profile exceeds motion limits');
    }
    for(let i=start;i<=s.end_frame;i++)if(!close(profileMotion(s,frames[i].time,shape).q,frames[i].q))throw new Error('Shared profile differs from checked references');
    const end=profileMotion(s,s.start_time_s+s.duration_s,shape);
    if(!close(end.velocity_rad_s,zero(),1e-8)||!close(end.acceleration_rad_s2,zero(),1e-8))throw new Error('Shared profile lacks a resting endpoint');
    v=end.velocity_rad_s;a=end.acceleration_rad_s2;start=s.end_frame;
  }
  if((segments.length&&start!==frames.length-1)||(!segments.length&&(frames.length!==1||!close(v,zero())||!close(a,zero()))))throw new Error('Incomplete shared profile');
  const boundaries=segments.length?segments.map(s=>s.start_time_s+s.duration_s):[endTime];
  return {smooth:true,endTime,sample(time){time=Math.max(0,Math.min(endTime,time));const s=segments.find(s=>time<=s.start_time_s+s.duration_s+1e-12)||segments.at(-1);
    return {...(s?profileMotion(s,time,shape):{q:frames[0].q.slice(),velocity_rad_s:zero(),acceleration_rad_s2:zero()}),time,index:frameIndex(frames,time)};},
    advance:(current,dt)=>advanceToBoundary(boundaries,current,dt),
    velocityPoints:segments.flatMap(s=>Array.from({length:Math.ceil(s.duration_s/.02)+1},(_,i)=>{
      const t=s.start_time_s+Math.min(s.duration_s,i*.02);return velocityPoint(t,profileMotion(s,t,shape).velocity_rad_s);
    })).concat(segments.length?[]:[velocityPoint(endTime,zero())])};
}
