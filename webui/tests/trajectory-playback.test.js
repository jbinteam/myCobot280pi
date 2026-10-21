import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createTrajectorySampler} from '../src/trajectory-playback.js';
import {bezierValue} from '../src/bezier-playback.js';

const degrees=180/Math.PI;
const fraction=u=>10*u**3-15*u**4+6*u**5;
function fixture() {
  const start=Array(6).fill(0),corner=[.1*degrees,0,0,0,0,0],end=[.1*degrees,.1*degrees,0,0,0,0];
  const frames=[{q:start,time:0}];
  for(const [offset,a,b] of [[0,start,corner],[1,corner,end]])for(let i=1;i<=10;i++)
    frames.push({time:offset+i/10,q:a.map((q,j)=>q+fraction(i/10)*(b[j]-q))});
  const smoothing={method:'collision_checked_shortcuts_quintic',collision_rechecked:true,starts_and_ends_at_rest:true,
    speed_limit_rad_s:1,acceleration_limit_rad_s2:3,
    segments:[{start_frame:0,end_frame:10,start_time_s:0,duration_s:1,start_deg:start,end_deg:corner},
      {start_frame:10,end_frame:20,start_time_s:1,duration_s:1,start_deg:corner,end_deg:end}]};
  return {frames,smoothing};
}
test('continuous quintic rendering preserves checked points, limits and corner geometry',()=>{
  const {frames,smoothing}=fixture(),sampler=createTrajectorySampler(frames,smoothing);
  for(const frame of frames)frame.q.forEach((q,j)=>assert.ok(Math.abs(q-sampler.sample(frame.time).q[j])<1e-9));
  for(let i=0;i<=2000;i++) {
    const value=sampler.sample(i/1000);
    assert.ok(Math.hypot(...value.velocity_rad_s)<=1);assert.ok(Math.hypot(...value.acceleration_rad_s2)<=3);
    if(i<=1000){assert.equal(value.q[1],0);assert.ok(value.q[0]>=0&&value.q[0]<=.1*degrees);}
    else assert.equal(value.q[0],.1*degrees);
  }
  assert.notEqual(sampler.sample(.123).q[0],sampler.sample(.1).q[0]);
  for(const time of [0,1,2]) {
    assert.ok(Math.hypot(...sampler.sample(time).velocity_rad_s)<1e-12);
    assert.ok(Math.hypot(...sampler.sample(time).acceleration_rad_s2)<1e-12);
  }
  assert.equal(sampler.advance(.9,.5),1);assert.equal(sampler.advance(1,.5),1.5);
  // Derivatives are continuous on both sides of the retained corner.
  for(const time of [1-1e-6,1+1e-6])assert.ok(Math.hypot(...sampler.sample(time).acceleration_rad_s2)<1e-5);
});
test('refuses changed geometry, unmatched frames and unsafe analytic timing',()=>{
  const {frames,smoothing}=fixture();
  for(const change of [s=>s.segments[0].end_deg[0]+=1,s=>s.segments.pop(),
    s=>s.segments[0].duration_s=.1,s=>s.collision_rechecked=false,
    s=>s.speed_limit_rad_s=.01]) {
    const bad=structuredClone(smoothing);change(bad);
    assert.throws(()=>createTrajectorySampler(frames,bad));
  }
  const bad=structuredClone(frames);bad[5].q[2]=.1;
  assert.throws(()=>createTrajectorySampler(bad,smoothing));
});
test('ordinary IK specification timing is smooth and respects the manufacturer joint cap',()=>{
  const {frames,smoothing}=fixture();
  const timing={...smoothing,method:'specification_checked_quintic',specification_checked:true,
    collision_rechecked:false,speed_limit_rad_s:2.6};
  const sampler=createTrajectorySampler(frames,timing);
  assert.equal(sampler.smooth,true);
  for(const time of [0,1,2]) {
    assert.ok(Math.hypot(...sampler.sample(time).velocity_rad_s)<1e-12);
    assert.ok(Math.hypot(...sampler.sample(time).acceleration_rad_s2)<1e-12);
  }
  for(const invalid of [{...timing,speed_limit_rad_s:3},
    {...timing,specification_checked:false},{...timing,acceleration_limit_rad_s2:4}])
    assert.throws(()=>createTrajectorySampler(frames,invalid));
});
test('fixed-target rendering uses per-joint selected allowance and signed coordinated velocities',()=>{
  const start=Array(6).fill(0),end=[0,0,0,0,-.1*degrees,.1*degrees],duration=2;
  const frames=Array.from({length:21},(_,i)=>({time:i/10,q:end.map(q=>fraction(i/20)*q)}));
  const timing={method:'specification_checked_quintic',specification_checked:true,
    starts_and_ends_at_rest:true,speed_limit_kind:'per_joint',speed_percent:5,
    joint_velocity_allowance_deg_s:7.5,speed_limit_rad_s:7.5/degrees,acceleration_limit_rad_s2:3,
    segments:[{start_frame:0,end_frame:20,start_time_s:0,duration_s:duration,start_deg:start,end_deg:end}]};
  const sampler=createTrajectorySampler(frames,timing),peak=sampler.sample(1).velocity_rad_s;
  assert.ok(Math.hypot(...peak)>timing.speed_limit_rad_s); // Vector norm is not the per-joint allowance.
  assert.ok(Math.max(...peak.map(Math.abs))<=timing.speed_limit_rad_s);
  assert.equal(peak[4],-peak[5]);
  assert.throws(()=>createTrajectorySampler(frames,{...timing,speed_percent:80}));
  const fast={...timing,speed_percent:80,joint_velocity_allowance_deg_s:120,speed_limit_rad_s:120/degrees};
  assert.equal(createTrajectorySampler(frames,fast).smooth,true);
  assert.throws(()=>createTrajectorySampler(frames,{...fast,speed_percent:81,joint_velocity_allowance_deg_s:121.5,speed_limit_rad_s:121.5/degrees}));
});
test('a checked held pose stays at rest and legacy interpolation preserves its corners',()=>{
  const {frames,smoothing}=fixture();
  const hold=createTrajectorySampler([frames[0]],{...smoothing,segments:[]});
  assert.deepEqual(hold.sample(10).q,frames[0].q);assert.deepEqual(hold.sample(10).velocity_rad_s,Array(6).fill(0));
  const legacy=createTrajectorySampler([frames[0],frames[10],frames[20]]);
  assert.equal(legacy.advance(.9,.5),1);assert.equal(legacy.sample(.5).q[1],0);
});

function movingFixture() {
  const row=x=>[x*degrees,0,0,0,0,0],controls=[row(0),row(.02),row(.05),row(.2),row(.2),row(.2)];
  const frames=Array.from({length:51},(_,i)=>({time:i*.02,q:bezierValue(controls,i/50)}));
  const smoothing={method:'collision_checked_bezier_c2',applied:true,collision_rechecked:true,ends_at_rest:true,
    speed_limit_rad_s:1,acceleration_limit_rad_s2:3,start_velocity_rad_s:[.1,0,0,0,0,0],start_acceleration_rad_s2:[.2,0,0,0,0,0],
    segments:[{start_frame:0,end_frame:50,start_time_s:0,duration_s:1,controls_deg:controls}]};
  return {frames,smoothing};
}
test('continuous mouse curves carry boundary motion and finish at rest',()=>{
  const {frames,smoothing}=movingFixture(),sampler=createTrajectorySampler(frames,smoothing);
  assert.ok(Math.abs(sampler.sample(0).velocity_rad_s[0]-.1)<1e-12);
  assert.ok(Math.abs(sampler.sample(0).acceleration_rad_s2[0]-.2)<1e-12);
  for(let i=0;i<=1000;i++){
    const value=sampler.sample(i/1000);assert.ok(Math.hypot(...value.velocity_rad_s)<=1);assert.ok(Math.hypot(...value.acceleration_rad_s2)<=3);
  }
  assert.ok(Math.hypot(...sampler.sample(1).velocity_rad_s)<1e-12);
  assert.ok(Math.hypot(...sampler.sample(1).acceleration_rad_s2)<1e-12);
  for(const frame of frames)frame.q.forEach((q,j)=>assert.ok(Math.abs(sampler.sample(frame.time).q[j]-q)<1e-10));
});
test('continuous mouse sampler rejects tampered boundary motion, curves and limits',()=>{
  const {frames,smoothing}=movingFixture();
  for(const mutate of [s=>s.start_velocity_rad_s[0]=0,s=>s.start_acceleration_rad_s2[0]=0,
    s=>s.segments[0].controls_deg[2][0]+=1,s=>s.ends_at_rest=false,s=>s.speed_limit_rad_s=.01]){
    const bad=structuredClone(smoothing);mutate(bad);assert.throws(()=>createTrajectorySampler(frames,bad));
  }
});

test('waypoint velocity profiles preserve signs, timing, corners and seeking',()=>{
  const start=[10,20,30,40,50,60];
  const frames=[{time:0,q:start},{time:.5,q:[11,18,30,40,50,60]},
    {time:2,q:[8,18,30,40,50,60]}];
  const sampler=createTrajectorySampler(frames);
  assert.ok(sampler.sample(.25).velocity_rad_s[0]>0);
  assert.ok(sampler.sample(.25).velocity_rad_s[1]<0);
  assert.ok(sampler.sample(1).velocity_rad_s[0]<0);
  for(const time of [1.25,.25,2,0,.5,.25]) {
    const sample=sampler.sample(time);
    const expected=time<=.5?[10+2*time,20-4*time,30,40,50,60]:[11-2*(time-.5),18,30,40,50,60];
    expected.forEach((q,j)=>assert.ok(Math.abs(sample.q[j]-q)<1e-12));
  }
  assert.deepEqual(sampler.sample(2).velocity_rad_s,Array(6).fill(0));
  assert.equal(sampler.advance(.25,1),.5);
  assert.equal(sampler.velocityPoints.length,5);
  for(const point of sampler.velocityPoints)assert.equal(point.speed_rad_s,Math.hypot(...point.velocity_rad_s));
});

test('checked profiles expose all six signed derivatives used to reconstruct motion',()=>{
  const {frames,smoothing}=fixture();
  // Move the first joint's travel into a negative third-joint movement.
  for(const frame of frames){frame.q[2]=-frame.q[0];frame.q[0]=0;}
  for(const segment of smoothing.segments)for(const key of ['start_deg','end_deg']){
    // Segment endpoints share arrays with frames in this fixture.
    segment[key]=frames[key==='start_deg'?segment.start_frame:segment.end_frame].q.slice();
  }
  for(const sampler of [createTrajectorySampler(frames,smoothing),
    createTrajectorySampler(movingFixture().frames,movingFixture().smoothing)]) {
    for(const point of sampler.velocityPoints){
      assert.equal(point.velocity_rad_s.length,6);
      point.velocity_rad_s.forEach((v,j)=>assert.ok(Math.abs(v-sampler.sample(point.time).velocity_rad_s[j])<1e-12));
    }
    const t=.37,epsilon=1e-5,before=sampler.sample(t-epsilon),after=sampler.sample(t+epsilon);
    sampler.sample(t).velocity_rad_s.forEach((v,j)=>
      assert.ok(Math.abs((after.q[j]-before.q[j])/degrees/(2*epsilon)-v)<1e-8));
  }
  assert.ok(createTrajectorySampler(frames,smoothing).sample(.5).velocity_rad_s[2]<0);
});

test('adjustable ramps preserve checked frames, derivatives and speed limits',async()=>{
  const {readFileSync}=await import('node:fs');
  const fixture=JSON.parse(readFileSync('../output/velocity_shape_ui_fixture.json','utf8'));
  const {frames,smoothing}=fixture.result,sampler=createTrajectorySampler(frames,smoothing);
  for(const frame of frames)frame.q.forEach((q,j)=>assert.ok(Math.abs(q-sampler.sample(frame.time).q[j])<1e-8));
  const segment=smoothing.segments[0],r=smoothing.reference_shape.rise_percent/100,d=smoothing.reference_shape.fall_percent/100;
  for(const u of [0,r,1-d,1]) {
    const value=sampler.sample(u*segment.duration_s);
    assert.ok(Math.hypot(...value.acceleration_rad_s2)<1e-9);
    if(u===0||u===1)assert.ok(Math.hypot(...value.velocity_rad_s)<1e-9);
  }
  const first=sampler.sample((r+.05)*segment.duration_s).velocity_rad_s;
  const second=sampler.sample((1-d-.05)*segment.duration_s).velocity_rad_s;
  first.forEach((v,j)=>assert.ok(Math.abs(v-second[j])<1e-10));
  for(let i=0;i<=1000;i++) {
    const time=sampler.endTime*i/1000,value=sampler.sample(time),h=1e-5;
    assert.ok(Math.max(...value.velocity_rad_s.map(Math.abs))<=smoothing.speed_limit_rad_s+1e-9);
    if(i>0&&i<1000) {
      const before=sampler.sample(time-h).q,after=sampler.sample(time+h).q;
      value.velocity_rad_s.forEach((v,j)=>assert.ok(Math.abs((after[j]-before[j])*Math.PI/180/(2*h)-v)<1e-6));
    }
  }
  for(const reference_shape of [{kind:'unknown'},{kind:'smooth_trapezoid',rise_percent:80,fall_percent:80}])
    assert.throws(()=>createTrajectorySampler(frames,{...smoothing,reference_shape}));
  const altered=structuredClone(frames);altered[3].q[0]+=.1;
  assert.throws(()=>createTrajectorySampler(altered,smoothing));
});
