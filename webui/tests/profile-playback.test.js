import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createTrajectorySampler} from '../src/trajectory-playback.js';
import {profilePieces} from '../src/profile-playback.js';
import {shapeValue} from '../src/reference-shape.js';
const root=new URL('../../output/shared_tracking_profile_20261008/',import.meta.url);
const fixture=name=>JSON.parse(readFileSync(new URL(name,root),'utf8'));
const near=(a,b,tol=1e-7)=>a.forEach((v,j)=>assert.ok(Math.abs(v-b[j])<tol,`${v} != ${b[j]}`));
test('shared resting tracking profile matches fixed-target rise/hold/fall and every preview frame',()=>{
  const r=fixture('resting_fixture.json'),sampler=createTrajectorySampler(r.frames,r.smoothing),s=r.smoothing.segments[0];
  for(const frame of r.frames)near(sampler.sample(frame.time).q,frame.q);
  for(const u of [0,.025,.05,.4,.9,.95,1]){
    const m=sampler.sample(s.duration_s*u),value=shapeValue(u,r.smoothing.reference_shape),delta=s.end_deg.map((v,j)=>v-s.start_deg[j]);
    near(m.q,delta.map((v,j)=>s.start_deg[j]+value.progress*v));
    near(m.velocity_rad_s,delta.map(v=>v*Math.PI/180*value.rate/s.duration_s));
  }
});
test('moving profile playback retains q/v/a boundary and reconstructs bounded polynomial pieces',()=>{
  const r=fixture('moving_fixture.json'),sampler=createTrajectorySampler(r.frames,r.smoothing),first=r.smoothing.segments[0];
  near(sampler.sample(0).q,r.frames[0].q);near(sampler.sample(0).velocity_rad_s,r.smoothing.start_velocity_rad_s);
  near(sampler.sample(0).acceleration_rad_s2,r.smoothing.start_acceleration_rad_s2);
  for(const frame of r.frames)near(sampler.sample(frame.time).q,frame.q);
  assert.equal(profilePieces(first,r.smoothing.reference_shape).length,3);
  near(sampler.sample(sampler.endTime).velocity_rad_s,Array(6).fill(0));
});
test('unchecked changes to ramp settings and initial motion fail playback',()=>{
  for(const corrupt of [r=>r.smoothing.reference_shape.rise_percent=20,r=>r.smoothing.segments[0].start_velocity_rad_s[0]+=.1]){
    const r=fixture('moving_fixture.json');corrupt(r);assert.throws(()=>createTrajectorySampler(r.frames,r.smoothing));
  }
});
