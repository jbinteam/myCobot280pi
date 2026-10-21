import {test} from 'node:test';
import assert from 'node:assert/strict';
import {hardwareVelocityProfiles} from '../src/hardware-velocity.js';

test('hardware profiles use signed radians and actual telemetry intervals',()=>{
  const start=[0,10,20,30,40,50],end=[2,9,20,30,40,50];
  const {reference,measured}=hardwareVelocityProfiles([
    {time:0,q:start,reference_velocity_deg_s:[8,-4,0,0,0,0]},
    {time:.5,q:end,reference_velocity_deg_s:[8,-4,0,0,0,0],command_velocity_deg_s:[4,-2,0,0,0,0]}
  ]);
  assert.equal(reference.length,2);assert.equal(measured.length,1);
  assert.equal(measured[0].time,.25);
  assert.equal(measured[0].velocity_rad_s[0],4*Math.PI/180);
  assert.equal(measured[0].velocity_rad_s[1],-2*Math.PI/180);
  assert.deepEqual(reference[1].velocity_rad_s,measured[0].velocity_rad_s);
});

test('missing, duplicate, stale and invalid telemetry does not invent velocities',()=>{
  const q=Array(6).fill(0);
  assert.deepEqual(hardwareVelocityProfiles([]),{reference:[],measured:[]});
  const result=hardwareVelocityProfiles([{time:0,q},{time:0,q},{time:3,q},
    {time:3.1,q:[NaN,0,0,0,0,0],reference_velocity_deg_s:[Infinity,0,0,0,0,0]},
    {time:3.2,q},{time:3.3,q}]);
  assert.equal(result.reference.length,0);assert.equal(result.measured.length,1);
  assert.deepEqual(result.measured[0].velocity_rad_s,Array(6).fill(0));
});
