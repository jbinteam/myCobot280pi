import {velocityPoint} from './joint-velocity.js';
const radians=Math.PI/180;
const vector=value=>Array.isArray(value)&&value.length===6&&value.every(Number.isFinite);

// Firmware reports joint angles. Estimate velocity over actual telemetry
// intervals rather than treating the SDK's speed setting as measured motion.
export function hardwareVelocityProfiles(samples) {
  const reference=[],measured=[];
  samples.forEach((sample,i)=>{
    if(!Number.isFinite(sample.time))return;
    const velocity=sample.command_velocity_deg_s??sample.reference_velocity_deg_s;
    if(vector(velocity))reference.push(velocityPoint(sample.time,velocity.map(v=>v*radians)));
    const previous=samples[i-1],dt=sample.time-previous?.time;
    if(previous&&dt>0&&dt<=2&&vector(sample.q)&&vector(previous.q))
      measured.push(velocityPoint((previous.time+sample.time)/2,
        sample.q.map((q,j)=>(q-previous.q[j])*radians/dt)));
  });
  return {reference,measured};
}
