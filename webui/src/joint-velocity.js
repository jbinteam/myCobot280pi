const degrees=180/Math.PI;

// Three-point Gauss quadrature integrates our quartic joint velocity curves
// exactly. Reconstruct from the segment anchor on every sample so seeking and
// irregular render intervals cannot accumulate integration drift.
export function positionFromVelocity(startDeg,velocityAt,duration) {
  const offset=Math.sqrt(3/5)/2;
  const nodes=[.5-offset,.5,.5+offset],weights=[5/18,4/9,5/18];
  const delta=Array(6).fill(0);
  nodes.forEach((node,i)=>velocityAt(duration*node).forEach((v,j)=>delta[j]+=weights[i]*v));
  return startDeg.map((q,j)=>q+duration*delta[j]*degrees);
}

export function velocityPoint(time,velocity_rad_s) {
  return {time,velocity_rad_s,speed_rad_s:Math.hypot(...velocity_rad_s)};
}
