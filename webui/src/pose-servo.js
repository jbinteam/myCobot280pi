import * as THREE from 'three';

const SCALE = .3, ORIENTATION_WEIGHT = .6;
const clamp = THREE.MathUtils.clamp;
const norm = values => Math.hypot(...values);

// Small positive-definite systems from damped least squares; no inverse needed.
function linearSolve(matrix, rhs) {
  const a = matrix.map((row, i) => [...row, rhs[i]]), n = rhs.length;
  for (let i = 0; i < n; i++) {
    let pivot = i;
    for (let k = i + 1; k < n; k++) if (Math.abs(a[k][i]) > Math.abs(a[pivot][i])) pivot = k;
    [a[i], a[pivot]] = [a[pivot], a[i]];
    if (Math.abs(a[i][i]) < 1e-14) return null;
    for (let k = i + 1; k < n; k++) {
      const factor = a[k][i] / a[i][i];
      for (let j = i; j <= n; j++) a[k][j] -= factor * a[i][j];
    }
  }
  const x = Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    x[i] = (a[i][n] - a[i].slice(i + 1, n).reduce((sum, v, j) => sum + v * x[i + 1 + j], 0)) / a[i][i];
  }
  return x;
}

// Same active-set box DLS and five-row task as simulation/pose5.py.
function boundedVelocity(j, error, q, lower, upper, dt, speed, damping) {
  const h = q.map((_, a) => q.map((_, b) => j.reduce((sum, row) => sum + row[a] * row[b], 0) + (a === b ? damping ** 2 : 0)));
  const rhs = q.map((_, a) => 3 * j.reduce((sum, row, i) => sum + row[a] * error[i], 0));
  const lo = q.map((v, i) => Math.max((lower[i] - v) / dt, -speed));
  const hi = q.map((v, i) => Math.min((upper[i] - v) / dt, speed));
  const v = q.map(() => 0), fixed = q.map(() => 0);
  for (let iteration = 0; iteration < 64; iteration++) {
    const free = q.map((_, i) => i).filter(i => !fixed[i]);
    const active = q.map((_, i) => i).filter(i => fixed[i]);
    if (free.length) {
      const solved = linearSolve(free.map(a => free.map(b => h[a][b])), free.map(a => rhs[a] - active.reduce((sum, b) => sum + h[a][b] * v[b], 0)));
      if (!solved) return q.map(() => 0);
      free.forEach((i, k) => { v[i] = solved[k]; });
    }
    const violations = v.map((value, i) => fixed[i] ? -1 : Math.max(lo[i] - value, value - hi[i]));
    const worst = Math.max(...violations), k = violations.indexOf(worst);
    if (worst > 1e-10) {
      fixed[k] = v[k] < lo[k] ? -1 : 1;
      v[k] = fixed[k] < 0 ? lo[k] : hi[k];
      continue;
    }
    const wrong = v.map((_, i) => {
      const gradient = h[i].reduce((sum, value, b) => sum + value * v[b], 0) - rhs[i];
      return fixed[i] === -1 ? -gradient : fixed[i] === 1 ? gradient : -1;
    });
    if (Math.max(...wrong) <= 1e-10) break;
    fixed[wrong.indexOf(Math.max(...wrong))] = 0;
  }
  const bounded = v.map((value, i) => clamp(value, lo[i], hi[i]));
  const factor = Math.min(1, speed / Math.max(norm(bounded), 1e-12));
  return bounded.map(value => value * factor);
}

export function pointingError(current, target) {
  const cross = new THREE.Vector3().crossVectors(current, target);
  const length = cross.length(), angle = Math.atan2(length, clamp(current.dot(target), -1, 1));
  const components = current.toArray().map(Math.abs), reference = new THREE.Vector3();
  reference.setComponent(components.indexOf(Math.min(...components)), 1);
  const b1 = new THREE.Vector3().crossVectors(current, reference).normalize();
  const b2 = new THREE.Vector3().crossVectors(current, b1);
  const vector = length > 1e-10 ? cross.multiplyScalar(angle / length) : b1.clone().multiplyScalar(current.dot(target) < 0 ? angle : 0);
  return { vector, basis: [b1, b2], angle };
}

export class PoseServo {
  constructor(robot, jointNames, limitsDeg) {
    this.model = robot.clone();
    this.joints = jointNames.map(name => this.model.joints[name]);
    this.joints.forEach(joint => { joint.ignoreLimits = true; });
    this.flange = this.model.links.joint6_flange;
    this.lower = limitsDeg.min.map(THREE.MathUtils.degToRad);
    this.upper = limitsDeg.max.map(THREE.MathUtils.degToRad);
  }

  pose(q) {
    this.joints.forEach((joint, i) => joint.setJointValue(q[i]));
    this.model.updateMatrixWorld(true);
    return {
      position: this.flange.getWorldPosition(new THREE.Vector3()),
      axis: new THREE.Vector3(0, 0, 1).transformDirection(this.flange.matrixWorld),
    };
  }

  task(q, target, targetAxis) {
    const pose = this.pose(q), pointing = pointingError(pose.axis, targetAxis);
    const delta = target.clone().sub(pose.position).divideScalar(SCALE);
    const error = [...delta.toArray(), ...pointing.basis.map(b => ORIENTATION_WEIGHT * b.dot(pointing.vector))];
    const j = Array.from({ length: 5 }, () => []);
    for (const joint of this.joints) {
      const origin = joint.getWorldPosition(new THREE.Vector3());
      const axis = joint.axis.clone().transformDirection(joint.matrixWorld);
      const linear = new THREE.Vector3().crossVectors(axis, pose.position.clone().sub(origin)).divideScalar(SCALE);
      linear.toArray().forEach((value, i) => j[i].push(value));
      pointing.basis.forEach((b, i) => j[3 + i].push(ORIENTATION_WEIGHT * b.dot(axis)));
    }
    return { ...pose, j, error, merit: error.reduce((sum, v) => sum + v * v, 0), positionErrorMm: delta.length() * SCALE * 1000, angleDeg: THREE.MathUtils.radToDeg(pointing.angle) };
  }

  step(q, target, targetAxis, dt, speed = 1.2, damping = .04) {
    if (q.length !== 6 || !q.every(Number.isFinite) || q.some((v, i) => v < this.lower[i] - 1e-9 || v > this.upper[i] + 1e-9)) return null;
    if (!target.toArray().every(Number.isFinite) || !targetAxis.toArray().every(Number.isFinite) || targetAxis.lengthSq() < 1e-12) return null;
    if (!(dt > 0 && speed > 0 && damping > 0)) return null;
    const axis = targetAxis.clone().normalize(), current = this.task(q, target, axis);
    const velocity = boundedVelocity(current.j, current.error, q, this.lower, this.upper, dt, speed, damping);
    let scale = dt;
    for (let attempt = 0; attempt <= 10; attempt++) {
      const next = q.map((value, i) => clamp(value + scale * velocity[i], this.lower[i], this.upper[i]));
      const result = this.task(next, target, axis);
      if (result.merit <= current.merit + 1e-14) return { ...result, q: next, travel: norm(next.map((value, i) => value - q[i])) };
      scale *= .5;
    }
    return { ...current, q: q.slice(), travel: 0 };
  }
}
