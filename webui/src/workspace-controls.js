import * as THREE from 'three';
import { readToolAxis, readReferenceTilts, directionInputIds } from './tool-direction.js';
const $ = id => document.getElementById(id);

// Boundary faces of occupied cells, leaving internal cavities and gaps intact.
export function voxelSurface(cells, size) {
  const occupied = new Set(cells.map(c => c.join(','))), vertices = [];
  const faces = [
    [[1,0,0],[[1,0,0],[1,1,0],[1,1,1],[1,0,1]]],
    [[-1,0,0],[[0,0,1],[0,1,1],[0,1,0],[0,0,0]]],
    [[0,1,0],[[0,1,1],[1,1,1],[1,1,0],[0,1,0]]],
    [[0,-1,0],[[0,0,0],[1,0,0],[1,0,1],[0,0,1]]],
    [[0,0,1],[[0,0,1],[1,0,1],[1,1,1],[0,1,1]]],
    [[0,0,-1],[[0,1,0],[1,1,0],[1,0,0],[0,0,0]]],
  ];
  for (const cell of cells) for (const [normal, corners] of faces) {
    if (occupied.has(cell.map((v,i) => v+normal[i]).join(','))) continue;
    for (const index of [0,1,2,0,2,3]) vertices.push(...cell.map((v,i) => (v+corners[index][i])*size/1000));
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
  geometry.computeVertexNormals();
  return geometry;
}

export function initWorkspace({ scene, frame, getStartPose }) {
  const group = new THREE.Group(); group.name = 'reachable-workspace'; scene.add(group);
  let cloud, shell, fkCloud, fkShell, generation = 0, liveState = null;
  let watchedPose = [...getStartPose()];
  window.addEventListener('robot-state', event => { liveState = event.detail; });
  const currentPose = () => liveState?.connected && Array.isArray(liveState.angles) ? liveState.angles : getStartPose();
  const button = $('generate-workspace'), status = $('workspace-status');
  // Fixed screen-size markers stay legible when the FK cloud sets a wide view.
  const dotCanvas = document.createElement('canvas'); dotCanvas.width = dotCanvas.height = 32;
  const dotContext = dotCanvas.getContext('2d');
  dotContext.fillStyle = '#ffffff'; dotContext.beginPath(); dotContext.arc(16,16,14,0,Math.PI*2); dotContext.fill();
  const dotTexture = new THREE.CanvasTexture(dotCanvas);
  function clear() {
    for (const object of [...group.children]) {
      group.remove(object); object.geometry.dispose(); object.material.dispose();
    }
    cloud = shell = fkCloud = fkShell = null;
  }
  function invalidate() {
    generation++;
    status.textContent = cloud
      ? 'Settings changed · showing the previous workspace. Generate again to update for the start pose, solver and tolerances.'
      : 'Settings changed · generate the workspace.';
  }
  for (const id of ['ik-task','target-rx','target-ry','ik-tolerance','ik-orientation-tolerance','workspace-samples','workspace-voxel','workspace-targets','ik-method','ik-speed','ik-steps','ik-damping',...directionInputIds]) {
    $(id).addEventListener(['ik-task','ik-method','workspace-targets','workspace-samples','direction-entry','direction-preset'].includes(id) ? 'change' : 'input', invalidate);
  }
  // Case/live-target buttons assign values without firing input events.
  const signature = () => ['ik-task','target-rx','target-ry','ik-tolerance','ik-orientation-tolerance','ik-method','ik-speed','ik-steps','ik-damping','workspace-targets','workspace-samples','workspace-voxel',...directionInputIds].map(id => $(id).value).join('|') + '|' + watchedPose.join(',');
  let lastSignature = signature();
  setInterval(() => {
    const pose = currentPose();
    if (Array.isArray(pose) && pose.some((v,i) => Math.abs(v-watchedPose[i]) > 1)) watchedPose = [...pose];
    const current = signature();
    if (current !== lastSignature) { lastSignature = current; invalidate(); }
  }, 300);
  function visibility() {
    const solved = $('show-workspace-cloud').checked, fk = $('show-fk-workspace-cloud').checked;
    const volume = $('show-workspace-volume').checked;
    if (cloud) cloud.visible = solved;
    if (shell) shell.visible = solved && volume;
    if (fkCloud) fkCloud.visible = fk;
    if (fkShell) fkShell.visible = fk && volume;
  }
  for (const id of ['show-workspace-cloud','show-fk-workspace-cloud','show-workspace-volume']) $(id).onchange = visibility;
  $('workspace-opacity').oninput = e => {
    if (shell) shell.material.opacity = Number(e.target.value);
    if (fkShell) fkShell.material.opacity = Number(e.target.value) / 3;
  };
  button.onclick = async () => {
    watchedPose = [...currentPose()];
    const requestGeneration = ++generation, requestSignature = signature();
    lastSignature = requestSignature;
    button.disabled = true; status.textContent = 'Testing target poses with the selected solver from one captured start…';
    try {
      if($('ik-task').value==='pose5')readToolAxis();
      const stateResponse = await fetch('/api/state');
      if (!stateResponse.ok) throw new Error('Simulation bridge unavailable');
      const state = await stateResponse.json();
      if (state.connected && (state.motion || state.moving)) throw new Error('Wait for the robot to stop before estimating solvable targets');
      if (state.connected && (!Array.isArray(state.angles) || state.angles.length !== 6 || !state.angles.every(Number.isFinite) || !Number.isFinite(state.updated) || Date.now()/1000-state.updated > 5)) throw new Error('Live joint telemetry is unavailable or stale');
      const start = state.connected ? state.angles : getStartPose();
      const response = await fetch('/api/sim/workspace', {method:'POST', headers:{'Content-Type':'application/json','X-Robot-Token':state.token}, body:JSON.stringify({
        start_deg:start, method:$('ik-method').value, steps:Number($('ik-steps').value), speed_rad_s:Number($('ik-speed').value), damping:Number($('ik-damping').value), targets:Number($('workspace-targets').value),
        task:$('ik-task').value, ...($('ik-task').value==='pose5'?{orientation_xy_deg:readReferenceTilts(),...($('direction-entry').value==='axis'?{target_axis:readToolAxis()}: {})}:{}),
        tolerance_mm:Number($('ik-tolerance').value), orientation_tolerance_deg:Number($('ik-orientation-tolerance').value),
        samples:Number($('workspace-samples').value), voxel_mm:Number($('workspace-voxel').value),
      })});
      const data = await response.json(); if (!response.ok) throw new Error(data.error || 'Workspace estimation failed');
      if (requestGeneration !== generation || signature() !== requestSignature) return;
      clear();
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(data.points_mm.flat().map(v => v/1000), 3));
      cloud = new THREE.Points(geometry, new THREE.PointsMaterial({color:'#0062ff', size:8, sizeAttenuation:false, map:dotTexture, alphaTest:.3, transparent:true, opacity:1, depthWrite:false, toneMapped:false}));
      cloud.renderOrder = 8;
      cloud.name = 'reachable-workspace-cloud'; cloud.visible = $('show-workspace-cloud').checked;
      shell = new THREE.Mesh(voxelSurface(data.voxels, data.voxel_mm), new THREE.MeshBasicMaterial({color:'#46b0b6', transparent:true, opacity:Number($('workspace-opacity').value), depthWrite:false, side:THREE.DoubleSide}));
      shell.name = 'reachable-workspace-volume'; shell.visible = $('show-workspace-volume').checked;
      shell.renderOrder = 3;
      const fkGeometry = new THREE.BufferGeometry();
      fkGeometry.setAttribute('position', new THREE.Float32BufferAttribute(data.fk_points_mm.flat().map(v => v/1000), 3));
      fkCloud = new THREE.Points(fkGeometry, new THREE.PointsMaterial({color:'#bd8434', size:2, sizeAttenuation:false, transparent:true, opacity:.3, depthWrite:false, toneMapped:false}));
      fkCloud.renderOrder = 2;
      fkCloud.name = 'fk-workspace-cloud';
      fkShell = new THREE.Mesh(voxelSurface(data.fk_voxels, data.voxel_mm), new THREE.MeshBasicMaterial({color:'#d6a45a', transparent:true, opacity:Number($('workspace-opacity').value)/3, depthWrite:false, side:THREE.DoubleSide}));
      fkShell.name = 'fk-workspace-volume';
      fkShell.renderOrder = 1;
      group.add(fkCloud, fkShell, cloud, shell); visibility();
      group.userData.estimate = {accepted:data.accepted, tested:data.tested, fk_count:data.fk_points_mm.length, start_deg:data.start_deg, method:data.method, voxel_mm:data.voxel_mm};
      $('solvable-cloud-count').textContent = `(${data.accepted.toLocaleString()} points)`;
      $('fk-cloud-count').textContent = `(${data.fk_points_mm.length.toLocaleString()} points)`;
      status.textContent = data.accepted
        ? `${data.accepted.toLocaleString()} / ${data.tested.toLocaleString()} targets solved · ${data.method} · start (${data.start_source}): [${data.start_deg.map(v=>v.toFixed(1)).join(', ')}]° · ${(data.elapsed_ms/1000).toFixed(2)} s · position radius ${data.tolerance_mm} mm${data.target_axis ? ` · direction ≤ ${data.orientation_tolerance_deg}°` : ' · any direction'} · voxel outer-shell error ≤ ${data.envelope_extra_mm_max.toFixed(1)} mm beyond tolerance balls.`
        : `Solvable cloud is empty: 0 / ${data.tested} targets converged with ${data.method} from the captured start for the selected direction. FK points do not imply solver convergence. Try another direction, solver or step budget; this does not prove unreachability.`;
      status.textContent += ` · FK reachable: ${data.fk_points_mm.length.toLocaleString()} points.`;
      const bounds = $('show-fk-workspace-cloud').checked ? data.fk_bounds_mm : data.bounds_mm;
      if (bounds) frame('perspective', [bounds.min.map(v=>v/1000), bounds.max.map(v=>v/1000)]);
    } catch (error) { if (requestGeneration === generation) status.textContent = error.message; }
    finally { button.disabled = false; }
  };
}
