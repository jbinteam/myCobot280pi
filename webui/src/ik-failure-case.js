import * as THREE from 'three';
import outwardFixture from '../../simulation/diffusion/tests/fixtures/ik_failed_reachable.json';
import inwardFixture from '../../simulation/diffusion/tests/fixtures/ik_failed_reachable_inward.json';
import final150Fixture from '../../simulation/diffusion/tests/fixtures/ik_failed_reachable_150.json';
import {coneDiskRegion, createTrackingRegionView, regionPositionMetrics} from './tracking-region.js';
import {createRRTSearchView} from './rrt-search-view.js';

const $ = id => document.getElementById(id);

// This is an inspection of a stored geometric proof. It never prepares a
// hardware plan or interprets the route as a timed execution reference.
export function initIKFailureCase({scene, getEEFPose, setPose, frame, stopTracking, currentCamera=null, startApproach=null}) {
  $('case-relax-orientation').checked=true;
  const cases = {'ik-failed-reachable-150':final150Fixture,'ik-failed-reachable':outwardFixture,'ik-failed-reachable-inward':inwardFixture};
  const requestedCase = new URLSearchParams(location.search).get('case');
  const caseKey = Object.hasOwn(cases,requestedCase) ? requestedCase : 'ik-failed-reachable-150';
  const fixture = cases[caseKey];
  $('case-run-approach').hidden = !fixture.approach;
  $('case-relax-option').hidden = !fixture.approach;
  const fov = camera => `${camera.horizontal_fov_deg}° × ${camera.vertical_fov_deg}°`;
  $('case-camera').textContent = `Saved route camera: ${fov(fixture.camera)}${currentCamera ? ` · current camera: ${fov(currentCamera)}` : ''}.`;
  if (currentCamera && JSON.stringify(currentCamera)!==JSON.stringify(fixture.camera)) {
    const checked = fixture.camera_revalidation;
    $('case-camera').textContent += checked && JSON.stringify(checked.camera)===JSON.stringify(currentCamera)
      ? checked.route_visibility_audit.passed ? ' Saved route rechecked: visible throughout under the current FOV.'
        : ' Saved historical route loses the object under the current FOV; it is a preview of the old proof.'
      : ' The saved proof uses its recorded camera; current-camera validity requires rechecking.';
  }
  $('case-selector').value = caseKey;
  $('case-selector').addEventListener('change', event => {
    if (Object.hasOwn(cases,event.target.value)) location.assign(`?case=${event.target.value}`);
  });
  let state = {}, active = false, pending = null;
  const region = coneDiskRegion(new THREE.Vector3().fromArray(fixture.request.cursor_mm).multiplyScalar(.001),
    fixture.request.target_axis, fixture.request.tracking_region);
  const view = createTrackingRegionView();
  view.group.name = 'ik-failure-region';
  scene.add(view.group);
  const initialRegion = fixture.approach ? coneDiskRegion(
    new THREE.Vector3().fromArray(fixture.request.cursor_mm).multiplyScalar(.001),
    fixture.request.target_axis,{...fixture.request.tracking_region,distance_mm:fixture.approach.measured_initial_distance_mm}) : null;
  const initialView = createTrackingRegionView();
  initialView.group.name = 'ik-failure-initial-disk';scene.add(initialView.group);
  initialView.group.getObjectByName('mouse-aim-cone').visible = false;
  initialView.group.getObjectByName('mouse-aim-acceptance-disk').visible = false;
  initialView.group.getObjectByName('mouse-aim-acceptance-ring').material.color.set('#8794a8');
  const markers = new THREE.Group();markers.name = 'ik-failure-markers';markers.visible = false;scene.add(markers);
  function marker(name, color, radius, position) {
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(radius, 20, 14),
      new THREE.MeshBasicMaterial({color, depthTest:false}));
    mesh.name = name;mesh.position.fromArray(position).multiplyScalar(.001);mesh.renderOrder = 30;markers.add(mesh);
    return mesh;
  }
  marker('ik-failure-object', '#e99129', .007, fixture.request.cursor_mm);
  const startPose = getEEFPose(fixture.request.start_deg), endPose = getEEFPose(fixture.witness.q_deg);
  $('case-title').textContent = initialRegion ? `Inward stage: ${initialRegion.distance_mm} → ${region.distance_mm} mm` : 'Reachable disk, missed by IK';
  $('case-description').textContent = initialRegion
    ? `The corrected approach reaches its current-distance disk at ${initialRegion.distance_mm} mm. The next ${region.distance_mm} mm disk has a recorded IK failure despite having a checked reachable route. The object stays fixed.`
    : 'Start 129 mm from the object and reposition outward to a disk 330 mm along the frozen axis. The bounded search misses a reachable pose.';
  if(fixture.full_approach_reach_check?.final_stage_impossible_for_any_axis)
    $('case-description').textContent += ' This is a 250 mm stage diagnostic: the full 150 mm finish is outside the robot reach bound and cannot be completed for this object.';
  if(fixture.final_stage_proof?.accepted)
    $('case-description').textContent += ' The 150 mm finish is independently checked, including a complete timed route. The current planner still has to discover a solution itself.';
  $('case-legend').textContent = `Orange: object · green: failed ${region.distance_mm} mm disk (radius ${region.radius_mm.toFixed(1)} mm) · magenta: start · blue: reachable endpoint and saved geometric route.${initialRegion ? ` Gray ring: completed ${initialRegion.distance_mm} mm disk.` : ''}`;
  const witnessRange=fixture.final_stage_proof?.acceptance.distance_mm ?? 288.1243938025135;
  $('case-distance-note').textContent = initialRegion
    ? `Disk distances are axial. The off-center witness is ${witnessRange.toFixed(1)} mm from the object in a straight line. The route is not required to move monotonically closer.` : 'Disk distance is axial; an off-center endpoint can be farther from the object in a straight line.';
  marker('ik-failure-start', '#bf008f', .005, startPose.position);
  marker('ik-failure-witness', '#0050ff', .006, endPose.position);
  const route = fixture.witness.route_rad.map(q => q.map(THREE.MathUtils.radToDeg));
  // Keep the displayed endpoints identical to their saved degree values;
  // a radians round trip can change the final floating-point digit.
  route[0] = fixture.request.start_deg.slice();
  route[route.length-1] = fixture.witness.q_deg.slice();
  // Project joint-interpolated edges with dense FK, rather than drawing
  // straight Cartesian chords between the eleven stored waypoints.
  const frames = [{position:startPose.position}];
  for (let i = 1; i < route.length; i++) {
    const delta = route[i].map((q,j) => q-route[i-1][j]);
    const steps = Math.ceil(Math.hypot(...delta.map(THREE.MathUtils.degToRad))/.005);
    for (let k = 1; k <= steps; k++) frames.push(getEEFPose(delta.map((dq,j) => route[i-1][j]+dq*k/steps)));
  }
  // Separate group from the live approach planner's display.
  const routeView = createRRTSearchView(scene, getEEFPose, $('case-show-route'));
  scene.children.at(-1).name = 'ik-failure-saved-route';
  const allowed = () => state.hardware_enabled === false && !state.connected && !state.motion && !state.moving;
  function controls() {
    $('ik-failure-case').hidden = state.hardware_enabled !== false;
    $('load-ik-failure-case').disabled = !allowed() || !!pending;
    for (const id of ['case-start', 'case-witness', 'case-progress', 'case-show-route']) $(id).disabled = !active || !allowed() || !!pending;
    $('case-reproduce').disabled = !active || !allowed() || !!pending;
    $('case-run-approach').disabled = !fixture.approach || !startApproach || !active || !allowed() || !!pending;
    $('case-relax-orientation').disabled = !fixture.approach || !active || !allowed() || !!pending;
    $('case-clear').disabled = !active;
  }
  function inspect(value) {
    if (!active || !allowed()) return;
    const progress = Number(value), index = Math.min(route.length-2, Math.floor(progress));
    const blend = progress-index;
    const q = route[index].map((angle,j) => angle+(route[index+1][j]-angle)*blend);
    setPose(q);$('case-progress').value = progress;
    const pose = getEEFPose(q), metrics = regionPositionMetrics(new THREE.Vector3().fromArray(pose.position).multiplyScalar(.001), region);
    const delta = new THREE.Vector3().fromArray(fixture.request.cursor_mm).sub(new THREE.Vector3().fromArray(pose.position));
    const pointing = THREE.MathUtils.radToDeg(new THREE.Vector3().fromArray(pose.axis).angleTo(delta));
    $('case-metrics').textContent = `Object distance ${delta.length().toFixed(1)} mm · plane error ${metrics.axialErrorMm.toFixed(2)} mm · radial ${metrics.radialMm.toFixed(1)} / ${region.radius_mm.toFixed(1)} mm · aim error ${pointing.toFixed(3)}°`;
    $('case-pose-label').textContent = progress === 0 ? 'Failed request: exact starting pose' : progress === route.length-1
      ? `Reachable witness: candidate ${fixture.witness.candidate_index+1}` : `Geometric route: segment ${index+1} / ${route.length-1}`;
  }
  function clear() {
    pending?.abort();pending = null;active = false;
    view.clear();initialView.clear();routeView.clear();markers.visible = false;
    document.body.classList.remove('inspecting-ik-case');controls();
  }
  function load() {
    if (!allowed() || pending) return;
    stopTracking();active = true;document.body.classList.add('inspecting-ik-case');
    view.show(region);markers.visible = true;routeView.accepted(frames, fixture.witness.q_deg);
    if (initialRegion) initialView.show(initialRegion);
    const expected = fixture.expected_baseline;
    $('case-status').textContent = `Saved result: ${expected.status} · ${expected.candidate_limit} candidate points · ${expected.candidates_filtered} filtered · ${expected.jobs_started} exhausted IK jobs · ${expected.route_attempts} route attempts`;
    controls();inspect(0);
    const bounds = [fixture.request.cursor_mm, ...frames.map(frame=>frame.position)].map(p => p.map(v => v/1000));
    for (const disk of [region,initialRegion].filter(Boolean)) {
      const center = new THREE.Vector3().fromArray(disk.center_mm).multiplyScalar(.001);
      for (let i = 0; i < 3; i++) for (const sign of [-1,1]) {
        const p = center.clone();p.setComponent(i,p.getComponent(i)+sign*disk.radius_mm/1000);bounds.push(p.toArray());
      }
    }
    frame('perspective', bounds);
  }
  $('load-ik-failure-case').addEventListener('click', load);
  $('case-start').addEventListener('click', () => inspect(0));
  $('case-witness').addEventListener('click', () => inspect(route.length-1));
  $('case-progress').max = route.length-1;
  $('case-progress').addEventListener('input', event => inspect(event.target.value));
  $('case-clear').addEventListener('click', clear);
  $('case-run-approach').addEventListener('click', () => {
    if(!fixture.approach || !startApproach || !active || !allowed() || pending) return;
    const demo = {start_deg:fixture.approach.initial_request.start_deg.slice(),
      cursor_mm:fixture.request.cursor_mm.slice(),distance_mm:fixture.approach.nominal_demo_distance_mm,
      half_angle_deg:fixture.request.tracking_region.half_angle_deg,
      orientation_mode:$('case-relax-orientation').checked?'strict_then_fov':'point_at_object'};
    clear();
    try {
      startApproach(demo);
      $('case-status').textContent = 'Approach started with this fixed object. Use Pause, Resume, and Stop approach above the scene. Failed searches hold and retry the current stage.'
        +(fixture.full_approach_reach_check?.final_stage_impossible_for_any_axis?' The 250 mm stage is reachable, but this diagnostic cannot complete the 150 mm finish.':'')
        +(fixture.final_stage_proof?.accepted?' The 150 mm finish has an independent checked witness; the planner receives no saved endpoint or route.':'');
    } catch(error) {$('case-status').textContent = `Cannot start approach: ${error.message}`;}
  });
  $('case-reproduce').addEventListener('click', async () => {
    if (!active || !allowed() || pending) return;
    inspect(0);const request = new AbortController();pending = request;controls();
    $('case-status').textContent = `Running Learned IK + RRT* for the ${region.distance_mm} mm disk from the exact fixture start…`;
    try {
      const response = await fetch('/api/sim/mouse-aim', {method:'POST',signal:request.signal,
        headers:{'Content-Type':'application/json','X-Robot-Token':state.token},
        body:JSON.stringify({...fixture.request,tracking_engine:'learned_rrt'})});
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
      if (pending !== request || !active) return;
      const sameModel = result.checkpoint_sha256 === fixture.checkpoint_sha256 &&
        result.environment?.robot?.urdf_sha256 === fixture.environment.robot.urdf_sha256 &&
        result.environment?.robot?.geometry_sha256 === fixture.environment.robot.geometry_sha256 &&
        JSON.stringify(result.visibility_model?.camera) === JSON.stringify(fixture.camera);
      $('case-status').textContent = `Fresh result: ${result.status} · ${result.region_candidates_filtered} points filtered · ${result.region_candidate_jobs_tested} IK jobs · ${result.region_route_attempts?.length ?? 0} route attempts · preview held at start${sameModel ? '' : ' · model differs from saved fixture; revalidate the witness'}`;
    } catch (error) {
      if (pending === request && error.name !== 'AbortError') $('case-status').textContent = `Planner request failed: ${error.message}`;
    } finally {if (pending === request) {pending = null;controls();}}
  });
  window.addEventListener('robot-state', ({detail}) => {
    state = detail;if (!allowed() && active) clear();controls();
  });
  window.addEventListener('control-mode-changing', clear);
  for (const event of ['preview-edited','mouse-aim-change']) window.addEventListener(event, event => {
    if (event.type === 'preview-edited' || event.detail?.active) clear();
  });
  for (const id of ['start-approach-simulation','load-visible-tracking-demo','zero-pose','rest-pose','solve-ik','compare-ik','plan-prm','plan-classical','plan-diffusion','play-ik'])
    $(id).addEventListener('click', clear, {capture:true});
  document.addEventListener('keydown', event => {if (event.key === 'Escape') clear();});
  fetch('/api/state').then(response => response.json()).then(next => {
    if (!state.token) state = next;controls();
    if (Object.hasOwn(cases,requestedCase)) load();
  }).catch(() => {$('case-status').textContent = 'Cannot read simulation state. Reload after the server is available.';});
  controls();
}
