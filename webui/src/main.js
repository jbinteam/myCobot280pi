import './style.css';
import './operator-layout.css';
import { initOperatorLayout } from './operator-layout.js';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import URDFLoader from 'urdf-loader';
import { loadColladaMesh } from './collada-mesh.js';
import { initSimulation } from './simulation-controls.js';
import { initRobotControls } from './robot-controls.js';
import { initMouseAim } from './mouse-aim.js';
import { initApproachSimulation } from './approach-controls.js';
import { initIKFailureCase } from './ik-failure-case.js';
import { createCameraFOVView } from './camera-fov-view.js';

const $ = (id) => document.getElementById(id);
initOperatorLayout();
const MODEL_URL = '/models/mycobot_description/urdf/mycobot_280_pi/mycobot_280_pi.urdf';
const REST = [0, -40, -130, 80, 0, 50];
const degrees = THREE.MathUtils.radToDeg;
const radians = THREE.MathUtils.degToRad;
let robot, mouseAim, approachSimulation, joints = [], rows = [], loadFailed = false;
let wasConnected = false;
const viewport = $('viewport');
const cameraFOV=createCameraFOVView({toggle:$('show-camera-fov'),sizeOutput:$('camera-fov-size'),objectOutput:$('camera-fov-object')});
const scene = new THREE.Scene();
scene.background = new THREE.Color('#eef2f6');
const camera = new THREE.PerspectiveCamera(36, 1, 0.001, 30);
camera.up.set(0, 0, 1);
let renderer;
try {
  renderer = new THREE.WebGLRenderer({ antialias: true });
} catch (error) {
  $('loading-message').textContent = 'WebGL is unavailable. Enable hardware acceleration in your browser.';
  $('model-status').textContent = '3D rendering unavailable';
  document.querySelector('.spinner').hidden = true;
  throw error;
}
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.25;
viewport.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 0.12;
controls.maxDistance = 3;
controls.target.set(0, 0, 0.23);
scene.add(new THREE.HemisphereLight(0xffffff, 0x86947b, 2.4));
const sun = new THREE.DirectionalLight(0xffffff, 3.2);
sun.position.set(0.5, -0.6, 1.2);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.left = sun.shadow.camera.bottom = -0.65;
sun.shadow.camera.right = sun.shadow.camera.top = 0.65;
sun.shadow.camera.near = 0.01;
sun.shadow.camera.far = 3;
sun.shadow.bias = -0.00005;
sun.shadow.normalBias = 0.002;
scene.add(sun);
const fill = new THREE.DirectionalLight(0xdce8f0, 1.3);
fill.position.set(-0.8, 0.6, 0.5);
scene.add(fill);
const floor = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), new THREE.ShadowMaterial({ opacity: 0.12 }));
floor.position.z = -0.035;
floor.receiveShadow = true;
scene.add(floor);
const grid = new THREE.GridHelper(1.6, 32, 0xb4c1a6, 0xd2dbca);
grid.rotation.x = Math.PI / 2;
grid.position.z = -0.034;
grid.material.transparent = true;
grid.material.opacity = 0.55;
scene.add(grid);
const axes = new THREE.AxesHelper(0.12);
axes.position.set(-0.17, -0.17, -0.032);
scene.add(axes);
const selectionAxes = new THREE.AxesHelper(0.055);
selectionAxes.visible = false;
scene.add(selectionAxes);
let selectedJoint = null;
const eefFrame = new THREE.Group();
eefFrame.name = 'flange-eef-frame';
const axisSpecs = [
  ['X', new THREE.Vector3(1, 0, 0), '#d4473e'],
  ['Y', new THREE.Vector3(0, 1, 0), '#30974c'],
  ['Z', new THREE.Vector3(0, 0, 1), '#377cd1'],
];
for (const [label, direction, color] of axisSpecs) {
  const arrow = new THREE.ArrowHelper(direction, new THREE.Vector3(), 0.055, color, 0.009, 0.006);
  for (const object of [arrow.line, arrow.cone]) {
    object.material.depthTest = false;
    object.material.depthWrite = false;
    object.renderOrder = 20;
  }
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 128;
  const context = canvas.getContext('2d');
  context.font = 'bold 86px system-ui';
  context.textAlign = 'center'; context.textBaseline = 'middle';
  context.lineWidth = 10; context.strokeStyle = 'white';
  context.strokeText(label, 64, 64);
  context.fillStyle = color; context.fillText(label, 64, 64);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthTest: false, depthWrite: false, toneMapped: false }));
  sprite.position.copy(direction).multiplyScalar(0.066);
  sprite.scale.set(0.018, 0.018, 0.018);
  sprite.renderOrder = 21;
  eefFrame.add(arrow, sprite);
}
eefFrame.traverse(object => { object.userData.isFrameVisualization = true; });
const flangePosition = new THREE.Vector3();
function refreshFlange() {
  if (!robot?.links.joint6_flange) return;
  robot.updateMatrixWorld(true);
  robot.links.joint6_flange.getWorldPosition(flangePosition);
  $('flange-xyz').textContent = flangePosition.toArray().map((v, i) => `${'XYZ'[i]} ${ (v * 1000).toFixed(1) }`).join(' · ');
  const direction = new THREE.Vector3(0, 0, 1).transformDirection(robot.links.joint6_flange.matrixWorld);
  $('flange-tool-axis').textContent = `[${direction.toArray().map(v => v.toFixed(4)).join(', ')}]`;
}

function frame(view = 'perspective', extraPoints = []) {
  let center = new THREE.Vector3(0, 0, 0.22), distance = 0.95;
  if (robot) {
    robot.updateMatrixWorld(true);
    const bounds = new THREE.Box3().setFromObject(robot);
    for (const point of extraPoints) bounds.expandByPoint(new THREE.Vector3(...point));
    if (!bounds.isEmpty()) {
      center = bounds.getCenter(new THREE.Vector3());
      const size = bounds.getSize(new THREE.Vector3());
      distance = Math.max(size.x, size.y, size.z) / (2 * Math.tan(radians(camera.fov / 2))) * 1.35;
      distance *= Math.max(1, 1 / camera.aspect);
    }
  }
  const directions = {
    perspective: new THREE.Vector3(0.8, -1, 0.65),
    front: new THREE.Vector3(0, -1, 0.08),
    side: new THREE.Vector3(1, 0, 0.08),
    top: new THREE.Vector3(0, -0.001, 1),
  };
  camera.position.copy(center).addScaledVector(directions[view].normalize(), distance);
  controls.target.copy(center);
  controls.update();
}
function resize() {
  const width = viewport.clientWidth, height = viewport.clientHeight;
  renderer.setSize(width, height);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}
window.addEventListener('robot-state',({detail:state})=>{
  if(state.connected&&!wasConnected)requestAnimationFrame(()=>frame());
  wasConnected=!!state.connected;
});
new ResizeObserver(resize).observe(viewport);
resize();
frame();
renderer.setAnimationLoop(time => {
  controls.update();
  mouseAim?.update(time);
  approachSimulation?.update(time);
  if(robot) {
    const target=['approach-fixed-object','ik-failure-object','mouse-aim-cursor'].map(name=>scene.getObjectByName(name)).find(object=>{
      if(!object)return false;
      for(let parent=object;parent;parent=parent.parent)if(!parent.visible)return false;
      return true;
    });
    cameraFOV.update(target?target.getWorldPosition(new THREE.Vector3()):null);
  }
  if (selectedJoint && selectionAxes.visible) {
    selectedJoint.getWorldPosition(selectionAxes.position);
    selectedJoint.getWorldQuaternion(selectionAxes.quaternion);
  }
  renderer.render(scene, camera);
});

function poseValues() {
  return joints.map((joint) => Number(degrees(joint.angle).toFixed(1)));
}
function refreshPose() {
  $('pose-value').textContent = JSON.stringify(poseValues());
  $('copy-feedback').textContent = '';
  refreshFlange();
  rows.forEach(({ slider, number }, index) => {
    const value = degrees(joints[index].angle);
    slider.value = value;
    number.value = Number(value.toFixed(1));
    slider.style.setProperty('--fill', `${100 * (value - Number(slider.min)) / (Number(slider.max) - Number(slider.min))}%`);
  });
}
function setPose(values, live = false) {
  joints.forEach((joint, index) => {
    joint.ignoreLimits = live;
    joint.setJointValue(radians(values[index]));
    joint.ignoreLimits = false;
  });
  refreshPose();
}
function buildControls() {
  joints = Object.values(robot.joints).filter((joint) => joint.jointType === 'revolute' || joint.jointType === 'continuous');
  joints.forEach((joint, index) => {
    const min = Math.ceil(degrees(joint.limit.lower) * 10) / 10;
    const max = Math.floor(degrees(joint.limit.upper) * 10) / 10;
    const row = document.createElement('div');
    row.className = 'joint-control';
    row.innerHTML = `<div class="joint-label"><label class="joint-name" for="joint-${index}-slider"><span class="joint-id">J${index + 1}</span>Joint ${index + 1}</label><label class="joint-input-wrap"><input id="joint-${index}-number" aria-label="Joint ${index + 1} angle in degrees" type="number" min="${min}" max="${max}" step="0.1" value="0"/>°</label></div><input id="joint-${index}-slider" aria-label="Joint ${index + 1}" type="range" min="${min}" max="${max}" step="0.1" value="0"/><div class="joint-limits"><span>${min}°</span><span>${max}°</span></div>`;
    row.title = joint.name;
    const slider = row.querySelector('input[type=range]');
    const number = row.querySelector('input[type=number]');
    const update = (input) => {
      if (input.value.trim() === '') return;
      const value = Number(input.value);
      if (!Number.isFinite(value)) return;
      joint.setJointValue(radians(Math.max(Number(input.min), Math.min(Number(input.max), value))));
      refreshPose();
      window.dispatchEvent(new Event('preview-edited'));
    };
    slider.addEventListener('input', () => update(slider));
    number.addEventListener('change', () => update(number));
    row.addEventListener('pointerenter', () => { selectedJoint = joint; selectionAxes.visible = axes.visible; });
    row.addEventListener('pointerleave', () => { selectionAxes.visible = false; });
    $('joint-controls').appendChild(row);
    rows.push({ slider, number });
  });
  // Traversal keeps the displayed link list in the URDF's kinematic chain order.
  robot.traverse((obj) => {
    if (!obj.isURDFLink) return;
    const item = document.createElement('div');
    item.className = 'link-item';
    const icon = document.createElement('span');
    icon.className = 'link-node';
    item.append(icon, document.createTextNode(obj.name));
    $('link-tree').appendChild(item);
  });
  $('link-count').textContent = `${Object.keys(robot.links).length} LINKS`;
  $('model-details').textContent = `${joints.length} movable joints · ${Object.keys(robot.links).length} links · Local assets`;
  ['zero-pose', 'rest-pose', 'copy-pose'].forEach((id) => { $(id).disabled = false; });
  refreshPose();
  initRobotControls({
    getPose: poseValues,
    setPose,
    setEditing(enabled) {
      for (const { slider, number } of rows) { slider.disabled = number.disabled = !enabled; }
      $('zero-pose').disabled = $('rest-pose').disabled = !enabled;
    },
  });
}

const manager = new THREE.LoadingManager();
manager.onProgress = (_url, loaded, total) => { if (!loadFailed) $('loading-message').textContent = `Loading model assets · ${loaded}/${total}`; };
manager.onError = (url) => {
  loadFailed = true;
  $('loading-message').textContent = `Could not load model asset: ${url.split('/').pop()}. Reload to retry.`;
  $('model-status').textContent = 'Model loading failed';
  document.querySelector('.spinner').hidden = true;
};
manager.onLoad = () => {
  if (!robot || loadFailed) return;
  robot.traverse((obj) => {
    if (obj.isMesh) {
      obj.castShadow = true;
      obj.receiveShadow = true;
      // The stock assets are double-sided; keep imported colors and textures.
      for (const material of Array.isArray(obj.material) ? obj.material : [obj.material]) material.side = THREE.DoubleSide;
    }
  });
  frame();
  $('model-status').textContent = 'Model ready';
  $('model-dot').classList.add('ready');
  $('loading-overlay').classList.add('hidden');
  const fkRobot = robot.clone();
  const fkJoints = joints.map(joint => fkRobot.joints[joint.name]);
  fkJoints.forEach(joint => { joint.ignoreLimits = true; });
  const getEEFPose = pose => {
    fkJoints.forEach((joint, i) => joint.setJointValue(radians(pose[i])));
    fkRobot.updateMatrixWorld(true);
    const matrix = fkRobot.links.joint6_flange.matrixWorld.elements;
    return {position:[matrix[12],matrix[13],matrix[14]].map(v=>v*1000),axis:[matrix[8],matrix[9],matrix[10]],
      rotation:[[matrix[0], matrix[4], matrix[8]], [matrix[1], matrix[5], matrix[9]], [matrix[2], matrix[6], matrix[10]]]};
  };
  const getEEFRotation = pose => getEEFPose(pose).rotation;
  mouseAim = initMouseAim({ robot, jointNames: joints.map(joint => joint.name),
    getPose: () => joints.map(joint => degrees(joint.angle)), setPose,
    scene, camera, canvas: renderer.domElement, controls });
  approachSimulation=initApproachSimulation({getPose:()=>joints.map(joint=>degrees(joint.angle)),
    getEEFPose,setPose,scene,stopMouseAim:()=>mouseAim.stop(),frame});
  initSimulation({ getPose: () => joints.map(joint => degrees(joint.angle)), getEEFRotation, getEEFPose, setPose, scene, frame })
    .then(info => { if (info) {mouseAim.configure(info);approachSimulation.configure(info);requestAnimationFrame(()=>{
      cameraFOV.configure(info.tracking_camera);
      resize();frame();
      initIKFailureCase({scene,getEEFPose,setPose,frame,currentCamera:info.tracking_camera,
        startApproach:demo=>approachSimulation.startCase(demo),stopTracking:()=>{mouseAim.stop();approachSimulation.runner.stop();}});
    });} });
};
const loader = new URDFLoader(manager);
loader.loadMeshCb = loadColladaMesh;
loader.packages = { mycobot_description: '/models/mycobot_description' };
loader.parseCollision = false;
// Some upstream axes have leading spaces. URDFLoader's tuple parser does not
// trim these, producing NaN joint axes when the corresponding joints rotate.
// Normalize numeric tuple whitespace in memory; keep downloaded files intact.
const parseURDF = loader.parse.bind(loader);
loader.parse = (content, workingPath) => {
  const xml = new DOMParser().parseFromString(content, 'text/xml');
  for (const node of xml.querySelectorAll('[xyz], [rpy], [scale]')) {
    for (const attribute of ['xyz', 'rpy', 'scale']) {
      if (node.hasAttribute(attribute)) node.setAttribute(attribute, node.getAttribute(attribute).trim().replace(/\s+/g, ' '));
    }
  }
  return parseURDF(new XMLSerializer().serializeToString(xml), workingPath);
};
loader.load(MODEL_URL, (result) => {
  robot = result;
  scene.add(robot);
  robot.links.joint6_flange.add(eefFrame);
  robot.links.joint6_flange.add(cameraFOV.group);
  buildControls();
}, undefined, (error) => {
  loadFailed = true;
  $('model-status').textContent = 'Model loading failed';
  $('loading-message').textContent = `Could not load the URDF: ${error.message || error}`;
  document.querySelector('.spinner').hidden = true;
});

$('zero-pose').addEventListener('click', () => { setPose([0, 0, 0, 0, 0, 0]); window.dispatchEvent(new Event('preview-edited')); });
$('rest-pose').addEventListener('click', () => { setPose(REST); frame(); window.dispatchEvent(new Event('preview-edited')); });
$('reset-view').addEventListener('click', () => frame());
for (const view of ['front', 'side', 'top']) $(view + '-view').addEventListener('click', () => frame(view));
$('show-eef-frame').addEventListener('change', event => { eefFrame.visible = event.target.checked; });
$('show-grid').addEventListener('change', (event) => { grid.visible = event.target.checked; });
$('show-axes').addEventListener('change', (event) => { axes.visible = event.target.checked; selectionAxes.visible = false; });
$('wireframe').addEventListener('change', (event) => {
  if (!robot) return;
  robot.traverse((obj) => {
    if (obj.isMesh && !obj.userData.isFrameVisualization) for (const material of Array.isArray(obj.material) ? obj.material : [obj.material]) material.wireframe = event.target.checked;
  });
});
$('copy-pose').addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(JSON.stringify(poseValues()));
    $('copy-feedback').textContent = 'Copied joint angles';
  } catch {
    $('copy-feedback').textContent = 'Select and copy the angles above.';
  }
});

export { robot, camera, scene, renderer, mouseAim, approachSimulation };
