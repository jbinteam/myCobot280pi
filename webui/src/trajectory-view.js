import * as THREE from 'three';

export function trajectoryView(scene,name) {
  const group=new THREE.Group();group.name=name;group.visible=false;scene.add(group);
  const points=new THREE.Points(new THREE.BufferGeometry(),new THREE.PointsMaterial({
    color:'#267d9a',size:6,sizeAttenuation:false,depthTest:false,depthWrite:false}));
  points.name=`${name}-points`;
  points.renderOrder=25;group.add(points);
  group.traverse(object=>object.userData.isFrameVisualization=true);
  const toggle=document.getElementById('show-trajectory-points');
  let populated=false;
  toggle.addEventListener('change',()=>group.visible=populated&&toggle.checked);
  return {
    show(frames,targetMm=frames.at(-1).position) {
      points.geometry.dispose();points.geometry=new THREE.BufferGeometry().setFromPoints(
        [new THREE.Vector3(...targetMm.map(v=>v/1000))]);
      populated=true;group.visible=toggle.checked;
    },
    clear(){populated=false;group.visible=false;points.geometry.dispose();points.geometry=new THREE.BufferGeometry();}
  };
}
