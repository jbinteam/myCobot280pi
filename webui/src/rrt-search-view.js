import * as THREE from 'three';
import {LineSegments2} from 'three/addons/lines/LineSegments2.js';
import {LineSegmentsGeometry} from 'three/addons/lines/LineSegmentsGeometry.js';
import {LineMaterial} from 'three/addons/lines/LineMaterial.js';

export function createRRTSearchView(scene,getEEFPose,toggle,onStatus=()=>{}) {
  const group=new THREE.Group();group.name='approach-rrt-search';scene.add(group);
  const outlines=new WeakMap();
  function wideLine(name,color,width,order) {
    const geometry=new LineSegmentsGeometry().setPositions([]);
    const material=(color,linewidth)=>new LineMaterial({color,linewidth,worldUnits:false,
      depthTest:false,depthWrite:false,toneMapped:false});
    const outline=new LineSegments2(geometry,material('#ffffff',width+3));
    const line=new LineSegments2(geometry,material(color,width));
    line.name=name;outline.name=`${name}-outline`;outlines.set(line,outline);
    outline.renderOrder=order;line.renderOrder=order+1;
    line.frustumCulled=outline.frustumCulled=false;
    group.add(outline,line);return line;
  }
  // Screen-space fat lines: WebGL ignores LineBasicMaterial.linewidth on most GPUs.
  const tree=wideLine('approach-rrt-branches','#bf008f',4,24);
  const route=wideLine('approach-accepted-route','#0050ff',6,26);
  group.traverse(object=>{object.userData.isFrameVisualization=true;});
  let nodes=[],dirty=false,populated=false,attempts=0,goalKey=null;
  const attemptsByGoal=new Map();
  const xyz=q=>new THREE.Vector3().fromArray(getEEFPose(q).position).multiplyScalar(.001);
  function geometry(object,points){
    const previous=object.geometry;
    object.geometry=new LineSegmentsGeometry().setPositions(points.flatMap(point=>point.toArray()));
    outlines.get(object).geometry=object.geometry;previous.dispose();
  }
  function visibility(){group.visible=toggle.checked&&populated;}
  toggle.addEventListener('change',visibility);visibility();
  return {
    events(events) {
      for(const event of events) {
        if(event.kind==='begin') {
          goalKey=Array.isArray(event.goal_deg)?JSON.stringify(event.goal_deg):null;
          attempts++;nodes=[{position:xyz(event.start_deg),parent:-1}];geometry(route,[]);
          onStatus(`RRT* attempt ${attempts} · searching…`);
        } else if(event.kind==='add') {
          nodes[event.index]={position:xyz(event.q_deg),parent:event.parent};
        } else if(event.kind==='rewire'&&nodes[event.index])nodes[event.index].parent=event.parent;
        else if(event.kind==='end') {
          if(event.status==='direct_optimal'&&event.path_deg?.length>1)
            nodes=[{position:xyz(event.path_deg[0]),parent:-1},{position:xyz(event.path_deg.at(-1)),parent:0}];
          if(goalKey!==null){
            attemptsByGoal.set(goalKey,nodes.map(node=>node&&({...node})));
            if(attemptsByGoal.size>16)attemptsByGoal.delete(attemptsByGoal.keys().next().value);
          }
          onStatus(event.status==='direct_optimal'?'Direct checked candidate · no RRT* expansion needed'
            :`RRT* attempt ${attempts} · ${event.nodes} nodes · ${event.rewires} rewires · ${event.status}`);
        }
        dirty=true;populated=true;
      }
      visibility();
    },
    update() {
      if(!dirty)return;
      const points=[];
      for(const node of nodes)if(node?.parent>=0&&nodes[node.parent])points.push(nodes[node.parent].position,node.position);
      geometry(tree,points);dirty=false;
    },
    accepted(frames,goal) {
      const selected=attemptsByGoal.get(JSON.stringify(goal));
      if(selected){nodes=selected;dirty=true;}
      const points=frames.map(frame=>new THREE.Vector3().fromArray(frame.position).multiplyScalar(.001)),segments=[];
      for(let i=1;i<points.length;i++)segments.push(points[i-1],points[i]);
      geometry(route,segments);populated=true;visibility();
    },
    clear(){nodes=[];attempts=0;goalKey=null;attemptsByGoal.clear();populated=false;dirty=false;geometry(tree,[]);geometry(route,[]);visibility();}
  };
}
