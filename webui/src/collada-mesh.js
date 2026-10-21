import { ColladaLoader } from 'three/addons/loaders/ColladaLoader.js';
import { FileLoader, LoaderUtils } from 'three';

// Rhino-exported parts use <polygons>, which Three.js ColladaLoader skips.
// Represent the same indexed faces as <polylist>; the loader triangulates the
// triangles and quads while retaining vertex, normal and texture indices.
export function normalizeColladaPolygons(text) {
  if (!text.includes('<polygons')) return text;
  const xml = new DOMParser().parseFromString(text, 'text/xml');
  const namespace = xml.documentElement.namespaceURI;
  const materialIds = Array.from(xml.getElementsByTagName('material')).map(node => node.id);
  for (const binding of xml.getElementsByTagName('instance_material')) {
    const target = binding.getAttribute('target').replace(/^#/, '');
    if (!materialIds.includes(target)) {
      // joint5 references a removed Rhino material, but defines a single
      // replacement material. Bind its faces to that existing material.
      if (materialIds.length !== 1) throw new Error(`Missing COLLADA material: ${target}`);
      binding.setAttribute('target', `#${materialIds[0]}`);
    }
  }
  for (const polygons of Array.from(xml.getElementsByTagName('polygons'))) {
    const children = Array.from(polygons.children);
    const inputs = children.filter(node => node.localName === 'input');
    const faces = children.filter(node => node.localName === 'p');
    if (children.some(node => node.localName === 'ph')) throw new Error('COLLADA polygons with holes are not supported');
    const stride = Math.max(...inputs.map(node => Number(node.getAttribute('offset') || 0))) + 1;
    const indices = faces.map(node => node.textContent.trim().split(/\s+/));
    const counts = indices.map(face => face.length / stride);
    if (counts.some(count => !Number.isInteger(count) || count < 3 || count > 4)) {
      throw new Error('Expected triangular or quadrilateral model faces');
    }
    const polylist = xml.createElementNS(namespace, 'polylist');
    for (const attribute of polygons.attributes) polylist.setAttribute(attribute.name, attribute.value);
    polylist.setAttribute('count', String(faces.length));
    for (const input of inputs) polylist.appendChild(input.cloneNode(true));
    const vcount = xml.createElementNS(namespace, 'vcount');
    vcount.textContent = counts.join(' ');
    const p = xml.createElementNS(namespace, 'p');
    p.textContent = indices.map(face => face.join(' ')).join(' ');
    polylist.append(vcount, p);
    polygons.replaceWith(polylist);
  }
  return new XMLSerializer().serializeToString(xml);
}

export function loadColladaMesh(path, manager, done) {
  new FileLoader(manager).load(path, text => {
    try {
      const loader = new ColladaLoader(manager);
      const result = loader.parse(normalizeColladaPolygons(text), LoaderUtils.extractUrlBase(path));
      let meshCount = 0;
      result.scene.traverse(object => { if (object.isMesh && object.geometry.attributes.position?.count > 0) meshCount++; });
      if (meshCount === 0) throw new Error('The model asset contains no renderable mesh');
      done(result.scene);
    } catch (error) {
      manager.itemError(path);
      done(null, error);
    }
  }, undefined, error => done(null, error));
}
