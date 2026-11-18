// Run with node checks_anchor.js. Geometry and final-pixel anchor regressions.
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {groundReference, projectedReference, anchoredCenter} from '../../../src/scrap_composite/frontend/static/contact_anchor.js';

const group = new THREE.Group();
group.add(new THREE.Mesh(new THREE.BoxGeometry(1.35,.7,.5)));
const base = groundReference(group);
assert.ok(base.distanceTo(new THREE.Vector3(0,-.35,0)) < 1e-6);
// Subdivision must not change the placement reference.
const divided = new THREE.Group();
divided.add(new THREE.Mesh(new THREE.BoxGeometry(1.35,.7,.5,8,8,8)));
group.rotation.set(.3,.8,.2); divided.rotation.copy(group.rotation);
assert.ok(groundReference(group).distanceTo(groundReference(divided)) < 1e-6);
const camera = new THREE.PerspectiveCamera(35,1,.01,50);
const anchor = [2340.25,1620.21];
for (const yaw of [0,80,180,360]) for (const elevation of [-45,35,45,55,89]) {
  const a = THREE.MathUtils.degToRad(yaw), e = THREE.MathUtils.degToRad(elevation);
  camera.position.set((1.35*2.35)*Math.sin(a)*Math.cos(e),(1.35*2.35)*Math.sin(e),(1.35*2.35)*Math.cos(a)*Math.cos(e));
  camera.lookAt(0,0,0); camera.updateMatrixWorld(true);
  const uv = projectedReference(groundReference(group), camera);
  for (const side of [256,512,1024]) {
    const p = anchoredCenter(anchor, uv, side);
    const actual = [Math.round(p.x-side/2)+uv[0]*side, Math.round(p.y-side/2)+uv[1]*side];
    assert.ok(Math.hypot(actual[0]-anchor[0], actual[1]-anchor[1]) <= Math.SQRT1_2);
  }
}
console.log('Contact anchor: subdivision invariance and 60 camera/resolution combinations passed.');
