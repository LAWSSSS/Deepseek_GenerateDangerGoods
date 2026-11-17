import * as THREE from 'three';

// glTF is normalized to a 1.35-unit longest axis by the placement viewer.
// Clip the actual triangles at the same near-ground band as Blender. The
// midpoint is only a placement reference; each foot still needs support.
export function groundReference(model, span = 1.35) {
  model.updateMatrixWorld(true);
  const meshes = [];
  let floor = Infinity;
  model.traverse(mesh => {
    if (!mesh.isMesh || !mesh.visible) return;
    const attr = mesh.geometry.attributes.position, points = [];
    for (let i = 0; i < attr.count; i++) {
      const point = new THREE.Vector3().fromBufferAttribute(attr, i).applyMatrix4(mesh.matrixWorld);
      floor = Math.min(floor, point.y);
      points.push(point);
    }
    meshes.push({points, index: mesh.geometry.index});
  });
  if (!Number.isFinite(floor)) return null;
  const height = floor + span * .012, bounds = new THREE.Box3();
  for (const {points, index} of meshes) {
    const count = index ? index.count : points.length;
    for (let i = 0; i + 2 < count; i += 3) {
      const triangle = [0, 1, 2].map(j => points[index ? index.getX(i + j) : i + j]);
      if (triangle.every(p => p.y > height)) continue;
      for (let j = 0; j < 3; j++) {
        const first = triangle[j], last = triangle[(j + 1) % 3];
        if (first.y <= height) bounds.expandByPoint(first);
        if ((first.y <= height) !== (last.y <= height)) {
          bounds.expandByPoint(first.clone().lerp(last, (height - first.y) / (last.y - first.y)));
        }
      }
    }
  }
  if (bounds.isEmpty()) return null;
  const point = bounds.getCenter(new THREE.Vector3());
  point.y = floor;
  return point;
}

export function projectedReference(point, camera) {
  if (!point) return null;
  const projected = point.clone().project(camera);
  return [(projected.x + 1) / 2, (1 - projected.y) / 2];
}

export function anchoredCenter(anchor, uv, side) {
  return {x: anchor[0] - (uv[0] - .5) * side,
          y: anchor[1] - (uv[1] - .5) * side};
}
