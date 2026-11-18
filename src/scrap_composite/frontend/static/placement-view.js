import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {groundReference, projectedReference, anchoredCenter} from './contact_anchor.js';

// Loaded on the first model selection. The image workspace also runs without WebGL.
export function createView(canvas) {
  const renderer = new THREE.WebGLRenderer({canvas, alpha: true, antialias: true, preserveDrawingBuffer: true});
  renderer.setSize(768, 768, false);
  renderer.setClearColor(0, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 1, .01, 50);
  // Match the standalone model review page's neutral studio light. The white
  // ambient-only setup washed out metal and made the live model look different
  // from the reviewed GLB. The preview renderer estimates scene lighting.
  scene.add(new THREE.AmbientLight(0xffffff,.75));
  for(const [lightPosition,intensity] of [[[4,6,5],1.4],[[-4,3,-4],.7]]){
    const light=new THREE.DirectionalLight(0xffffff,intensity);
    light.position.set(...lightPosition);
    scene.add(light);
  }
  // A softbox-shaped reflection environment, copied from the review page so the
  // same glTF material has the same metal/roughness response in both views.
  const environmentWidth=512,environmentHeight=256;
  const environmentPixels=new Uint8Array(environmentWidth*environmentHeight*4);
  const panels=[[.13,.46,.045,.28,.82],[.39,.43,.055,.34,.92],[.72,.47,.035,.25,.74],[.91,.30,.09,.11,.52]];
  for(let y=0;y<environmentHeight;y++)for(let x=0;x<environmentWidth;x++){
    const u=x/environmentWidth,v=y/environmentHeight;let value=.30+.10*Math.cos((v-.35)*Math.PI);
    for(const [cx,cy,sx,sy,power] of panels){
      const d=Math.abs(u-cx),dx=Math.min(d,1-d)/sx,dy=Math.abs(v-cy)/sy;
      value+=power*Math.exp(-Math.pow(dx,6)-Math.pow(dy,6));
    }
    const channel=Math.round(Math.min(1,value)*255),i=(y*environmentWidth+x)*4;
    environmentPixels[i]=environmentPixels[i+1]=environmentPixels[i+2]=channel;
    environmentPixels[i+3]=255;
  }
  const environmentTexture=new THREE.DataTexture(environmentPixels,environmentWidth,environmentHeight,THREE.RGBAFormat);
  environmentTexture.colorSpace=THREE.SRGBColorSpace;
  environmentTexture.mapping=THREE.EquirectangularReflectionMapping;
  environmentTexture.needsUpdate=true;
  const environmentGenerator=new THREE.PMREMGenerator(renderer);
  scene.environment=environmentGenerator.fromEquirectangular(environmentTexture).texture;
  environmentTexture.dispose();environmentGenerator.dispose();

  let model = null;
  let reference = null;
  const loader = new GLTFLoader();

  function clear() {
    if (!model) return;
    scene.remove(model);
    const textures = new Set();
    model.traverse(node => {
      node.geometry?.dispose();
      for (const material of [].concat(node.material || [])) {
        for (const texture of Object.values(material)) {
          if (texture?.isTexture) textures.add(texture);
        }
        material.dispose();
      }
    });
    textures.forEach(texture => texture.dispose());
    model = null;
    reference = null;
  }

  async function load(url) {
    const gltf = await loader.loadAsync(url);
    clear();
    model = new THREE.Group();
    gltf.scene.rotation.x = -Math.PI / 2;
    model.add(gltf.scene);
    model.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(model, true);
    const size = box.getSize(new THREE.Vector3());
    gltf.scene.position.copy(box.getCenter(new THREE.Vector3())).multiplyScalar(-1);
    model.scale.setScalar(1.35 / Math.max(size.x, size.y, size.z));
    scene.add(model);
  }

  function rotate(pose) {
    if (!model) return;
    const radians = THREE.MathUtils.degToRad;
    const rotation = new THREE.Matrix4().makeRotationZ(radians(pose.rot_z));
    rotation.multiply(new THREE.Matrix4().makeRotationY(radians(pose.rot_y)));
    rotation.multiply(new THREE.Matrix4().makeRotationX(radians(pose.rot_x)));
    model.quaternion.setFromRotationMatrix(rotation);
    reference = groundReference(model);
  }

  function draw(pose) {
    const a = THREE.MathUtils.degToRad(pose.azimuth);
    const e = THREE.MathUtils.degToRad(pose.elevation);
    const distance = 1.35 * 2.35;
    camera.position.set(distance * Math.sin(a) * Math.cos(e), distance * Math.sin(e), distance * Math.cos(a) * Math.cos(e));
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld(true);
    renderer.render(scene, camera);
  }

  return {load, clear, draw, rotate, anchoredCenter, reference: () => projectedReference(reference, camera)};
}
