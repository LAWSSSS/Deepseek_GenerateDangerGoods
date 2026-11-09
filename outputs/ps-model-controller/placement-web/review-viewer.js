import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {RoomEnvironment} from 'three/addons/environments/RoomEnvironment.js';

const $ = id => document.getElementById(id);
const canvas = $('canvas');
const viewport = $('viewport');
const loader = new GLTFLoader();
const clay = new THREE.MeshStandardMaterial({color:0x858d8c, roughness:.8, metalness:0});
let renderer, scene, camera, controls, model = null;
let items = [], current = '', mode = 'material', loadSerial = 0;

function setMessage(message, show=true) {
  $('loading').textContent = message;
  $('loading').hidden = !show;
}

function initScene() {
  renderer = new THREE.WebGLRenderer({canvas, antialias:true, alpha:false, preserveDrawingBuffer:true});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  scene = new THREE.Scene();
  scene.background = new THREE.Color('#e8edef');
  camera = new THREE.PerspectiveCamera(38, 1, .01, 100);
  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.minDistance = 1.8;
  controls.maxDistance = 15;
  scene.add(new THREE.AmbientLight(0xffffff, .65));
  const key = new THREE.DirectionalLight(0xffffff, 1.45);
  key.position.set(4, 7, 5);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xffffff, .65);
  fill.position.set(-5, 3, -4);
  scene.add(fill);
  const pmrem = new THREE.PMREMGenerator(renderer);
  const room = new RoomEnvironment();
  scene.environment = pmrem.fromScene(room, .04).texture;
  scene.environmentIntensity = .75;
  room.dispose();
  pmrem.dispose();
  controls.target.set(0,0,0);
  resetView();
  resize();
  if (window.ResizeObserver) new ResizeObserver(resize).observe(viewport);
  else window.addEventListener('resize', resize);
  const animate = () => {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  };
  animate();
}

function resize() {
  if (!renderer) return;
  const width = Math.max(1, viewport.clientWidth);
  const height = Math.max(1, viewport.clientHeight);
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

function resetView() {
  if (!controls) return;
  camera.position.set(3.6, 2.2, 4.4).normalize().multiplyScalar(4.8);
  controls.target.set(0,0,0);
  controls.update();
}

function discard(group) {
  if (!group) return;
  const geometries = new Set(), materials = new Set(), textures = new Set();
  group.traverse(obj => {
    if (!obj.isMesh) return;
    if (obj.geometry) geometries.add(obj.geometry);
    const original = obj.userData.reviewMaterial || obj.material;
    for (const material of (Array.isArray(original) ? original : [original])) {
      if (material && material !== clay) materials.add(material);
    }
  });
  for (const material of materials) {
    for (const value of Object.values(material)) if (value && value.isTexture) textures.add(value);
    material.dispose();
  }
  for (const texture of textures) texture.dispose();
  for (const geometry of geometries) geometry.dispose();
}

function applyMode() {
  if (!model) return;
  model.traverse(obj => {
    if (obj.isMesh) obj.material = mode === 'clay' ? clay : obj.userData.reviewMaterial;
  });
  for (const id of ['material','clay']) {
    $(id).classList.toggle('active', mode === id);
    $(id).setAttribute('aria-pressed', String(mode === id));
  }
  if (window.reviewViewer) window.reviewViewer.mode = mode;
}

function itemFor(id) { return items.find(item => item.id === String(id)); }
function label(item) { return `${item.id.includes('-') ? item.id : item.id.padStart(2,'0')}  ${item.name}`; }

async function load(id) {
  const item = itemFor(id);
  if (!item) return;
  current = item.id;
  const index = items.indexOf(item);
  $('item').value = item.id;
  $('title').textContent = label(item);
  $('review-status').textContent = item.human_review === 'approved' ? '模型已由你确认通过；场景仍待检查'
    : item.human_review === 'revision_pending_review' ? '按你的意见修订，每次显示一件；待你复核'
    : '新模型候选，供人工检查';
  $('reference-note').textContent = item.reference_missing
    ? '37 号没有可靠对应的六视图参考；隐藏部位仍需核对。'
    : item.number === 19 ? '做锈原图包含六只气瓶；当前三维模型只显示所选的一只。'
    : '不同图片的观察角度可能不同；六视图参考未标定。';
  $('rust').src = `/api/rust/${item.id}`;
  $('board').href = `/api/board/${item.id}`;
  $('previous').disabled = index <= 0;
  $('next').disabled = index >= items.length - 1;
  const page = new URL(window.location.href);
  page.searchParams.set('item', item.id);
  history.replaceState(null, '', page);
  const serial = ++loadSerial;
  if (!item.available) {
    if (model) { scene.remove(model); discard(model); model = null; }
    setMessage(item.reason || '模型文件不可用');
    return;
  }
  setMessage('正在加载三维模型…');
  try {
    const gltf = await loader.loadAsync(`/api/model/${item.id}`);
    const next = new THREE.Group();
    gltf.scene.rotation.x = -Math.PI / 2;
    next.add(gltf.scene);
    next.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(next);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const longest = Math.max(size.x,size.y,size.z);
    if (!Number.isFinite(longest) || longest <= 0) throw Error('模型尺寸无效');
    const scale = 2.25 / longest;
    next.scale.setScalar(scale);
    next.position.copy(center).multiplyScalar(-scale);
    let meshCount = 0;
    next.traverse(obj => {
      if (obj.isMesh) { obj.userData.reviewMaterial = obj.material; meshCount++; }
    });
    if (serial !== loadSerial) { discard(next); return; }
    if (!meshCount) { discard(next); throw Error('模型没有可显示的网格'); }
    if (model) { scene.remove(model); discard(model); }
    model = next;
    scene.add(model);
    applyMode();
    resetView();
    window.reviewViewer = {id:item.id, number:item.number, mode, meshCount, ready:true};
    setMessage('',false);
  } catch (error) {
    if (serial !== loadSerial) return;
    if (model) { scene.remove(model); discard(model); model = null; }
    window.reviewViewer = {id:item.id, number:item.number, mode, ready:false};
    setMessage('模型无法加载，请尝试重新打开查看器。');
    console.error('Model loading failed', error);
  }
}

$('item').addEventListener('change', event => load(event.target.value));
$('previous').addEventListener('click', () => { const i=items.findIndex(x=>x.id===current); if(i>0)load(items[i-1].id); });
$('next').addEventListener('click', () => { const i=items.findIndex(x=>x.id===current); if(i>=0 && i<items.length-1)load(items[i+1].id); });
$('reset').addEventListener('click', resetView);
for (const value of ['material','clay']) $(value).addEventListener('click', () => { mode = value; applyMode(); });

try {
  initScene();
  const response = await fetch('/api/items');
  if (!response.ok) throw Error('模型清单不可用');
  items = await response.json();
  for (const item of items) {
    const option = document.createElement('option');
    option.value = item.id;
    option.textContent = label(item) + (item.available ? '' : '（文件不可用）');
    $('item').append(option);
  }
  let requested = new URLSearchParams(location.search).get('item') || '2';
  if (requested === '19' && itemFor('19-1')) requested = '19-1';
  if (/^0\d$/.test(requested)) requested = String(Number(requested));
  await load(itemFor(requested) ? requested : '2');
} catch (error) {
  setMessage('查看器无法启动，请确认浏览器支持三维显示并重新打开。');
  console.error('Viewer failed', error);
}
