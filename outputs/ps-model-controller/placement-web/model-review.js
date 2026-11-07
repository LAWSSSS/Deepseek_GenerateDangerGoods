import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

const $=id=>document.getElementById(id);
const renderer=new THREE.WebGLRenderer({canvas:$('canvas'),antialias:true,alpha:false,preserveDrawingBuffer:true});
renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.setPixelRatio(Math.min(devicePixelRatio,2));
const scene=new THREE.Scene();scene.background=new THREE.Color('#d9dedd');
const camera=new THREE.PerspectiveCamera(35,1,.01,100);
const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
scene.add(new THREE.AmbientLight(0xffffff,1.0));
for(const [position,intensity] of [[[4,6,5],2.0],[[-4,3,-4],1.0]]){
  const light=new THREE.DirectionalLight(0xffffff,intensity);light.position.set(...position);scene.add(light);
}
// Metals need a neutral reflected environment as well as direct lights.
const pixels=new Uint8Array(64*32*4).fill(255), texture=new THREE.DataTexture(pixels,64,32,THREE.RGBAFormat);
texture.colorSpace=THREE.SRGBColorSpace;texture.mapping=THREE.EquirectangularReflectionMapping;texture.needsUpdate=true;
const generator=new THREE.PMREMGenerator(renderer);scene.environment=generator.fromEquirectangular(texture).texture;
texture.dispose();generator.dispose();
const clay=new THREE.MeshStandardMaterial({color:0xb4b5b0,roughness:.85,metalness:0});
let current=null, items={}, mode='material', loadSerial=0;
function resize(){const rect=$('canvas').getBoundingClientRect();renderer.setSize(rect.width,rect.height,false);camera.aspect=rect.width/rect.height;camera.updateProjectionMatrix()}
function reset(){camera.position.set(2.5,1.6,3.8);camera.position.normalize().multiplyScalar(3.1);controls.target.set(0,0,0);controls.update()}
function styleModel(){if(!current)return;current.traverse(obj=>{if(obj.isMesh)obj.material=mode==='clay'?clay:obj.userData.originalMaterial})}
function mark(group,id){for(const name of group)$(name).classList.toggle('selected',name===id)}
async function load(which){
  const row=items[which];if(!row){$('status').textContent='模型文件尚未准备好。';return}
  const serial=++loadSerial;$('loading').textContent='正在加载模型…';$('loading').hidden=false;
  try{
    const gltf=await new GLTFLoader().loadAsync('/api/model/'+row.id);if(serial!==loadSerial)return;
    if(current){scene.remove(current);current.traverse(o=>{if(o.isMesh)o.geometry.dispose()})}
    const model=new THREE.Group();gltf.scene.rotation.x=-Math.PI/2;model.add(gltf.scene);
    model.updateMatrixWorld(true);const box=new THREE.Box3().setFromObject(model),center=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3());
    model.position.sub(center);model.scale.setScalar(1.35/Math.max(size.x,size.y,size.z));model.position.multiplyScalar(model.scale.x);
    model.traverse(obj=>{if(obj.isMesh)obj.userData.originalMaterial=obj.material});current=model;scene.add(model);styleModel();
    mark(['old','new'],which);$('loading').hidden=true;
    $('status').textContent=which==='new'?'新版按六视图重建；模型和场景均待人工核验。':'旧版由单张做锈素材重建；源视图结构待审。';
    window.reviewState={which,mode,id:row.id};
  }catch(error){$('loading').textContent='加载失败：'+error.message;$('status').textContent='模型加载失败'}
}
for(const which of ['new','old'])$(which).onclick=()=>load(which);
for(const m of ['clay','material'])$(m).onclick=()=>{mode=m;styleModel();mark(['clay','material'],m);if(window.reviewState)window.reviewState.mode=m};
$('reset').onclick=reset;
window.addEventListener('resize',resize);resize();reset();
function animate(){requestAnimationFrame(animate);controls.update();renderer.render(scene,camera)}animate();
try{
  const response=await fetch('/api/assets');if(!response.ok)throw Error('模型库不可用');
  const rows=await response.json();items.old=rows.find(x=>x.number===1&&!x.manual_only);items.new=rows.find(x=>x.number===1&&x.shape_basis==='user_six_view_reference');
  if(items.new)$('downloadModel').href='/api/model/'+items.new.id;
  await load(items.new?'new':'old');
}catch(error){$('loading').textContent='无法读取模型库：'+error.message}
