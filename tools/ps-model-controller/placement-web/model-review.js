import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

const $=id=>document.getElementById(id);
const NUMBER=Number(document.body.dataset.number||1);
const NEW_VERSION=document.body.dataset.newVersion||'';
const PREVIOUS_VERSION=document.body.dataset.previousVersion||'';
const renderer=new THREE.WebGLRenderer({canvas:$('canvas'),antialias:true,alpha:false,preserveDrawingBuffer:true});
renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.setPixelRatio(Math.min(devicePixelRatio,2));
const scene=new THREE.Scene();scene.background=new THREE.Color('#d9dedd');
const camera=new THREE.PerspectiveCamera(35,1,.01,100);
const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
scene.add(new THREE.AmbientLight(0xffffff,.75));
for(const [position,intensity] of [[[4,6,5],1.4],[[-4,3,-4],.7]]){
  const light=new THREE.DirectionalLight(0xffffff,intensity);light.position.set(...position);scene.add(light);
}
// Neutral softbox-shaped reflections reveal metal without embedding light in
// the model. This environment is used only in the standalone review page.
const width=512,height=256,pixels=new Uint8Array(width*height*4);
const panels=[[.13,.46,.045,.28,.82],[.39,.43,.055,.34,.92],[.72,.47,.035,.25,.74],[.91,.30,.09,.11,.52]];
for(let y=0;y<height;y++)for(let x=0;x<width;x++){
  const u=x/width,v=y/height;let value=.30+.10*Math.cos((v-.35)*Math.PI);
  for(const [cx,cy,sx,sy,power] of panels){const d=Math.abs(u-cx),dx=Math.min(d,1-d)/sx,dy=Math.abs(v-cy)/sy;value+=power*Math.exp(-Math.pow(dx,6)-Math.pow(dy,6))}
  const channel=Math.round(Math.min(1,value)*255),i=(y*width+x)*4;pixels[i]=pixels[i+1]=pixels[i+2]=channel;pixels[i+3]=255;
}
const texture=new THREE.DataTexture(pixels,width,height,THREE.RGBAFormat);
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
    mark(['old','previous','new'],which);$('loading').hidden=true;
    $('status').textContent=which==='new'?(document.body.dataset.newStatus||'新版：模型和场景均待人工核验。'):which==='previous'?'上一版六视图重建：模型与场景待审。':'原始模型由单张做锈素材重建；源视图结构待审。';
    window.reviewState={which,mode,id:row.id};
  }catch(error){$('loading').textContent='加载失败：'+error.message;$('status').textContent='模型加载失败'}
}
for(const which of ['new','previous','old'])$(which).onclick=()=>load(which);
for(const m of ['clay','material'])$(m).onclick=()=>{mode=m;styleModel();mark(['clay','material'],m);if(window.reviewState)window.reviewState.mode=m};
$('reset').onclick=reset;
window.addEventListener('resize',resize);resize();reset();
function animate(){requestAnimationFrame(animate);controls.update();renderer.render(scene,camera)}animate();
try{
  const response=await fetch('/api/assets');if(!response.ok)throw Error('模型库不可用');
  const rows=await response.json();items.old=rows.find(x=>x.number===NUMBER&&!x.manual_only);
  items.previous=PREVIOUS_VERSION?rows.find(x=>x.number===NUMBER&&x.version===PREVIOUS_VERSION):null;
  items.new=rows.find(x=>x.number===NUMBER&&x.version===NEW_VERSION)||null;
  $('previous').disabled=!items.previous;
  if(items.new)$('downloadModel').href='/api/model/'+items.new.id;
  await load(items.new?'new':'old');
}catch(error){$('loading').textContent='无法读取模型库：'+error.message}
