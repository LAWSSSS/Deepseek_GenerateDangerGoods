import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {groundReference, projectedReference, anchoredCenter} from './contact_anchor.js';

const $=id=>document.getElementById(id), ui={};
let session=null, assets=[], model=null, position={x:0,y:0,size:300,azimuth:80,elevation:45,rot_x:0,rot_y:0,rot_z:0}, syncTimer, loadSerial=0, syncQueue=Promise.resolve(), interactionMode='move';
let referencePoint=null, contactAnchor=null;
const renderer=new THREE.WebGLRenderer({canvas:$('modelCanvas'),alpha:true,antialias:true,preserveDrawingBuffer:true});
renderer.setSize(768,768,false); renderer.setClearColor(0,0); renderer.outputColorSpace=THREE.SRGBColorSpace;
const scene=new THREE.Scene(), camera=new THREE.PerspectiveCamera(35,1,.01,50);
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
async function api(path, options={}) {let r=await fetch(path,options);let data=await r.json();if(!r.ok)throw Error(data.error||'请求失败');return data}
function post(data){return {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)}}
function endpoint(tail=''){return `/api/sessions/${session.id}${tail}`}
function say(t){$('status').textContent=t}
function error(e){say('操作未完成：'+e.message)}
function currentReference(){return projectedReference(referencePoint,camera)}
function captureAnchor(){const uv=currentReference();contactAnchor=uv?[position.x+(uv[0]-.5)*Math.round(position.size),position.y+(uv[1]-.5)*Math.round(position.size)]:null}
function render(){let a=THREE.MathUtils.degToRad(position.azimuth), e=THREE.MathUtils.degToRad(position.elevation);camera.position.set((1.35*2.35)*Math.sin(a)*Math.cos(e),(1.35*2.35)*Math.sin(e),(1.35*2.35)*Math.cos(a)*Math.cos(e));camera.lookAt(0,0,0);camera.updateProjectionMatrix();camera.updateMatrixWorld(true);const uv=currentReference();if($('lockContact').checked&&uv){if(!contactAnchor)captureAnchor();else Object.assign(position,anchoredCenter(contactAnchor,uv,Math.round(position.size)))}renderer.render(scene,camera);layout()}
function applyModelRotation(){if(!model)return;const rx=THREE.MathUtils.degToRad(position.rot_x),ry=THREE.MathUtils.degToRad(position.rot_y),rz=THREE.MathUtils.degToRad(position.rot_z);const m=new THREE.Matrix4().makeRotationZ(rz);m.multiply(new THREE.Matrix4().makeRotationY(ry));m.multiply(new THREE.Matrix4().makeRotationX(rx));model.quaternion.setFromRotationMatrix(m);referencePoint=groundReference(model);render()}
async function loadModel(){if(!session||!$('asset').value)return;const serial=++loadSerial;try{let gltf=await new GLTFLoader().loadAsync('/api/model/'+$('asset').value);if(serial!==loadSerial)return;if(model)scene.remove(model);contactAnchor=null;referencePoint=null;model=new THREE.Group();gltf.scene.rotation.x=-Math.PI/2;model.add(gltf.scene);model.updateMatrixWorld(true);let box=new THREE.Box3().setFromObject(model,true), center=box.getCenter(new THREE.Vector3()), size=box.getSize(new THREE.Vector3());gltf.scene.position.copy(center).multiplyScalar(-1);model.scale.setScalar(1.35/Math.max(size.x,size.y,size.z));scene.add(model);applyModelRotation();$('object').hidden=false;layout();const a=assets.find(x=>x.id===$('asset').value);const reviewLink=$('modelReview');if(a.review_url){reviewLink.href=a.review_url;reviewLink.textContent=`查看第 ${String(a.number).padStart(2,'0')} 件完整模型检查页`;reviewLink.hidden=false}else{reviewLink.hidden=true}const review=a.model_review==='approved'?'模型已由你确认通过':a.manual_only?'候选模型待审':'原模型';const evidence=a.shape_basis==='user_feedback_revision'?'修订后的源视图待核':a.passed?'源视图检查通过':'源视图未通过或待核';$('assetStatus').textContent=review+'；'+evidence+'；'+(a.reason||'姿态和场景融合仍待复核。');await sync()}catch(e){error(e)}}
function layout(){if(!session)return;let w=$('stage').clientWidth, scale=w/session.width;let obj=$('object');obj.style.width=position.size*scale+'px';obj.style.height=position.size*scale+'px';obj.style.left=(position.x-position.size/2)*scale+'px';obj.style.top=(position.y-position.size/2)*scale+'px';$('coordinates').textContent=`中心 (${Math.round(position.x)}, ${Math.round(position.y)}) · 占位 ${Math.round(position.size)} px`;for(let k of ['azimuth','elevation','size'])$(k).value=Math.round(position[k]);$('rotX').value=Math.round(position.rot_x);$('rotY').value=Math.round(position.rot_y);$('rotZ').value=Math.round(position.rot_z);const anchor=$('contactMarker');anchor.hidden=!model||!$('lockContact').checked||!contactAnchor;if(!anchor.hidden){anchor.style.left=contactAnchor[0]*scale+'px';anchor.style.top=contactAnchor[1]*scale+'px';$('coordinates').textContent+=` · 接触参考点 (${Math.round(contactAnchor[0])}, ${Math.round(contactAnchor[1])})`}}
async function sync(){if(!session||!model)return;const sid=session.id, state={...position,asset:$('asset').value,...($('lockContact').checked&&contactAnchor?{contact_anchor:[...contactAnchor]}:{})};syncQueue=syncQueue.catch(()=>{}).then(async()=>{let updated=await api(`/api/sessions/${sid}/state`,post(state));if(session?.id!==sid)return;session=updated;$('result').hidden=true;$('download').hidden=true;$('mask').hidden=true;$('regions').innerHTML='';$('checks').textContent='人工位置与视角已记录；落稳、尺度、可见面、保护区和遮挡均待核验。'});return syncQueue}
function schedule(){clearTimeout(syncTimer);syncTimer=setTimeout(()=>{syncTimer=null;sync().catch(error)},250)}
async function setup(file){try{let f=new FormData();if(file)f.append('background',file);session=await api('/api/sessions',{method:'POST',body:f});contactAnchor=null;referencePoint=null;position={x:session.width*.5,y:session.height*.65,size:Math.round(Math.min(session.width,session.height)*.16),azimuth:80,elevation:45,rot_x:0,rot_y:0,rot_z:0};$('background').src=endpoint('/background');$('empty').hidden=true;$('stage').style.display='block';$('result').hidden=true;$('regions').innerHTML='';layout();await loadModel();say('已加载背景，请放置物品')}catch(e){error(e)}}
function candidates(){
  const root=$('regions');
  root.innerHTML='<p class="hint">先看与物体相交的候选。悬停查看原图区域，勾选前景，圆点选择候选承托面。</p>';
  const priority=c=>c.object_overlap_pixels>0 ? (c.kind==='thin_line_proposal' ? 2 : 1) : 0;
  const ranked=[...session.candidates].sort((a,b)=>priority(b)-priority(a)
    || (b.object_overlap_pixels||0)-(a.object_overlap_pixels||0));
  for(const c of ranked){
    const row=document.createElement('label');
    row.className='region';
    const input=document.createElement('input');
    input.type='checkbox';
    input.checked=session.selected.includes(c.id);
    input.dataset.id=c.id;
    input.onchange=applySelection;
    const radio=document.createElement('input');
    radio.type='radio';
    radio.name='support';
    radio.checked=session.support===c.id;
    radio.title='设为承托面';
    radio.disabled=c.kind==='thin_line_proposal';
    radio.onchange=async()=>{
      try{
        session=await api(endpoint('/support'),post({revision:session.revision,support:c.id}));
        showLive();
        say('已选择承托面，重新预览后核对接触斑');
      }catch(e){error(e)}
    };
    const labels=[`${radio.disabled?'细钢件':'区域'} ${c.id}`];
    if(c.object_overlap_pixels>0)labels.push(input.checked?'已设为前景':'与物体相交，前后待确认');
    if(c.support_score>0)labels.push(`接触斑覆盖 ${Math.round(c.support_score*100)}%`);
    row.append(input,radio,document.createTextNode(labels.join(' · ')));
    row.onmouseenter=()=>showMask(c.id);
    row.onmouseleave=()=>$('mask').hidden=true;
    root.append(row);
  }
  if(!ranked.length)root.textContent='没有检测到候选区域，遮挡仍需人工检查。';
}
function showMask(n){$('mask').src=endpoint('/mask/'+n);$('mask').hidden=false}
async function applySelection(){try{let ids=[...document.querySelectorAll('#regions input[type=checkbox]:checked')].map(x=>Number(x.dataset.id));session=await api(endpoint('/selection'),post({revision:session.revision,selected:ids}));showLive();say('已更新前景区域，可撤销')}catch(e){error(e)}}
async function generate(){
  try{
    await syncPending();
    if(!session||!model)throw Error('请先选择背景和物品');
    const {id,revision}=session;
    say('正在核对投影、接触斑和场景光色…');
    renderer.render(scene,camera);
    const updated=await api(endpoint('/preview'),post({revision,image:renderer.domElement.toDataURL('image/png')}));
    if(session.id!==id||session.revision!==revision||session.generation>updated.generation||syncTimer)return;
    session=updated;
    candidates();
    $('exports').innerHTML='';
    $('result').src=endpoint('/preview.png')+'?t='+Date.now();
    $('result').hidden=false;
    $('object').hidden=true;
    $('mask').hidden=true;
    $('download').href=$('result').src;
    $('download').hidden=false;
    $('checks').textContent=`投影长边 ${session.rendered_major_px} px，网页与 Blender 的轮廓 IoU 为 ${session.projection_iou}。`+session.warnings.join(' ');
    say('待审预览已生成');
  }catch(e){error(e)}
}
async function syncPending(){if(syncTimer){clearTimeout(syncTimer);syncTimer=null;await sync()}else await syncQueue}
function showLive(){$('result').hidden=true;$('object').hidden=!model;$('download').hidden=true;$('exports').innerHTML=''}
function drag(handle,mode){handle.addEventListener('contextmenu',ev=>{if(mode==='move')ev.preventDefault()});handle.addEventListener('pointerdown',ev=>{if(!session)return;ev.preventDefault();ev.stopPropagation();showLive();const activeMode=mode==='move'&&(interactionMode==='rotate'||ev.button===2||ev.shiftKey)?'rotate':mode;const roll=ev.button===2||ev.shiftKey;handle.setPointerCapture(ev.pointerId);let start={...position},x=ev.clientX,y=ev.clientY;const move=ev=>{let scale=session.width/$('stage').clientWidth,dx=ev.clientX-x,dy=ev.clientY-y;if(activeMode==='move'){position.x=Math.max(0,Math.min(session.width,start.x+dx*scale));position.y=Math.max(0,Math.min(session.height,start.y+dy*scale));captureAnchor();layout()}else if(activeMode==='rotate'){if(ev.altKey){position.azimuth=((start.azimuth+dx*.35)%360+360)%360;position.elevation=Math.max(-89,Math.min(89,start.elevation-dy*.25));render()}else if(roll){position.rot_z=Math.max(-180,Math.min(180,start.rot_z+dx*.35));applyModelRotation()}else{position.rot_y=Math.max(-180,Math.min(180,start.rot_y+dx*.35));position.rot_x=Math.max(-180,Math.min(180,start.rot_x-dy*.25));applyModelRotation()}layout()}else{position.size=Math.max(16,Math.min(Math.max(session.width,session.height),start.size+(dx+dy)*scale));render()}};const end=()=>{handle.removeEventListener('pointermove',move);handle.removeEventListener('pointerup',end);handle.removeEventListener('pointercancel',end);schedule()};handle.addEventListener('pointermove',move);handle.addEventListener('pointerup',end);handle.addEventListener('pointercancel',end)})}
drag($('object'),'move');drag($('rotate'),'rotate');drag($('resize'),'resize');window.addEventListener('resize',layout);
function setInteractionMode(mode){interactionMode=mode;const moving=mode==='move';$('moveMode').classList.toggle('active',moving);$('rotateMode').classList.toggle('active',!moving);$('object').classList.toggle('rotating',!moving);if(model)showLive();say(moving?'拖动模式：左键移动物体':'拖动模式：左键旋转物体 X/Y；右键或 Shift 拖动旋转 Z；Alt+拖动调整视角')}
$('lockContact').onchange=()=>{captureAnchor();showLive();layout();schedule();say($('lockContact').checked?'已固定接触参考点，转动视角和缩放时保持位置；落稳仍需核对':'已恢复按图片中心调整')};
$('moveMode').onclick=()=>setInteractionMode('move');$('rotateMode').onclick=()=>setInteractionMode('rotate');
$('resetView').onclick=()=>{if(!session)return;showLive();position.azimuth=80;position.elevation=45;render();$('azimuth').value=80;$('elevation').value=45;schedule()};
for(let k of ['azimuth','elevation','size'])$(k).onchange=()=>{if(!session)return;position[k]=Math.max(k==='elevation'?-89:k==='size'?16:0,Math.min(k==='elevation'?89:k==='size'?Math.max(session.width,session.height):360,Number($(k).value)));showLive();layout();render();schedule()};
for(const [id,key] of [['rotX','rot_x'],['rotY','rot_y'],['rotZ','rot_z']])$(id).onchange=()=>{position[key]=Math.max(-180,Math.min(180,Number($(id).value)||0));showLive();layout();applyModelRotation();schedule()};
$('resetObject').onclick=()=>{position.rot_x=position.rot_y=position.rot_z=0;showLive();layout();applyModelRotation();schedule()};
$('backgroundInput').onchange=e=>setup(e.target.files[0]);$('sceneButton').onclick=()=>setup();$('asset').onchange=()=>loadModel();
$('analyze').onclick=async()=>{try{await syncPending();say('正在分析场景，首次可能需要数分钟…');session=await api(endpoint('/analyze'),{method:'POST'});candidates();showLive();say('请选择前景区域，或与 AI 讨论')}catch(e){error(e)}};
$('undo').onclick=async()=>{try{session=await api(endpoint('/undo'),{method:'POST'});candidates();showLive();say('已撤销')}catch(e){error(e)}};
$('preview').onclick=generate;
$('export').onclick=async()=>{try{say('正在调用 Photoshop 导出…');let result=await api(endpoint('/export'),{method:'POST'});$('exports').innerHTML=`<a href="${endpoint('/export/manual.png')}">下载 Photoshop PNG</a><br><a href="${endpoint('/export/manual.psd')}">下载分层 PSD</a>`;$('checks').textContent='Photoshop 编辑范围外像素变化：'+result.outside_changed_pixels+'。'+result.limitations.join('；');say('Photoshop 待审文件已导出')}catch(e){error(e)}};
$('chat').onclick=async()=>{try{let text=$('chatText').value;if(!text)return;let d=await api(endpoint('/chat'),post({text,revision:session.revision}));session=d.session;$('messages').textContent='AI：'+d.reply;$('chatText').value='';candidates();showLive();await generate();say('AI 变更已更新预览，请检查')}catch(e){error(e)}};
function tab(manual){$('manualControls').hidden=!manual;$('automaticControls').hidden=manual;$('manualTab').classList.toggle('active',manual);$('autoTab').classList.toggle('active',!manual);$('stageTitle').textContent=manual?'人工放置 · 实时三维占位':'自动放置';}
$('manualTab').onclick=()=>tab(true);$('autoTab').onclick=()=>tab(false);
$('runAuto').onclick=async()=>{try{await syncPending();let r=await api(endpoint('/automatic'),{method:'POST'});$('autoResult').textContent='已启动，结果目录：'+r.output;let id=setInterval(async()=>{try{let s=await api(endpoint());if(s.automatic?.status!=='running'){clearInterval(id);$('autoResult').textContent='运行结束：'+s.automatic.status+'。结果目录：'+s.automatic.output;let link=document.createElement('a');link.textContent=' 查看运行记录';link.href=endpoint('/automatic/result.json');$('autoResult').append(link)}}catch(e){clearInterval(id);error(e)}},5000)}catch(e){error(e)}};
try{assets=await api('/api/assets?view=placement');$('asset').innerHTML='';for(const a of assets){let option=document.createElement('option');option.value=a.id;option.disabled=!a.available;const status=a.model_review==='approved'?'（模型已通过）':a.manual_only||!a.passed?'（待审）':'';const label=String(a.id).includes('-')?a.id:String(a.number).padStart(2,'0');option.textContent=`${label} ${a.name} ${status}`;$('asset').append(option)}say(`模型库：${assets.length} 个选项；请选择背景`)}catch(e){error(e)}
