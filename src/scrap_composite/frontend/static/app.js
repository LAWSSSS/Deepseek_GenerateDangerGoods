const $ = id => document.getElementById(id);
const poseFields = {azimuth: 'azimuth', elevation: 'elevation', size: 'size', rotX: 'rot_x', rotY: 'rot_y', rotZ: 'rot_z'};
let session, view, activeAssetId, position, contactAnchor, syncTimer, autoTimer;
let assets = [], modelReady = false, busy = false, syncQueue = Promise.resolve();

async function api(path, options) {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => null);
  if (!response.ok || !data) throw Error(data?.error || `本地服务请求失败（HTTP ${response.status}）`);
  return data;
}
function post(data) { return {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)}; }
function endpoint(tail = '') { return `/api/sessions/${session.id}${tail}`; }
function say(text, failed = false) {
  $('status').textContent = text;
  $('status').toggleAttribute('data-error', failed);
}
function error(e) { say('操作未完成。' + e.message, true); }

function updateControls() {
  const ready = modelReady && !busy;
  const canChoose = assets.some(asset => asset.available);
  $('asset').disabled = busy || !canChoose;
  for (const id of ['chooseBackground', 'browseFile', 'backgroundInput', 'sceneButton', 'retryModel']) $(id).disabled = busy;
  for (const id of ['poseControls', 'preview', 'resetView', 'analyze', 'chatText', 'undo']) $(id).disabled = !ready;
  $('chat').disabled = !ready || !$('chatText').value.trim();
  $('export').disabled = !ready || $('result').hidden;
  $('runAuto').disabled = !ready || session?.automatic?.status === 'running';
  $('regions').inert = busy;
}

// Serialize session mutations, so delayed responses cannot replace a newly selected background or model.
async function runAction(action) {
  if (busy) return;
  busy = true;
  updateControls();
  try { await action(); } catch (e) { error(e); }
  finally { busy = false; updateControls(); }
}
function showLive() {
  $('result').hidden = true;
  $('object').hidden = !modelReady;
  $('mask').hidden = true;
  $('download').hidden = true;
  $('exports').replaceChildren();
  updateControls();
}

function captureAnchor() {
  const uv = view?.reference();
  contactAnchor = uv ? [position.x + (uv[0] - .5) * Math.round(position.size),
    position.y + (uv[1] - .5) * Math.round(position.size)] : null;
}
function layout() {
  if (!session) return;
  const scale = $('stage').clientWidth / session.width;
  const obj = $('object');
  obj.style.width = obj.style.height = position.size * scale + 'px';
  obj.style.left = (position.x - position.size / 2) * scale + 'px';
  obj.style.top = (position.y - position.size / 2) * scale + 'px';
  for (const [id, key] of Object.entries(poseFields)) $(id).value = Math.round(position[key]);
  let caption = `${session.width} × ${session.height} px`;
  if (modelReady) caption += ` · 中心 (${Math.round(position.x)}, ${Math.round(position.y)}) · 占位 ${Math.round(position.size)} px`;
  const marker = $('contactMarker');
  marker.hidden = !modelReady || !$('lockContact').checked || !contactAnchor;
  if (!marker.hidden) {
    marker.style.left = contactAnchor[0] * scale + 'px';
    marker.style.top = contactAnchor[1] * scale + 'px';
    caption += ` · 接触参考点 (${Math.round(contactAnchor[0])}, ${Math.round(contactAnchor[1])})`;
  }
  $('coordinates').textContent = caption;
}
function render() {
  view.draw(position);
  const uv = view.reference();
  if ($('lockContact').checked && uv) {
    if (!contactAnchor) captureAnchor();
    else Object.assign(position, view.anchoredCenter(contactAnchor, uv, Math.round(position.size)));
  }
  layout();
}
function rotate() { view.rotate(position); render(); }

function sync() {
  if (!session || !modelReady) return;
  const sid = session.id;
  const state = {...position, asset: activeAssetId};
  if ($('lockContact').checked && contactAnchor) state.contact_anchor = [...contactAnchor];
  syncQueue = syncQueue.catch(() => {}).then(async () => {
    const updated = await api(`/api/sessions/${sid}/state`, post(state));
    if (session.id !== sid) return;
    session = updated;
    $('regions').replaceChildren();
    $('checks').textContent = '已记录摆位。接触、尺度、可见面、保护区和遮挡均待核验。';
  });
  return syncQueue;
}
function schedule() {
  clearTimeout(syncTimer);
  syncTimer = setTimeout(() => { syncTimer = null; sync().catch(error); }, 250);
}
async function syncPending() {
  if (!syncTimer) return syncQueue;
  clearTimeout(syncTimer);
  syncTimer = null;
  return sync();
}
async function loadModel() {
  const asset = assets.find(row => row.id === $('asset').value);
  $('retryModel').hidden = true;
  $('modelReview').hidden = !asset?.review_url;
  if (asset?.review_url) $('modelReview').href = asset.review_url;
  if (!session || !asset?.available) return;
  modelReady = false;
  contactAnchor = null;
  view?.clear();
  showLive();
  layout();
  $('regions').replaceChildren();
  $('checks').textContent = '模型载入后可核对摆位。';
  $('assetStatus').textContent = '正在载入三维模型…';
  try {
    if (!view) {
      let module;
      try { module = await import('./placement-view.js'); }
      catch { throw Error('无法载入三维组件，请检查前端依赖是否已安装，再重新选择模型。'); }
      try { view = module.createView($('modelCanvas')); }
      catch { throw Error('浏览器无法初始化三维画布，请启用硬件加速或更换支持 WebGL 2 的浏览器。'); }
    }
    await view.load('/api/model/' + encodeURIComponent(asset.id));
    activeAssetId = asset.id;
    modelReady = true;
    rotate();
    await sync();
    showLive();
    const review = asset.model_review === 'approved' ? '模型已确认' : asset.manual_only ? '候选模型待审' : '原模型';
    const evidence = asset.shape_basis === 'user_feedback_revision' ? '修订后的源视图待核' : asset.passed ? '源视图检查通过' : '源视图未通过或待核';
    $('assetStatus').textContent = [review, evidence, asset.reason || '姿态和场景融合仍待复核。'].join(' · ');
    say('模型已载入，可在画布上摆放。');
  } catch (e) {
    modelReady = false;
    showLive();
    layout();
    $('assetStatus').textContent = '模型载入失败，可重试或重新选择。';
    $('retryModel').hidden = false;
    throw e;
  }
}
async function setup(file) {
  if (file && file.size > 30 * 1024 * 1024) throw Error('照片超过 30 MB，请选择较小的文件。');
  await syncPending().catch(() => {});
  const form = new FormData();
  if (file) form.append('background', file);
  say('正在读取场景背景…');
  try { session = await api('/api/sessions', {method: 'POST', body: form}); }
  catch (e) {
    if (!file) throw Error('现有监控背景读取失败，请检查本机场景配置，或选择一张背景照片。');
    throw e;
  }
  clearTimeout(autoTimer);
  syncQueue = Promise.resolve();
  view?.clear();
  modelReady = false;
  contactAnchor = null;
  position = {x: session.width * .5, y: session.height * .65, size: Math.max(16, Math.round(Math.min(session.width, session.height) * .16)),
    azimuth: 80, elevation: 45, rot_x: 0, rot_y: 0, rot_z: 0};
  $('backgroundName').textContent = file?.name || '现有监控背景';
  $('stage').hidden = false;
  $('empty').hidden = true;
  $('background').src = endpoint('/background');
  for (const id of ['regions', 'messages', 'autoResult']) $(id).replaceChildren();
  $('chatText').value = '';
  $('checks').textContent = '背景已载入，选择可用模型后可检查摆位。';
  showLive();
  layout();
  say('背景已载入。' + (!assets.some(a => a.available) ? '请在建模工作台发布可用模型。' : ''));
  await loadModel();
}

function candidates() {
  const root = $('regions');
  root.replaceChildren();
  const priority = c => c.object_overlap_pixels > 0 ? (c.kind === 'thin_line_proposal' ? 2 : 1) : 0;
  const ranked = [...session.candidates].sort((a, b) => priority(b) - priority(a) || (b.object_overlap_pixels || 0) - (a.object_overlap_pixels || 0));
  if (!ranked.length) {
    root.textContent = '没有候选区域，遮挡仍需人工检查。';
    return;
  }
  const hint = document.createElement('p');
  hint.className = 'hint';
  hint.textContent = '勾选位于物体前方的区域，用圆点选择承托面。悬停或聚焦可查看区域。';
  root.append(hint);
  for (const c of ranked) {
    const row = document.createElement('div');
    row.className = 'region';
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.checked = session.selected.includes(c.id);
    input.dataset.id = c.id;
    input.setAttribute('aria-label', `区域 ${c.id} 位于物体前方`);
    const radio = document.createElement('input');
    radio.type = 'radio';
    radio.name = 'support';
    radio.checked = session.support === c.id;
    radio.disabled = c.kind === 'thin_line_proposal';
    radio.setAttribute('aria-label', `设区域 ${c.id} 为承托面`);
    function saveRegion(control, path, data, message) {
      return runAction(async () => {
        try {
          session = await api(endpoint(path), post({revision: session.revision, ...data}));
          showLive();
          say(message);
        } finally {
          // Restore server-confirmed choices even when foreground and support conflict.
          candidates();
        }
      }).then(() => {
        const selector = 'input[type=' + control.type + '][data-id="' + c.id + '"]';
        root.querySelector(selector)?.focus({preventScroll: true});
      });
    }
    radio.dataset.id = c.id;
    radio.onchange = () => saveRegion(radio, '/support', {support: c.id}, '已选择承托面，重新预览后可核对接触斑。');
    input.onchange = () => {
      const selected = Array.from(root.querySelectorAll('input[type=checkbox]:checked'), item => Number(item.dataset.id));
      return saveRegion(input, '/selection', {selected}, '已更新前景区域。');
    };
    const labels = [`${radio.disabled ? '细钢件' : '区域'} ${c.id}`];
    if (c.object_overlap_pixels > 0) labels.push(input.checked ? '已设为前景' : '与物体相交，前后待确认');
    if (c.support_score > 0) labels.push(`接触斑覆盖 ${Math.round(c.support_score * 100)}%`);
    row.append(input, radio, document.createTextNode(labels.join(' · ')));
    row.onmouseenter = row.onfocusin = () => { $('mask').src = endpoint('/mask/' + c.id); $('mask').hidden = false; };
    row.onmouseleave = row.onfocusout = () => { $('mask').hidden = true; };
    root.append(row);
  }
}

async function generate() {
  await syncPending();
  say('正在核对投影、接触斑和场景光色…');
  view.draw(position);
  session = await api(endpoint('/preview'), post({revision: session.revision, image: $('modelCanvas').toDataURL('image/png')}));
  candidates();
  $('exports').replaceChildren();
  $('result').src = endpoint('/preview.png') + '?t=' + Date.now();
  $('result').hidden = false;
  $('object').hidden = $('mask').hidden = true;
  $('download').href = $('result').src;
  $('download').hidden = false;
  $('checks').textContent = `投影长边 ${session.rendered_major_px} px，网页与 Blender 的轮廓 IoU 为 ${session.projection_iou}。` + session.warnings.join(' ');
  say('待审预览已生成。');
}

function drag(handle, mode) {
  handle.addEventListener('contextmenu', event => event.preventDefault());
  handle.addEventListener('pointerdown', event => {
    if (busy || !modelReady) return;
    event.preventDefault();
    event.stopPropagation();
    showLive();
    const roll = event.button === 2 || event.shiftKey;
    const action = mode === 'move' && ($('rotateMode').getAttribute('aria-pressed') === 'true' || roll) ? 'rotate' : mode;
    handle.setPointerCapture(event.pointerId);
    const start = {...position}, x = event.clientX, y = event.clientY;
    const move = ev => {
      if (busy) return;
      const scale = session.width / $('stage').clientWidth, dx = ev.clientX - x, dy = ev.clientY - y;
      if (action === 'move') {
        position.x = Math.max(0, Math.min(session.width, start.x + dx * scale));
        position.y = Math.max(0, Math.min(session.height, start.y + dy * scale));
        captureAnchor();
        layout();
      } else if (action === 'resize') {
        position.size = Math.max(16, Math.min(Math.max(session.width, session.height), start.size + (dx + dy) * scale));
        render();
      } else if (ev.altKey) {
        position.azimuth = ((start.azimuth + dx * .35) % 360 + 360) % 360;
        position.elevation = Math.max(-89, Math.min(89, start.elevation - dy * .25));
        render();
      } else {
        if (roll) position.rot_z = Math.max(-180, Math.min(180, start.rot_z + dx * .35));
        else {
          position.rot_y = Math.max(-180, Math.min(180, start.rot_y + dx * .35));
          position.rot_x = Math.max(-180, Math.min(180, start.rot_x - dy * .25));
        }
        rotate();
      }
    };
    const end = () => {
      handle.removeEventListener('pointermove', move);
      handle.removeEventListener('pointerup', end);
      handle.removeEventListener('pointercancel', end);
      schedule();
    };
    handle.addEventListener('pointermove', move);
    handle.addEventListener('pointerup', end);
    handle.addEventListener('pointercancel', end);
  });
}
drag($('object'), 'move');
drag($('rotate'), 'rotate');
drag($('resize'), 'resize');
new ResizeObserver(layout).observe($('stage'));
const direction = {ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1]};
$('object').onkeydown = event => {
  const delta = direction[event.key];
  if (!delta || busy || !modelReady) return;
  event.preventDefault();
  const step = event.shiftKey ? 10 : 1;
  position.x = Math.max(0, Math.min(session.width, position.x + step * delta[0]));
  position.y = Math.max(0, Math.min(session.height, position.y + step * delta[1]));
  captureAnchor();
  showLive();
  layout();
  schedule();
};

for (const id of ['moveMode', 'rotateMode']) $(id).onclick = () => {
  $('moveMode').setAttribute('aria-pressed', String(id === 'moveMode'));
  $('rotateMode').setAttribute('aria-pressed', String(id === 'rotateMode'));
  $('object').classList.toggle('rotating', id === 'rotateMode');
  showLive();
};
$('lockContact').onchange = () => { captureAnchor(); showLive(); layout(); schedule(); };
for (const [id, key] of Object.entries(poseFields)) $(id).onchange = () => {
  const field = $(id);
  const max = id === 'size' ? Math.max(session.width, session.height) : Number(field.max);
  position[key] = Math.max(Number(field.min), Math.min(max, Number(field.value) || 0));
  showLive();
  if (key.startsWith('rot_')) rotate(); else render();
  schedule();
};
$('resetView').onclick = () => { position.azimuth = 80; position.elevation = 45; showLive(); render(); schedule(); };
$('resetObject').onclick = () => { position.rot_x = position.rot_y = position.rot_z = 0; showLive(); rotate(); schedule(); };

$('chooseBackground').onclick = $('browseFile').onclick = () => $('backgroundInput').click();
$('retryModel').onclick = () => runAction(loadModel);
$('backgroundInput').onchange = event => {
  const file = event.target.files[0];
  if (file) runAction(() => setup(file));
  event.target.value = '';
};
$('sceneButton').onclick = () => runAction(() => setup());
$('asset').onchange = () => runAction(async () => { await syncPending().catch(() => {}); await loadModel(); });
const viewport = $('viewport');
viewport.ondragover = event => { event.preventDefault(); if (!busy) viewport.classList.add('drag-over'); };
viewport.ondragleave = event => { if (!viewport.contains(event.relatedTarget)) viewport.classList.remove('drag-over'); };
viewport.ondrop = event => {
  event.preventDefault();
  viewport.classList.remove('drag-over');
  const file = event.dataTransfer.files[0];
  if (file) runAction(() => setup(file));
};
$('analyze').onclick = () => runAction(async () => {
  await syncPending();
  say('正在分析场景，首次可能需要数分钟…');
  session = await api(endpoint('/analyze'), {method: 'POST'});
  candidates(); showLive(); say('场景分析已结束，请检查候选区域。');
});
$('undo').onclick = () => runAction(async () => {
  const generation = session.generation;
  session = await api(endpoint('/undo'), {method: 'POST'});
  if (session.generation === generation) { say('没有可撤销的遮挡调整。'); return; }
  candidates(); showLive(); say('已撤销上次遮挡调整。');
});
$('preview').onclick = () => runAction(generate);
$('export').onclick = () => runAction(async () => {
  say('正在调用 Photoshop 导出…');
  const result = await api(endpoint('/export'), {method: 'POST'});
  $('exports').replaceChildren();
  for (const ext of ['png', 'psd']) {
    const link = document.createElement('a');
    link.href = endpoint('/export/manual.' + ext);
    link.textContent = ext === 'psd' ? '下载分层 PSD' : '下载 Photoshop PNG';
    $('exports').append(link);
  }
  $('checks').textContent = '编辑范围外变化的像素数为 ' + result.outside_changed_pixels + '。' + result.limitations.join('。');
  say('Photoshop 待审文件已导出。');
});
$('chatText').oninput = updateControls;
$('chat').onclick = () => runAction(async () => {
  await syncPending();
  const data = await api(endpoint('/chat'), post({text: $('chatText').value, revision: session.revision}));
  session = data.session;
  $('messages').textContent = 'AI：' + data.reply;
  $('chatText').value = '';
  candidates(); showLive(); await generate();
});
for (const id of ['manualTab', 'autoTab']) $(id).onclick = () => {
  const manual = id === 'manualTab';
  $('manualControls').hidden = !manual;
  $('automaticControls').hidden = manual;
  $('manualTab').setAttribute('aria-pressed', String(manual));
  $('autoTab').setAttribute('aria-pressed', String(!manual));
};

async function pollAutomatic(sid) {
  try {
    const current = await api(`/api/sessions/${sid}`);
    if (session?.id !== sid) return;
    session.automatic = current.automatic;
    if (current.automatic?.status === 'running') {
      autoTimer = setTimeout(() => pollAutomatic(sid), 5000);
      return;
    }
    const record = current.automatic;
    if (!record) throw Error('未找到本次自动放置记录。');
    $('autoResult').textContent = (record.status === 'finished' ? '运行结束，结果待核对。' : '运行已停止，结果待审。') + '输出目录：' + record.output;
    const link = document.createElement('a');
    link.textContent = '查看运行记录';
    link.href = endpoint('/automatic/result.json');
    $('autoResult').append(' ', link);
    updateControls();
  } catch (e) {
    if (session?.id !== sid) return;
    error(e);
    autoTimer = setTimeout(() => pollAutomatic(sid), 5000);
  }
}
$('runAuto').onclick = () => runAction(async () => {
  await syncPending();
  session.automatic = await api(endpoint('/automatic'), {method: 'POST'});
  $('autoResult').textContent = '已启动，输出目录：' + session.automatic.output;
  const sid = session.id;
  autoTimer = setTimeout(() => pollAutomatic(sid), 5000);
});

async function readAssets() {
  assets = await api('/api/assets?view=placement');
  const choices = assets.filter(a => a.available);
  $('asset').replaceChildren();
  if (!choices.length) $('asset').append(new Option('当前没有可用模型', ''));
  for (const a of assets) {
    const status = a.available ? a.model_review === 'approved' ? '（模型已通过）' : a.manual_only || !a.passed ? '（待审）' : '' : '（文件不可用）';
    const label = String(a.id).includes('-') ? a.id : String(a.number).padStart(2, '0');
    const option = new Option(`${label} ${a.name} ${status}`, a.id);
    option.disabled = !a.available;
    $('asset').append(option);
  }
  if (choices.length) {
    $('asset').value = choices[0].id;
    say(`模型库中有 ${choices.length} 个可用选项，请选择背景照片。`);
  } else {
    say('可先载入背景照片。');
    $('assetStatus').textContent = '模型文件尚未就绪。';
    const link = document.createElement('a');
    link.href = '/modeling'; link.textContent = '前往建模工作台';
    $('assetStatus').append(' ', link);
  }
}
await runAction(readAssets);
if (!assets.length && $('status').hasAttribute('data-error')) {
  $('asset').replaceChildren(new Option('模型库读取失败', ''));
  $('assetStatus').textContent = '请检查本地服务后刷新本页。';
}
