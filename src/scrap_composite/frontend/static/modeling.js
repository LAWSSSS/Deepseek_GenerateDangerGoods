const field = (id) => document.getElementById(id);
const labels = {
  queued: '排队中', running: '正在建模', interrupted: '执行中断',
  ready: '候选可预览', failed: '建模失败', reconstruction: '照片重建',
  import: '检查导入素材', material: '匹配部件材质', blender_validation: 'Blender 生成与回读',
};
const downloadNames = {preview: '下载 GLB', render: '下载 Blender 模型', verification: '查看核验 JSON'};
let assets = [];
let selection;
let timer;
let viewRevision = 0;

async function request(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.error || `请求失败（HTTP ${response.status}）`);
  }
  return response.json();
}

function feedback(id, text = '', error = false) {
  field(id).textContent = text;
  field(id).dataset.error = String(error);
}

function syncInputs() {
  const newAsset = !field('assets').value;
  field('name-field').hidden = !newAsset;
  field('asset-name').required = newAsset;
  field('source').required = newAsset;
  field('source-help').textContent = newAsset ? '支持 PNG、JPG、WebP、GLB 和 Blender 文件。'
    : '留空可沿用已有素材，也可上传新照片、GLB 或 Blender 文件。';
}

function assetDescription(asset) {
  if (asset.archived) return '归档';
  if (asset.available) return '可用';
  return asset.published ? '发布文件不可用' : '未发布';
}

function listAssets(selectedId) {
  field('assets').replaceChildren(new Option('新增物品', ''));
  for (const asset of assets) {
    field('assets').append(new Option(`${asset.id} · ${asset.name} · ${assetDescription(asset)}`, asset.id));
  }
  field('assets').value = selectedId;
  syncInputs();
}

async function loadAssets(assetId) {
  assets = await request('/api/modeling/assets');
  listAssets(assetId ?? field('assets').value);
}

function placeholder(title, hint) {
  field('board').hidden = true;
  field('preview-title').textContent = title;
  field('preview-hint').textContent = hint;
  field('preview-placeholder').hidden = false;
}

function recordLabel(record, asset) {
  const date = record.created ? new Date(record.created * 1000).toLocaleString() : record.id.slice(0, 8);
  const state = record.id === asset?.published ? '当前发布版本' : labels[record.state] || record.state;
  return `${date} · ${state}`;
}

function show(record) {
  clearTimeout(timer);
  selection = record;
  const asset = assets.find((item) => item.id === field('assets').value);
  const ready = record?.state === 'ready';
  const published = ready && asset?.published === record.id;
  field('candidate-status').hidden = !record;
  field('publish').hidden = !ready || published;
  field('resume').hidden = !['failed', 'interrupted', 'queued'].includes(record?.state);
  field('verification').hidden = !Object.keys(record?.checks || {}).length;
  field('checks').textContent = record ? JSON.stringify(record.checks, null, 2) : '';
  field('downloads').replaceChildren();
  feedback('message');
  if (!record) {
    field('board').removeAttribute('src');
    placeholder('尚无候选预览', asset ? '为此物品创建候选后，可在这里核对模型。' : '创建候选或选择已有物品，查看模型的正面与斜侧视图。');
    return;
  }
  const state = labels[record.state] || record.state;
  field('candidate-status').textContent = published ? '当前发布版本' : state;
  field('candidate-status').dataset.state = record.state;
  if (field('history').selectedOptions[0]?.value === record.id) {
    field('history').selectedOptions[0].textContent = recordLabel(record, asset);
  }
  if (ready) {
    const board = field('board');
    if (board.getAttribute('src') !== record.urls.board) {
      placeholder('正在读取预览图', '读取完成后显示模型视图。');
      board.src = record.urls.board;
    }
    for (const [kind, name] of Object.entries(downloadNames)) {
      const link = document.createElement('a');
      link.href = record.urls[kind];
      if (kind !== 'verification') link.download = kind === 'render' ? 'model.blend' : 'preview.glb';
      link.textContent = name;
      field('downloads').append(link);
    }
    if (!published) feedback('message', '核对视图和检查记录后，可手动发布此版本。');
    return;
  }
  field('board').removeAttribute('src');
  const inProgress = ['queued', 'running'].includes(record.state);
  const stage = record.stage !== record.state && labels[record.stage];
  placeholder(inProgress ? '候选正在准备中' : '暂无可用预览', stage ? `当前步骤：${stage}` : state);
  if (record.error) feedback('message', record.error, true);
  if (inProgress) {
    const revision = viewRevision;
    timer = setTimeout(async () => {
      try {
        const updated = await request('/api/modeling/builds/' + record.id);
        if (viewRevision === revision) show(updated);
      } catch (error) {
        if (viewRevision === revision) feedback('message', `状态更新失败，请刷新页面重试。${error.message}`, true);
      }
    }, 1500);
  }
}

field('board').onload = () => {
  if (selection?.state !== 'ready') return;
  field('board').hidden = false;
  field('preview-placeholder').hidden = true;
};
field('board').onerror = () => {
  if (selection?.state === 'ready') placeholder('预览图读取失败', '请检查建模文件后刷新页面重试。');
};

async function loadHistory(job) {
  const revision = ++viewRevision;
  const assetId = field('assets').value;
  const asset = assets.find((item) => item.id === assetId);
  const history = field('history');
  show(null);
  history.disabled = true;
  history.replaceChildren(new Option(assetId ? '正在读取…' : '暂无建模记录', ''));
  if (!assetId) return;
  try {
    const items = await request(`/api/modeling/assets/${encodeURIComponent(assetId)}/builds`);
    if (revision !== viewRevision) return;
    if (!items.length) {
      history.replaceChildren(new Option('暂无建模记录', ''));
      return;
    }
    history.disabled = false;
    history.replaceChildren(...items.map((item) => new Option(recordLabel(item, asset), item.id)));
    const current = items.find((item) => item.id === job) || items[0];
    history.value = current.id;
    show(current);
  } catch (error) {
    if (revision !== viewRevision) return;
    history.replaceChildren(new Option('记录读取失败', ''));
    placeholder('建模记录读取失败', '请重新选择物品或刷新页面。');
    feedback('message', error.message, true);
  }
}

field('assets').onchange = () => {
  syncInputs();
  feedback('build-feedback');
  loadHistory();
};
field('operation').onchange = () => {
  if (field('operation').value === 'retexture') field('material-settings').open = true;
};
field('history').onchange = async () => {
  const revision = ++viewRevision;
  show(null);
  placeholder('正在读取建模记录', '读取完成后显示候选状态。');
  try {
    const current = await request('/api/modeling/builds/' + field('history').value);
    if (revision === viewRevision) show(current);
  } catch (error) {
    if (revision === viewRevision) {
      placeholder('建模记录读取失败', '请重新选择记录或刷新页面。');
      feedback('message', error.message, true);
    }
  }
};
field('build').onsubmit = async (event) => {
  event.preventDefault();
  field('submit').disabled = true;
  feedback('build-feedback', '正在提交素材…');
  let record;
  try {
    const form = new FormData(event.currentTarget);
    const options = {operation: field('operation').value};
    for (const key of ['subject', 'dimension-source', 'material-library']) {
      if (field(key).value.trim()) options[key.replaceAll('-', '_')] = field(key).value.trim();
    }
    if (field('major-mm').value) options.major_mm = Number(field('major-mm').value);
    form.set('options', JSON.stringify(options));
    record = await request('/api/modeling/builds', {method: 'POST', body: form});
    feedback('build-feedback', '候选任务已提交。');
    await loadAssets(record.asset.id);
    await loadHistory(record.id);
  } catch (error) {
    const detail = record ? '任务已提交，物品列表刷新失败。' : '';
    feedback('build-feedback', detail + error.message, true);
  } finally { field('submit').disabled = false; }
};
field('resume').onclick = async () => {
  const record = selection;
  field('resume').disabled = true;
  try {
    await request(`/api/modeling/builds/${record.id}/resume`, {method: 'POST'});
    if (selection?.id === record.id) await loadHistory(record.id);
  } catch (error) {
    if (selection?.id === record.id) feedback('message', error.message, true);
  } finally { field('resume').disabled = false; }
};
field('publish').onclick = async () => {
  const record = selection;
  field('publish').disabled = true;
  try {
    await request(`/api/modeling/assets/${encodeURIComponent(record.asset.id)}/publish`, {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({version: record.id}),
    });
    const asset = assets.find((item) => item.id === record.asset.id);
    if (asset) Object.assign(asset, {published: record.id, available: true});
    listAssets(field('assets').value);
    if (selection?.id === record.id) {
      await loadHistory(record.id);
      if (selection?.id === record.id) feedback('message', '此版本已发布到物品库。');
    }
  } catch (error) {
    if (selection?.id === record.id) feedback('message', error.message, true);
  } finally { field('publish').disabled = false; }
};

loadAssets().then(() => loadHistory()).catch((error) => feedback('build-feedback', `物品库读取失败。${error.message}`, true));
