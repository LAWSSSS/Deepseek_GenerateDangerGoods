const field = (id) => document.getElementById(id);
const labels = {
  queued: '排队中', running: '正在建模', interrupted: '执行中断',
  ready: '候选可预览', failed: '建模失败', reconstruction: '照片重建',
  import: '检查导入素材', material: '匹配部件材质', blender_validation: 'Blender 生成与回读',
};
const downloadNames = {preview: '下载 GLB', render: '下载 Blender 模型', verification: '核验 JSON'};
let selection;
let timer;

async function request(path, options) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error);
  return data;
}

function message(text) {
  field('message').textContent = text;
}

function syncInputs() {
  const newAsset = !field('assets').value;
  field('name-field').hidden = !newAsset;
  field('asset-name').required = newAsset;
  field('source').required = newAsset;
}

async function loadAssets(assetId = field('assets').value) {
  const assets = await request('/api/modeling/assets');
  field('assets').replaceChildren(new Option('新增物品', ''));
  for (const asset of assets) {
    const marker = asset.archived ? ' · 归档' : asset.available ? ' · 可用' : ' · 未发布';
    field('assets').append(new Option(asset.id + ' · ' + asset.name + marker, asset.id));
  }
  field('assets').value = assetId;
  syncInputs();
}

async function show(record) {
  selection = record;
  clearTimeout(timer);
  const ready = record?.state === 'ready';
  field('publish').hidden = !ready;
  field('resume').hidden = !['failed', 'interrupted', 'queued'].includes(record?.state);
  field('board').hidden = !ready;
  field('verification').hidden = !record || !Object.keys(record.checks || {}).length;
  field('checks').textContent = record ? JSON.stringify(record.checks, null, 2) : '';
  field('downloads').replaceChildren();
  if (!record) {
    message('请选择物品或创建候选');
    return;
  }
  message([labels[record.state], record.stage !== record.state && labels[record.stage], record.error]
    .filter(Boolean).join(' · '));
  if (ready) {
    field('board').src = record.urls.board;
    for (const [kind, name] of Object.entries(downloadNames)) {
      const link = document.createElement('a');
      link.href = record.urls[kind];
      if (kind !== 'verification') link.download = kind === 'render' ? 'model.blend' : 'preview.glb';
      link.textContent = name;
      field('downloads').append(link);
    }
  }
  if (['queued','running'].includes(record.state)) {
    timer = setTimeout(async () => {
      try { await show(await request('/api/modeling/builds/' + record.id)); }
      catch (error) { message(error.message); }
    }, 1500);
  }
}

async function loadHistory(job) {
  const asset = field('assets').value;
  const items = asset ? await request('/api/modeling/assets/' + asset + '/builds') : [];
  field('history').replaceChildren(...items.map((item) =>
    new Option(item.created
      ? new Date(item.created * 1000).toLocaleString()
      : item.id.slice(0, 8), item.id)));
  const current = items.find((item) => item.id === job) || items[0];
  if (current) field('history').value = current.id;
  await show(current);
}

field('assets').onchange = async () => {
  syncInputs();
  try { await loadHistory(); } catch (error) { message(error.message); }
};
field('history').onchange = async () => {
  try { await show(await request('/api/modeling/builds/' + field('history').value)); }
  catch (error) { message(error.message); }
};
field('build').onsubmit = async (event) => {
  event.preventDefault();
  field('submit').disabled = true;
  try {
    const form = new FormData(event.currentTarget);
    const options = {operation: field('operation').value};
    for (const key of ['subject','dimension-source','material-library']) {
      if (field(key).value.trim()) options[key.replaceAll('-', '_')] = field(key).value.trim();
    }
    if (field('major-mm').value) options.major_mm = Number(field('major-mm').value);
    form.set('options', JSON.stringify(options));
    const record = await request('/api/modeling/builds', {method: 'POST', body: form});
    await loadAssets(record.asset.id);
    await loadHistory(record.id);
  } catch (error) { message(error.message); }
  finally { field('submit').disabled = false; }
};
field('resume').onclick = async () => {
  try {
    await request('/api/modeling/builds/' + selection.id + '/resume', {method: 'POST'});
    await loadHistory(selection.id);
  } catch (error) { message(error.message); }
};
field('publish').onclick = async () => {
  const record = selection;
  try {
    await request('/api/modeling/assets/' + record.asset.id + '/publish', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({version: record.id}),
    });
    await loadAssets(record.asset.id);
    message('已发布。新的生成任务可以选择此版本。');
    field('publish').hidden = true;
  } catch (error) { message(error.message); }
};

loadAssets().then(() => loadHistory()).catch((error) => message(error.message));
