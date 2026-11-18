'use strict';

const field = id => document.getElementById(id);
const selectedValues = select => Array.from(select.selectedOptions, option => option.value);
let batch = localStorage.getItem('displayBatchId');
let pollTimer;
let latestStatus;
let actionPending = false;
let statusPending = Boolean(batch);
let directoryVersion = 0;

async function requestJson(path, options) {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => null);
  if (!response.ok || data === null) {
    const error = new Error(data?.error || `本地服务返回异常（HTTP ${response.status}），请检查服务后重试。`);
    error.status = response.status;
    throw error;
  }
  return data;
}

function updateActions() {
  field('start').disabled = actionPending || statusPending || Boolean(latestStatus?.running);
  field('resume').hidden = !latestStatus?.resumable;
  field('resume').disabled = !latestStatus?.resumable || actionPending || statusPending;
  field('refresh').disabled = actionPending || statusPending;
}

function setState(text, tone = '') {
  field('batchState').textContent = text;
  field('batchState').dataset.tone = tone;
}

const extraEffectIds = ['fogExtra', 'fogMode', 'fogGain', 'weathering', 'scaleMode', 'contactMode',
  'chromaMode', 'poolKernel', 'surfaceDust', 'adjustEffects'];

function updateMode() {
  const mode = field('testMode').value;
  const test = mode !== 'batch';
  field('count').max = test ? '15' : '1000';
  if (test) {
    if (Number(field('count').value) > 15) field('count').value = 1;
    field('appearance').value = 'local';
    field('photoStyle').value = 'dusty_scrap';
  }
  for (const id of extraEffectIds) field(id).disabled = mode !== 'test';
  for (const id of ['vehicleWidth', 'vehicleHeight']) field(id).disabled = mode !== 'rebuild';
}

field('testMode').onchange = updateMode;
field('photoStyle').onchange = () => {
  if (field('photoStyle').value === 'dusty_scrap') field('appearance').value = 'local';
};
field('appearance').onchange = () => {
  if (field('appearance').value !== 'local' && field('photoStyle').value === 'dusty_scrap') {
    field('photoStyle').value = 'scrap_gray';
  }
};
field('reviewProfile').onchange = () => {
  if (field('reviewProfile').value === 'photographer') field('integrationMode').value = 'evidence_v1';
};
field('integrationMode').onchange = () => {
  if (field('integrationMode').value !== 'evidence_v1') field('reviewProfile').value = 'legacy';
};

async function checkModes() {
  try {
    const options = await requestJson('/api/batches/effects');
    const ready = Boolean(options.geometry_rebuild_trial);
    field('testMode').querySelector('[value="rebuild"]').disabled = !ready;
    field('integrationMode').querySelector('[value="coherent"]').disabled =
      !options.integration_modes?.options?.some(option => option.value === 'coherent');
    field('integrationMode').querySelector('[value="evidence_v1"]').disabled = !options.workflows?.includes('evidence_v1');
    field('reviewProfile').querySelector('[value="photographer"]').disabled = !options.review_profiles?.includes('photographer');
    field('geometryAvailability').textContent = ready
      ? '尺度与摆位重生成可用，该模式按车厢内宽重新计算尺度。'
      : '当前服务未提供尺度与摆位重生成模式，可使用原批量流程。';
  } catch {
    field('geometryAvailability').textContent = '无法读取扩展模式。请检查本地服务后刷新页面。';
  }
}

function syncModelSelection() {
  const ids = new Set(field('assets').value.split(',').map(value => value.trim()));
  for (const option of field('assetChoices').options) option.selected = ids.has(option.value);
}

async function listAssets() {
  try {
    const models = await requestJson('/api/modeling/assets');
    const available = models.filter(model => model.available && !model.archived);
    field('assetChoices').replaceChildren(...available.map(model => new Option(`${model.id} · ${model.name}`, model.id)));
    field('assetChoices').hidden = available.length === 0;
    field('assetListStatus').textContent = available.length
      ? `可选模型 ${available.length} 个，支持多选。`
      : '目录中暂无可选的已发布模型，可在建模工作台查看。';
    syncModelSelection();
  } catch (error) {
    field('assetListStatus').textContent = `模型目录读取失败。${error.message}`;
  }
}

field('assets').oninput = syncModelSelection;
field('assetChoices').onchange = () => {
  field('assets').value = selectedValues(field('assetChoices')).join(',');
};

function resetBackgroundList() {
  directoryVersion++;
  field('backgroundFiles').replaceChildren();
  field('backgroundFilter').hidden = true;
  field('listBackgrounds').disabled = false;
  field('backgroundNote').textContent = '默认使用目录内全部背景图。';
}

field('backgrounds').oninput = resetBackgroundList;
field('listBackgrounds').onclick = async () => {
  const directory = field('backgrounds').value.trim();
  if (!directory) {
    field('backgroundNote').textContent = '请先填写运行服务所在电脑上的背景目录。';
    field('backgrounds').focus();
    return;
  }
  const version = ++directoryVersion;
  field('listBackgrounds').disabled = true;
  field('backgroundNote').textContent = '正在读取背景目录…';
  try {
    const names = await requestJson('/api/backgrounds?directory=' + encodeURIComponent(directory));
    if (version !== directoryVersion) return;
    const selected = new Set(selectedValues(field('backgroundFiles')));
    field('backgroundFiles').replaceChildren(...names.map(name => new Option(name, name, false, selected.has(name))));
    field('backgroundFilter').hidden = !names.length;
    field('backgroundNote').textContent = names.length
      ? `找到 ${names.length} 张背景图，可多选；留空使用全部。`
      : '这个目录里没有可用的背景图。';
  } catch (error) {
    if (version !== directoryVersion) return;
    field('backgroundFiles').replaceChildren();
    field('backgroundFilter').hidden = true;
    field('backgroundNote').textContent = `背景目录读取失败。${error.message}`;
  } finally {
    if (version === directoryVersion) field('listBackgrounds').disabled = false;
  }
};

function batchPayload() {
  const value = id => field(id).value;
  const number = id => Number(value(id));
  const payload = {
    backgrounds: value('backgrounds').trim(), background_files: selectedValues(field('backgroundFiles')),
    count: number('count'), max_iterations: number('iterations'), seed: number('seed'),
    assets: value('assets').split(',').map(id => id.trim()).filter(Boolean), render_device: value('device'),
    photo_style: value('photoStyle'), appearance: value('appearance'), atmosphere: value('atmosphere'),
    placement_mode: value('placementMode'), uncertain_policy: value('uncertainPolicy'),
    integration_mode: value('integrationMode'), photoshop_mode: value('photoshopMode'),
  };
  if (payload.integration_mode === 'evidence_v1') {
    payload.workflow = 'evidence_v1';
    payload.integration_mode = 'coherent';
  }
  if (value('reviewProfile') === 'photographer') payload.review_profile = 'photographer';
  if (value('realismPreset') === 'realism_v1') payload.preset = 'realism_v1';
  if (value('testMode') !== 'batch') {
    payload.scale_mode = value('scaleMode');
    payload.contact_mode = value('contactMode');
    payload.chroma_mode = value('chromaMode');
    payload.effects = {
      fog_mode: value('fogMode'), fog_gain: number('fogGain'), weathering: value('weathering'),
      shadow_mode: 'matched_receiver', lighting: 'local', fog_extra: number('fogExtra') / 100,
      pool_kernel: number('poolKernel'), surface_dust: number('surfaceDust'), ai_adjust: value('adjustEffects') === 'yes',
    };
  }
  if (value('testMode') === 'rebuild') {
    payload.geometry_preset = 'rebuild';
    payload.vehicle_reference = {
      vehicle_class: 'semitrailer', internal_width_m: number('vehicleWidth'), nominal_internal_height_m: number('vehicleHeight'),
      source: {kind: 'user_supplied', reference: '用户在网页填写的车厢内宽；内高仅辅助估计'},
    };
  }
  return payload;
}

const stageNames = {
  photoshop: 'Photoshop 导出', asset: '准备模型', scene: '分析背景',
  review: '复核成图', review_request: '复核图片', queued: '等待执行',
  launch: '启动工作进程', prepare_inputs: '准备输入与队列', preflight: '运行前检查',
  photoshop_preflight: '检查 Photoshop', reuse: '复用已有结果', provider_probe: '检查视觉服务',
  dispatch: '分派任务', close_renderer: '关闭渲染器', close_provider: '关闭视觉服务',
  summary: '写入批次摘要', complete: '处理结束',
};

function showRunDetails(data) {
  const list = field('runDetails');
  list.replaceChildren();
  function row(label, value) {
    const term = document.createElement('dt');
    term.textContent = label;
    const description = document.createElement('dd');
    description.textContent = value;
    list.append(term, description);
  }
  const run = data.run;
  if (run?.id) row('本次运行编号', run.id);
  if (run?.stage) row('最后记录阶段', stageNames[run.stage] || run.stage);
  if (run?.job_id) row('对应任务', run.job_id);
  if (run?.remaining != null) row('剩余任务', String(run.remaining));
  const previous = data.previous_interruption;
  if (previous) {
    const details = [previous.id, stageNames[previous.stage] || previous.stage, previous.job_id, previous.error];
    if (previous.remaining != null) details.push(`当时剩余 ${previous.remaining} 项`);
    row('上一次中断', details.filter(Boolean).join(' · '));
  }
  const trace = data.insertion_traces;
  if (trace) row('插入痕迹', `未见 ${trace.none ?? 0} · 不确定 ${trace.uncertain ?? 0} · 可见 ${trace.visible ?? 0}`);
  const cost = data.cost_total;
  if (cost) {
    const format = amount => Number.isFinite(amount) ? amount.toFixed(4) : null;
    const minimum = format(cost.estimated_api_min);
    const maximum = format(cost.estimated_api_max);
    const range = minimum || maximum ? `${cost.currency || ''} ${minimum ?? '待核算'}–${maximum ?? '待核算'}` : '未计价';
    row('累计 API 估算', `${range} · ${cost.actual_requests ?? '未知'} 次请求 · ${cost.reported_tokens ?? '未知'} token（含重试）`);
  }
  if (data.photoshop_mode) row('Photoshop', {off: '关闭', always: '全部', on_issue: '按问题'}[data.photoshop_mode] || data.photoshop_mode);
  if (data.integration_mode) row('合成流程', data.workflow === 'evidence_v1' ? '凹处与结构化复核（小样）'
    : data.integration_mode === 'coherent' ? '局部受光与反射协同（小样）' : '原融合流程');
  if (data.geometry_preset === 'rebuild') {
    row('尺度与摆位重生成', `车厢内宽 ${data.vehicle_reference?.internal_width_m ?? '未知'} 米`);
    if (data.effective_geometry) row('本次几何设置', JSON.stringify(data.effective_geometry));
  }
  if (data.run?.seconds != null) row('本次耗时', `${Math.round(data.run.seconds)} 秒`);
  if (data.output) row('保存位置', data.output);
}

function showStatus(data) {
  const counts = data.counts || {};
  const total = Object.values(counts).reduce((sum, n) => sum + n, 0);
  const done = (counts.accepted_display || 0) + (counts.review || 0) + (counts.failed || 0);
  const error = data.run?.error || data.process_error;
  if (data.running) setState('运行中', 'running');
  else if (data.run?.state === 'interrupted') setState('已中断', 'warning');
  else if (data.run?.state?.startsWith('paused_')) setState('已暂停', 'warning');
  else if (error) setState('运行异常', 'warning');
  else if (data.resumable) setState('可继续', 'warning');
  else if (total && done === total) setState('处理结束');
  else setState('未运行');

  field('progress').hidden = false;
  field('noBatch').hidden = true;
  field('batchIdentity').textContent = `批次 ${data.id || batch}`;
  field('countsPanel').hidden = !total;
  if (total) {
    field('completion').textContent = `${done} / ${total} 项`;
    field('jobMeter').max = total;
    field('jobMeter').value = done;
    field('accepted').textContent = counts.accepted_display || 0;
    field('review').textContent = counts.review || 0;
    field('failed').textContent = counts.failed || 0;
    field('queued').textContent = `排队 ${counts.queued || 0} 项 · ${data.running ? '执行中' : '中断待续'} ${counts.running || 0} 项 · 展示合格率 ${(100 * (counts.accepted_display || 0) / total).toFixed(1)}%`;
  }
  const imageRate = data.image_rate == null ? '' : ` · 有图率 ${(100 * data.image_rate).toFixed(1)}%`;
  field('images').textContent = data.generated_pngs == null ? '' : `已出图 ${data.generated_pngs} 张${imageRate}`;
  field('activeJobs').replaceChildren(...(data.active || []).map(job => {
    const item = document.createElement('li');
    item.textContent = `${job.id} · ${stageNames[job.stage] || job.stage}`;
    return item;
  }));
  const location = [data.run?.job_id, stageNames[data.run?.stage] || data.run?.stage].filter(Boolean);
  const remaining = data.run?.remaining;
  if (remaining != null) location.push(`剩余 ${remaining} 项`);
  const failure = error ? [...location, error].join(' · ') : '';
  const legacyNote = data.legacy_runtime_required ? '历史批次请在创建它的原实现版本续跑。' : '';
  field('runError').textContent = [failure, legacyNote].filter(Boolean).join(' ');
  field('results').hidden = !data.has_results;
  field('results').href = `/api/batches/${encodeURIComponent(batch)}/files/index.html`;
  showRunDetails(data);
}

async function refreshStatus() {
  clearTimeout(pollTimer);
  if (!batch) return;
  const id = batch;
  statusPending = true;
  updateActions();
  try {
    const data = await requestJson('/api/batches/' + encodeURIComponent(id));
    if (id !== batch) return;
    latestStatus = data;
    showStatus(data);
    field('message').textContent = '';
    field('refresh').hidden = true;
    if (data.running) pollTimer = setTimeout(refreshStatus, 2500);
  } catch (error) {
    if (id !== batch) return;
    latestStatus = null;
    field('noBatch').hidden = true;
    field('progress').hidden = true;
    field('results').hidden = true;
    field('refresh').hidden = false;
    setState('状态未知', 'warning');
    field('message').textContent = `批次 ${id} 状态读取失败。${error.message}`;
    if (error.status !== 404) pollTimer = setTimeout(refreshStatus, 6000);
  } finally {
    if (id === batch) {
      statusPending = false;
      updateActions();
    }
  }
}

async function runAction(action) {
  if (actionPending) return;
  actionPending = true;
  clearTimeout(pollTimer);
  updateActions();
  try {
    await action();
    await refreshStatus();
  } catch (error) {
    field('message').textContent = error.message;
    if (batch) field('refresh').hidden = false;
  } finally {
    actionPending = false;
    updateActions();
  }
}

field('form').onsubmit = event => {
  event.preventDefault();
  if (field('start').disabled) return;
  const payload = batchPayload();
  const endpoint = field('testMode').value === 'batch' ? '/api/batches' : '/api/batches/test';
  if (window.matchMedia('(max-width: 839px)').matches) field('runPanel').scrollIntoView();
  runAction(async () => {
    field('message').textContent = '正在创建批次…';
    const result = await requestJson(endpoint, {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload),
    });
    batch = result.id;
    localStorage.setItem('displayBatchId', batch);
    localStorage.setItem('displayBackgrounds', payload.backgrounds);
    latestStatus = null;
    field('progress').hidden = true;
    field('noBatch').hidden = true;
    field('results').hidden = true;
    setState('读取状态');
  });
};
field('resume').onclick = () => runAction(async () => {
  field('message').textContent = '正在继续此批次…';
  await requestJson(`/api/batches/${encodeURIComponent(batch)}/resume`, {method: 'POST'});
});
field('refresh').onclick = refreshStatus;
field('backgrounds').value = localStorage.getItem('displayBackgrounds') || '';
updateMode();
updateActions();
checkModes();
listAssets();
if (batch) {
  setState('读取状态');
  field('noBatch').hidden = true;
  refreshStatus();
}
