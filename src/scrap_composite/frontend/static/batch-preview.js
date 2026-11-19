import {readLocal, removeLocal, writeLocal} from './batch-local.js';

const field = id => document.getElementById(id);
const pageSize = 12;
const reviewLabels = {accepted_display: '展示合格', review: '待审', failed: '失败'};
const reasonLabels = {
  visual_ok_evidence_unresolved: '视觉复核通过，现场证据待补。',
  no_improvement: '后续修正未改善结果。',
  no_effective_image_change: '后续修正未改变成图。',
  no_effective_action: '暂无可执行的修正。',
  review_invalid_after_one_repair: '视觉复核暂未得到有效结果。',
};

export function batchArtifactUrl(id, output, filename) {
  if (typeof output !== 'string' || !output || typeof filename !== 'string') return null;
  const root = output.replaceAll('\\', '/').replace(/\/+$/, '') + '/';
  const path = filename.replaceAll('\\', '/');
  const windowsPath = /^[a-z]:\//i.test(root) || root.startsWith('//');
  const inside = windowsPath ? path.toLowerCase().startsWith(root.toLowerCase()) : path.startsWith(root);
  if (!inside) return null;
  const segments = path.slice(root.length).split('/');
  if (segments.some(segment => !segment || segment === '.' || segment === '..')) return null;
  return '/api/batches/' + encodeURIComponent(id) + '/files/' + segments.map(encodeURIComponent).join('/');
}

function parseResults(id, output, text) {
  const images = [];
  let incomplete = false;
  for (const line of text.split('\n')) {
    if (!line.trim()) continue;
    try {
      const row = JSON.parse(line);
      if (!row.image) continue;
      const image = batchArtifactUrl(id, output, row.image);
      if (!image || typeof row.job_id !== 'string') {
        incomplete = true;
        continue;
      }
      images.push({
        key: JSON.stringify([id, row.job_id, row.best_iteration ?? null]),
        batch: id, job: row.job_id, asset: row.asset_id, iteration: row.best_iteration,
        review: reviewLabels[row.status] || '已成图',
        tone: row.status === 'accepted_display' ? 'success' : row.status === 'failed' ? 'error' : 'warning',
        reason: reasonLabels[row.reason] || row.reason || '',
        image, crop: batchArtifactUrl(id, output, row.crop), psd: batchArtifactUrl(id, output, row.psd),
      });
    } catch {
      incomplete = true;
    }
  }
  return {images, incomplete};
}

function resultName(result) {
  return result.asset ? '物品 ' + result.asset + ' · ' + result.job : result.job;
}

function exportName(result, format) {
  const stem = (result.batch + '-' + result.job).replace(/[<>:"/\\|?*\u0000-\u001f]/g, '_');
  return stem.slice(0, 120) + '.' + format;
}

export function setupPreview(exportFile) {
  const canvas = document.createElement('img');
  canvas.id = 'preview-image';
  canvas.alt = '';
  canvas.hidden = true;
  field('preview-canvas').append(canvas);
  const thumbnails = field('result-thumbnails');
  const agent = field('agent-panel');
  const editor = field('agent-draft');
  const drafts = new Map();
  let activeBatch;
  let loadVersion = 0;
  let loadedSummary;
  let results = [];
  let selected = null;
  let page = 0;
  let exporting = false;
  let fallbackFormat;

  function empty(headline, hint) {
    field('preview-headline').textContent = headline;
    field('preview-hint').textContent = hint;
    canvas.hidden = true;
    field('preview-empty').hidden = false;
  }

  function setAgentOpen(open) {
    agent.hidden = !open;
    field('toggle-agent').setAttribute('aria-expanded', String(open));
  }

  function updateDraftControls() {
    field('copy-agent-draft').disabled = !editor.value.trim();
  }

  function draftContext(result) {
    const iteration = result.iteration === 0 ? '初稿'
      : result.iteration == null ? '迭代未记录' : '第 ' + result.iteration + ' 轮';
    return '批次 ' + result.batch + ' · ' + resultName(result) + ' · ' + iteration;
  }

  async function showDraft(result) {
    const key = result.key;
    field('agent-context').textContent = draftContext(result);
    editor.value = drafts.get(key) || '';
    updateDraftControls();
    field('agent-feedback').textContent = '';
    if (drafts.has(key)) return;
    try {
      const saved = await readLocal('drafts', key);
      if (!drafts.has(key)) drafts.set(key, saved || '');
      if (selected?.key === key) {
        editor.value = drafts.get(key);
        updateDraftControls();
      }
    } catch {
      if (selected?.key === key) field('agent-feedback').textContent = '暂时无法读取已保存的草稿，仍可输入并复制。';
    }
  }

  function showSelected(result, retry = false) {
    const previousKey = selected?.key;
    const previousImage = selected?.image;
    selected = result;
    field('preview-tools').hidden = !result;
    if (!result) {
      setAgentOpen(false);
      canvas.removeAttribute('src');
      return;
    }
    const changed = result.key !== previousKey || result.image !== previousImage;
    if (changed || retry && canvas.hidden) {
      canvas.alt = resultName(result) + '的合成图';
      canvas.src = result.image;
      canvas.hidden = false;
      field('preview-empty').hidden = true;
    }
    if (changed) {
      field('export-feedback').textContent = '';
      field('export-fallback').hidden = true;
      fallbackFormat = undefined;
      void showDraft(result);
    }
    field('preview-name').textContent = resultName(result);
    field('preview-original').href = result.image;
    field('preview-reason').textContent = result.reason;
    const status = field('preview-status');
    status.textContent = result.review;
    status.dataset.tone = result.tone;
    field('export-psd').hidden = !result.psd;
  }

  function updateSelection() {
    for (const button of thumbnails.querySelectorAll('button')) {
      button.setAttribute('aria-pressed', String(button.dataset.key === selected?.key));
    }
  }

  function renderGallery() {
    field('result-gallery').hidden = !results.length;
    const start = page * pageSize;
    const items = results.slice(start, start + pageSize);
    thumbnails.replaceChildren(...items.map(result => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'result-thumbnail';
      button.title = resultName(result);
      button.setAttribute('aria-label', '预览 ' + resultName(result));
      button.dataset.key = result.key;
      const image = document.createElement('img');
      image.src = result.crop || result.image;
      image.alt = '';
      image.loading = 'lazy';
      image.decoding = 'async';
      image.addEventListener('error', () => { image.hidden = true; });
      const label = document.createElement('span');
      label.textContent = String(result.asset || result.job);
      button.append(image, label);
      button.addEventListener('click', () => {
        showSelected(result, true);
        updateSelection();
      });
      return button;
    }));
    updateSelection();
    field('result-pagination').hidden = results.length <= pageSize;
    field('previous-results').disabled = page === 0;
    field('next-results').disabled = start + pageSize >= results.length;
    field('result-range').textContent = (start + 1) + '–' + (start + items.length) + ' / ' + results.length;
  }

  function replaceResults(images, force = false) {
    const priorIndex = results.findIndex(result => result.key === selected?.key);
    results = images;
    const found = results.findIndex(result => result.key === selected?.key);
    const index = found >= 0 ? found : 0;
    if (found < 0 || Math.floor(priorIndex / pageSize) === page) page = Math.floor(index / pageSize);
    page = Math.min(page, Math.floor(Math.max(0, results.length - 1) / pageSize));
    showSelected(results[index] || null, force);
    renderGallery();
  }

  function reset(id) {
    activeBatch = id;
    loadVersion += 1;
    loadedSummary = undefined;
    page = 0;
    replaceResults([]);
    field('preview-feedback').textContent = '';
    field('export-feedback').textContent = '';
    field('export-fallback').hidden = true;
    empty(id ? '等待成图' : '暂无成图', id
      ? '任务完成后会逐张显示。' : '开始生图后，已完成的图片会显示在这里。');
  }

  function showEmptyBatch(status) {
    if (results.length) return;
    if (status.running) empty('等待成图', '任务完成后会逐张显示。');
    else if (status.resumable) empty('暂无成图', '继续此批次后可查看后续成图。');
    else if (status.run) empty('本批次暂无成图', status.has_results
      ? '可在批次报告中查看各项结果。' : '可在运行明细中查看任务状态。');
  }

  async function sync(id, status, force = false) {
    if (id !== activeBatch) reset(id);
    if (!status.has_results) {
      showEmptyBatch(status);
      return true;
    }
    const summary = JSON.stringify([status.counts, status.generated_pngs, status.run?.state]);
    if (!force && loadedSummary === summary) return true;
    const version = ++loadVersion;
    try {
      const prefix = '/api/batches/' + encodeURIComponent(id) + '/files/';
      const response = await fetch(prefix + 'results.jsonl', {cache: 'no-store'});
      if (!response.ok) throw new Error('结果文件暂不可用');
      const parsed = parseResults(id, status.output, await response.text());
      if (version !== loadVersion) return true;
      loadedSummary = parsed.incomplete ? undefined : summary;
      replaceResults(parsed.images, force);
      showEmptyBatch(status);
      field('preview-feedback').textContent = parsed.incomplete ? '部分结果记录无法读取，可查看批次报告。' : '';
      return !parsed.incomplete;
    } catch {
      if (version !== loadVersion) return true;
      field('preview-feedback').textContent = '暂时无法读取成图列表，请刷新批次状态。';
      showEmptyBatch(status);
      return false;
    }
  }

  async function exportSelected(format, browserDownload = false) {
    const result = selected;
    const url = format === 'png' ? result?.image : result?.psd;
    if (!url || exporting) return;
    const buttons = ['export-png', 'export-psd', 'export-fallback'].map(field);
    exporting = true;
    for (const button of buttons) button.disabled = true;
    const notice = field('export-feedback');
    notice.textContent = '正在导出文件…';
    try {
      const message = await exportFile(url, exportName(result, format), browserDownload);
      if (result.key !== selected?.key) return;
      notice.textContent = message;
      field('export-fallback').hidden = true;
    } catch (error) {
      if (result.key !== selected?.key) return;
      notice.textContent = error.message;
      fallbackFormat = format;
      field('export-fallback').hidden = !error.canDownload;
    } finally {
      exporting = false;
      for (const button of buttons) button.disabled = false;
    }
  }

  canvas.addEventListener('error', () => {
    if (selected) empty('暂时无法查看成图', '可点击查看原尺寸，或刷新批次状态。');
  });
  field('previous-results').addEventListener('click', () => {
    page -= 1;
    renderGallery();
  });
  field('next-results').addEventListener('click', () => {
    page += 1;
    renderGallery();
  });
  field('export-png').addEventListener('click', () => exportSelected('png'));
  field('export-psd').addEventListener('click', () => exportSelected('psd'));
  field('export-fallback').addEventListener('click', () => exportSelected(fallbackFormat, true));
  field('toggle-agent').addEventListener('click', () => {
    setAgentOpen(agent.hidden);
    if (!agent.hidden) editor.focus();
  });
  editor.addEventListener('input', () => {
    if (selected) drafts.set(selected.key, editor.value);
    updateDraftControls();
  });
  field('save-agent-draft').addEventListener('click', async () => {
    if (!selected) return;
    const {key} = selected;
    const text = editor.value;
    const notice = field('agent-feedback');
    const button = field('save-agent-draft');
    button.disabled = true;
    drafts.set(key, text);
    try {
      if (text.trim()) await writeLocal('drafts', key, text);
      else await removeLocal('drafts', key);
      if (selected?.key === key) notice.textContent = text.trim() ? '草稿已保存到此浏览器。' : '已清空保存的草稿。';
    } catch {
      if (selected?.key === key) notice.textContent = '暂时无法保存草稿，可复制指令。';
    } finally {
      button.disabled = false;
    }
  });
  field('copy-agent-draft').addEventListener('click', async () => {
    if (!selected || !editor.value.trim()) return;
    const result = selected;
    const text = draftContext(result) + '\n\n' + editor.value.trim();
    try {
      await navigator.clipboard.writeText(text);
      if (selected?.key === result.key) field('agent-feedback').textContent = '已复制指令和成图信息。';
    } catch {
      if (selected?.key !== result.key) return;
      editor.focus();
      editor.select();
      field('agent-feedback').textContent = '请使用系统复制快捷键复制选中的修改指令。';
    }
  });

  return {reset, sync};
}
