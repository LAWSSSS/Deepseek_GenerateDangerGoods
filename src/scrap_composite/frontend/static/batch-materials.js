import {loadMaterials, removeLocal, writeLocal} from './batch-local.js';

const field = id => document.getElementById(id);
const formats = '.jpg,.jpeg,.png,.webp';
const directions = {
  single: [['photo', '主照片']],
  six: [['front', '正面'], ['back', '背面'], ['left', '左侧'], ['right', '右侧'], ['top', '顶面'], ['bottom', '底面']],
};

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

function action(label, callback) {
  const button = element('button', 'text-button', label);
  button.type = 'button';
  button.addEventListener('click', callback);
  return button;
}

async function validateImage(file) {
  const extension = file.name.match(/\.(png|jpe?g|webp)$/i)?.[1].toLowerCase();
  if (!extension) throw new Error('请使用 JPG、PNG 或 WebP 图片。');
  const bytes = new Uint8Array(await file.slice(0, 12).arrayBuffer());
  const signature = String.fromCharCode(...bytes);
  const valid = {
    png: [137, 80, 78, 71, 13, 10, 26, 10].every((byte, index) => bytes[index] === byte),
    jpg: bytes[0] === 255 && bytes[1] === 216 && bytes[2] === 255,
    webp: signature.startsWith('RIFF') && signature.endsWith('WEBP'),
  };
  if (!valid[extension === 'jpeg' ? 'jpg' : extension]) throw new Error('文件内容与图片格式不符。');

  const url = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    return file;
  } catch {
    throw new Error('无法读取这张图片，请检查文件是否损坏。');
  } finally {
    URL.revokeObjectURL(url);
  }
}

function storageError(error) {
  return error.name === 'QuotaExceededError'
    ? '浏览器存储空间不足，请移除部分本机素材后重试。'
    : '暂时无法写入本机素材库，请检查浏览器存储权限后重试。';
}

function feedback(node, text, error = false) {
  node.textContent = text;
  node.dataset.tone = error ? 'error' : '';
}

export function setupMaterials({requestJson, onChange, savedDirectory, rememberDirectory}) {
  const directory = field('backgrounds');
  const backgroundNote = field('backgroundNote');
  const objectNote = field('object-import-status');
  const backgroundInput = field('background-upload');
  const replacementInput = field('background-replacement');
  const backgroundStatus = field('background-import-status');
  const objectForm = field('object-import-form');
  const objectName = field('object-name');
  const dropZone = field('background-drop');
  const urls = new Map();
  const records = new Map();
  const removing = new Set();
  const sources = {
    background: {items: [], selected: new Set(), state: 'loading'},
    object: {items: [], selected: new Set(), state: 'loading'},
  };
  let requestNumber = 0;
  let loadedDirectory = '';
  let currentReplacement;
  let importingBackground = false;
  let savingObject = false;
  let draft = {files: {}, pending: new Map()};
  let materialsReady;

  const mode = kind => document.querySelector('input[name="' + kind + '-mode"]:checked').value;
  const photoMode = () => objectForm.querySelector('input[name="photo-mode"]:checked').value;
  const localRecords = kind => [...records.values()].filter(record => record.kind === kind)
    .sort((first, second) => second.created - first.created);
  const cover = record => record.files.photo || record.files.front;

  function imageUrl(file) {
    if (!urls.has(file)) urls.set(file, URL.createObjectURL(file));
    return urls.get(file);
  }

  function thumbnail(file) {
    const image = element('img');
    image.src = imageUrl(file);
    image.alt = '';
    image.decoding = 'async';
    image.loading = 'lazy';
    return image;
  }

  function releaseUnusedUrls() {
    const files = new Set([...records.values()].flatMap(record => Object.values(record.files)));
    for (const file of Object.values(draft.files)) files.add(file);
    for (const [file, url] of urls) {
      if (files.has(file)) continue;
      URL.revokeObjectURL(url);
      urls.delete(file);
    }
  }

  function updatePicker(kind) {
    const source = sources[kind];
    const count = source.items.length;
    const specified = mode(kind) === 'specified';
    const label = kind === 'background' ? '背景图' : '模型';
    let text = '可用' + label + ' ' + count + ' ' + (kind === 'background' ? '张' : '个');
    if (specified && count) {
      const selected = source.items.filter(item => source.selected.has('server:' + item.id)).length;
      text += '，已指定 ' + selected;
    }
    if (source.state === 'loading') text = '正在读取' + label + '目录…';
    else if (source.state === 'unread') text = '请在导入页读取新的背景目录。';
    else if (source.state === 'error') text = kind === 'background'
      ? '背景目录暂不可用，可在导入页重新读取。'
      : '模型目录读取失败，请刷新目录。';
    else if (!count) text = '暂无可用' + label + '。';
    field(kind + '-availability').textContent = text;
    field(kind + '-choices').hidden = !specified;
  }

  function renderPicker(kind) {
    const source = sources[kind];
    const local = localRecords(kind);
    const list = field(kind + '-choices');
    list.replaceChildren();

    function appendOptions(items, origin, heading) {
      if (!items.length) return;
      list.append(element('p', 'choices-heading', heading));
      for (const item of items) {
        const key = origin + ':' + item.id;
        const label = element('label', 'choice-row');
        const input = element('input');
        input.type = 'checkbox';
        input.value = key;
        input.checked = source.selected.has(key);
        input.addEventListener('change', () => {
          if (input.checked) source.selected.add(key);
          else source.selected.delete(key);
          updatePicker(kind);
          onChange();
        });
        label.append(input);
        if (origin === 'local') label.append(thumbnail(cover(item)));
        label.append(element('span', 'choice-name', item.name));
        if (origin === 'local') label.append(element('small', 'local-mark', '仅本机'));
        list.append(label);
      }
    }

    appendOptions(source.items, 'server', kind === 'background' ? '服务端背景' : '可用模型');
    appendOptions(local, 'local', '本机素材');
    if (!source.items.length && !local.length) list.append(element('p', 'choices-heading', '暂无可选素材'));
    updatePicker(kind);
  }

  function renderLibrary(kind) {
    const list = field(kind === 'background' ? 'local-backgrounds' : 'local-objects');
    const local = localRecords(kind);
    list.replaceChildren();
    if (kind === 'background') field('local-background-count').textContent = local.length ? local.length + ' 张本机图片' : '';
    for (const record of local) {
      const item = element('li', 'material-card');
      const preview = element('a', 'material-preview');
      preview.href = imageUrl(cover(record));
      preview.target = '_blank';
      preview.rel = 'noopener';
      preview.setAttribute('aria-label', '查看「' + record.name + '」原图');
      preview.append(thumbnail(cover(record)));
      const info = element('div', 'material-info');
      const name = element('strong', null, record.name);
      name.title = record.name;
      info.append(name, element('small', 'muted', kind === 'background'
        ? '仅本机' : (record.mode === 'six' ? '六视图' : '单张照片') + ' · 仅本机'));
      const controls = element('div', 'material-actions');
      const edit = action(kind === 'background' ? '替换' : '编辑', () => {
        if (removing.has(record.id)) return;
        if (kind === 'object') editObject(record);
        else if (!importingBackground) {
          currentReplacement = record.id;
          replacementInput.click();
        }
      });
      const remove = action('移除', async () => {
        if (removing.has(record.id) || (kind === 'background' ? importingBackground : savingObject)) return;
        removing.add(record.id);
        remove.disabled = true;
        edit.disabled = true;
        try {
          await removeLocal('materials', record.id);
          records.delete(record.id);
          sources[kind].selected.delete('local:' + record.id);
          if (draft.id === record.id) resetObjectForm();
          renderLibrary(kind);
          feedback(kind === 'background' ? backgroundStatus : objectNote, '已移除「' + record.name + '」。');
        } catch (error) {
          feedback(kind === 'background' ? backgroundStatus : objectNote, storageError(error), true);
        } finally {
          removing.delete(record.id);
          remove.disabled = false;
          edit.disabled = false;
        }
      });
      edit.setAttribute('aria-label', edit.textContent + '「' + record.name + '」');
      remove.setAttribute('aria-label', '移除「' + record.name + '」');
      controls.append(edit, remove);
      item.append(preview, info, controls);
      list.append(item);
    }
    renderPicker(kind);
    releaseUnusedUrls();
    onChange();
  }

  async function importBackgrounds(files, replacement) {
    if (!files.length || importingBackground) return;
    importingBackground = true;
    backgroundInput.disabled = true;
    const errors = field('background-errors');
    errors.replaceChildren();
    feedback(backgroundStatus, '正在导入背景图…');
    let imported = 0;
    try {
      await materialsReady;
      for (const file of files) {
        try {
          await validateImage(file);
        } catch (error) {
          errors.append(element('li', null, file.name + ' · ' + error.message));
          continue;
        }
        const id = replacement || crypto.randomUUID();
        const prior = records.get(id);
        if (replacement && (!prior || removing.has(replacement))) {
          errors.append(element('li', null, '原背景已移除，请重新导入「' + file.name + '」。'));
          continue;
        }
        const record = {id, kind: 'background', name: file.name, files: {photo: file}, created: prior?.created || Date.now()};
        try {
          await writeLocal('materials', id, record);
          records.set(id, record);
          imported += 1;
        } catch (error) {
          errors.append(element('li', null, file.name + ' · ' + storageError(error)));
        }
      }
      if (imported) renderLibrary('background');
      feedback(backgroundStatus, replacement && imported ? '已替换本机背景图。'
        : imported ? '已导入 ' + imported + ' 张背景图。' : '没有图片导入。', !imported);
    } finally {
      importingBackground = false;
      backgroundInput.disabled = false;
    }
  }

  backgroundInput.addEventListener('change', () => {
    const files = Array.from(backgroundInput.files);
    backgroundInput.value = '';
    void importBackgrounds(files);
  });
  replacementInput.addEventListener('change', () => {
    const files = Array.from(replacementInput.files);
    replacementInput.value = '';
    void importBackgrounds(files, currentReplacement);
  });
  dropZone.addEventListener('dragover', event => {
    event.preventDefault();
    event.dataTransfer.dropEffect = importingBackground ? 'none' : 'copy';
    dropZone.classList.toggle('is-dragging', !importingBackground);
  });
  dropZone.addEventListener('dragleave', event => {
    if (!dropZone.contains(event.relatedTarget)) dropZone.classList.remove('is-dragging');
  });
  dropZone.addEventListener('drop', event => {
    event.preventDefault();
    dropZone.classList.remove('is-dragging');
    void importBackgrounds(Array.from(event.dataTransfer.files));
  });

  function updateObjectSave() {
    const slots = directions[photoMode()];
    const ready = slots.every(([direction]) => draft.files[direction] && !draft.pending.has(direction));
    field('save-object').disabled = savingObject || !ready;
    field('save-object').textContent = draft.id ? '保存更改' : '保存物品';
    field('cancel-object-edit').hidden = !draft.id;
    field('cancel-object-edit').disabled = savingObject;
  }

  function renderSlots() {
    const slots = field('object-slots');
    slots.replaceChildren();
    slots.dataset.mode = photoMode();
    for (const [direction, name] of directions[photoMode()]) {
      const cell = element('div', 'upload-cell');
      const label = element('label', 'upload-slot');
      const input = element('input');
      input.type = 'file';
      input.accept = formats;
      input.name = direction;
      input.id = 'photo-' + direction;
      label.htmlFor = input.id;
      const title = element('strong', null, name);
      const image = element('img');
      image.alt = '';
      const empty = element('span', 'upload-symbol');
      empty.setAttribute('aria-hidden', 'true');
      const fileName = element('small', 'slot-name');
      const error = element('p', 'slot-error');
      error.id = input.id + '-error';
      error.setAttribute('role', 'status');
      input.setAttribute('aria-describedby', error.id);
      // Keep the input in the label for the pointer target, and give it a stable name for keyboard use.
      input.setAttribute('aria-label', name);
      let remove;
      function showSlot() {
        const file = draft.files[direction];
        const busy = draft.pending.has(direction);
        image.hidden = !file;
        if (file) image.src = imageUrl(file);
        else image.removeAttribute('src');
        empty.hidden = Boolean(file);
        fileName.textContent = busy ? '正在检查…' : file?.name || '选择图片';
        remove.hidden = !file;
        label.classList.toggle('has-photo', Boolean(file));
        updateObjectSave();
      }
      remove = action('移除', () => {
        draft.pending.delete(direction);
        delete draft.files[direction];
        error.textContent = '';
        showSlot();
        input.focus();
        releaseUnusedUrls();
      });
      remove.setAttribute('aria-label', '移除' + name + '照片');
      label.append(input, title, image, empty, fileName);
      cell.append(label, remove, error);
      slots.append(cell);
      input.addEventListener('change', async () => {
        const file = input.files[0];
        input.value = '';
        if (!file || savingObject) return;
        const target = draft;
        const check = validateImage(file);
        target.pending.set(direction, check);
        error.textContent = '';
        showSlot();
        try {
          const photo = await check;
          if (target.pending.get(direction) === check) target.files[direction] = photo;
        } catch (problem) {
          if (target === draft && target.pending.get(direction) === check && cell.isConnected) {
            error.textContent = file.name + ' · ' + problem.message;
          }
        } finally {
          if (target.pending.get(direction) === check) {
            target.pending.delete(direction);
            if (target === draft) {
              if (cell.isConnected) showSlot();
              else renderSlots();
              releaseUnusedUrls();
            }
          }
        }
      });
      showSlot();
    }
  }

  function resetObjectForm(record) {
    draft = {id: record?.id, files: {...record?.files}, pending: new Map()};
    const selectedMode = record?.mode || photoMode();
    objectForm.querySelector('input[name="photo-mode"][value="' + selectedMode + '"]').checked = true;
    objectName.value = record?.name || '';
    renderSlots();
    releaseUnusedUrls();
  }

  function editObject(record) {
    if (savingObject) return;
    resetObjectForm(record);
    feedback(objectNote, '正在编辑「' + record.name + '」。');
    objectName.focus();
  }

  for (const control of objectForm.querySelectorAll('input[name="photo-mode"]')) {
    control.addEventListener('change', () => {
      feedback(objectNote, '');
      renderSlots();
    });
  }
  field('cancel-object-edit').addEventListener('click', () => {
    resetObjectForm();
    feedback(objectNote, '已取消编辑。');
    objectName.focus();
  });
  objectForm.addEventListener('submit', async event => {
    event.preventDefault();
    if (field('save-object').disabled || removing.has(draft.id)) return;
    const kind = photoMode();
    const files = Object.fromEntries(directions[kind].map(([direction]) => [direction, draft.files[direction]]));
    const id = draft.id || crypto.randomUUID();
    const current = records.get(id);
    const defaultName = (files.photo || files.front).name.replace(/\.[^.]+$/, '');
    const name = objectName.value.trim() || defaultName;
    const record = {id, kind: 'object', mode: kind, name, files, created: current?.created || Date.now()};
    savingObject = true;
    for (const input of objectForm.querySelectorAll('input, .upload-cell button')) input.disabled = true;
    updateObjectSave();
    try {
      await materialsReady;
      await writeLocal('materials', id, record);
      records.set(id, record);
      resetObjectForm();
      renderLibrary('object');
      feedback(objectNote, '已保存「' + name + '」。');
    } catch (error) {
      feedback(objectNote, storageError(error), true);
    } finally {
      savingObject = false;
      for (const input of objectForm.querySelectorAll('input, .upload-cell button')) input.disabled = false;
      updateObjectSave();
    }
  });

  function updateDirectory(state, text) {
    sources.background.state = state;
    feedback(backgroundNote, text, state === 'error');
    renderPicker('background');
    onChange();
  }

  async function readDirectory() {
    const value = directory.value.trim();
    const number = ++requestNumber;
    const source = sources.background;
    if (!value) {
      updateDirectory('unread', '请填写运行生成服务的电脑上的背景目录。');
      directory.focus();
      return;
    }
    source.items = [];
    field('listBackgrounds').disabled = true;
    updateDirectory('loading', '正在读取背景目录…');
    try {
      const names = await requestJson('/api/backgrounds?directory=' + encodeURIComponent(value));
      if (number !== requestNumber) return;
      source.items = names.map(name => ({id: name, name}));
      loadedDirectory = value;
      rememberDirectory(value);
      updateDirectory('ready', names.length ? '找到 ' + names.length + ' 张可用于生图的背景。' : '此目录暂无可用背景图。');
    } catch (error) {
      if (number !== requestNumber) return;
      updateDirectory('error', '读取失败。' + error.message);
    } finally {
      if (number === requestNumber) field('listBackgrounds').disabled = false;
    }
  }

  async function readModels() {
    const source = sources.object;
    const refresh = field('refresh-models');
    refresh.disabled = true;
    source.state = 'loading';
    renderPicker('object');
    onChange();
    try {
      const models = await requestJson('/api/modeling/assets');
      source.items = models.filter(model => model.available && !model.archived)
        .map(model => ({id: String(model.id), name: model.id + ' · ' + model.name}));
      source.state = 'ready';
      field('model-directory-note').textContent = source.items.length
        ? '可用于生图的模型 ' + source.items.length + ' 个。' : '发布模型后，在这里刷新目录。';
    } catch (error) {
      source.items = [];
      source.state = 'error';
      field('model-directory-note').textContent = '读取模型失败。' + error.message;
    } finally {
      refresh.disabled = false;
      renderPicker('object');
      onChange();
    }
  }

  field('listBackgrounds').addEventListener('click', readDirectory);
  field('refresh-models').addEventListener('click', readModels);
  directory.addEventListener('input', () => {
    requestNumber += 1;
    loadedDirectory = '';
    const source = sources.background;
    source.items = [];
    source.selected = new Set([...source.selected].filter(key => key.startsWith('local:')));
    field('listBackgrounds').disabled = false;
    updateDirectory('unread', '点击「读取目录」更新可用背景图。');
  });
  directory.addEventListener('keydown', event => {
    if (event.key === 'Enter' && !field('listBackgrounds').disabled) void readDirectory();
  });

  for (const kind of Object.keys(sources)) {
    for (const control of document.querySelectorAll('input[name="' + kind + '-mode"]')) {
      control.addEventListener('change', () => {
        updatePicker(kind);
        onChange();
      });
    }
  }

  function selection() {
    const result = {backgrounds: loadedDirectory, background_files: [], assets: []};
    for (const [kind, key] of [['background', 'background_files'], ['object', 'assets']]) {
      const source = sources[kind];
      const specified = mode(kind) === 'specified';
      const label = kind === 'background' ? '背景' : '物品';
      const local = specified ? localRecords(kind).filter(item => source.selected.has('local:' + item.id)) : [];
      if (local.length) {
        const names = local.slice(0, 2).map(item => '「' + item.name + '」').join('、');
        return {error: names + (local.length > 2 ? '等 ' + local.length + ' 个素材' : '')
          + '仅在本机。请取消勾选并选择可用于生图的' + label + '。'};
      }
      if (source.state === 'loading') return {error: '正在读取可用' + label + '…'};
      if (source.state !== 'ready' || !source.items.length) {
        return {error: kind === 'background' ? '请在导入页读取可用背景目录。' : '请先发布物品模型，再刷新模型目录。'};
      }
      if (specified) {
        result[key] = source.items.filter(item => source.selected.has('server:' + item.id)).map(item => item.id);
        if (!result[key].length) return {error: '请至少指定一个可用于生图的' + label + '。'};
      }
    }
    return result;
  }

  function start() {
    directory.value = savedDirectory || 'external/backgrounds';
    resetObjectForm();
    materialsReady = loadMaterials().then(items => {
      for (const item of items) records.set(item.id, item);
      renderLibrary('background');
      renderLibrary('object');
    }).catch(() => {
      feedback(backgroundStatus, '暂时无法读取此浏览器的素材库。', true);
      feedback(objectNote, '暂时无法读取此浏览器的素材库。', true);
    });
    void readDirectory();
    void readModels();
  }

  return {start, selection};
}
