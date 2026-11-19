const databaseName = 'scrap-composite-workbench';
let openingDatabase;

function database() {
  if (!openingDatabase) {
    openingDatabase = new Promise((resolve, reject) => {
      const request = indexedDB.open(databaseName, 1);
      request.onupgradeneeded = () => {
        const db = request.result;
        db.createObjectStore('materials');
        db.createObjectStore('preferences');
        db.createObjectStore('drafts');
      };
      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const db = request.result;
        db.onversionchange = () => {
          db.close();
          openingDatabase = undefined;
        };
        resolve(db);
      };
    }).catch(error => {
      openingDatabase = undefined;
      throw error;
    });
  }
  return openingDatabase;
}

async function useStore(name, mode, operation) {
  const db = await database();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(name, mode);
    const request = operation(transaction.objectStore(name));
    transaction.oncomplete = () => resolve(request.result);
    transaction.onabort = () => reject(transaction.error || new Error('浏览器存储已中断'));
  });
}

export const loadMaterials = () => useStore('materials', 'readonly', store => store.getAll());
export const readLocal = (name, key) => useStore(name, 'readonly', store => store.get(key));
export const writeLocal = (name, key, value) => useStore(name, 'readwrite', store => store.put(value, key));
export const removeLocal = (name, key) => useStore(name, 'readwrite', store => store.delete(key));

async function unusedFilename(folder, filename) {
  const split = filename.lastIndexOf('.');
  const stem = filename.slice(0, split);
  const extension = filename.slice(split);
  for (let number = 1; ; number += 1) {
    const candidate = number === 1 ? filename : stem + '-' + number + extension;
    try {
      await folder.getFileHandle(candidate);
    } catch (error) {
      if (error.name === 'NotFoundError') return candidate;
      if (error.name !== 'TypeMismatchError') throw error;
    }
  }
}

async function saveInFolder(folder, filename, response) {
  const name = await unusedFilename(folder, filename);
  const file = await folder.getFileHandle(name, {create: true});
  try {
    const stream = await file.createWritable();
    await response.body.pipeTo(stream);
  } catch (error) {
    // An aborted write can leave the newly created entry empty.
    try {
      const pending = await folder.getFileHandle(name);
      if (await file.isSameEntry(pending) && (await pending.getFile()).size === 0) {
        await folder.removeEntry(name);
      }
    } catch {
      // Preserve the original write error if cleanup also fails.
    }
    throw error;
  }
  return name;
}

async function download(filename, response) {
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30_000);
}

export function setupExportSettings() {
  const field = id => document.getElementById(id);
  const choose = field('choose-export-folder');
  const clear = field('clear-export-folder');
  const feedback = field('settings-feedback');
  const supported = typeof window.showDirectoryPicker === 'function';
  const preference = 'export-folder';
  let folder = null;

  function showFolder() {
    choose.hidden = !supported;
    choose.textContent = folder ? '更换文件夹' : '选择文件夹';
    clear.hidden = !folder;
    field('export-location-name').textContent = folder?.name || '使用浏览器下载';
    field('export-location-note').textContent = folder
      ? '点击成图的导出按钮时保存到此文件夹。'
      : supported ? '保存位置由浏览器的下载设置决定。' : '当前浏览器通过下载保存文件。';
  }

  choose.disabled = true;
  showFolder();
  const ready = (supported ? readLocal('preferences', preference) : Promise.resolve())
    .then(saved => { folder = saved || null; })
    .catch(() => { feedback.textContent = '暂时无法读取保存位置，你仍可选择文件夹。'; })
    .finally(() => {
      choose.disabled = false;
      showFolder();
    });

  choose.addEventListener('click', async () => {
    choose.disabled = true;
    clear.disabled = true;
    feedback.textContent = '';
    try {
      folder = await window.showDirectoryPicker({id: 'scrap-composite-exports', mode: 'readwrite'});
      showFolder();
      try {
        await writeLocal('preferences', preference, folder);
        feedback.textContent = '已保存导出位置。';
      } catch {
        feedback.textContent = '当前页面可使用此位置。刷新后可能需要重新选择。';
      }
    } catch (error) {
      if (error.name !== 'AbortError') feedback.textContent = '无法选择文件夹，请重试或使用浏览器下载。';
    } finally {
      choose.disabled = false;
      clear.disabled = false;
    }
  });

  clear.addEventListener('click', async () => {
    choose.disabled = true;
    clear.disabled = true;
    folder = null;
    showFolder();
    try {
      await removeLocal('preferences', preference);
      feedback.textContent = '已切换到浏览器下载。';
    } catch {
      feedback.textContent = '当前页面已切换到浏览器下载，刷新后可能需要重新设置。';
    } finally {
      choose.disabled = false;
      clear.disabled = false;
    }
  });

  function folderError(directory, message) {
    return Object.assign(new Error('「' + directory.name + '」' + message + '。请重新授权或使用浏览器下载。'), {
      canDownload: true,
    });
  }

  return async function exportFile(url, name, browserDownload = false) {
    await ready;
    const directory = browserDownload ? null : folder;
    if (directory) {
      const options = {mode: 'readwrite'};
      try {
        if (await directory.queryPermission(options) !== 'granted'
            && await directory.requestPermission(options) !== 'granted') {
          throw new Error('Permission denied');
        }
      } catch {
        throw folderError(directory, '暂未获得写入权限');
      }
    }
    const response = await fetch(url, {cache: 'no-store'});
    if (!response.ok || !response.body) throw new Error('成图文件读取失败，请刷新批次后重试。');
    if (!directory) {
      await download(name, response);
      return '已开始浏览器下载。';
    }
    try {
      const save = () => saveInFolder(directory, name, response);
      const saved = navigator.locks ? await navigator.locks.request('scrap-composite-export', save) : await save();
      return '已保存到「' + directory.name + '」的 ' + saved + '。';
    } catch {
      await response.body.cancel().catch(() => {});
      throw folderError(directory, '写入失败');
    }
  };
}
