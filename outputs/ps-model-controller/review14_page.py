"""Standalone, dependency-free review page and Chrome check for item 14.

The page uses only pre-rendered evidence images and links to the generated GLB /
Blender file.  It never talks to the shared placement server and does not modify
shared pages.  Chrome is invoked synchronously in headless mode for a page-load
and interaction self-test.
"""
import html
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

CHROME_CANDIDATES = (
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
)

CSS = """
:root{--ink:#18242b;--muted:#53656c;--line:#cbd5d7;--paper:#f2f5f5;--card:#ffffff;--blue:#1757b4;--warn:#8a2b20;--ok:#1d5c3a}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 "Microsoft YaHei",system-ui,sans-serif}
header{background:linear-gradient(120deg,#143b73,#1b6bb8);color:#fff;padding:22px 30px 18px}
header h1{margin:0 0 6px;font-size:26px}header p{margin:0;color:#dbe8f7;max-width:1120px}
.badge{display:inline-block;margin-top:10px;padding:3px 9px;border:1px solid #ffd88a;border-radius:999px;color:#ffe7b0;font-size:12px}
main{display:grid;grid-template-columns:minmax(0,1fr) 380px;gap:18px;padding:18px 22px 40px;max-width:1500px;margin:auto}
section,aside{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;box-shadow:0 2px 8px rgba(20,40,50,.05)}
h2{margin:4px 0 10px;font-size:18px}h3{margin:14px 0 6px;font-size:15px}
.notice{background:#fff6df;border:1px solid #eccf91;color:#5b4316;border-radius:10px;padding:10px 14px;margin:14px 22px 0;max-width:1450px}
.controls{display:flex;flex-wrap:wrap;gap:14px;align-items:flex-start;margin-bottom:12px}
.group{display:flex;gap:6px;flex-wrap:wrap}button{font:inherit;padding:6px 11px;border:1px solid #aebbc0;background:#f8fafb;color:#24353c;border-radius:7px;cursor:pointer}
button.active{background:var(--blue);border-color:var(--blue);color:#fff}
button:hover{border-color:var(--blue)}
.viewer figure{margin:0;background:#e3e8e8;border-radius:10px;padding:10px;text-align:center}
#view-img{max-width:100%;height:560px;object-fit:contain;background:#dde3e3;border-radius:6px}
figcaption{color:var(--muted);font-size:13px;margin-top:6px}
#turn-wrap{background:#dde3e3;border-radius:10px;padding:10px;text-align:center;cursor:grab;touch-action:none;user-select:none}
#turn-wrap:active{cursor:grabbing}
#turn-img{max-width:100%;height:520px;object-fit:contain;pointer-events:none}
#turn-range{width:100%;margin-top:10px}
.gallery{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-top:6px}
.gallery figure{margin:0;background:#eef2f2;border:1px solid var(--line);border-radius:8px;padding:6px;text-align:center}
.gallery img{width:100%;height:170px;object-fit:contain}
.gallery figcaption{font-size:12px;min-height:32px}
.boards img{width:100%;border:1px solid var(--line);border-radius:8px;margin:6px 0 14px}
.links a{display:block;color:#155397;padding:2px 0}
.mono{font-family:Consolas,monospace;font-size:12px;word-break:break-all}
.ok{color:var(--ok);font-weight:700}.bad{color:var(--warn);font-weight:700}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{border-bottom:1px solid #e3e8e9;padding:5px 4px;text-align:left;vertical-align:top}
#selftest{white-space:pre-wrap;background:#111820;color:#cfe6d6;padding:12px 16px;margin:18px 22px;border-radius:8px;font:12px/1.5 Consolas,monospace;max-width:1450px}
@media(max-width:1100px){main{grid-template-columns:1fr}.gallery{grid-template-columns:repeat(2,minmax(0,1fr))}}
"""

JS = r"""
const state={which:'new',mode:'material',view:'front',turn:0};
const data=JSON.parse(document.getElementById('review-data').textContent);
const $=id=>document.getElementById(id);
const viewNames={front:'正视',back:'背视',left:'左视',right:'右视',top:'顶视',bottom:'底视'};
function imgFor(){
  if(state.mode==='clay'&&state.view==='front')return 'renders/new-clay-front.png';
  return `renders/${state.which}-${state.view}.png`;
}
function applyView(){
  const src=imgFor();$('view-img').src=src;$('view-img').alt=src;
  $('view-caption').textContent=`${state.which==='new'?'新候选':'原模型'} · ${viewNames[state.view]||state.view} · ${state.mode==='clay'?'灰模':'材质'}`;
  document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view===state.view));
  document.querySelectorAll('[data-which]').forEach(b=>b.classList.toggle('active',b.dataset.which===state.which));
  document.querySelectorAll('[data-mode]').forEach(b=>b.classList.toggle('active',b.dataset.mode===state.mode));
}
function applyTurn(){
  const src=`renders/turn-${String(state.turn).padStart(2,'0')}.png`;
  $('turn-img').src=src;$('turn-range').value=state.turn;
  $('turn-caption').textContent=`第 ${state.turn+1}/24 方向（拖动或滑块切换；仅预渲染转台，不是交互式 3D 引擎）`;
}
document.querySelectorAll('[data-which]').forEach(b=>b.onclick=()=>{state.which=b.dataset.which;applyView();});
document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>{state.mode=b.dataset.mode;applyView();});
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{state.view=b.dataset.view;applyView();});
$('turn-range').oninput=e=>{state.turn=Number(e.target.value);applyTurn();};
let drag=null;
const wrap=$('turn-wrap');
wrap.addEventListener('pointerdown',e=>{drag={x:e.clientX,turn:state.turn};wrap.setPointerCapture?.(e.pointerId);});
wrap.addEventListener('pointermove',e=>{
  if(!drag)return;
  const delta=Math.round((e.clientX-drag.x)/16);
  state.turn=(drag.turn+delta+24)%24;applyTurn();
});
wrap.addEventListener('pointerup',()=>{drag=null;});
wrap.addEventListener('pointercancel',()=>{drag=null;});
applyView();applyTurn();
function waitImages(){
  const imgs=[...document.images];
  return Promise.all(imgs.map(img=>{
    if(img.complete)return img.naturalWidth?Promise.resolve():Promise.reject(new Error('broken '+img.src));
    return new Promise((res,rej)=>{img.onload=res;img.onerror=()=>rej(new Error('load '+img.src));});
  }));
}
window.addEventListener('load',async()=>{
  const errors=[];window.addEventListener('error',e=>errors.push(String(e.message||e.error||e)));
  try{
    await waitImages();
    const initialTurn=state.turn;
    const evt=(type,x)=>new PointerEvent(type,{bubbles:true,clientX:x,pointerId:1,pointerType:'mouse'});
    wrap.dispatchEvent(evt('pointerdown',120));
    wrap.dispatchEvent(evt('pointermove',300));
    wrap.dispatchEvent(evt('pointerup',300));
    const turnChanged=state.turn!==initialTurn;
    document.querySelector('[data-view="front"]').click();
    document.querySelector('[data-which="old"]').click();
    const oldSrc=$('view-img').getAttribute('src');
    document.querySelector('[data-which="new"]').click();
    const newSrc=$('view-img').getAttribute('src');
    document.querySelector('[data-mode="clay"]').click();
    const claySrc=$('view-img').getAttribute('src');
    document.querySelector('[data-mode="material"]').click();
    document.querySelector('[data-view="top"]').click();
    const topSrc=$('view-img').getAttribute('src');
    document.querySelector('[data-view="front"]').click();
    const result={
      images_loaded:true,error_count:errors.length,errors,
      old_new_differ:oldSrc!==newSrc,
      clay_material_differ:claySrc!==newSrc,
      turntable_drag_changed:turnChanged,
      view_switch_changed:topSrc!==newSrc,
      final_state:{...state},
      link_count:document.querySelectorAll('aside a').length,
      has_glb:!!document.querySelector('a[href="mesh.glb"]'),
      has_blend:!!document.querySelector('a[href="model.blend"]'),
      has_board:!!document.querySelector('a[href="六视图-新旧对照.jpg"]')
    };
    result.passed=!!(result.images_loaded&&!result.errors.length&&result.old_new_differ&&
      result.clay_material_differ&&result.turntable_drag_changed&&result.has_glb&&
      result.has_blend&&result.has_board);
    document.getElementById('selftest').textContent=JSON.stringify(result);
    document.body.dataset.selftest=result.passed?'passed':'failed';
  }catch(e){
    const result={passed:false,images_loaded:false,errors:[...errors,String(e&&e.message||e)]};
    document.getElementById('selftest').textContent=JSON.stringify(result);
    document.body.dataset.selftest='failed';
  }
});
"""


def _value(record, keys, default='—'):
    node = record
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _yes(value):
    return '<span class="ok">通过</span>' if value is True else \
        ('<span class="bad">未通过</span>' if value is False else str(value))


def build_page(out):
    out = Path(out)
    result = json.loads((out / 'result.json').read_text(encoding='utf8'))
    checks = result['checks']
    opening = result.get('isolated_opening_checks') or {}
    reflection = result.get('reflection_checks') or {}
    metal = reflection.get('metal_response', {})
    views = result.get('reference_views', [])
    detail_items = (
        ('detail-front', '整体正视'), ('detail-cap-front', '盖正面/喷雾缺口'),
        ('detail-cap-top', '盖顶 U 形槽'), ('detail-cap-well', '槽内底板'),
        ('detail-nozzle', '按压帽/喷孔'), ('detail-label-logo', '正面字标/标志'),
        ('detail-label-lower', '标签下部'), ('detail-label-side', '侧面标签'),
        ('detail-cap-seam', '盖缝/划痕'), ('detail-bottom', '罐底凹面'),
        ('detail-bottom-code', '冲压编码'), ('detail-bottom-rim', '底部卷边'),
        ('detail-body-side', '罐身/金属底圈'), ('detail-surface', '标签表面'),
        ('detail-cap-scratch', '盖面划痕'), ('new-clay-close', '灰模几何'),
    )
    gallery = ''.join(
        f'<figure><img src="renders/{name}.png" alt="{html.escape(label)}">'
        f'<figcaption>{html.escape(label)}</figcaption></figure>'
        for name, label in detail_items)
    view_buttons = ''.join(
        f'<button data-view="{view}">{label}</button>' for view, label in
        (('front', '正视'), ('back', '背视'), ('left', '左视'),
         ('right', '右视'), ('top', '顶视'), ('bottom', '底视')))
    ref_rows = ''.join(
        f'<tr><td>{row["view"]}</td><td>{row.get("reference_new_iou")}</td>'
        f'<td>{row.get("reference_old_iou")}</td><td>{row.get("old_new_iou")}</td></tr>'
        for row in views)
    pending = ''.join(f'<li>{html.escape(item)}</li>' for item in result['pending_review'])
    limitations = ''.join(f'<li>{html.escape(item)}</li>' for item in result['limitations'][:6])
    check_rows = ''.join(
        f'<tr><td>{html.escape(key)}</td><td>{_yes(value)}</td></tr>'
        for key, value in checks.items())
    board_links = (
        ('六视图-新旧对照.jpg', '六视图新旧对照'), ('局部检查.jpg', '关键局部检查'),
        ('结构开口-隔离检查.jpg', '隔离结构与开口'), ('材质-同光照对照.jpg', '同光照材质'),
        ('材质-变光检查.jpg', '材质变光'), ('金属反射-变光对照.jpg', '金属反射变光'),
        ('转台-24方向.jpg', '24 方向转台'))
    boards = ''.join(
        f'<h3>{label}</h3><a href="{name}"><img src="{name}" alt="{label}"></a>'
        for name, label in board_links)
    data_block = json.dumps({
        'number': 14, 'manual_only': True, 'status': result['status'],
        'version': result['version'], 'checks': checks,
        'opening': opening, 'reflection': reflection,
        'reference_role': result['reference_role'],
        'pending_review': result['pending_review'],
    }, ensure_ascii=False)
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>14 杀虫剂罐 · 独立检查页</title><style>{CSS}</style></head>
<body data-number="14">
<header><h1>14 杀虫剂罐 · 独立检查页</h1>
<p>蓝色榄菊杀虫气雾罐的分部件重建候选。六视图为未标定的外观参考；原模型、标签包裹、盖体、按压帽与冲压底均为独立部件。页面只展示预渲染证据，不连接共享放置服务。</p>
<span class="badge">manual_only · 待人工复核 · 不进入自动放置/正式合成</span></header>
<div class="notice"><b>未验收声明：</b>六视图没有相机标定；候选的尺寸、标签油墨、盖内配合、喷嘴内部阀件、冲压深度、落稳和现场尺度均未验收。所有轮廓 IoU 都是按包围盒对齐的诊断值，不是测量精度。原模型比较时按新候选高度统一缩放，库文件未改动。</div>
<main>
<div>
<section class="viewer"><h2>六方向新旧与材质</h2>
<div class="controls">
<div class="group"><button data-which="new" class="active">新候选</button><button data-which="old">原模型</button></div>
<div class="group"><button data-mode="material" class="active">材质</button><button data-mode="clay">灰模（仅正视）</button></div>
<div class="group">{view_buttons}</div>
</div>
<figure><img id="view-img" src="renders/new-front.png" alt="new front"><figcaption id="view-caption">新候选 · 正视 · 材质</figcaption></figure>
</section>
<section class="turntable" style="margin-top:18px"><h2>预渲染 24 方向转台</h2>
<div id="turn-wrap"><img id="turn-img" src="renders/turn-00.png" alt="turntable"></div>
<input id="turn-range" type="range" min="0" max="23" value="0">
<p id="turn-caption" style="color:#53656c;font-size:13px">第 1/24 方向（拖动或滑块切换；仅预渲染转台，不是交互式 3D 引擎）</p>
<p style="color:#53656c;font-size:13px">此页没有 WebGL/GLB 实时查看器；GLB 与 Blender 文件用于后续串行整合和人工检查。</p>
</section>
<section class="gallery" style="margin-top:18px"><h2 style="grid-column:1/-1">关键局部</h2>{gallery}</section>
<section class="boards" style="margin-top:18px"><h2>对照板</h2>{boards}</section>
</div>
<aside>
<h2>检查摘要</h2>
<table>{check_rows}</table>
<h3>几何开口（射线）</h3>
<table>
<tr><th>项目</th><th>结果</th></tr>
<tr><td>喷雾槽深度</td><td>{opening.get('well_depth_m')} m {_yes(opening.get('well_recess_passed'))}</td></tr>
<tr><td>盖底中空差</td><td>{opening.get('cap_hollow_difference_m')} m {_yes(opening.get('cap_bottom_opening_passed'))}</td></tr>
<tr><td>喷嘴通孔</td><td>{_yes(opening.get('nozzle_bore_passed'))}</td></tr>
</table>
<h3>金属底反射</h3>
<p>平均亮度差：{metal.get('mean_abs_luma_difference', '—')}<br>高光质心偏移：{metal.get('highlight_shift_px', '—')} px {_yes(metal.get('changed'))}</p>
<h3>参考轮廓诊断（非验收）</h3>
<table><tr><th>视角</th><th>参考/新</th><th>参考/旧</th><th>旧/新</th></tr>{ref_rows}</table>
<h3>文件</h3>
<div class="links">
<a href="mesh.glb">下载 GLB（分部件重建候选）</a>
<a href="model.blend">下载可编辑 Blender 文件</a>
<a href="result.json">完整检查记录 result.json</a>
<a href="hashes.json">输入/输出哈希</a>
<a href="worker-result.json">Blender worker 记录</a>
<a href="request.json">复现请求/贴图路径</a>
</div>
<h3>待复核</h3><ul>{pending}</ul>
<h3>主要限制</h3><ul>{limitations}</ul>
<p class="mono">version: {html.escape(result['version'])}<br>reference: {html.escape(result['reference_name'])}</p>
</aside>
</main>
<pre id="selftest">pending</pre>
<script id="review-data" type="application/json">{data_block}</script>
<script>{JS}</script>
</body></html>"""
    path = out / '检查页.html'
    path.write_text(page, encoding='utf8')
    return path


def _chrome():
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    found = shutil.which('chrome') or shutil.which('msedge')
    if found:
        return found
    return None


def browser_check(out):
    out = Path(out)
    page = out / '检查页.html'
    if not page.is_file():
        build_page(out)
    chrome = _chrome()
    record = {
        'status': 'not_run',
        'checked_at': datetime.now(timezone.utc).isoformat(),
        'url': page.as_uri(),
        'method': ('Standalone HTML self-test in headless Chrome via file://; '
                   'checks image loading, old/new switch, clay/material switch, '
                   'view switch and synthetic turntable drag. It does not verify '
                   'WebGL rendering, GLB parsing, scene placement or model acceptance.'),
        'result': {},
    }
    if chrome is None:
        record['reason'] = 'Chrome/Edge headless executable not found'
        (out / '浏览器核验.json').write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding='utf8')
        return record
    common = [chrome, '--headless=new', '--disable-gpu', '--no-sandbox',
              '--allow-file-access-from-files', '--virtual-time-budget=12000',
              '--window-size=1400,1000']
    try:
        dumped = subprocess.run(common + ['--dump-dom', page.as_uri()],
                                capture_output=True, text=True, encoding='utf-8',
                                errors='replace', timeout=120)
        screenshot = out / '浏览器核验.png'
        shot = subprocess.run(common + ['--screenshot=' + str(screenshot),
                                        page.as_uri()],
                              capture_output=True, text=True, encoding='utf-8',
                              errors='replace', timeout=120)
        match = re.search(r'<pre id="selftest"[^>]*>(.*?)</pre>', dumped.stdout,
                          re.S | re.I)
        payload = html.unescape(match.group(1)) if match else ''
        try:
            result = json.loads(payload)
        except Exception:
            result = {'passed': False, 'parse_error': True,
                      'payload': payload[:2000]}
        result['chrome_returncode_dump_dom'] = dumped.returncode
        result['chrome_returncode_screenshot'] = shot.returncode
        result['screenshot'] = str(screenshot) if screenshot.is_file() else None
        record['result'] = result
        record['status'] = 'passed' if result.get('passed') else 'failed'
        if record['status'] != 'passed':
            record['reason'] = 'self-test did not pass'
    except Exception as exc:
        record['status'] = 'failed'
        record['reason'] = str(exc)
    (out / '浏览器核验.json').write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding='utf8')
    return record
