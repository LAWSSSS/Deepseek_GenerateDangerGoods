"""Generate the standalone review page for the item-39 oil-receiver candidate.

The page is written inside the candidate output directory and uses only local
pre-rendered evidence.  It never modifies the shared placement server, shared
index or any other item's page.
"""
import html
from pathlib import Path


def _row(label, value):
    return f'<tr><th>{html.escape(str(label))}</th><td>{html.escape(str(value))}</td></tr>'


def write_page(out, result, tables):
    out = Path(out)
    checks = tables['checks']
    fidelity = tables['fidelity']
    reference_views = tables['reference_views']
    textures = tables['textures']
    metal = tables['metal']
    paint = tables['paint']
    openings = tables['openings']
    structural = tables['structural']
    browser = tables['browser']
    pending = tables['pending']
    luma = tables['luma']

    rows = [
        _row('严格原相机轮廓 IoU（未标定，仅诊断）', f"{fidelity['iou']:.4f}"),
        _row('原相机诊断填孔数', fidelity['filled_holes']),
        _row('窄结构召回（诊断）', fidelity['narrow_recall']),
        _row('网格部件 / 闭合', f"{result['mesh_audit']['parts']} / "
             f"{result['mesh_audit']['closed_after_welding_seams']}"),
        _row('同部件自交', result['worker']['self_intersecting_components'] or '无'),
        _row('隔离渲染内孔：正视 / 顶视',
             f"{openings['silhouette_front_holes']} / {openings['silhouette_top_holes']}"),
        _row('PBR 贴图数值', textures['passed']),
        _row('金属件高光随光位移动', f"{metal['passed']}（平均差 {metal['mean_abs_difference']}）"),
        _row('漆面高光随光位移动', f"{paint['changed']}（平均差 {paint['mean_abs_difference']}）"),
        _row('新正视图亮度/黑块占比',
             f"{luma['new-front.png']['median_luma']} / {luma['new-front.png']['dark_fraction']}"),
        _row('浏览器页面', browser.get('status', '未执行')),
        _row('落稳 / 现场尺度 / 场景放置', '未验证'),
        _row('状态', 'manual_only · 待人工复核'),
    ]

    reference_rows = ''.join(
        '<tr><td>' + html.escape(str(row['view'])) + '</td><td>'
        + ('' if row['reference_new_mask_iou'] is None else str(row['reference_new_mask_iou']))
        + '</td><td>'
        + ('' if row['old_new_silhouette_iou'] is None else str(row['old_new_silhouette_iou']))
        + '</td><td>' + html.escape(str(row['note'])) + '</td></tr>'
        for row in reference_views)

    texture_rows = ''.join(
        f'<tr><th>{html.escape(key)}</th><td>{html.escape(str(value))}</td></tr>'
        for key, value in textures['summary'].items())

    pending_html = ''.join(f'<li>{html.escape(item)}</li>' for item in pending)
    structural_html = html.escape(str(structural['counts']))
    turn_frames = ','.join(f"'renders/turn-{i:02d}.png'" for i in range(24))

    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>39 集油器2 · 三维模型检查（待审）</title>
<style>
:root{{--ink:#17242b;--muted:#5b6b72;--line:#d2dada;--panel:#fbfcfc;--accent:#0f6f70;--warn:#8a2b20}}
*{{box-sizing:border-box}}body{{margin:0;font:15px/1.6 "Microsoft YaHei",system-ui,sans-serif;color:var(--ink);background:#edf1f1}}
header{{background:linear-gradient(120deg,#0d3d43,#127d7c);color:#f2fbfa;padding:26px 30px}}header h1{{margin:0 0 6px;font-size:25px}}
header p{{margin:0;color:#d3eeeb}}main{{max-width:1240px;margin:24px auto 60px;padding:0 18px}}
section{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:20px;margin:18px 0}}
h2{{margin:0 0 12px;font-size:19px;color:var(--accent)}}h3{{margin:16px 0 8px;font-size:16px}}
.notice{{border-left:5px solid #b5762a;background:#fff7e8;padding:12px 16px;border-radius:6px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px}}
.card{{display:block;background:#fff;border:1px solid var(--line);border-radius:9px;overflow:hidden;color:inherit;text-decoration:none}}
.card img{{width:100%;height:220px;object-fit:contain;background:#e7ebeb;display:block}}
.card span{{display:block;padding:8px 10px;font-size:13px}}
.viewer{{display:grid;grid-template-columns:1fr 300px;gap:18px}}@media(max-width:850px){{.viewer{{grid-template-columns:1fr}}}}
.turnbox{{background:#e2e8e8;border-radius:10px;min-height:420px;display:flex;align-items:center;justify-content:center;user-select:none;touch-action:none;cursor:grab}}
.turnbox img{{max-height:520px;max-width:100%}}
aside{{font-size:13px;color:var(--muted)}}button{{font:inherit;padding:7px 14px;border:1px solid var(--line);border-radius:8px;background:#fff;cursor:pointer}}
button:hover{{border-color:var(--accent)}}table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;border-bottom:1px solid var(--line);padding:8px 10px;vertical-align:top}}
th{{width:360px;color:var(--muted);font-weight:500}}code{{background:#e9efef;padding:1px 5px;border-radius:4px}}
ul{{margin:8px 0 0 20px;padding:0}}a{{color:var(--accent)}}.tag{{display:inline-block;background:#cfe8e5;color:#0b5555;border-radius:20px;padding:2px 10px;font-size:12px}}
</style></head><body>
<header><h1>39 集油器2 · 六视图分部件重建（{html.escape(result['version'])}） <span class="tag">manual_only · 待审</span></h1>
<p>立式青绿色 HPEOK PHO-10 油分离/集油压力罐 · 筒体/封头/焊缝/双视镜口/标签/上下阀组/侧向服务接头 · 六视图为未标定视觉匹配参考</p></header>
<main>
<section><div class="notice"><b>不能进入正式合成。</b> 本候选不替换原模型，不进入自动放置或正式合成。
原照与六视图均未标定；所有 IoU、尺寸、阀门朝向与背面结构均为诊断/推断，不是验收通过。壁厚、内部通道、螺纹、落稳、现场尺度与场景光照仍待人工复核。</div></section>

<section><h2>旋转检查（24 方向实测渲染）</h2><div class="viewer">
<div class="turnbox" id="turnbox"><img id="turn" src="renders/turn-00.png" alt="转台"></div>
<aside><p>拖动图片旋转；也可自动播放。转台只覆盖外观，不证明隐藏结构、零件配合或落稳。</p>
<p><button id="play">自动旋转</button> <button id="reset">回到 000°</button></p>
<p>当前角度：<b id="angle">000°</b></p>
<p><a href="mesh.glb">下载 GLB</a><br><a href="model.blend">下载 Blender 可编辑文件</a><br>
<a href="result.json">result.json</a> · <a href="hashes.json">hashes.json</a> ·
<a href="request.json">Blender request</a> · <a href="worker-result.json">worker 记录</a></p></aside></div></section>

<section><h2>对照图</h2><div class="grid">
<a class="card" href="六视图-新旧对照.jpg" target="_blank"><img src="六视图-新旧对照.jpg" alt="六方向新旧对照"><span>六方向新旧对照（参考 / 原模型 / 新候选）</span></a>
<a class="card" href="原照-新候选-参考.jpg" target="_blank"><img src="原照-新候选-参考.jpg" alt="原照对照"><span>原照 / 原模型 / 新候选 / 参考正视</span></a>
<a class="card" href="局部检查.jpg" target="_blank"><img src="局部检查.jpg" alt="关键局部"><span>关键局部：上下阀组/视镜口/标签/焊缝/侧接头</span></a>
<a class="card" href="结构开口-隔离检查.jpg" target="_blank"><img src="结构开口-隔离检查.jpg" alt="分部件隔离"><span>分部件隔离与开口检查</span></a>
<a class="card" href="材质-同光照对照.jpg" target="_blank"><img src="材质-同光照对照.jpg" alt="材质对照"><span>同光照材质：原模型 / 新候选</span></a>
<a class="card" href="材质-变光检查.jpg" target="_blank"><img src="材质-变光检查.jpg" alt="变光"><span>漆面主光方向变化</span></a>
<a class="card" href="表面反射-变光对照.jpg" target="_blank"><img src="表面反射-变光对照.jpg" alt="金属反射"><span>不锈钢/黄铜件高光变光</span></a>
<a class="card" href="PBR贴图-对照.jpg" target="_blank"><img src="PBR贴图-对照.jpg" alt="PBR 贴图"><span>PBR 贴图与标签矫正</span></a>
<a class="card" href="转台-24方向.jpg" target="_blank"><img src="转台-24方向.jpg" alt="24 方向转台"><span>24 方向转台</span></a>
</div></section>

<section><h2>六方向诊断对照</h2>
<table><tr><th>视角</th><th>参考-新轮廓 IoU</th><th>旧-新轮廓 IoU</th><th>说明</th></tr>
{reference_rows}</table>
<p class="notice" style="margin-top:12px">参考图为未标定视觉匹配图，各面板缩放不一致；IoU 按最长边等比对齐，只用于发现形体差异，不能当作测量精度或验收。</p></section>

<section><h2>实际执行的检查</h2><table>{''.join(rows)}</table>
<p class="notice" style="margin-top:14px">贴图数值检查只证明贴图内容与参数；金属变光只证明不锈钢/黄铜件高光随灯位移动，不证明环境匹配；
漆面变光只覆盖可见漆面。未执行项已明确写为“未执行/未验证”。</p></section>

<section><h2>PBR 贴图数值</h2><table>{texture_rows}</table></section>

<section><h2>分部件结构</h2><table>
<tr><th>部件计数</th><td><code>{structural_html}</code></td></tr>
<tr><th>材质分组</th><td><code>{html.escape(str(structural['materials']))}</code></td></tr>
<tr><th>说明</th><td>{html.escape(structural['note'])}</td></tr></table></section>

<section><h2>待人工复核</h2><ul>{pending_html}</ul></section>

<section><h2>文件与记录</h2>
<p><a href="result.json">result.json</a> · <a href="hashes.json">hashes.json</a> ·
<a href="request.json">Blender request</a> · <a href="worker-result.json">worker 记录</a> ·
<a href="texture-checks.json">贴图数值检查</a> · <a href="source-measurements.json">原照测量/假设</a> ·
<a href="old-model-audit.json">旧模型审计</a> · <a href="checks-39.json">独立检查记录</a></p>
<p>贴图：<a href="body_base.png">筒体底色</a> · <a href="body_mr.png">筒体金属/粗糙度</a> ·
<a href="body_normal.png">筒体法线</a> · <a href="paint_base.png">小件漆面图集</a> ·
<a href="label_base.png">HPEOK 标签</a> · <a href="metal_base.png">不锈钢</a> ·
<a href="brass_base.png">黄铜</a> · <a href="black_base.png">黑色塑料</a></p>
<p>候选为 <code>{html.escape(result['version'])}</code>，参考角色 <code>visual_match_unverified</code>；
原始模型、共享页面与共享索引未被修改。串行整合后再接入共享查看器与人工验收。</p></section>
</main>
<script>
const frames=[{turn_frames}];
let index=0, timer=null, dragging=false, startX=0, startIndex=0;
const img=document.getElementById('turn'), angle=document.getElementById('angle');
function show(i){{ index=(i+frames.length)%frames.length; img.src=frames[index];
  angle.textContent=String(index*15).padStart(3,'0')+'°'; }}
document.getElementById('play').onclick=()=>{{
  if(timer){{clearInterval(timer);timer=null;return;}}
  timer=setInterval(()=>show(index+1),140);
}};
document.getElementById('reset').onclick=()=>show(0);
const box=document.getElementById('turnbox');
box.addEventListener('pointerdown',e=>{{dragging=true;startX=e.clientX;startIndex=index;box.setPointerCapture(e.pointerId);}});
box.addEventListener('pointermove',e=>{{if(!dragging)return;
  const step=Math.round((e.clientX-startX)/18); if(step) show(startIndex-step);}});
box.addEventListener('pointerup',()=>{{dragging=false;}});
</script>
</body></html>
"""
    (out / 'review-39.html').write_text(page, encoding='utf8')
    return out / 'review-39.html'
