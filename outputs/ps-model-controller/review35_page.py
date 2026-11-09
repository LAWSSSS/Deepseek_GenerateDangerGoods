"""Generate the standalone review page for the item-35 extinguisher candidate.

The page is written inside the candidate output directory and uses only local
rendered evidence.  It never modifies the shared placement server, shared index
or any other item's page.
"""
import html
import json
from pathlib import Path


def _row(label, value):
    return f'<tr><th>{html.escape(str(label))}</th><td>{html.escape(str(value))}</td></tr>'


def write_page(out, result, tables):
    out = Path(out)
    checks = tables['checks']
    fidelity = tables['fidelity']
    aligned = tables['aligned']
    best = tables['best_aligned']
    metal = tables['metal']
    openings = tables['openings']
    render_openings = tables['render_openings']
    label = tables['label']
    dial = tables['dial']
    texture = tables['texture']
    structural = tables['structural']
    pending = tables['pending']
    ref_rows = tables['reference_views']

    rows = [
        _row('严格原相机轮廓 IoU（未标定，仅诊断）', f"{fidelity['iou']:.4f}"),
        _row('原相机诊断填孔数', fidelity['filled_holes']),
        _row('包围盒对齐 IoU（仅诊断）', None if aligned['iou'] is None else f"{aligned['iou']:.4f}"),
        _row('24 方向最优诊断 IoU',
             None if best['iou'] is None else f"{best['iou']:.4f} @ {best['best_turntable_view_deg']}°"),
        _row('网格部件 / 闭合', f"{result['mesh_audit']['parts']} / {result['mesh_audit']['closed_after_welding_seams']}"),
        _row('同部件自交', result['worker']['self_intersecting_components'] or '无'),
        _row('喇叭口通孔（射线）', openings['nozzle_bore_ray']['passed']),
        _row('保险销环（射线）', openings['safety_pin_ring']['passed']),
        _row('凹底（射线）', openings['concave_bottom']['passed']),
        _row('隔离渲染孔数：喇叭口 / 保险销环',
             f"{render_openings['nozzle_through_interior_holes']} / {render_openings['safety_ring_interior_holes']}"),
        _row('PBR 图集数值', texture['passed']),
        _row('标签纹理数值', label['passed']),
        _row('表盘纹理数值', dial['passed']),
        _row('铬件高光随光位移动', f"{metal['passed']}（位移 {metal.get('shift_px')} px）"),
        _row('网格边界', result['mesh_audit']['bounds']),
        _row('浏览器交互检查', '未执行'),
        _row('落稳 / 现场尺度 / 场景放置', '未验证'),
        _row('状态', 'manual_only · 待人工复核'),
    ]

    reference_rows = ''
    for row in ref_rows:
        reference_rows += ('<tr><td>' + html.escape(row['view']) + '</td><td>'
                           + str(row['reference_new_mask_iou']) + '</td><td>'
                           + str(row['old_new_silhouette_iou']) + '</td><td>'
                           + html.escape(row['note']) + '</td></tr>')

    pending_html = ''.join(f'<li>{html.escape(item)}</li>' for item in pending)
    turn_frames = ','.join(f"'renders/turn-{i:02d}.png'" for i in range(24))

    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>35 灭火器2 · 三维模型检查（待审）</title>
<style>
:root{{--ink:#1d2320;--muted:#5f6a64;--line:#d5d9d5;--panel:#fbfcfa;--accent:#8a2a1f}}
*{{box-sizing:border-box}}body{{margin:0;font:15px/1.65 "Microsoft YaHei",system-ui,sans-serif;color:var(--ink);background:#eef1ee}}
header{{background:#2c2422;color:#f7efe9;padding:26px 30px}}header h1{{margin:0 0 6px;font-size:25px}}
header p{{margin:0;color:#ddcbc3}}main{{max-width:1220px;margin:24px auto 60px;padding:0 18px}}
section{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:20px;margin:18px 0}}
h2{{margin:0 0 12px;font-size:19px;color:var(--accent)}}h3{{margin:16px 0 8px;font-size:16px}}
.notice{{border-left:5px solid #b5762a;background:#fff7e8;padding:12px 16px;border-radius:6px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px}}
.card{{display:block;background:#fff;border:1px solid var(--line);border-radius:9px;overflow:hidden;color:inherit;text-decoration:none}}
.card img{{width:100%;height:220px;object-fit:contain;background:#e7eae6;display:block}}
.card span{{display:block;padding:8px 10px;font-size:13px}}
.viewer{{display:grid;grid-template-columns:1fr 280px;gap:18px}}@media(max-width:820px){{.viewer{{grid-template-columns:1fr}}}}
.turnbox{{background:#e3e7e2;border-radius:10px;min-height:400px;display:flex;align-items:center;justify-content:center;user-select:none;touch-action:none;cursor:grab}}
.turnbox img{{max-height:460px;max-width:100%}}
aside{{font-size:13px;color:var(--muted)}}button{{font:inherit;padding:7px 14px;border:1px solid var(--line);border-radius:8px;background:#fff;cursor:pointer}}
button:hover{{border-color:var(--accent)}}table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;border-bottom:1px solid var(--line);padding:8px 10px;vertical-align:top}}
th{{width:380px;color:var(--muted);font-weight:500}}code{{background:#eef1ee;padding:1px 5px;border-radius:4px}}
ul{{margin:8px 0 0 20px;padding:0}}a{{color:var(--accent)}}.tag{{display:inline-block;background:#f0d9cf;color:#7a2a1a;border-radius:20px;padding:2px 10px;font-size:12px}}
</style></head><body>
<header><h1>35 灭火器2 · 六视图分部件重建（{html.escape(result['version'])}） <span class="tag">manual_only · 待审</span></h1>
<p>红色 MFZ/ABC3 手提式干粉灭火器 · 筒体/卷边/凹底/阀颈/阀体/表盘/手柄/保险销/喇叭口/曲面标签 · 六视图为未标定视觉匹配参考</p></header>
<main>
<section><div class="notice"><b>不能进入正式合成。</b> 本候选不替换原模型，不进入自动放置或正式合成。
原照与六视图均未标定；所有 IoU 与尺寸均为诊断/推断，不是验收通过。背面、内部通道、壁厚、落稳和现场光照仍待人工复核。</div></section>

<section><h2>旋转检查（24 方向实测渲染）</h2><div class="viewer">
<div class="turnbox" id="turnbox"><img id="turn" src="renders/turn-00.png" alt="转台"></div>
<aside><p>拖动图片旋转；也可自动播放。转台只覆盖外观，不证明隐藏结构、零件配合或落稳。</p>
<p><button id="play">自动旋转</button> <button id="reset">回到 000°</button></p>
<p>当前角度：<b id="angle">000°</b></p>
<p><a href="mesh.glb">下载 GLB</a><br><a href="model.blend">下载 Blender 可编辑文件</a><br>
<a href="result.json">result.json</a> · <a href="hashes.json">hashes.json</a> · <a href="report.md">报告</a></p></aside></div></section>

<section><h2>对照图</h2><div class="grid">
<a class="card" href="六视图-新旧对照.jpg" target="_blank"><img src="六视图-新旧对照.jpg" alt="六方向新旧对照"><span>六方向新旧对照（参考 / 原模型 / 新候选）</span></a>
<a class="card" href="局部检查.jpg" target="_blank"><img src="局部检查.jpg" alt="关键局部"><span>关键局部：表盘/手柄/喇叭口/标签/卷边/凹底</span></a>
<a class="card" href="结构开口-隔离检查.jpg" target="_blank"><img src="结构开口-隔离检查.jpg" alt="分部件隔离与开孔"><span>分部件隔离与开孔</span></a>
<a class="card" href="材质-同光照对照.jpg" target="_blank"><img src="材质-同光照对照.jpg" alt="材质与灰模"><span>同光照材质 / 灰模</span></a>
<a class="card" href="材质-变光检查.jpg" target="_blank"><img src="材质-变光检查.jpg" alt="变光"><span>主光方向改变后的漆面响应</span></a>
<a class="card" href="表面反射-变光对照.jpg" target="_blank"><img src="表面反射-变光对照.jpg" alt="金属反射"><span>仅铬件的金属高光变光</span></a>
<a class="card" href="原照-新候选对照.jpg" target="_blank"><img src="原照-新候选对照.jpg" alt="原照对照"><span>原照 / 原模型 / 新候选 / 参考</span></a>
<a class="card" href="转台-24方向.jpg" target="_blank"><img src="转台-24方向.jpg" alt="24 方向转台"><span>24 方向转台</span></a>
<a class="card" href="材质图集-布局.jpg" target="_blank"><img src="材质图集-布局.jpg" alt="PBR 图集"><span>PBR 图集布局</span></a>
<a class="card" href="展签-表盘-贴图.jpg" target="_blank"><img src="展签-表盘-贴图.jpg" alt="标签与表盘"><span>矫正后的标签与表盘纹理</span></a>
</div></section>

<section><h2>六方向诊断对照</h2>
<table><tr><th>视角</th><th>参考-新轮廓 IoU</th><th>旧-新轮廓 IoU</th><th>说明</th></tr>
{reference_rows}</table>
<p class="notice" style="margin-top:12px">IoU 均按最长边等比对齐，参考图未标定且不是同一台实物；只用于发现形体差异，不能当作测量精度或验收。</p></section>

<section><h2>实际执行的检查</h2><table>{''.join(rows)}</table>
<p class="notice" style="margin-top:14px">图集/标签/表盘数值检查只证明贴图内容与参数；开孔射线只覆盖喇叭口、保险销环与凹底；
金属变光只证明铬件高光移动，不证明环境匹配。未执行项已明确写为“未执行/未验证”。</p></section>

<section><h2>分部件结构</h2><table>
<tr><th>部件计数</th><td><code>{html.escape(json.dumps(structural['counts'], ensure_ascii=False))}</code></td></tr>
<tr><th>铬件计数</th><td>{structural['gauge_chrome_parts']}</td></tr>
<tr><th>说明</th><td>{html.escape(structural['note'])}</td></tr></table></section>

<section><h2>待人工复核</h2><ul>{pending_html}</ul></section>

<section><h2>文件与记录</h2>
<p><a href="result.json">result.json</a> · <a href="hashes.json">hashes.json</a> ·
<a href="source-measurements.json">原照测量/假设</a> · <a href="old-model-audit.json">旧模型审计</a> ·
<a href="request.json">Blender request</a> · <a href="worker-result.json">worker 记录</a> ·
<a href="report.md">中文报告</a></p>
<p>贴图：<a href="base_color.png">底色图集</a> · <a href="metallic_roughness.png">金属/粗糙度</a> ·
<a href="normal.png">法线</a> · <a href="label_base.png">标签</a> · <a href="dial_base.png">表盘</a></p>
<p>候选为 <code>{html.escape(result['version'])}</code>，参考角色 <code>visual_match_unverified</code>；
原始模型和共享页面未被修改。串行整合后再接入共享查看器与人工验收。</p></section>
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
    (out / 'index.html').write_text(page, encoding='utf8')
