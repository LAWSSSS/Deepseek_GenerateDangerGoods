"""Build the inspectable before/after index for the refined model library."""
import argparse
from collections import Counter
import html
import json
from pathlib import Path


def link(path,label):
    if not path or not Path(path).is_file(): return ""
    uri=html.escape(Path(path).resolve().as_uri(),quote=True)
    return f'<a href="{uri}">{html.escape(label)}</a>'


def thumb(path,label):
    if not path or not Path(path).is_file(): return ""
    uri=html.escape(Path(path).resolve().as_uri(),quote=True)
    return f'<figure><img loading="lazy" src="{uri}" alt="{html.escape(label)}"><figcaption>{html.escape(label)}</figcaption></figure>'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before",type=Path,default=Path("outputs/三维模型库/results.jsonl"))
    p.add_argument("--refined",type=Path,default=Path("outputs/三维模型库-圆润复验/results.jsonl"))
    p.add_argument("--six-review",type=Path,default=Path("outputs/轴承座-六视图圆润对照-v4/03-六视图前后对照.jpg"))
    args=p.parse_args()
    rows=[json.loads(x) for x in args.refined.read_text(encoding="utf8").splitlines() if x.strip()]
    before={x["asset_id"]:x for x in (json.loads(s) for s in args.before.read_text(encoding="utf8").splitlines() if s.strip())}
    if any(row["asset_id"] not in before for row in rows):raise ValueError("新旧模型清单不一致")
    states=Counter(row["surface_refinement"]["state"] for row in rows)
    passed=sum(bool(row.get("reconstruction",{}).get("fidelity",{}).get("passed")) for row in rows)
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>三维模型圆润复验</title>',
           '<style>body{font:16px/1.5 sans-serif;margin:2em;max-width:1200px;background:#f7f8fa;color:#20252a}',
           'section{border-top:1px solid #ccd2d7;padding:1em 0}.views{display:flex;gap:1em;flex-wrap:wrap}',
           'figure{margin:.2em}img{width:250px;height:220px;object-fit:contain;background:#e5e8eb}',
           '.note{color:#5b6470}a{margin-right:1em}</style><h1>三维模型圆润复验</h1>',
           f'<p>共 {len(rows)} 件；接受平滑 {states.get("accepted",0)} 件；源视图几何通过 {passed} 件。</p>',
           '<p>六视图图片未标定，按外观做初步关联，不能当作精确尺寸，也尚未用于多视图重建。',
           '平滑不补齐缺失的结构，原本的待审状态继续有效。</p>',
           '<p>'+link(args.six_review,"轴承座六方向对照")+'</p>']
    for row in rows:
        old=before[row["asset_id"]]; refine=row["surface_refinement"];ref=refine.get("reference",{})
        status="平滑已接受" if refine["state"]=="accepted" else "保留原网格"
        source_status="源视图几何通过，结构待审" if row["reconstruction"]["fidelity"]["passed"] else "源视图几何待审"
        parts.append(f'<section><h2>{row["number"]:02} {html.escape(row["file"])}</h2><p>{status} · {source_status}</p>')
        if refine["state"]=="accepted":
            metric=refine["details"];b=metric["normal_laplacian_before_p50_p90_p99"][1];a=metric["normal_laplacian_after_p50_p90_p99"][1]
            parts.append(f'<p>高频起伏 P90：{b:.6f} → {a:.6f}；材质保留：{"是" if refine.get("texture_preserved") else "否"}。</p>')
        else:
            parts.append('<p>'+html.escape(refine.get("reason",""))+'</p>')
        parts.append('<div class="views">'+thumb(old.get("source_view"),"平滑前源视图")+thumb(row.get("source_view"),"平滑后源视图")+'</div>')
        parts.append('<p>'+link(row.get("mesh"),"三维网格 GLB")+link(ref.get("path"),"六视图参考")+
                     link(args.refined.parent/"items"/f'{row["number"]:02}-{row["asset_id"]}'/"result.json","检查记录")+'</p>')
        parts.append('<p class="note">参考关系：'+html.escape(ref.get("note","未找到可确认的参考"))+'</p></section>')
    parts.append('</html>')
    out=args.refined.parent/"index.html";temp=out.with_suffix(".tmp");temp.write_text(''.join(parts),encoding="utf8");temp.replace(out)
    print(out.resolve())


if __name__=="__main__":main()
