"""Make a source-checked review candidate without a tiny floating fragment."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh

from mesh_checks import source_fidelity
from mesh_detail import render_source
from scene_v2 import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--number',type=int,choices=[8,19],default=8)
    p.add_argument('--library',type=Path,default=Path('outputs/三维模型库-圆润复验/results.jsonl'))
    p.add_argument('--blender',type=Path,default=Path('work/blender45/blender-4.5.3-windows-x64/blender.exe'))
    args=p.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError('请选择新的输出目录')
    rows=[json.loads(x) for x in args.library.read_text(encoding='utf8').splitlines() if x.strip()]
    row=next(r for r in rows if r['number']==args.number)
    source=Path(row['mesh']);alpha=Path(row['source_view']).parent/'source-alpha.png'
    if digest(source)!=row['mesh_sha256'] or digest(alpha)!=row['source_alpha_sha256']:
        raise ValueError('源网格或透明度哈希不匹配')
    original=trimesh.load(source,force='scene').to_geometry()
    parts=sorted(original.split(only_watertight=False),key=lambda m:m.area,reverse=True)
    removed=parts[1] if len(parts)>1 else None
    ratio=float(removed.area/sum(p.area for p in parts)) if removed else 0
    if args.number==8:
        if len(parts)!=2 or ratio>.02:
            raise ValueError('散件数量或面积不符合已核对的 08 号小漂浮碎片')
        candidate=parts[0]
    else:
        # The 19 source contains six bottles; the independently floating
        # rear wedge is not a bottle. Keep both bottle-bearing components.
        if len(parts)!=3 or not .10<=ratio<=.22 or parts[1].bounds[1,0]>=parts[0].bounds[0,0]:
            raise ValueError('散件不符合已核对的 19 号背部浮片')
        candidate=trimesh.util.concatenate((parts[0],parts[2]))
    target=out/'mesh.glb';candidate.export(target)
    check=out/'sourcecheck'
    render_source(candidate,check,row['reconstruction']['source_camera'],args.blender.resolve(),'CPU')
    with Image.open(alpha) as a,Image.open(check/'source-view.png') as b:
        fidelity=source_fidelity(np.asarray(a.convert('L')),np.asarray(b.getchannel('A')))
    info={'number':args.number,'asset_id':row['asset_id'],'source_sha256':row['mesh_sha256'],
          'mesh':str(target),'mesh_sha256':digest(target),'fidelity':fidelity,
          'removed_area_fraction':round(ratio,5),
          'status':'source_pass_pending_structure' if fidelity['passed'] else 'review_source_failed',
          'limitations':['仅移除原图未见的漂浮碎片，隐藏结构、独立部件关系和物理姿态仍待复核']}
    (out/'result.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(info,ensure_ascii=False))


if __name__=='__main__':main()
