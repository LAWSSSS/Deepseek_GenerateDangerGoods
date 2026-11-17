"""Try source-corroborated opening repair without replacing the model library."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import trimesh

from mesh_carve import carve
from mesh_checks import source_fidelity
from mesh_detail import render_source
from mesh_library import holes_corroborated
from reconstruct_worker import normalized_input
from scene_v2 import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--library',type=Path,default=Path('outputs/三维模型库-圆润复验/results.jsonl'))
    p.add_argument('--blender',type=Path,default=Path('work/blender45/blender-4.5.3-windows-x64/blender.exe'))
    p.add_argument('--ids',type=int,nargs='+',default=[2,7,29])
    p.add_argument('--padding-factor',type=float,default=.85)
    p.add_argument('--resolution',type=int,choices=[192,256,384])
    args=p.parse_args();args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    rows={r['number']:r for r in (json.loads(x) for x in args.library.read_text(encoding='utf8').splitlines() if x.strip())}
    reports=[]
    for number in args.ids:
        row=rows[number];folder=args.out/f'{number:02d}';folder.mkdir(exist_ok=False)
        info={'number':number,'asset_id':row['asset_id'],'original_mesh_sha256':row['mesh_sha256'],
              'status':'review','started':time.time()}
        try:
            mesh=Path(row['mesh']);rust=Path(row['source_view']).parent/'source-alpha.png'
            if digest(mesh)!=row['mesh_sha256'] or digest(rust)!=row['source_alpha_sha256'] or digest(Path(row['original']))!=row['clean_sha256']:
                raise ValueError('模型或原始证据的哈希不匹配')
            clean=folder/'clean';clean.mkdir();normalized_input(Path(row['original']),clean)
            with Image.open(rust) as a,Image.open(clean/'source-alpha.png') as b:
                corroboration=holes_corroborated(np.asarray(a.convert('L')),np.asarray(b.convert('L')))
            info['source_hole_corroboration']=corroboration
            if not corroboration['corroborated']:
                raise ValueError('原貌素材没有支持全部透明孔，停止修复')
            trial=folder/'candidate'
            carve(mesh,rust,Path(row['source_view']),row['reconstruction']['source_camera'],trial,
                  resolution=args.resolution,padding_factor=args.padding_factor)
            candidate=trimesh.load(trial/'mesh.glb',force='mesh',process=False)
            source=folder/'sourcecheck'
            render_source(candidate,source,row['reconstruction']['source_camera'],args.blender.resolve(),'CPU')
            with Image.open(rust) as a, Image.open(source/'source-view.png') as b:
                fidelity=source_fidelity(np.asarray(a.convert('L')),np.asarray(b.getchannel('A')))
            info.update(fidelity=fidelity,mesh=str((trial/'mesh.glb').resolve()),mesh_sha256=digest(trial/'mesh.glb'))
            info['status']='source_pass_pending_structure' if fidelity['passed'] else 'review_source_failed'
            info['limitations']=['孔深和隐藏结构仍为推断','模型外观与物理落稳尚未验收']
        except Exception as exc:
            info['reason']=str(exc)[:400]
        info['elapsed_seconds']=round(time.time()-info.pop('started'),2)
        (folder/'result.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf8')
        reports.append(info)
        (args.out/'results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in reports),encoding='utf8')
        print(number,info['status'],info.get('reason',''),flush=True)


if __name__=='__main__':main()
