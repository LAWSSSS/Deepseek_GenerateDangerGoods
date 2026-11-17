"""Bounded, source-checked depth corrections for reference-supported cylinders."""
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
    p.add_argument('--library',type=Path,default=Path('outputs/三维模型库-圆润复验/results.jsonl'))
    p.add_argument('--references',type=Path,default=Path('references/六视图'))
    p.add_argument('--map',type=Path,default=Path('tools/ps-model-controller/six-view-map.json'))
    p.add_argument('--blender',type=Path,default=Path('work/blender45/blender-4.5.3-windows-x64/blender.exe'))
    args=p.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    rows={r['number']:r for r in (json.loads(x) for x in args.library.read_text(encoding='utf8').splitlines() if x.strip())}
    refs={r['number']:r for r in json.loads(args.map.read_text(encoding='utf8'))['views']}
    reports=[]
    for number in (15,16,20,36):
        row=rows[number];folder=out/f'{number:02d}';folder.mkdir(exist_ok=False)
        info={'number':number,'asset_id':row['asset_id'],'original_mesh_sha256':row['mesh_sha256'],'status':'review'}
        try:
            source=Path(row['mesh']);alpha=Path(row['source_view']).parent/'source-alpha.png'
            reference=args.references/refs[number]['file']
            if not reference.is_file() or digest(source)!=row['mesh_sha256'] or digest(alpha)!=row['source_alpha_sha256']:
                raise ValueError('参考图或源模型证据不可用')
            mesh=trimesh.load(source,force='scene').to_geometry()
            x,y,z=map(float,mesh.extents)
            if z<1.7*max(x,y) or x/y<1.3:
                raise ValueError('网格不满足竖向圆柱的受限修复条件')
            scale=float(np.clip(y/x,.4,1.0))
            # Z is the cylinder axis. Approximate a circular X/Y section from
            # the uncalibrated reference; keep front-view geometry gated below.
            vertices=np.asarray(mesh.vertices).copy();center=float((mesh.bounds[0,0]+mesh.bounds[1,0])/2)
            vertices[:,0]=center+(vertices[:,0]-center)*scale
            candidate=mesh.copy();candidate.vertices=vertices
            path=folder/'mesh.glb';candidate.export(path)
            check=folder/'sourcecheck'
            render_source(candidate,check,row['reconstruction']['source_camera'],args.blender.resolve(),'CPU')
            with Image.open(alpha) as a,Image.open(check/'source-view.png') as b:
                fidelity=source_fidelity(np.asarray(a.convert('L')),np.asarray(b.getchannel('A')))
            info.update(mesh=str(path),mesh_sha256=digest(path),reference=str(reference.resolve()),
                        reference_sha256=digest(reference),depth_scale=round(scale,4),fidelity=fidelity,
                        status='source_pass_pending_structure' if fidelity['passed'] else 'review_source_failed',
                        limitations=['六视图未标定，圆形截面及侧背面为近似推断','材质、落稳与真实结构尚待复核'])
        except Exception as exc:info['reason']=str(exc)[:300]
        (folder/'result.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf8')
        reports.append(info)
        (out/'results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in reports),encoding='utf8')
        print(number,info['status'],info.get('fidelity',{}),info.get('reason',''),flush=True)


if __name__=='__main__':main()
