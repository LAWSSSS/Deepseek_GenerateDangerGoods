"""Build a review-only bearing-housing candidate from uncalibrated six-view evidence."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import trimesh
from PIL import Image

from scene_v2 import digest
from mesh_carve import interior_holes, opening_clearance, project, cut_opening
from mesh_detail import render_source
from mesh_checks import source_fidelity


def make_candidate(original, alpha, camera, pitch=.003):
    """Mirror source-facing geometry onto the inferred back of the housing.

    The six-view reference suggests front/back symmetry. The mirror is an
    explicitly unverified approximation, and the source-facing half remains
    derived from the existing mesh. The result still requires source checks.
    """
    if not .002<=pitch<=.006:
        raise ValueError('网格采样间距超出范围')
    bounds=np.asarray(original.bounds)
    span=float(np.max(bounds[1]-bounds[0]))
    if span<=0:raise ValueError('源模型尺寸为空')
    step=span*pitch
    grid=original.voxelized(step).fill()
    matrix=grid.matrix.copy()
    origin=np.asarray(grid.transform[:3,3])
    plane=float((bounds[0,0]+bounds[1,0])/2)
    indices=np.arange(matrix.shape[0])
    mirror=np.rint((2*plane-(origin[0]+indices*step)-origin[0])/step).astype(int)
    old=matrix.copy()
    # The visible nipple sits off the estimated mirror plane. Mirroring it
    # would invent a second fitting; preserve the original top of the casting.
    cap_z=bounds[1,2]-.07*span
    cap=origin[2]+np.arange(matrix.shape[2])*step >= cap_z
    top=np.asarray(original.vertices)
    top=top[top[:,2]>=bounds[1,2]-.02*span]
    if not len(top):raise ValueError('无法定位顶部油嘴')
    nipple_y=float(np.median(top[:,1]))
    nipple_band=np.abs(origin[1]+np.arange(matrix.shape[1])*step-nipple_y)<.07*span
    for i,j in enumerate(mirror):
        if origin[0]+i*step<plane:
            matrix[i]=old[j] if 0<=j<len(old) else False
            matrix[i][np.ix_(nipple_band,cap)]=old[i][np.ix_(nipple_band,cap)]
    holes,count=interior_holes(alpha)
    if not count:
        raise ValueError('源图中没有可验证的孔洞')
    field=opening_clearance(holes)
    focal=256/np.tan(np.deg2rad(camera['fov_deg'])/2)
    occupied=np.argwhere(matrix)
    removed=0
    for start in range(0,len(occupied),150000):
        points=occupied[start:start+150000]
        u,v,depth=project(grid.indices_to_points(points),camera)
        selected=points[cut_opening(field,u,v,depth,step,focal)]
        matrix[selected[:,0],selected[:,1],selected[:,2]]=False
        removed+=len(selected)
    if removed/len(occupied)>.25:
        raise ValueError('源图透明孔需要删除过多网格')
    rebuilt=trimesh.voxel.VoxelGrid(matrix,transform=grid.transform)
    result=rebuilt.marching_cubes
    result.vertices=np.asarray(result.vertices)*step+origin
    # Sample the supplied material onto vertices. Interpolation avoids UV
    # seam streaks; mirrored surface appearance remains explicitly inferred.
    mapped=np.asarray(result.vertices).copy()
    preserve=(mapped[:,2]>=cap_z)&(np.abs(mapped[:,1]-nipple_y)<.07*span)
    rear=(mapped[:,0]<plane)&~preserve
    mapped[rear,0]=2*plane-mapped[rear,0]
    source=np.asarray(original.vertices)
    valid=np.flatnonzero(~((source[:,2]>=cap_z)&(np.abs(source[:,1]-nipple_y)<.1*span)))
    distances,nearest=cKDTree(source).query(mapped,k=8,workers=4)
    rear_distance,rear_nearest=cKDTree(source[valid]).query(mapped[rear],k=8,workers=4)
    nearest[rear]=valid[rear_nearest]
    distances[rear]=rear_distance
    source_colors=original.visual.to_color().vertex_colors
    weights=1/np.maximum(distances,span*.0015)**2
    colors=np.sum(source_colors[nearest].astype(np.float32)*weights[:,:,None],axis=1)/weights.sum(axis=1)[:,None]
    # The original oil fitting must not be painted a second time on the
    # inferred back. Blend that small inferred patch into adjacent casting
    # colors; it remains inferred rather than a photographed texture.
    vertices=np.asarray(result.vertices)
    neighbor=(source[:,2]>cap_z-.10*span)&(source[:,2]<cap_z-.02*span)&(np.abs(source[:,1]-nipple_y)>.12*span)&(np.abs(source[:,1]-nipple_y)<.22*span)
    if neighbor.sum()>20:
        palette=np.median(source_colors[neighbor].astype(np.float32),axis=0)
        vertical=(vertices[:,2]-(cap_z-.035*span))/(.055*span)
        horizontal=(vertices[:,1]-nipple_y)/(.10*span)
        blend=np.exp(-2*(vertical**2+horizontal**2))*rear
        colors=colors*(1-blend[:,None])+palette*blend[:,None]
    result.visual=trimesh.visual.ColorVisuals(mesh=result,vertex_colors=np.clip(colors,0,255).astype(np.uint8))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--library',type=Path,default=Path('outputs/三维模型库-圆润复验/results.jsonl'))
    p.add_argument('--blender',type=Path,default=Path('work/blender45/blender-4.5.3-windows-x64/blender.exe'))
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'result.json').exists():raise FileExistsError('请选择新的输出目录')
    records=[json.loads(line) for line in args.library.read_text(encoding='utf8').splitlines() if line.strip()]
    rows=[row for row in records if row.get('mesh_sha256')==digest(args.source)]
    if len(rows)!=1:raise ValueError('源模型与模型库记录不一致')
    row=rows[0]
    if row['number']!=3 or args.reference.name!='轴承座六视图工程图.png':
        raise ValueError('本修复方法仅适用于 03 轴承座及其对应六视图')
    alpha_path=Path(row['source_view']).parent/'source-alpha.png'
    if digest(alpha_path)!=row['source_alpha_sha256']:
        raise ValueError('源透明度与模型库记录不一致')
    alpha=np.asarray(Image.open(alpha_path).convert('L'))
    mesh=trimesh.load(args.source,force='scene').to_geometry()
    result=make_candidate(mesh,alpha,row['reconstruction']['source_camera'])
    if not result.is_watertight or result.body_count!=1:
        raise ValueError('候选网格没有形成单个封闭体')
    target=args.out/'mesh.glb';result.export(target)
    folder=(args.out/'sourcecheck').resolve()
    render_source(result,folder,row['reconstruction']['source_camera'],args.blender.resolve(),'CPU')
    with Image.open(folder/'source-view.png') as rendered:
        fidelity=source_fidelity(alpha,np.asarray(rendered.getchannel('A')))
    (folder/'fidelity.json').write_text(json.dumps(fidelity,ensure_ascii=False,indent=2),encoding='utf8')
    info={'status':'review','source':str(args.source.resolve()),'source_sha256':digest(args.source),
          'reference':str(args.reference.resolve()),'reference_sha256':digest(args.reference),
          'mesh':str(target.resolve()),'mesh_sha256':digest(target),
          'method':'reference_informed_mirrored_back_with_original_front_texture',
          'vertices':len(result.vertices),'faces':len(result.faces),'watertight':bool(result.is_watertight),
          'dimensions':result.extents.tolist(),
          'source_fidelity':fidelity,
          'limitations':['参考六视图未标定，背面对称和细节为推测','背面纹理由源正面镜像推断，不是实拍',
                         '顶部保留原油嘴处的过渡可能产生接缝，需要人工复核',
                         '物理落稳、遮挡、尺度及背面真实性仍待验；源视图检查结果见 source_fidelity']}
    (args.out/'result.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(info,ensure_ascii=False))


if __name__=='__main__':main()
