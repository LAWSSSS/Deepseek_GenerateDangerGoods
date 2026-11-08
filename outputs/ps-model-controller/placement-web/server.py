"""Local, review-first placement UI. Run from the project root."""
import asyncio
import base64
import hashlib
import io
import json
import os
import re
import secrets
import sys
import time
from pathlib import Path

import cv2
import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
from scene_v2 import Perception, digest
from relight import ambient_match
from occlusion import refine_thin_mask, thin_line_seeds
from lighting_estimate import estimate_light_from_rods

LIBRARY = ROOT / 'outputs/三维模型库-圆润复验/results.jsonl'
SCENE = ROOT / 'outputs/固定场景-39件尺寸标注/场景配置.json'
DATA = ROOT / 'work/placement-web'
DATA.mkdir(parents=True, exist_ok=True)
ROWS = {r['asset_id']: r for r in (json.loads(x) for x in LIBRARY.read_text(encoding='utf8').splitlines() if x.strip())}


def register_review_candidate():
    """Offer the verified-source candidate separately; never replace the library mesh."""
    folder = ROOT/'outputs/轴承座-六视图背面修复候选-v8'
    info_file, proof_file = folder/'result.json', folder/'sourcecheck/fidelity.json'
    if not info_file.is_file() or not proof_file.is_file():
        return
    try:
        info = json.loads(info_file.read_text(encoding='utf8'))
        proof = json.loads(proof_file.read_text(encoding='utf8'))
        original = next(r for r in ROWS.values() if r['number']==3)
        mesh = folder/'mesh.glb'
        if (not proof.get('passed') or proof.get('iou',0)<.90 or proof.get('filled_holes')!=0
                or proof.get('narrow_recall',0)<.80 or not mesh.is_file()
                or digest(mesh)!=info['mesh_sha256'] or info['source_sha256']!=original['mesh_sha256']):
            return
        key=original['asset_id']+'-sixview'
        ROWS[key]={**original,'asset_id':key,'file':'轴承座（六视图背面修复候选）',
                   'mesh':str(mesh),'mesh_sha256':info['mesh_sha256'], 'status':'review',
                   'reason':'背面依据未标定参考镜像推断；顶部过渡接缝和落稳待复核',
                   'reconstruction':{**original['reconstruction'],'fidelity':proof},
                   'manual_only':True}
    except (OSError,ValueError,KeyError,StopIteration):
        return


register_review_candidate()


def register_opening_candidate():
    """Expose only a source-verified candidate with corroborated source holes."""
    folder=ROOT/'outputs/三维模型-已证实孔洞修复试验-v4/29'
    file=folder/'result.json'
    if not file.is_file():return
    try:
        info=json.loads(file.read_text(encoding='utf8'))
        original=next(r for r in ROWS.values() if r['number']==29 and not r.get('manual_only'))
        mesh=folder/'candidate/mesh.glb';proof=info['fidelity']
        if (info['status']!='source_pass_pending_structure' or not info['source_hole_corroboration']['corroborated']
                or not proof['passed'] or proof['iou']<.90 or proof['filled_holes']!=0
                or proof['narrow_recall']<.80 or not mesh.is_file() or digest(mesh)!=info['mesh_sha256']
                or info['original_mesh_sha256']!=original['mesh_sha256']):return
        key=original['asset_id']+'-hole-repair'
        ROWS[key]={**original,'asset_id':key,'file':'液压减震器2（孔洞修复候选）',
                   'mesh':str(mesh),'mesh_sha256':info['mesh_sha256'],'status':'review',
                   'reason':'已核对源图孔洞；隐藏结构、外观和落稳仍待复核',
                   'reconstruction':{**original['reconstruction'],'fidelity':proof},'manual_only':True}
    except (OSError,ValueError,KeyError,StopIteration):return


register_opening_candidate()


def register_fragment_candidate(number,folder_name,name):
    folder=ROOT/'outputs'/folder_name;file=folder/'result.json'
    if not file.is_file():return
    try:
        info=json.loads(file.read_text(encoding='utf8'))
        original=next(r for r in ROWS.values() if r['number']==number and not r.get('manual_only'))
        mesh=folder/'mesh.glb';proof=info['fidelity']
        if (info['status']!='source_pass_pending_structure' or not proof['passed'] or proof['iou']<.90
                or proof['filled_holes']!=0 or proof['narrow_recall']<.80 or not mesh.is_file()
                or digest(mesh)!=info['mesh_sha256'] or info['source_sha256']!=original['mesh_sha256']):return
        key=original['asset_id']+'-fragment-repair'
        ROWS[key]={**original,'asset_id':key,'file':name,
                   'mesh':str(mesh),'mesh_sha256':info['mesh_sha256'],'status':'review',
                   'reason':'原图未见的小漂浮碎片已移除；隐藏结构和落稳仍待复核',
                   'reconstruction':{**original['reconstruction'],'fidelity':proof},'manual_only':True}
    except (OSError,ValueError,KeyError,StopIteration):return


register_fragment_candidate(8,'分离过滤器-漂浮碎片修复候选','分离过滤器（漂浮碎片修复候选）')
register_fragment_candidate(19,'气瓶组-背部浮片修复候选','气瓶组（背部浮片修复候选）')


def register_cylinder_candidate(number):
    folder=ROOT/'outputs/圆柱物品-形体修复试验'/f'{number:02d}';file=folder/'result.json'
    if not file.is_file():return
    try:
        info=json.loads(file.read_text(encoding='utf8'))
        original=next(r for r in ROWS.values() if r['number']==number and not r.get('manual_only'))
        mesh=folder/'mesh.glb';proof=info['fidelity']
        if (info['status']!='source_pass_pending_structure' or not proof['passed'] or proof['iou']<.90
                or proof['filled_holes']!=0 or proof['narrow_recall']<.80 or not mesh.is_file()
                or digest(mesh)!=info['mesh_sha256'] or info['original_mesh_sha256']!=original['mesh_sha256']):return
        key=original['asset_id']+'-cylinder-repair'
        ROWS[key]={**original,'asset_id':key,'file':original['file']+'（圆柱形体修复候选）',
                   'mesh':str(mesh),'mesh_sha256':info['mesh_sha256'],'status':'review',
                   'reason':'依据参考图近似调整截面；侧背面和落稳仍待复核',
                   'reconstruction':{**original['reconstruction'],'fidelity':proof},'manual_only':True}
    except (OSError,ValueError,KeyError,StopIteration):return


for _number in (15,16,36):register_cylinder_candidate(_number)


def register_rebuilt_candidate(folder=None, variant='v6'):
    """Register 01 separately; source-photo fidelity is diagnostic for this version."""
    if variant not in ('v6','v7'):return
    folder = folder or ROOT/f'outputs/01阀门-六视图重建-{variant}'
    file = folder/'result.json'
    if not file.is_file(): return
    try:
        info=json.loads(file.read_text(encoding='utf8'))
        original=next(r for r in ROWS.values() if r['number']==1 and not r.get('manual_only'))
        mesh=folder/'mesh.glb'
        reference=Path(info['reference'])
        if (info.get('draft') or info.get('status')!='review' or info.get('shape_basis')!='user_six_view_reference'
                or info.get('version')!=f'valve01-parametric-{variant}'
                or info.get('number')!=1 or info.get('asset_id')!=original['asset_id']
                or info.get('original_mesh_sha256')!=original['mesh_sha256']
                or info.get('inputs',{}).get('rust')!=original['rust_sha256']
                or not mesh.is_file() or digest(mesh)!=info.get('mesh_sha256')
                or not reference.is_file() or digest(reference)!=info['inputs']['reference']):return
        key=original['asset_id']+'-rebuild-'+variant
        ROWS[key]={**original, 'asset_id':key, 'file':f'长阀杆阀门（六视图重建 {variant}）',
                   'mesh':str(mesh), 'mesh_sha256':info['mesh_sha256'], 'status':'review',
                   'reason':'六视图未标定；侧背面与物理姿态仍待人工复核',
                   'reconstruction':{**original['reconstruction'], 'fidelity':info['source_fidelity_diagnostic']},
                   'manual_only':True, 'version':info['version'], 'shape_basis':info['shape_basis'],
                   'model_review':'pending', 'scene_validation':'pending'}
    except (OSError, ValueError, KeyError, StopIteration, TypeError):return


register_rebuilt_candidate()
register_rebuilt_candidate(variant='v7')


def register_rebuilt_candidate_02(folder=None, variant='v1'):
    """Register the DN200 candidate separately; keep it manual-only."""
    if variant != 'v1':
        return
    folder = folder or ROOT/f'outputs/02球阀-六视图重建-{variant}'
    file = folder/'result.json'
    if not file.is_file():
        return
    try:
        info=json.loads(file.read_text(encoding='utf8'))
        original=next(r for r in ROWS.values() if r['number']==2 and not r.get('manual_only'))
        mesh=folder/'mesh.glb'
        reference=Path(info['reference'])
        if (info.get('draft') or info.get('status')!='review'
                or info.get('shape_basis')!='user_six_view_reference'
                or info.get('version')!=f'valve02-parametric-{variant}'
                or info.get('number')!=2 or info.get('asset_id')!=original['asset_id']
                or info.get('original_mesh_sha256')!=original['mesh_sha256']
                or info.get('inputs',{}).get('rust')!=original['rust_sha256']
                or not mesh.is_file() or digest(mesh)!=info.get('mesh_sha256')
                or not reference.is_file() or digest(reference)!=info.get('reference_sha256')):return
        key=original['asset_id']+'-rebuild-'+variant
        ROWS[key]={**original, 'asset_id':key, 'file':f'DN200 球阀（六视图重建 {variant}）',
                   'mesh':str(mesh), 'mesh_sha256':info['mesh_sha256'], 'status':'review',
                   'reason':'六视图未标定；剖分法兰方向、支脚形式和物理姿态待人工复核',
                   'reconstruction':{**original['reconstruction'], 'fidelity':info['source_fidelity_diagnostic']},
                   'manual_only':True, 'version':info['version'], 'shape_basis':info['shape_basis'],
                   'model_review':'pending', 'scene_validation':'pending'}
    except (OSError, ValueError, KeyError, StopIteration, TypeError):return


register_rebuilt_candidate_02()


def register_rebuilt_candidate_03(folder=None, variant='v1'):
    """Register the P210 pillow-block candidate separately; keep it manual-only."""
    if variant != 'v1':
        return
    folder = folder or ROOT/'outputs/03轴承座-六视图重建-v3'
    file = folder/'result.json'
    if not file.is_file():
        return
    try:
        info=json.loads(file.read_text(encoding='utf8'))
        original=next(r for r in ROWS.values() if r['number']==3 and not r.get('manual_only'))
        mesh=folder/'mesh.glb'
        reference=Path(info['reference'])
        if (info.get('draft') or info.get('status')!='review'
                or info.get('shape_basis')!='user_six_view_reference'
                or info.get('version')!=f'bearing03-parametric-{variant}'
                or info.get('number')!=3 or info.get('asset_id')!=original['asset_id']
                or info.get('original_mesh_sha256')!=original['mesh_sha256']
                or info.get('inputs',{}).get('rust')!=original['rust_sha256']
                or not mesh.is_file() or digest(mesh)!=info.get('mesh_sha256')
                or not reference.is_file() or digest(reference)!=info.get('reference_sha256')):return
        key=original['asset_id']+'-rebuild-'+variant
        ROWS[key]={**original, 'asset_id':key, 'file':f'P210 轴承座（六视图重建 {variant}）',
                   'mesh':str(mesh), 'mesh_sha256':info['mesh_sha256'], 'status':'review',
                   'reason':'六视图未标定且拼图各向异性；轴承内圈偏置、紧定螺钉角度和背面细节待人工复核',
                   'reconstruction':{**original['reconstruction'], 'fidelity':info['source_fidelity_diagnostic']},
                   'manual_only':True, 'version':info['version'], 'shape_basis':info['shape_basis'],
                   'model_review':'pending', 'scene_validation':'pending'}
    except (OSError, ValueError, KeyError, StopIteration, TypeError):return


register_rebuilt_candidate_03()


def register_rebuilt_candidate_04(folder=None):
    """Register the SN513 split plummer-block candidate separately; keep it manual-only."""
    folder = folder or ROOT/'outputs/04轴承座-六视图重建-v2'
    file = folder/'result.json'
    if not file.is_file():
        return
    try:
        info=json.loads(file.read_text(encoding='utf8'))
        original=next(r for r in ROWS.values() if r['number']==4 and not r.get('manual_only'))
        mesh=folder/'mesh.glb'
        reference=Path(info['reference'])
        if (info.get('draft') or info.get('status')!='review'
                or info.get('shape_basis')!='user_six_view_reference'
                or info.get('version')!='bearing04-parametric-v1'
                or info.get('number')!=4 or info.get('asset_id')!=original['asset_id']
                or info.get('original_mesh_sha256')!=original['mesh_sha256']
                or info.get('inputs',{}).get('rust')!=original['rust_sha256']
                or not mesh.is_file() or digest(mesh)!=info.get('mesh_sha256')
                or not reference.is_file() or digest(reference)!=info.get('reference_sha256')):return
        key=original['asset_id']+'-rebuild-v2'
        ROWS[key]={**original, 'asset_id':key, 'file':'SN513 轴承座（六视图重建 v1）',
                   'mesh':str(mesh), 'mesh_sha256':info['mesh_sha256'], 'status':'review',
                   'reason':'六视图未标定且侧面板几何矛盾；剖分面、油嘴、轴承内部与背面细节待人工复核',
                   'reconstruction':{**original['reconstruction'], 'fidelity':info['source_fidelity_diagnostic']},
                   'manual_only':True, 'version':info['version'], 'shape_basis':info['shape_basis'],
                   'model_review':'pending', 'scene_validation':'pending'}
    except (OSError, ValueError, KeyError, StopIteration, TypeError):return


register_rebuilt_candidate_04()


def register_rebuilt_candidate_05(folder=None):
    """Register the vertical sanitary-tank candidate separately; keep it manual-only."""
    folder = folder or ROOT/'outputs/05储罐-六视图重建-v1'
    file = folder/'result.json'
    if not file.is_file():
        return
    try:
        info=json.loads(file.read_text(encoding='utf8'))
        original=next(r for r in ROWS.values() if r['number']==5 and not r.get('manual_only'))
        mesh=folder/'mesh.glb'
        reference=Path(info['reference'])
        if (info.get('draft') or info.get('status')!='review'
                or info.get('shape_basis')!='user_six_view_reference'
                or info.get('version')!='tank05-parametric-v1'
                or info.get('number')!=5 or info.get('asset_id')!=original['asset_id']
                or info.get('original_mesh_sha256')!=original['mesh_sha256']
                or info.get('inputs',{}).get('rust')!=original['rust_sha256']
                or not mesh.is_file() or digest(mesh)!=info.get('mesh_sha256')
                or not reference.is_file() or digest(reference)!=info.get('reference_sha256')):return
        key=original['asset_id']+'-rebuild-v1'
        ROWS[key]={**original, 'asset_id':key, 'file':'立式不锈钢储罐（六视图重建 v1）',
                   'mesh':str(mesh), 'mesh_sha256':info['mesh_sha256'], 'status':'review',
                   'reason':'六视图未标定且侧管口镜像不一致；下部前管口、侧小管口、支脚长槽与孔深为推断；物理姿态待复核',
                   'reconstruction':{**original['reconstruction'], 'fidelity':info['source_fidelity_diagnostic']},
                   'manual_only':True, 'version':info['version'], 'shape_basis':info['shape_basis'],
                   'model_review':'pending', 'scene_validation':'pending'}
    except (OSError, ValueError, KeyError, StopIteration, TypeError):return


register_rebuilt_candidate_05()


def register_rebuilt_candidate_06(folder=None):
    """Register the horizontal stainless pressure-tank candidate separately; keep it manual-only."""
    folder = folder or ROOT/'outputs/06压力罐-六视图重建-v2'
    file = folder/'result.json'
    if not file.is_file():
        return
    try:
        info=json.loads(file.read_text(encoding='utf8'))
        original=next(r for r in ROWS.values() if r['number']==6 and not r.get('manual_only'))
        mesh=folder/'mesh.glb'
        reference=Path(info['reference'])
        if (info.get('draft') or info.get('status')!='review'
                or info.get('shape_basis')!='user_six_view_reference'
                or info.get('version')!='tank06-parametric-v1'
                or info.get('number')!=6 or info.get('asset_id')!=original['asset_id']
                or info.get('original_mesh_sha256')!=original['mesh_sha256']
                or info.get('inputs',{}).get('rust')!=original['rust_sha256']
                or not mesh.is_file() or digest(mesh)!=info.get('mesh_sha256')
                or not reference.is_file() or digest(reference)!=info.get('reference_sha256')):return
        key=original['asset_id']+'-rebuild-v1'
        ROWS[key]={**original, 'asset_id':key, 'file':'卧式不锈钢压力罐（六视图重建 v1）',
                   'mesh':str(mesh), 'mesh_sha256':info['mesh_sha256'], 'status':'review',
                   'reason':'六视图未标定且左视侧管口、底视排污位置和顶视支撑带互相矛盾；下部右端法兰口、支撑加强筋和隐藏面为推断；物理姿态待复核',
                   'reconstruction':{**original['reconstruction'], 'fidelity':info['source_fidelity_diagnostic']},
                   'manual_only':True, 'version':info['version'], 'shape_basis':info['shape_basis'],
                   'model_review':'pending', 'scene_validation':'pending'}
    except (OSError, ValueError, KeyError, StopIteration, TypeError):return


register_rebuilt_candidate_06()


def register_rebuilt_candidate_07(folder=None):
    """Register the vertical mixing-tank candidate separately; keep it manual-only."""
    folder = folder or ROOT/'outputs/07储罐-六视图重建-v2'
    file = folder/'result.json'
    if not file.is_file():
        return
    try:
        info=json.loads(file.read_text(encoding='utf8'))
        original=next(r for r in ROWS.values() if r['number']==7 and not r.get('manual_only'))
        mesh=folder/'mesh.glb'
        reference=Path(info['reference'])
        if (info.get('draft') or info.get('status')!='review'
                or info.get('shape_basis')!='source_photo_with_uncalibrated_category_reference'
                or info.get('reference_role')!='category_reference'
                or info.get('version')!='tank07-parametric-v1'
                or info.get('number')!=7 or info.get('asset_id')!=original['asset_id']
                or info.get('original_mesh_sha256')!=original['mesh_sha256']
                or info.get('inputs',{}).get('rust')!=original['rust_sha256']
                or not mesh.is_file() or digest(mesh)!=info.get('mesh_sha256')
                or not reference.is_file() or digest(reference)!=info.get('reference_sha256')):return
        key=original['asset_id']+'-rebuild-v1'
        ROWS[key]={**original, 'asset_id':key, 'file':'立式不锈钢搅拌罐（六视图重建 v2）',
                   'mesh':str(mesh), 'mesh_sha256':info['mesh_sha256'], 'status':'review',
                   'reason':'六视图为相近类别参考且顶部搅拌件不一致；三腿、支架背面、视镜、阀门与隐藏面为推断；物理姿态待复核',
                   'reconstruction':{**original['reconstruction'], 'fidelity':info['source_fidelity_diagnostic']},
                   'manual_only':True, 'version':info['version'], 'shape_basis':info['shape_basis'],
                   'model_review':'pending', 'scene_validation':'pending'}
    except (OSError, ValueError, KeyError, StopIteration, TypeError):return


register_rebuilt_candidate_07()


def register_rebuilt_candidate_08(folder=None):
    """Register the separator-filter candidate separately; keep it manual-only."""
    folder = folder or ROOT/'outputs/08分离过滤器-六视图重建-v3'
    file = folder/'result.json'
    if not file.is_file():
        return
    try:
        info=json.loads(file.read_text(encoding='utf8'))
        original=next(r for r in ROWS.values() if r['number']==8 and not r.get('manual_only'))
        mesh=folder/'mesh.glb'
        reference=Path(info['reference'])
        if (info.get('draft') or info.get('status')!='review'
                or info.get('shape_basis')!='source_photo_with_uncalibrated_visual_match_reference'
                or info.get('reference_role')!='visual_match_unverified'
                or info.get('version')!='separator08-parametric-v1'
                or info.get('number')!=8 or info.get('asset_id')!=original['asset_id']
                or info.get('original_mesh_sha256')!=original['mesh_sha256']
                or info.get('inputs',{}).get('rust')!=original['rust_sha256']
                or not mesh.is_file() or digest(mesh)!=info.get('mesh_sha256')
                or not reference.is_file() or digest(reference)!=info.get('reference_sha256')):return
        key=original['asset_id']+'-rebuild-v1'
        ROWS[key]={**original, 'asset_id':key, 'file':'分离过滤器（六视图重建 v3）',
                   'mesh':str(mesh), 'mesh_sha256':info['mesh_sha256'], 'status':'review',
                   'reason':'六视图未标定；背面法兰/视窗、法兰面、孔深、排水器内部与隐藏面为推断；无可见支撑，落稳待复核',
                   'reconstruction':{**original['reconstruction'], 'fidelity':info['source_fidelity_diagnostic']},
                   'manual_only':True, 'version':info['version'], 'shape_basis':info['shape_basis'],
                   'model_review':'pending', 'scene_validation':'pending'}
    except (OSError, ValueError, KeyError, StopIteration, TypeError):return


register_rebuilt_candidate_08()


def register_rebuilt_candidate_09(folder=None):
    """Register the separator-filter 1 candidate separately; keep it manual-only."""
    folder = folder or ROOT/'outputs/09分离过滤器1-六视图重建-v4'
    file = folder/'result.json'
    if not file.is_file():
        return
    try:
        info=json.loads(file.read_text(encoding='utf8'))
        original=next(r for r in ROWS.values() if r['number']==9 and not r.get('manual_only'))
        mesh=folder/'mesh.glb'
        reference=Path(info['reference'])
        if (info.get('draft') or info.get('status')!='review'
                or info.get('shape_basis')!='source_photo_with_uncalibrated_visual_match_reference'
                or info.get('reference_role')!='visual_match_unverified'
                or info.get('version')!='separator09-parametric-v1'
                or info.get('number')!=9 or info.get('asset_id')!=original['asset_id']
                or info.get('original_mesh_sha256')!=original['mesh_sha256']
                or info.get('inputs',{}).get('rust')!=original['rust_sha256']
                or not mesh.is_file() or digest(mesh)!=info.get('mesh_sha256')
                or not reference.is_file() or digest(reference)!=info.get('reference_sha256')):return
        key=original['asset_id']+'-rebuild-v1'
        ROWS[key]={**original, 'asset_id':key, 'file':'分离过滤器 1（六视图重建 v1）',
                   'mesh':str(mesh), 'mesh_sha256':info['mesh_sha256'], 'status':'review',
                   'reason':'六视图未标定；照片侧法兰 3/4 视角与对称四腿投影不一致，候选按对称正视解释；背面、四腿/支脚、排水器内部、吊架、铭牌文字与隐藏面为推断；落稳待复核',
                   'reconstruction':{**original['reconstruction'], 'fidelity':info['source_fidelity_diagnostic']},
                   'manual_only':True, 'version':info['version'], 'shape_basis':info['shape_basis'],
                   'model_review':'pending', 'scene_validation':'pending'}
    except (OSError, ValueError, KeyError, StopIteration, TypeError):return


register_rebuilt_candidate_09()


def register_rebuilt_candidate_10(folder=None):
    """Register the 10 t bottle-jack candidate separately; keep it manual-only."""
    folder = folder or ROOT/'outputs/10千斤顶-六视图重建-v2'
    file = folder/'result.json'
    if not file.is_file():
        return
    try:
        info=json.loads(file.read_text(encoding='utf8'))
        original=next(r for r in ROWS.values() if r['number']==10 and not r.get('manual_only'))
        mesh=folder/'mesh.glb'
        reference=Path(info['reference'])
        if (info.get('draft') or info.get('status')!='review'
                or info.get('shape_basis')!='source_photo_with_uncalibrated_visual_match_reference'
                or info.get('reference_role')!='visual_match_unverified'
                or info.get('version')!='jack10-parametric-v1'
                or info.get('number')!=10 or info.get('asset_id')!=original['asset_id']
                or info.get('original_mesh_sha256')!=original['mesh_sha256']
                or info.get('inputs',{}).get('rust')!=original['rust_sha256']
                or not mesh.is_file() or digest(mesh)!=info.get('mesh_sha256')
                or not reference.is_file() or digest(reference)!=info.get('reference_sha256')):return
        key=original['asset_id']+'-rebuild-v1'
        ROWS[key]={**original, 'asset_id':key, 'file':'10 吨立式油压千斤顶（分部件重建 v2）',
                   'mesh':str(mesh), 'mesh_sha256':info['mesh_sha256'], 'status':'review',
                   'reason':'六视图未标定；原照为 3/4 产品图且手柄筒悬空，候选按推断的倾斜站立姿态表示；底座凸台数量/背面、泵内结构、握把插座深处、标签重绘与隐藏面待人工复核；落稳待验证',
                   'reconstruction':{**original['reconstruction'], 'fidelity':info['source_fidelity_diagnostic']},
                   'manual_only':True, 'version':info['version'], 'shape_basis':info['shape_basis'],
                   'model_review':'pending', 'scene_validation':'pending'}
    except (OSError, ValueError, KeyError, StopIteration, TypeError):return


register_rebuilt_candidate_10()


def register_rebuilt_candidate_11(folder=None):
    """Register the QL5 screw-jack candidate separately; keep it manual-only."""
    folder = folder or ROOT/'outputs/11千斤顶1-六视图重建-v3'
    file = folder/'result.json'
    if not file.is_file():
        return
    try:
        info=json.loads(file.read_text(encoding='utf8'))
        original=next(r for r in ROWS.values() if r['number']==11 and not r.get('manual_only'))
        mesh=folder/'mesh.glb'
        reference=Path(info['reference'])
        if (info.get('draft') or info.get('status')!='review'
                or info.get('shape_basis')!='source_photo_with_uncalibrated_visual_match_reference'
                or info.get('reference_role')!='visual_match_unverified'
                or info.get('version')!='jack11-parametric-v1'
                or info.get('number')!=11 or info.get('asset_id')!=original['asset_id']
                or info.get('original_mesh_sha256')!=original['mesh_sha256']
                or info.get('inputs',{}).get('rust')!=original['rust_sha256']
                or not mesh.is_file() or digest(mesh)!=info.get('mesh_sha256')
                or not reference.is_file() or digest(reference)!=info.get('reference_sha256')):return
        key=original['asset_id']+'-rebuild-v1'
        ROWS[key]={**original, 'asset_id':key, 'file':'QL5 螺旋千斤顶（分部件重建 v1）',
                   'mesh':str(mesh), 'mesh_sha256':info['mesh_sha256'], 'status':'review',
                   'reason':'六视图未标定且面板不一致（杠杆圆管/矩形杆、提环侧向）；齿轮座内部、拉钮、提环背面、5T 浮雕、标签重绘与隐藏面为推断；落稳与现场尺度未验证',
                   'reconstruction':{**original['reconstruction'], 'fidelity':info['source_fidelity_diagnostic']},
                   'manual_only':True, 'version':info['version'], 'shape_basis':info['shape_basis'],
                   'model_review':'pending', 'scene_validation':'pending'}
    except (OSError, ValueError, KeyError, StopIteration, TypeError):return


register_rebuilt_candidate_11()


def register_rebuilt_candidate_12(folder=None):
    """Register the 2 t trolley-jack candidate separately; keep it manual-only."""
    folder = folder or ROOT/'outputs/12千斤顶2-六视图重建-v3'
    file = folder/'result.json'
    if not file.is_file():
        return
    try:
        info=json.loads(file.read_text(encoding='utf8'))
        original=next(r for r in ROWS.values() if r['number']==12 and not r.get('manual_only'))
        mesh=folder/'mesh.glb'
        reference=Path(info['reference'])
        if (info.get('draft') or info.get('status')!='review'
                or info.get('shape_basis')!='source_photo_with_uncalibrated_visual_match_reference'
                or info.get('reference_role')!='visual_match_unverified'
                or info.get('version')!='jack12-parametric-v1'
                or info.get('number')!=12 or info.get('asset_id')!=original['asset_id']
                or info.get('original_mesh_sha256')!=original['mesh_sha256']
                or info.get('inputs',{}).get('rust')!=original['rust_sha256']
                or not mesh.is_file() or digest(mesh)!=info.get('mesh_sha256')
                or not reference.is_file() or digest(reference)!=info.get('reference_sha256')):return
        key=original['asset_id']+'-rebuild-v1'
        ROWS[key]={**original, 'asset_id':key, 'file':'2 吨卧式液压千斤顶（分部件重建 v1）',
                   'mesh':str(mesh), 'mesh_sha256':info['mesh_sha256'], 'status':'review',
                   'reason':'六视图未标定且正视/背视与左右视对手柄姿态不一致；泵内结构、插座锁定、焊缝、轮胎花纹、标签重绘与隐藏面为推断；落稳与现场尺度未验证',
                   'reconstruction':{**original['reconstruction'], 'fidelity':info['source_fidelity_diagnostic']},
                   'manual_only':True, 'version':info['version'], 'shape_basis':info['shape_basis'],
                   'model_review':'pending', 'scene_validation':'pending'}
    except (OSError, ValueError, KeyError, StopIteration, TypeError):return


register_rebuilt_candidate_12()


def register_rebuilt_candidate_13(folder=None):
    """Register the two-can sealant-canister candidate separately; keep it manual-only."""
    folder = folder or ROOT/'outputs/13密封胶罐-六视图重建-v2'
    file = folder/'result.json'
    if not file.is_file():
        return
    try:
        info=json.loads(file.read_text(encoding='utf8'))
        original=next(r for r in ROWS.values() if r['number']==13 and not r.get('manual_only'))
        mesh=folder/'mesh.glb'
        reference=Path(info['reference'])
        if (info.get('draft') or info.get('status')!='review'
                or info.get('shape_basis')!='source_photo_with_uncalibrated_category_reference'
                or info.get('reference_role')!='category_reference_unverified'
                or info.get('version')!='sealant13-parametric-v1'
                or info.get('number')!=13 or info.get('asset_id')!=original['asset_id']
                or info.get('original_mesh_sha256')!=original['mesh_sha256']
                or info.get('inputs',{}).get('rust')!=original['rust_sha256']
                or not mesh.is_file() or digest(mesh)!=info.get('mesh_sha256')
                or not reference.is_file() or digest(reference)!=info.get('reference_sha256')
                or not info.get('manual_only')):return
        key=original['asset_id']+'-rebuild-v1'
        ROWS[key]={**original, 'asset_id':key, 'file':'13 密封胶罐（两只白色塑料罐，分部件重建 v1）',
                   'mesh':str(mesh), 'mesh_sha256':info['mesh_sha256'], 'status':'review',
                   'reason':'六视图为单只干净罐的未标定类别参考，原照为两只使用过的罐；罐间距/朝向、盖内配合、壁厚、卡扣机构、底脚功能、回收标志细节与隐藏面为推断；全塑料，金属反射不适用；落稳与现场尺度未验证',
                   'reconstruction':{**original['reconstruction'], 'fidelity':info['source_fidelity_diagnostic']},
                   'manual_only':True, 'version':info['version'], 'shape_basis':info['shape_basis'],
                   'model_review':'pending', 'scene_validation':'pending'}
    except (OSError, ValueError, KeyError, StopIteration, TypeError):return


register_rebuilt_candidate_13()
SESSIONS = {}
ENGINE = None
ENGINE_LOCK = asyncio.Lock()
CHAT_ACTIVE = set()
RENDER_ALGORITHM = 'scene-light-canonical-axis-v4'
ANALYSIS_ALGORITHM = 'native-thin-occlusion-v1'


def err(message, status=400):
    return JSONResponse({'error': str(message)}, status_code=status)


def session(request):
    sid = request.path_params['sid']
    if not re.fullmatch(r'[0-9a-f]{32}', sid):
        raise ValueError('会话不存在，请重新选择背景')
    if sid not in SESSIONS:
        path = DATA / sid / 'session.json'
        if not path.is_file():
            raise ValueError('会话不存在，请重新选择背景')
        restored = json.loads(path.read_text(encoding='utf8'))
        if restored.get('id') != sid or Path(restored.get('directory','')).resolve() != path.parent.resolve():
            raise ValueError('会话记录无效')
        restored['lock'] = asyncio.Lock()
        SESSIONS[sid] = restored
    return SESSIONS[sid]


def save(s):
    serial = {k: v for k, v in s.items() if k not in ('lock',)}
    tmp = Path(s['directory']) / 'session.tmp'
    tmp.write_text(json.dumps(serial, ensure_ascii=False, indent=2), encoding='utf8')
    tmp.replace(Path(s['directory']) / 'session.json')


def snapshot(s):
    return {k: v for k, v in s.items() if k not in ('lock', 'background', 'directory', 'history')}


async def index(request):
    return FileResponse(HERE / 'index.html')


async def assets(request):
    return JSONResponse([{'id': r['asset_id'], 'number': r['number'], 'name': r['file'],
                          'status': r.get('status'), 'reason': r.get('reason'),
                          'manual_only': bool(r.get('manual_only')),
                          'available': Path(r.get('mesh', '')).is_file(),
                          'version':r.get('version', 'library-v1'),
                          'shape_basis':r.get('shape_basis', 'single_image_reconstruction'),
                          'model_review':r.get('model_review', 'pending'),
                          'scene_validation':r.get('scene_validation', 'pending'),
                          'passed': bool(r.get('reconstruction', {}).get('fidelity', {}).get('passed'))}
                         for r in sorted(ROWS.values(),key=lambda row:(row['number'],bool(row.get('manual_only'))))])


async def model(request):
    row = ROWS.get(request.path_params['asset'])
    if not row or not Path(row.get('mesh', '')).is_file():
        return err('模型不存在', 404)
    path = Path(row['mesh'])
    if digest(path) != row.get('mesh_sha256'):
        return err('模型哈希与库记录不一致', 409)
    return FileResponse(path, media_type='model/gltf-binary')


async def model_review_page(request):
    return FileResponse(HERE/'model-review.html')


async def model_review_page_02(request):
    return FileResponse(HERE/'model-review-02.html')


async def model_review_page_03(request):
    return FileResponse(HERE/'model-review-03.html')


async def model_review_page_04(request):
    return FileResponse(HERE/'model-review-04.html')


async def model_review_page_05(request):
    return FileResponse(HERE/'model-review-05.html')


async def model_review_page_06(request):
    return FileResponse(HERE/'model-review-06.html')


async def model_review_page_07(request):
    return FileResponse(HERE/'model-review-07.html')


async def model_review_page_08(request):
    return FileResponse(HERE/'model-review-08.html')


async def model_review_page_09(request):
    return FileResponse(HERE/'model-review-09.html')


async def model_review_page_10(request):
    return FileResponse(HERE/'model-review-10.html')


async def model_review_page_11(request):
    return FileResponse(HERE/'model-review-11.html')


async def model_review_page_12(request):
    return FileResponse(HERE/'model-review-12.html')


async def model_review_page_13(request):
    return FileResponse(HERE/'model-review-13.html')


REVIEW_FOLDERS = {
    1: 'outputs/01阀门-六视图重建-v7',
    2: 'outputs/02球阀-六视图重建-v1',
    3: 'outputs/03轴承座-六视图重建-v3',
    4: 'outputs/04轴承座-六视图重建-v2',
    5: 'outputs/05储罐-六视图重建-v1',
    6: 'outputs/06压力罐-六视图重建-v2',
    7: 'outputs/07储罐-六视图重建-v2',
    8: 'outputs/08分离过滤器-六视图重建-v3',
    9: 'outputs/09分离过滤器1-六视图重建-v4',
    10: 'outputs/10千斤顶-六视图重建-v2',
    11: 'outputs/11千斤顶1-六视图重建-v3',
    12: 'outputs/12千斤顶2-六视图重建-v3',
    13: 'outputs/13密封胶罐-六视图重建-v2',
}
REVIEW_FILES = {
    1: {'board':'六视图-新旧对照.jpg', 'turntable':'转台-24方向.jpg', 'record':'result.json',
        'browser':'浏览器核验.json', 'blend':'model.blend', 'metal':'金属材质-同光照对照.jpg',
        'front':'renders/new-front.png', 'wheel':'renders/detail-wheel.png',
        'stem':'renders/detail-stem.png', 'side':'renders/detail-side.png',
        'surface':'renders/detail-surface.png'},
    2: {'board':'六视图-新旧对照.jpg', 'turntable':'转台-24方向.jpg', 'record':'result.json',
        'browser':'浏览器核验.json', 'blend':'model.blend', 'metal':'材质-同光照对照.jpg',
        'light':'材质-变光检查.jpg', 'local':'局部检查.jpg', 'isolated':'结构开口-隔离检查.jpg',
        'front':'renders/new-front.png', 'wheel':'renders/detail-wheel.png',
        'flange':'renders/detail-flange.png', 'body':'renders/detail-body.png',
        'foot':'renders/detail-foot.png', 'bottom':'renders/detail-bottom.png',
        'surface':'renders/detail-surface.png'},
    3: {'board':'六视图-新旧对照.jpg', 'turntable':'转台-24方向.jpg', 'record':'result.json',
        'browser':'浏览器核验.json', 'blend':'model.blend', 'metal':'材质-同光照对照.jpg',
        'light':'材质-变光检查.jpg', 'metalreflect':'金属反射-变光对照.jpg',
        'local':'局部检查.jpg', 'isolated':'结构开口-隔离检查.jpg',
        'front':'renders/new-front.png', 'bearing':'renders/detail-bearing.png',
        'set-screw':'renders/detail-set-screw.png', 'zerk':'renders/detail-zerk.png',
        'label-asahi':'renders/detail-label-asahi.png',
        'label-p210':'renders/detail-label-p210.png',
        'slot':'renders/detail-slot.png', 'bottom':'renders/detail-bottom.png',
        'surface':'renders/detail-surface.png'},
    4: {'board':'六视图-新旧对照.jpg', 'turntable':'转台-24方向.jpg', 'record':'result.json',
        'browser':'浏览器核验.json', 'blend':'model.blend', 'metal':'材质-同光照对照.jpg',
        'light':'材质-变光检查.jpg', 'metalreflect':'金属反射-变光对照.jpg',
        'local':'局部检查.jpg', 'isolated':'结构开口-隔离检查.jpg',
        'front':'renders/new-front.png', 'bearing':'renders/detail-bearing.png',
        'bolt':'renders/detail-bolt.png', 'zerk':'renders/detail-zerk.png',
        'label':'renders/detail-label.png', 'split':'renders/detail-split.png',
        'slot':'renders/detail-slot.png', 'bottom':'renders/detail-bottom.png',
        'surface':'renders/detail-surface.png'},
    5: {'board':'六视图-新旧对照.jpg', 'turntable':'转台-24方向.jpg', 'record':'result.json',
        'browser':'浏览器核验.json', 'blend':'model.blend', 'metal':'材质-同光照对照.jpg',
        'light':'材质-变光检查.jpg', 'metalreflect':'金属反射-变光对照.jpg',
        'local':'局部检查.jpg', 'isolated':'结构开口-隔离检查.jpg',
        'front':'renders/new-front.png', 'manway':'renders/detail-manway.png',
        'clamps':'renders/detail-clamps.png', 'topnozzles':'renders/detail-top-nozzles.png',
        'frontport':'renders/detail-front-port.png', 'lowerport':'renders/detail-lower-port.png',
        'legs':'renders/detail-legs.png', 'foot':'renders/detail-foot.png',
        'drain':'renders/detail-drain.png', 'seam':'renders/detail-seam.png',
        'surface':'renders/detail-surface.png'},
    6: {'board':'六视图-新旧对照.jpg', 'turntable':'转台-24方向.jpg', 'record':'result.json',
        'browser':'浏览器核验.json', 'blend':'model.blend', 'metal':'材质-同光照对照.jpg',
        'light':'材质-变光检查.jpg', 'metalreflect':'金属反射-变光对照.jpg',
        'local':'局部检查.jpg', 'isolated':'结构开口-隔离检查.jpg',
        'front':'renders/new-front.png', 'manway':'renders/detail-manway.png',
        'clamps':'renders/detail-clamps.png', 'topnozzles':'renders/detail-top-nozzles.png',
        'rightports':'renders/detail-right-ports.png', 'righthead':'renders/detail-right-head.png',
        'support':'renders/detail-support.png', 'foot':'renders/detail-foot.png',
        'drain':'renders/detail-drain.png', 'seam':'renders/detail-seam.png',
        'surface':'renders/detail-surface.png'},
    7: {'board':'六视图-新旧对照.jpg', 'turntable':'转台-24方向.jpg', 'record':'result.json',
        'browser':'浏览器核验.json', 'blend':'model.blend', 'metal':'材质-同光照对照.jpg',
        'light':'材质-变光检查.jpg', 'metalreflect':'金属反射-变光对照.jpg',
        'local':'局部检查.jpg', 'isolated':'结构开口-隔离检查.jpg',
        'front':'renders/new-front.png', 'topport':'renders/detail-top-port.png',
        'liftlug':'renders/detail-lift-lug.png', 'motor':'renders/detail-motor.png',
        'skid':'renders/detail-skid.png', 'pipe':'renders/detail-side-pipe.png',
        'valve':'renders/detail-pipe-valve.png',
        'sightglass':'renders/detail-sight-glass.png',
        'flange':'renders/detail-cone-flange.png', 'legs':'renders/detail-legs.png',
        'foot':'renders/detail-foot.png', 'seam':'renders/detail-seam.png',
        'surface':'renders/detail-surface.png'},
    8: {'board':'六视图-新旧对照.jpg', 'turntable':'转台-24方向.jpg', 'record':'result.json',
        'browser':'浏览器核验.json', 'blend':'model.blend', 'metal':'材质-同光照对照.jpg',
        'light':'材质-变光检查.jpg', 'metalreflect':'金属反射-变光对照.jpg',
        'local':'局部检查.jpg', 'isolated':'结构开口-隔离检查.jpg',
        'front':'renders/new-front.png', 'lifteye':'renders/detail-lift-eye.png',
        'lifteyeside':'renders/detail-lift-eye-side.png',
        'flangeleft':'renders/detail-flange-left.png',
        'flangeright':'renders/detail-flange-right.png',
        'nozzleweld':'renders/detail-nozzle-weld.png',
        'topseam':'renders/detail-top-seam.png',
        'bottomseam':'renders/detail-bottom-seam.png',
        'drain':'renders/detail-drain.png',
        'drainwindow':'renders/detail-drain-window.png',
        'drainknurl':'renders/detail-drain-knurl.png',
        'drainbottom':'renders/detail-drain-bottom.png',
        'surface':'renders/detail-surface.png'},
    9: {'board':'六视图-新旧对照.jpg', 'turntable':'转台-24方向.jpg', 'record':'result.json',
        'browser':'浏览器核验.json', 'blend':'model.blend', 'metal':'材质-同光照对照.jpg',
        'light':'材质-变光检查.jpg', 'metalreflect':'金属反射-变光对照.jpg',
        'local':'局部检查.jpg', 'isolated':'结构开口-隔离检查.jpg',
        'front':'renders/new-front.png', 'topflange':'renders/detail-top-flange.png',
        'topface':'renders/detail-top-face.png', 'sideflange':'renders/detail-side-flange.png',
        'sideport':'renders/detail-side-port.png', 'vent':'renders/detail-vent.png',
        'nameplate':'renders/detail-nameplate.png',
        'upperseams':'renders/detail-upper-seams.png',
        'lowerseams':'renders/detail-lower-seams.png',
        'leggusset':'renders/detail-leg-gusset.png', 'foot':'renders/detail-foot.png',
        'drain':'renders/detail-drain.png', 'drainmount':'renders/detail-drain-mount.png',
        'draincage':'renders/detail-drain-cage.png',
        'drainbottom':'renders/detail-drain-bottom.png',
        'surface':'renders/detail-surface.png'},
    10: {'board':'六视图-新旧对照.jpg', 'turntable':'转台-24方向.jpg', 'record':'result.json',
        'browser':'浏览器核验.json', 'blend':'model.blend', 'metal':'材质-同光照对照.jpg',
        'light':'材质-变光检查.jpg', 'metalreflect':'金属反射-变光对照.jpg',
        'local':'局部检查.jpg', 'isolated':'结构开口-隔离检查.jpg',
        'front':'renders/new-front.png', 'saddle':'renders/detail-saddle.png',
        'ramhex':'renders/detail-ram-hex.png', 'shoulder':'renders/detail-shoulder.png',
        'label':'renders/detail-label.png', 'loop':'renders/detail-loop.png',
        'pumphead':'renders/detail-pump-head.png',
        'pumpsocket':'renders/detail-pump-socket.png',
        'pumplink':'renders/detail-pump-link.png',
        'basefront':'renders/detail-base-front.png',
        'basebottom':'renders/detail-base-bottom.png',
        'releasevalve':'renders/detail-release-valve.png',
        'handlebar':'renders/detail-handle-bar.png',
        'surface':'renders/detail-surface.png'},
    11: {'board':'六视图-新旧对照.jpg', 'turntable':'转台-24方向.jpg', 'record':'result.json',
        'browser':'浏览器核验.json', 'blend':'model.blend', 'metal':'材质-同光照对照.jpg',
        'light':'材质-变光检查.jpg', 'metalreflect':'金属反射-变光对照.jpg',
        'local':'局部检查.jpg', 'isolated':'结构开口-隔离检查.jpg',
        'front':'renders/new-front.png', 'levertip':'renders/detail-lever-tip.png',
        'leversocket':'renders/detail-lever-socket.png',
        'knob':'renders/detail-knob.png',
        'mechanismside':'renders/detail-mechanism-side.png',
        'loop':'renders/detail-loop.png',
        'loopmounts':'renders/detail-loop-mounts.png',
        'collar':'renders/detail-collar.png', 'screw':'renders/detail-screw.png',
        'cap':'renders/detail-cap.png', 'label':'renders/detail-label.png',
        'emboss':'renders/detail-emboss.png',
        'basefront':'renders/detail-base-front.png',
        'basebottom':'renders/detail-base-bottom.png',
        'surface':'renders/detail-surface.png'},
    12: {'board':'六视图-新旧对照.jpg', 'turntable':'转台-24方向.jpg', 'record':'result.json',
        'browser':'浏览器核验.json', 'blend':'model.blend', 'metal':'材质-同光照对照.jpg',
        'light':'材质-变光检查.jpg', 'metalreflect':'金属反射-变光对照.jpg',
        'local':'局部检查.jpg', 'isolated':'结构开口-隔离检查.jpg',
        'front':'renders/new-front.png', 'saddle':'renders/detail-saddle.png',
        'saddletop':'renders/detail-saddle-top.png',
        'armlabel':'renders/detail-arm-label.png',
        'loop':'renders/detail-carry-loop.png',
        'loopmounts':'renders/detail-loop-mounts.png',
        'pump':'renders/detail-pump-cover.png',
        'pumplabel':'renders/detail-pump-label.png',
        'socket':'renders/detail-handle-socket.png',
        'grip':'renders/detail-handle-grip.png',
        'frontwheel':'renders/detail-front-wheel.png',
        'rearcaster':'renders/detail-rear-caster.png',
        'frameside':'renders/detail-frame-side.png',
        'armpivot':'renders/detail-arm-pivot.png',
        'armlink':'renders/detail-arm-link.png',
        'basebottom':'renders/detail-base-bottom.png',
        'rearframe':'renders/detail-rear-frame.png',
        'surface':'renders/detail-surface.png'},
    13: {'board':'六视图-新旧对照.jpg', 'turntable':'转台-24方向.jpg', 'record':'result.json',
         'browser':'浏览器核验.json', 'blend':'model.blend',
         'metal':'材质-同光照对照.jpg', 'light':'材质-变光检查.jpg',
         'surfacereflect':'表面反射-变光对照.jpg',
         'local':'局部检查.jpg', 'isolated':'结构开口-隔离检查.jpg',
         'front':'renders/new-front.png', 'singlefront':'renders/single-front.png',
         'lidtop':'renders/detail-lid-top.png', 'lidrim':'renders/detail-lid-rim.png',
         'seamlatch':'renders/detail-seam-latch.png',
         'frontlatch':'renders/detail-front-latch.png',
         'bottom':'renders/detail-bottom.png', 'bottommark':'renders/detail-bottom-mark.png',
         'cangap':'renders/detail-can-gap.png', 'body':'renders/detail-body.png',
         'surface':'renders/detail-surface.png'},
}


async def model_review_file(request):
    number=request.path_params['number']
    folder_name=REVIEW_FOLDERS.get(number)
    folder=ROOT/folder_name if folder_name else None
    allowed=REVIEW_FILES.get(number, {})
    name=allowed.get(request.path_params['name'])
    path=folder/name if folder and name else None
    if not path or not path.is_file():return err('检查文件不存在',404)
    return FileResponse(path,filename=path.name,
                        content_disposition_type='attachment' if path.suffix.lower()=='.blend' else 'inline')


async def create_session(request):
    form = await request.form()
    sid = secrets.token_hex(16)
    folder = DATA / sid
    folder.mkdir()
    upload = form.get('background')
    try:
        if upload and getattr(upload, 'filename', ''):
            raw = await upload.read(30 * 1024 * 1024 + 1)
            await form.close()
            if len(raw) > 30 * 1024 * 1024:
                raise ValueError('图片超过 30 MB')
            source = io.BytesIO(raw)
        else:
            source = Path(json.loads(SCENE.read_text(encoding='utf8'))['background'])
        with Image.open(source) as im:
            im = ImageOps.exif_transpose(im)
            if im.width * im.height > 50_000_000 or min(im.size) < 64:
                raise ValueError('图片尺寸超出允许范围')
            if upload and getattr(upload, 'filename', ''):
                im.convert('RGB').save(folder / 'background.png', icc_profile=im.info.get('icc_profile'))
            width, height = im.size
        background_path = folder / 'background.png' if upload and getattr(upload, 'filename', '') else source
        s = {'id': sid, 'directory': str(folder), 'background': str(background_path),
             'background_sha256': digest(background_path), 'width': width, 'height': height,
             'revision': 0, 'generation': 0, 'state': None, 'candidates': [], 'selected': [], 'support': None, 'support_mode': None, 'history': [],
             'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'accounted_tokens': 0},
             'preview_revision': None, 'warnings': [], 'lock': asyncio.Lock()}
        SESSIONS[sid] = s
        save(s)
        return JSONResponse(snapshot(s))
    except (OSError, ValueError) as e:
        return err(e)


async def background(request):
    try:
        return FileResponse(session(request)['background'])
    except ValueError as e:
        return err(e, 404)


def validate_state(data, s):
    if data.get('asset') not in ROWS:
        raise ValueError('请选择模型库中的物品')
    values = {}
    for name, lower, upper in [('x', 0, s['width']), ('y', 0, s['height']),
                               ('size', 16, max(s['width'], s['height'])),
                               ('azimuth', -3600, 3600), ('elevation', 10, 85)]:
        value = float(data[name])
        if not np.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f'{name} 超出范围')
        values[name] = value
    return {'asset': data['asset'], **values}


async def update_state(request):
    try:
        s = session(request)
        data = await request.json()
        value = validate_state(data, s)
        async with s['lock']:
            if value != s['state']:
                s['state'] = value
                s['revision'] += 1
                s['generation'] += 1
                s['selected'] = []
                s['support'] = None
                s['support_mode'] = None
                s['candidates'] = []
                s['analysis_algorithm'] = None
                s['history'] = []
                s['preview_revision'] = None
                s.pop('export', None)
                save(s)
        return JSONResponse(snapshot(s))
    except (ValueError, KeyError, TypeError) as e:
        return err(e)


def compute_candidates(s, revision, state):
    global ENGINE
    if ENGINE is None:
        ENGINE = Perception(ROOT / 'work/v2-models', ROOT / 'work/v2-cache')
    rgb, depth, masks, original = ENGINE.scene(Path(s['background']))
    sx, sy = rgb.shape[1] / original[0], rgb.shape[0] / original[1]
    x, y, radius = state['x'] * sx, state['y'] * sy, state['size'] * max(sx, sy) * .65
    yy, xx = np.ogrid[:rgb.shape[0], :rgb.shape[1]]
    roi = ((xx-x)**2 + (yy-y)**2) < radius**2
    # Provisional contact neighborhood only: projected bottom of the square
    # is not a physical contact point and cannot prove a support relation.
    contact = (np.abs(xx-x)<radius*.50)&(np.abs(yy-(y+radius*.42))<radius*.22)
    candidates = []
    folder = Path(s['directory']) / f'analysis-{revision}'
    folder.mkdir(exist_ok=True)
    for mask in masks:
        overlap = int((mask & roi).sum())
        area = int(mask.sum())
        if overlap < 20 or area > roi.size * .18 or overlap / max(1, area) < .03:
            continue
        if any(np.count_nonzero(mask & old) / max(1, np.count_nonzero(mask | old)) > .75 for old in [c[0] for c in candidates]):
            continue
        candidates.append((mask, overlap))
    candidates.sort(key=lambda item: item[1], reverse=True)
    results = []
    with Image.open(s['background']) as source:
        frame=ImageOps.exif_transpose(source).convert('RGB')
    side=state['size'];cx=state['x'];cy=state['y']
    rect=(max(0,int(cx-side)),max(0,int(cy-side)),min(original[0],int(cx+side)),min(original[1],int(cy+side)))
    for i, (mask, _) in enumerate(candidates[:12], 1):
        ys, xs = np.nonzero(mask)
        full = cv2.resize(mask.astype('uint8'), original, interpolation=cv2.INTER_NEAREST)>0
        full, refinement=refine_thin_mask(ENGINE,frame,full,rect,radius=min(16,max(5,round(2/max(sx,sy)))))
        Image.fromarray(full.astype('uint8')*255).save(folder / f'{i}.png')
        results.append({'id': i, 'bounds': [round(xs.min()/sx), round(ys.min()/sy), round((xs.max()+1)/sx), round((ys.max()+1)/sy)],
                        'support_score':round(float((mask & contact).sum()/max(1,contact.sum())),3),
                        'boundary_refinement':refinement,
                        'relative_depth': round(float(np.median(depth[mask])), 3), 'order': 'unknown'})
    local=frame.crop(rect)
    for seed in thin_line_seeds(local,(cx-rect[0],cy-rect[1]),side,limit=4):
        full=np.zeros((original[1],original[0]),dtype=bool)
        full[rect[1]:rect[3],rect[0]:rect[2]]=seed
        full,refinement=refine_thin_mask(ENGINE,frame,full,rect,radius=6)
        # Reject almost-identical duplicates, but retain distinct crossings.
        part=full[rect[1]:rect[3],rect[0]:rect[2]]
        duplicate=False
        for c in results:
            with Image.open(folder/f"{c['id']}.png") as im:
                other=np.asarray(im.crop(rect))>0
            if np.count_nonzero(part&other)/max(1,np.count_nonzero(part|other))>.65:
                duplicate=True;break
        if duplicate:
            continue
        ys,xs=np.nonzero(full)
        if not len(xs):continue
        i=len(results)+1
        Image.fromarray(full.astype('uint8')*255).save(folder/f'{i}.png')
        depthmask=cv2.resize(full.astype('uint8'),(depth.shape[1],depth.shape[0]),interpolation=cv2.INTER_NEAREST)>0
        results.append({'id':i,'bounds':[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)],
                        'support_score':0,'boundary_refinement':refinement,'kind':'thin_line_proposal',
                        'relative_depth':round(float(np.median(depth[depthmask])),3) if depthmask.any() else None,
                        'order':'unknown'})
    return results


async def analyze(request):
    try:
        s = session(request)
        if not s['state']:
            raise ValueError('请先选择并放置物品')
        rev, generation, state = s['revision'], s['generation'], s['state'].copy()
        async with ENGINE_LOCK:
            candidates = await asyncio.to_thread(compute_candidates, s, rev, state)
        async with s['lock']:
            if rev != s['revision'] or generation != s['generation']:
                return err('摆位或遮挡已变化，忽略旧分析', 409)
            s['candidates'], s['selected'] = candidates, []
            s['analysis_algorithm'] = ANALYSIS_ALGORITHM
            s['support'] = None
            s['support_mode'] = None
            s['generation'] += 1
            s['preview_revision'] = None
            s.pop('export', None)
            s['warnings'] = ['相对深度不能独立证明前后顺序，请确认前景区域。']
            save(s)
        return JSONResponse(snapshot(s))
    except Exception as e:
        return err('场景分析失败：' + str(e), 503)


async def mask_image(request):
    try:
        s = session(request)
        n = int(request.path_params['number'])
        if n not in [x['id'] for x in s['candidates']]:
            raise ValueError('区域不存在')
        return FileResponse(Path(s['directory']) / f"analysis-{s['revision']}" / f'{n}.png', media_type='image/png')
    except (ValueError, OSError) as e:
        return err(e, 404)


async def selection(request):
    try:
        s = session(request)
        data = await request.json()
        selected = data.get('selected')
        if not isinstance(selected, list) or any(type(x) is not int or x not in [c['id'] for c in s['candidates']] for x in selected):
            raise ValueError('遮挡区域无效')
        if s.get('support') in selected and s.get('support_mode') != 'estimated':
            raise ValueError('同一区域不能同时作为前景和承托面')
        async with s['lock']:
            if data.get('revision') != s['revision']:
                return err('摆位已变化，请重新分析', 409)
            s['history'].append(s['selected'].copy())
            s['selected'] = sorted(set(selected))
            if s.get('support') in selected:
                s['support'] = None
                s['support_mode'] = None
            s['generation'] += 1
            s['preview_revision'] = None
            s.pop('export', None)
            save(s)
        return JSONResponse(snapshot(s))
    except (ValueError, TypeError) as e:
        return err(e)


async def undo(request):
    try:
        s = session(request)
        async with s['lock']:
            if s['history']:
                s['selected'] = s['history'].pop()
                s['generation'] += 1
                s['preview_revision'] = None
                s.pop('export', None)
                save(s)
        return JSONResponse(snapshot(s))
    except ValueError as e:
        return err(e)


async def support_selection(request):
    try:
        s = session(request)
        data = await request.json()
        n = data.get('support')
        if n is not None and (type(n) is not int or n not in [c['id'] for c in s['candidates']]):
            raise ValueError('承托面区域无效')
        if n is not None and next(c for c in s['candidates'] if c['id']==n).get('kind')=='thin_line_proposal':
            raise ValueError('细钢件候选不能作为承托面')
        if n in s['selected']:
            raise ValueError('同一区域不能同时作为前景和承托面')
        async with s['lock']:
            if data.get('revision') != s['revision']:
                return err('摆位已变化，请重新分析',409)
            s['support'] = n
            s['support_mode'] = 'user_confirmed' if n else None
            s['generation'] += 1
            s['preview_revision'] = None
            s.pop('export',None)
            save(s)
        return JSONResponse(snapshot(s))
    except (ValueError,TypeError) as e:
        return err(e)


def foreground(s):
    mask = Image.new('L', (s['width'], s['height']))
    for n in s['selected']:
        with Image.open(Path(s['directory']) / f"analysis-{s['revision']}" / f'{n}.png') as im:
            mask = Image.fromarray(np.maximum(np.asarray(mask), np.asarray(im.convert('L'))))
    return mask


def match_local_photo(s, obj, x, y):
    """Use selected support, or explicitly unverified nearby scene pixels."""
    n = s.get('support')
    side = obj.width
    pad = round(side*.5)
    crop = (max(0,x-pad), max(0,y-pad), min(s['width'],x+side+pad), min(s['height'],y+side+pad))
    if n:
        path = Path(s['directory']) / f"analysis-{s['revision']}" / f'{n}.png'
        with Image.open(path) as im:
            mask = np.asarray(im.convert('L').crop(crop)) > 127
    else:
        # A surrounding-scene sample is a fallback for light and color only;
        # it is not evidence of a common support or of a 3-D light direction.
        mask = np.ones((crop[3]-crop[1],crop[2]-crop[0]),dtype=bool)
    front = np.asarray(foreground(s).crop(crop)) >= 128
    with Image.open(s['background']) as im:
        frame = im.convert('RGB').crop(crop)
    result, photo = ambient_match(frame, mask, front, obj, x-crop[0], y-crop[1])
    photo['support_verification'] = (s.get('support_mode') or 'surrounding_scene_unverified')
    return result, photo


async def preview(request):
    try:
        s = session(request)
        data = await request.json()
        revision = data.get('revision')
        if revision != s['revision'] or not s['state']:
            return err('摆位已变化，请重新生成', 409)
        encoded = data.get('image', '')
        if not encoded.startswith('data:image/png;base64,') or len(encoded) > 12_000_000:
            raise ValueError('三维占位图格式或大小不正确')
        with Image.open(io.BytesIO(base64.b64decode(encoded.split(',', 1)[1], validate=True))) as im:
            if im.width > 2048 or im.height > 2048 or im.width != im.height:
                raise ValueError('占位图尺寸不正确')
            obj = im.convert('RGBA')
        if not obj.getchannel('A').getbbox():
            raise ValueError('占位图为空')
        if not s['candidates']:
            old_generation = s['generation']
            state = s['state'].copy()
            try:
                async with ENGINE_LOCK:
                    found = await asyncio.to_thread(compute_candidates,s,revision,state)
                async with s['lock']:
                    if revision != s['revision'] or old_generation != s['generation']:
                        return err('摆位或遮挡已变化，忽略旧分析',409)
                    s['candidates'] = found
                    s['analysis_algorithm'] = ANALYSIS_ALGORITHM
                    s['generation'] += 1
                    save(s)
            except (FileNotFoundError,RuntimeError,ValueError):
                # Continue as an explicitly unlit, review-only preview.
                pass
        if s.get('support') is None:
            ranked = sorted((c for c in s['candidates'] if c['id'] not in s['selected']),
                            key=lambda c:c.get('support_score',0),reverse=True)
            if ranked and ranked[0].get('support_score',0)>=.25 and (len(ranked)==1 or
                    ranked[0]['support_score']-ranked[1].get('support_score',0)>=.08):
                s['support'] = ranked[0]['id']
                s['support_mode'] = 'estimated'
                s['generation'] += 1
                save(s)
        generation = s['generation']
        st = s['state'].copy()
        row = ROWS[st['asset']]
        mesh = Path(row['mesh'])
        if digest(mesh) != row.get('mesh_sha256'):
            raise ValueError('模型哈希与库记录不一致')
        run_id = secrets.token_hex(8)
        folder = Path(s['directory']) / ('render-' + run_id)
        folder.mkdir()
        output = folder / 'object.png'
        side=round(st['size']);cx=st['x'];cy=st['y']
        light_rect=(max(0,int(cx-side)),max(0,int(cy-side)),min(s['width'],int(cx+side)),min(s['height'],int(cy+side)))
        with Image.open(s['background']) as background:
            light_frame=ImageOps.exif_transpose(background).convert('RGB').crop(light_rect)
        lighting=await asyncio.to_thread(estimate_light_from_rods,light_frame,(cx-light_rect[0],cy-light_rect[1]),side)
        req = {'mesh':str(mesh), 'azimuth':st['azimuth'], 'elevation':st['elevation'], 'output':str(output),
               'lighting':lighting}
        (folder/'request.json').write_text(json.dumps(req), encoding='utf8')
        blender = ROOT/'work/blender45/blender-4.5.3-windows-x64/blender.exe'
        if not blender.is_file():
            raise ValueError('找不到 Blender，无法生成实际渲染')
        with (folder/'blender.log').open('wb') as log:
            process = await asyncio.create_subprocess_exec(str(blender), '--background', '--factory-startup',
                '--python-exit-code', '1', '--python', str(HERE/'render_preview.py'), '--', str(folder/'request.json'),
                cwd=str(ROOT), stdout=log, stderr=log)
            try:
                code = await asyncio.wait_for(process.wait(), timeout=180)
            except asyncio.TimeoutError:
                process.kill(); await process.wait()
                raise ValueError('Blender 渲染超时')
        if code or not output.is_file():
            raise ValueError('Blender 渲染失败，请查看会话目录中的日志')
        with Image.open(output) as image:
            blender_image = image.convert('RGBA')
        one = np.asarray(obj.getchannel('A').resize((512,512), Image.Resampling.LANCZOS)) >= 128
        two = np.asarray(blender_image.getchannel('A').resize((512,512), Image.Resampling.LANCZOS)) >= 128
        union = np.count_nonzero(one | two)
        iou = float(np.count_nonzero(one & two) / union) if union else 0
        if iou < .95:
            raise ValueError(f'网页与 Blender 的投影不一致（IoU {iou:.3f}），停止生成')
        obj = blender_image
        side = round(st['size'])
        obj = obj.resize((side, side), Image.Resampling.LANCZOS)
        x, y = round(st['x']-side/2), round(st['y']-side/2)
        matched, photo = await asyncio.to_thread(match_local_photo,s,obj,x,y)
        base = Image.open(s['background']).convert('RGBA')
        result = base.copy()
        result.alpha_composite(matched, (x, y))
        front_mask = foreground(s)
        object_pixels = np.asarray(obj.getchannel('A')) >= 128
        overlap = np.asarray(front_mask.crop((x,y,x+side,y+side))) >= 128
        visible_ratio = 1-float(np.count_nonzero(object_pixels & overlap))/max(1,int(object_pixels.sum()))
        overlaps={}
        for candidate in s['candidates']:
            with Image.open(Path(s['directory'])/f"analysis-{revision}"/f"{candidate['id']}.png") as im:
                mask=np.asarray(im.convert('L').crop((x,y,x+side,y+side)))>=128
            overlaps[candidate['id']]=int(np.count_nonzero(mask&object_pixels))
        result.paste(base, (0, 0), front_mask)
        async with s['lock']:
            if revision != s['revision'] or generation != s['generation']:
                return err('摆位已变化，忽略旧预览', 409)
            obj.save(Path(s['directory']) / 'object.png')
            matched.save(Path(s['directory']) / 'object-matched.png')
            result.convert('RGB').save(Path(s['directory']) / 'preview.png')
            s['preview_revision'] = revision
            s['preview_generation'] = generation
            s['render_algorithm'] = RENDER_ALGORITHM
            s.pop('export',None)
            s['projection_iou'] = round(iou, 4)
            s['lighting']=lighting
            s['object_visible_fraction'] = round(visible_ratio,4)
            for candidate in s['candidates']:
                candidate['object_overlap_pixels']=overlaps[candidate['id']]
            bounds = obj.getchannel('A').getbbox()
            s['rendered_bounds_px'] = [x+bounds[0], y+bounds[1], x+bounds[2], y+bounds[3]]
            s['rendered_major_px'] = max(bounds[2]-bounds[0], bounds[3]-bounds[1])
            s['tone'] = None
            s['photometry'] = photo
            s['warnings'] = ['Blender 视角渲染，未验证刚体、接触、尺度和遮挡；不是 Photoshop 最终结果。',
                             ('已依据'+('自动估计的承托区域' if s.get('support_mode')=='estimated' else '人工选择的区域' if s.get('support') else '周边像素（承托面未确认）')+'估算局部环境光和颜色；实际光源方向、承托面及保护区仍待确认。') if photo and photo.get('status')=='measured' else '可用的环境像素不足，尚未重建环境光和调色。',
                             '没有经过验证的接触点，尚未生成接触阴影。']
            if visible_ratio<.5:
                s['warnings'].append(f'遮挡后物体仅露出约 {visible_ratio:.0%}，请确认前景选择。')
            if s['selected'] and all(overlaps.get(n,0)==0 for n in s['selected']):
                s['warnings'].append('所选前景区域没有与物体相交，未形成遮挡。')
            if s['candidates'] and s.get('analysis_algorithm')!=ANALYSIS_ALGORITHM:
                s['warnings'].append('当前遮挡候选来自旧版分析；请重新识别以获得细钢件候选。')
            s['warnings'].append('光向由局部钢件明暗低置信度估计，非实测。' if lighting['status']=='estimated'
                                 else '未找到足够一致的光向线索，使用中性环境光。')
            save(s)
        return JSONResponse(snapshot(s))
    except (ValueError, OSError, TypeError) as e:
        return err(e)


async def preview_image(request):
    try:
        s = session(request)
        if (s['preview_revision'] != s['revision'] or s.get('preview_generation') != s['generation']
                or s.get('render_algorithm') != RENDER_ALGORITHM):
            raise ValueError('预览已过期')
        return FileResponse(Path(s['directory']) / 'preview.png', media_type='image/png')
    except ValueError as e:
        return err(e, 404)


async def photoshop_export(request):
    """Export a clearly provisional layer stack; never claim physical acceptance."""
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from batch_v2 import Budget, Runner, save_mask
        from run import Config, load
        s = session(request)
        rev, generation = s['revision'], s['generation']
        if (s['preview_revision'] != rev or s.get('preview_generation') != generation
                or s.get('render_algorithm') != RENDER_ALGORITHM):
            raise ValueError('请先生成当前版本的预览')
        folder = Path(s['directory']) / ('export-' + secrets.token_hex(8))
        folder.mkdir()
        st = s['state'].copy()
        with Image.open(Path(s['directory'])/'object-matched.png') as im:
            asset = im.convert('RGBA')
        bbox = asset.getchannel('A').getbbox()
        if not bbox:
            raise ValueError('物体透明度为空')
        ox, oy = round(st['x']-st['size']/2), round(st['y']-st['size']/2)
        l,t,r,b = (ox+bbox[0],oy+bbox[1],ox+bbox[2],oy+bbox[3])
        if l < 0 or t < 0 or r > s['width'] or b > s['height']:
            raise ValueError('物体超出背景边界，不能导出')
        asset.crop(bbox).save(folder/'asset.png')
        with Image.open(Path(s['directory'])/'object.png') as im:
            im.convert('RGBA').crop(bbox).save(folder/'neutral.png')
        mask = np.asarray(foreground(s))
        entry = save_mask(mask, folder/'foreground.png', [0,0,s['width'],s['height']], (255,255,255))
        spec = {'id':'manual','asset':str(folder/'asset.png'),'center':[(l+r)/2,(t+b)/2],
                'width':r-l,'angle':0,'canvas':[s['width'],s['height']], 'crop':[0,0,s['width'],s['height']],
                'unmatched_asset':str(folder/'neutral.png'),
                'tone':{'brightness':0,'warmth':0,'saturation':0,'blur_px':0},'shade':None,'occlusion':entry,
                'png':str(folder/'manual.png'),'psd':str(folder/'manual.psd')}
        mcp = ROOT/'work/photoshop-mcp-env/Scripts/photoshop-mcp-server.exe'
        if not mcp.is_file():
            raise ValueError('未找到 Photoshop MCP')
        async with httpx.AsyncClient(trust_env=False) as http:
            runner = Runner(None,http,load(Config,HERE.parent/'batch-config.example.json'),None,
                            ROOT/'work/v2-cache',Budget(folder/'usage.json',0))
            with (folder/'photoshop.log').open('w',encoding='utf8') as log:
                async with stdio_client(StdioServerParameters(command=str(mcp)),errlog=log) as (reader,writer):
                    async with ClientSession(reader,writer) as client:
                        await asyncio.wait_for(client.initialize(),timeout=30)
                        runner.session = client
                        await runner.ps({'mode':'reference','source':s['background'],'png':str(folder/'reference.png')})
                        outcome = (await runner.ps({'mode':'compose','source':s['background'],'candidates':[spec]}))[0]
        if outcome.get('error') or not (folder/'manual.png').is_file() or not (folder/'manual.psd').is_file():
            raise ValueError('Photoshop 导出失败：'+str(outcome.get('error','输出文件缺失')))
        with Image.open(folder/'reference.png') as bg, Image.open(folder/'manual.png') as result:
            if result.size != bg.size:
                raise ValueError('Photoshop 导出尺寸变化')
            before = np.asarray(bg.convert('RGB')); after = np.asarray(result.convert('RGB'))
            changed = np.any(before != after,axis=2)
            changed[t:b,l:r] = False
            outside = int(changed.sum())
        if outside:
            raise ValueError(f'编辑范围外有 {outside} 像素变化，导出未通过验证')
        async with s['lock']:
            if rev != s['revision'] or generation != s['generation']:
                return err('摆位或遮挡已变化，导出结果已过期',409)
            s['export'] = {'directory':str(folder),'status':'review','outside_changed_pixels':outside,
                           'projection_iou':s.get('projection_iou'),
                           'limitations':['落稳、支撑、尺度、保护区、同层调色、阴影及前后顺序待确认']}
            save(s)
        return JSONResponse(s['export'])
    except Exception as e:
        return err('Photoshop 导出未完成：'+str(e)[:300],503)


async def export_file(request):
    try:
        s = session(request)
        name = request.path_params['name']
        if name not in ('manual.png','manual.psd') or not s.get('export'):
            raise ValueError('导出文件不存在')
        return FileResponse(Path(s['export']['directory'])/name,filename=name)
    except ValueError as e:
        return err(e,404)


async def chat(request):
    try:
        s = session(request)
    except ValueError as e:
        return err(e,404)
    if s['id'] in CHAT_ACTIVE:
        return err('已有对话正在处理，请稍候',409)
    CHAT_ACTIVE.add(s['id'])
    try:
        return await chat_once(request)
    finally:
        CHAT_ACTIVE.discard(s['id'])


async def chat_once(request):
    try:
        s = session(request)
        data = await request.json()
        prompt = str(data.get('text', '')).strip()[:1000]
        if not prompt or data.get('revision') != s['revision']:
            raise ValueError('请输入说明并确保摆位未变化')
        api_key = os.environ.get('PS_DEEPSEEK_KEY')
        if not api_key:
            return err('未配置 PS_DEEPSEEK_KEY，AI 对话不可用；可手动选择区域。', 503)
        if s['usage'].get('accounted_tokens',s['usage']['total_tokens']) + 4096 > 30000:
            return err('本会话模型预算已达上限', 429)
        rev, generation = s['revision'], s['generation']
        img = Image.open(s['background']).convert('RGB')
        img.thumbnail((512, 512))
        draw = ImageDraw.Draw(img)
        state = s['state']
        scale_x, scale_y = img.width/s['width'], img.height/s['height']
        half = state['size']/2
        draw.rectangle(((state['x']-half)*scale_x,(state['y']-half)*scale_y,
                        (state['x']+half)*scale_x,(state['y']+half)*scale_y),outline='cyan',width=2)
        for c in s['candidates']:
            l,t,r,b = c['bounds']
            sx, sy = img.width/s['width'], img.height/s['height']
            draw.rectangle((l*sx,t*sy,r*sx,b*sy), outline='yellow', width=2)
            draw.text((l*sx,t*sy), str(c['id']), fill='yellow', stroke_width=1, stroke_fill='black')
        buf = io.BytesIO(); img.save(buf, format='JPEG', quality=75)
        image_url = 'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode('ascii')
        instruction = ('青色框是物体占位范围，黄色框是候选区域。仅根据用户的明确指示，从候选编号中判断哪些是前景。不要猜测不明确的遮挡。'
                       '只返回 JSON：{"reply":"简短回答或追问","selected":[整数编号]}。'
                       '若不明确，selected 保持当前值。不能调整物品位置、大小或视角。'
                       f'当前前景编号：{s["selected"]}；有效编号：{[c["id"] for c in s["candidates"]]}。')
        # Reserve before sending; ambiguous failures retain the reservation.
        s['usage']['accounted_tokens'] = s['usage'].get('accounted_tokens',s['usage']['total_tokens']) + 4096
        save(s)
        async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
            response = await client.post('https://api.deepseek.com/chat/completions', headers={'Authorization': 'Bearer '+api_key},
                json={'model':'deepseek-flash','max_tokens':256,'response_format':{'type':'json_object'},
                      'messages':[{'role':'system','content':instruction},{'role':'user','content':[
                          {'type':'text','text':prompt},{'type':'image_url','image_url':{'url':image_url}}]}]})
            response.raise_for_status(); answer = response.json()
        usage = answer.get('usage', {})
        async with s['lock']:
            for key in ('prompt_tokens','completion_tokens','total_tokens'):
                s['usage'][key] += int(usage.get(key, 0) or 0)
            if isinstance(usage.get('total_tokens'),int):
                s['usage']['accounted_tokens'] += usage['total_tokens'] - 4096
            save(s)
            if rev != s['revision'] or generation != s['generation']:
                return err('摆位或遮挡已变化，忽略旧对话结果', 409)
            content = json.loads(answer['choices'][0]['message']['content'])
            chosen = content.get('selected')
            allowed = [c['id'] for c in s['candidates']]
            if not isinstance(chosen, list) or any(type(n) is not int or n not in allowed for n in chosen):
                raise ValueError('AI 返回了无效区域，未应用修改')
            s['history'].append(s['selected'].copy())
            s['selected'] = sorted(set(chosen)); s['generation'] += 1; s['preview_revision'] = None
            s.pop('export', None); save(s)
        return JSONResponse({'reply':str(content.get('reply','请确认前景区域。'))[:500], 'session':snapshot(s)})
    except Exception as e:
        return err('对话未完成：'+str(e)[:200], 503)


async def automatic(request):
    try:
        s = session(request)
        if not s['state']:
            raise ValueError('请先选择物品')
        if ROWS[s['state']['asset']].get('manual_only'):
            raise ValueError('六视图修复候选仅供人工预览，尚未通过自动放置的物理校验')
        out = DATA / f"auto-{s['id']}-{int(time.time())}"
        command = [sys.executable, '-X', 'utf8', '-B', str(HERE.parent / 'mesh_location_first.py'),
                   '--asset', s['state']['asset'], '--background', s['background'], '--out', str(out), '--budget', '0']
        with (Path(s['directory'])/'automatic.log').open('ab') as log:
            process = await asyncio.create_subprocess_exec(*command, cwd=str(ROOT), stdout=log, stderr=log)
        s['automatic'] = {'pid': process.pid, 'output': str(out), 'status': 'running'}; save(s)
        async def wait():
            code = await process.wait()
            s['automatic']['status'] = 'finished' if code == 0 else 'review'
            s['automatic']['exit_code'] = code
            save(s)
        asyncio.create_task(wait())
        return JSONResponse(s['automatic'])
    except (ValueError, OSError) as e:
        return err(e)


async def automatic_file(request):
    try:
        s = session(request)
        name = request.path_params['name']
        if name not in ('result.json','场景合成.png','场景合成.psd','位置优先-对照.jpg') or not s.get('automatic'):
            raise ValueError('自动放置文件不存在')
        path = Path(s['automatic']['output'])/name
        if not path.is_file():
            raise ValueError('自动放置尚未生成该文件')
        return FileResponse(path,filename=name)
    except ValueError as e:
        return err(e,404)


async def status(request):
    try:
        return JSONResponse(snapshot(session(request)))
    except ValueError as e:
        return err(e, 404)


routes = [Route('/', index), Route('/review/01', model_review_page),
          Route('/review/02', model_review_page_02),
          Route('/review/03', model_review_page_03),
          Route('/review/04', model_review_page_04),
          Route('/review/05', model_review_page_05),
          Route('/review/06', model_review_page_06),
          Route('/review/07', model_review_page_07),
          Route('/review/08', model_review_page_08),
          Route('/review/09', model_review_page_09),
          Route('/review/10', model_review_page_10),
          Route('/review/11', model_review_page_11),
          Route('/review/12', model_review_page_12),
          Route('/review/13', model_review_page_13),
          Route('/api/review/{number:int}/{name}', model_review_file),
          Route('/api/assets', assets), Route('/api/model/{asset}', model),
          Route('/api/sessions', create_session, methods=['POST']),
          Route('/api/sessions/{sid}', status), Route('/api/sessions/{sid}/background', background),
          Route('/api/sessions/{sid}/state', update_state, methods=['POST']),
          Route('/api/sessions/{sid}/analyze', analyze, methods=['POST']),
          Route('/api/sessions/{sid}/mask/{number}', mask_image),
          Route('/api/sessions/{sid}/selection', selection, methods=['POST']),
          Route('/api/sessions/{sid}/support', support_selection, methods=['POST']),
          Route('/api/sessions/{sid}/undo', undo, methods=['POST']),
          Route('/api/sessions/{sid}/preview', preview, methods=['POST']),
          Route('/api/sessions/{sid}/preview.png', preview_image),
          Route('/api/sessions/{sid}/export', photoshop_export, methods=['POST']),
          Route('/api/sessions/{sid}/export/{name}', export_file),
          Route('/api/sessions/{sid}/chat', chat, methods=['POST']),
          Route('/api/sessions/{sid}/automatic', automatic, methods=['POST']),
          Route('/api/sessions/{sid}/automatic/{name}', automatic_file),
          Mount('/static', StaticFiles(directory=str(HERE)), name='static')]
async def local_only(request, call_next):
    if request.client and request.client.host not in ('127.0.0.1', '::1', 'testclient'):
        return err('仅允许本机访问', 403)
    host = request.headers.get('host', '').split(':')[0]
    if host not in ('localhost', '127.0.0.1', 'testserver'):
        return err('无效的本机地址', 403)
    origin = request.headers.get('origin')
    if origin and origin not in (f'http://{request.headers.get("host")}', f'https://{request.headers.get("host")}'):
        return err('拒绝跨站请求', 403)
    return await call_next(request)


app = Starlette(routes=routes, middleware=[Middleware(BaseHTTPMiddleware, dispatch=local_only)])


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8765)
