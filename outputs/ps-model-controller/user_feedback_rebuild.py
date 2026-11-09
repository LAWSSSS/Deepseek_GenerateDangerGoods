"""Apply the 13/19/31 review feedback to new, review-only model versions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BLENDER = ROOT / 'work/blender45/blender-4.5.3-windows-x64/blender.exe'
WORKER = HERE / 'user_feedback_blender.py'
SEQUENCE = ROOT / 'work/direct-deepseek/sequence.json'
DECISIONS = ROOT / 'outputs/人工逐件审阅-20261109/用户审阅结论.json'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def glb(path):
    with path.open('rb') as f:
        magic, version, total = struct.unpack('<4sII', f.read(12))
        length, kind = struct.unpack('<II', f.read(8))
        if magic != b'glTF' or version != 2 or total != path.stat().st_size or kind != 0x4e4f534a:
            raise ValueError('Invalid GLB header')
        return json.loads(f.read(length).decode('utf8'))


def font(size):
    path = Path(r'C:\Windows\Fonts\msyh.ttc')
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def fitted(path, size):
    with Image.open(path) as image:
        image = image.convert('RGBA')
        background = Image.new('RGBA', image.size, (235, 239, 239, 255))
        background.alpha_composite(image)
        return ImageOps.contain(background.convert('RGB'), size, Image.Resampling.LANCZOS)


def board(output, title):
    canvas = Image.new('RGB', (1500, 1100), '#f5f7f7')
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 15), title, font=font(29), fill='#253b42')
    draw.text((24, 57), '人工反馈修订候选；场景尺度、接触和光照尚待检查', font=font(16), fill='#61777a')
    names = [('front','正面'),('back','背面'),('left','左侧'),('right','右侧'),('top','顶部'),('bottom','底部')]
    for i, (name,label) in enumerate(names):
        x, y = 20 + (i % 3) * 495, 93 + (i // 3) * 500
        draw.rounded_rectangle((x,y,x+478,y+480),radius=10,fill='#ffffff')
        image = fitted(output/'renders'/f'{name}.png',(456,435))
        canvas.paste(image,(x+(478-image.width)//2,y+11+(435-image.height)//2))
        draw.text((x+15,y+449),label,font=font(20),fill='#30474e')
    canvas.save(output/'六视图-修订后.jpg',quality=90,subsampling=0)


def compare(old, new, path, title, old_label, new_label):
    canvas = Image.new('RGB',(1400,760),'#f5f7f7')
    draw=ImageDraw.Draw(canvas)
    draw.text((24,20),title,font=font(30),fill='#263e45')
    for i,(image_path,label) in enumerate(((old,old_label),(new,new_label))):
        x=20+i*690
        draw.rounded_rectangle((x,80,x+670,740),radius=10,fill='#fff')
        image=fitted(image_path,(640,600))
        canvas.paste(image,(x+(670-image.width)//2,94+(600-image.height)//2))
        draw.text((x+16,700),label,font=font(22),fill='#30474e')
    canvas.save(path,quality=90,subsampling=0)


def group_board(cases, path):
    canvas=Image.new('RGB',(1800,900),'#f5f7f7')
    draw=ImageDraw.Draw(canvas)
    draw.text((24,18),'19 号气瓶：六个单独模型',font=font(32),fill='#263e45')
    draw.text((24,58),'每个模型只包含一只气瓶；大小比例是各模型的检查预览，不是现场尺度',
              font=font(19),fill='#61777a')
    for i,item in enumerate(cases):
        x=20+(i%3)*595; y=100+(i//3)*390
        draw.rounded_rectangle((x,y,x+575,y+370),radius=10,fill='#fff')
        image=fitted(Path(item['out'])/'renders/3q.png',(550,320))
        canvas.paste(image,(x+(575-image.width)//2,y+8+(320-image.height)//2))
        draw.text((x+14,y+333),f"{item['id']}  {item['name']}",font=font(20),fill='#30474e')
    canvas.save(path,quality=90,subsampling=0)


def run_worker(request, out, name):
    request_file = out / f'{name}-request.json'
    request_file.write_text(json.dumps(request,ensure_ascii=False,indent=2),encoding='utf8')
    log = out / f'{name}-blender.log'
    print('Starting',name,'; log:',log,flush=True)
    with log.open('w',encoding='utf8') as stream:
        completed = subprocess.run([str(BLENDER),'--background','--python',str(WORKER),'--',str(request_file)],
                                   cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=1800)
    if completed.returncode:
        raise RuntimeError(f'{name} failed; see {log}')
    print('Finished',name,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    out=args.out.resolve()
    if not out.is_relative_to((ROOT/'outputs').resolve()):
        raise ValueError('Output must be under the project outputs directory')
    out.mkdir(parents=True,exist_ok=False)
    sequence=json.loads(SEQUENCE.read_text(encoding='utf8'))
    source={n:json.loads(Path(sequence['items'][f'{n:02}']['record']).read_text(encoding='utf8'))
            for n in (13,19,31)}
    rows={r['number']:r for r in (json.loads(line) for line in
          (ROOT/'outputs/三维模型库-圆润复验/results.jsonl').read_text(encoding='utf8').splitlines())}
    config19=json.loads((HERE/'cylinders19_config.json').read_text(encoding='utf8'))
    config31=json.loads((HERE/'cylinder31_config.json').read_text(encoding='utf8'))
    case31={'id':'31','number':31,'name':'液压缸1（完整缸壁）','kind':'cylinder',
            'source_blend':source[31]['blend'],'out':str(out/'31-opaque'),'barrel':config31['dimensions']['barrel']}
    case13={'id':'13','number':13,'name':'密封胶罐（单只）','kind':'can','prefix':'Can rear',
            'source_blend':source[13]['blend'],'out':str(out/'13-single')}
    bottles=[{'id':f'19-{i}','number':19,'name':c['label']+'（单只）','kind':'bottle',
              'prefix':c['label'],'source_blend':source[19]['blend'],'out':str(out/f'19-{i}-single')}
             for i,c in enumerate(config19['cylinders'],1)]
    if len(bottles)!=6 or len({case['prefix'] for case in bottles})!=6:
        raise ValueError('Expected six distinct source bottles')
    run_worker({'out':str(out),'items':[case31],'summary':'31-worker-summary.json'},out,'31')
    run_worker({'out':str(out),'items':[case13,*bottles],'summary':'13-19-worker-summary.json'},out,'13-19')
    cases=[case13,*bottles,case31]
    records=[]
    for case in cases:
        folder=Path(case['out']); mesh=folder/'mesh.glb'; blend=folder/'model.blend'
        evidence=json.loads((folder/'geometry-check.json').read_text(encoding='utf8'))
        scene=glb(mesh)
        names=[node.get('name','') for node in scene.get('nodes',[])]
        if case['kind']=='can' and (names.count('Can rear body')!=1 or any(n.startswith('Can front ') for n in names)):
            raise RuntimeError('13 GLB does not contain exactly one rear can')
        if case['kind']=='bottle':
            if names.count(case['prefix']+' 筒身')!=1 or any(
                    name.startswith(other['prefix']+' ') for name in names for other in bottles if other is not case):
                raise RuntimeError(case['id']+' GLB contains an incorrect bottle set')
        if case['kind']=='cylinder':
            if 'barrel_full_opaque' not in names or any(n=='piston_body' or n.startswith(('piston_ring_','gland_ring_')) for n in names):
                raise RuntimeError('31 GLB contains a cutaway or exposed internal components')
            if evidence['outer_shell_rays_passed']!=evidence['outer_shell_rays_total']:
                raise RuntimeError('31 barrel is not fully opaque')
        if any('uri' in a and not str(a['uri']).startswith('data:')
               for a in scene.get('buffers',[])+scene.get('images',[])):
            raise RuntimeError('External GLB resource found')
        board(folder,f"{case['id']}  {case['name']}")
        records.append({'id':case['id'],'number':case['number'],'name':case['name'],
                        'quantity':1,'status':'revision_pending_review','mesh':str(mesh),'mesh_sha256':sha(mesh),
                        'blend':str(blend),'blend_sha256':sha(blend),
                        'board':str(folder/'六视图-修订后.jpg'),'rust':rows[case['number']]['rust'],
                        'source_blend_sha256':sha(Path(case['source_blend'])),'evidence':str(folder/'geometry-check.json')})
    compare(Path(source[31]['out'])/'renders/detail-overall-3q.png',out/'31-opaque/renders/3q.png',
            out/'31-修改前后对照.jpg','31 号液压缸：去除剖视开窗','修改前：剖视窗口','修改后：完整不透明缸壁')
    compare(Path(source[13]['out'])/'renders/new-front.png',out/'13-single/renders/front.png',
            out/'13-修改前后对照.jpg','13 号密封胶罐：仅保留一只','修改前：两只','修改后：一只')
    group_board(bottles,out/'19-拆分总览.jpg')
    result={'schema_version':1,'status':'revision_pending_review','user_accepted_other_numbers':[x for x in range(1,40) if x not in (13,19,31)],
            'scene_validation':'pending','items':records,'created':time.strftime('%Y-%m-%d %H:%M:%S')}
    (out/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print('Revision complete:',out,'; candidates:',len(records),flush=True)


if __name__=='__main__':
    main()
