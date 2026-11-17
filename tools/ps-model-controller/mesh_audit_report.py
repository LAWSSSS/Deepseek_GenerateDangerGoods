"""Assemble six-view inspection boards and conservative per-model findings."""
import argparse
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont

VIEWS=('front','back','left','right','top','bottom')
OBSERVATIONS={
    8:'侧视存在与主体分离的漂浮碎片；已有待审修复候选。',
    10:'手柄与主体为独立网格，装配或散件关系仍需确认。',
    15:'俯视截面明显偏椭圆；已有按参考图近似圆化的待审候选。',
    16:'俯视截面明显偏椭圆；已有按参考图近似圆化的待审候选。',
    19:'侧视背部出现原图未见的漂浮块；已有待审修复候选。',
    27:'侧面弹簧与连接部的厚度和完整性仍需核对。',
    29:'原模型源图孔洞被填；已有通过源图检查的开孔候选。',
    30:'侧视杆体与端部安装结构仍需核对；参考图型号并不确定。',
    34:'侧视与背面文字及表面有拉伸，需检查参考纹理。',
    36:'俯视截面偏椭圆；已有按参考图近似圆化的待审候选。',
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--library',type=Path,default=Path('outputs/三维模型库-圆润复验/results.jsonl'))
    args=p.parse_args()
    rows={r['number']:r for r in (json.loads(x) for x in args.library.read_text(encoding='utf8').splitlines() if x.strip())}
    renders=json.loads((args.out/'render-results.json').read_text(encoding='utf8'))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    reports=[]
    for result in renders:
        row=rows[result['number']]
        entry={'number':row['number'],'asset_id':row['asset_id'],'name':row['file'],
               'mesh_sha256':row['mesh_sha256'],'source_fidelity':row.get('reconstruction',{}).get('fidelity',{}),
               'render_status':result['status'],'views':{},'warnings':[]}
        board=Image.new('RGB',(1000,710),(238,240,241));draw=ImageDraw.Draw(board)
        draw.text((12,6),f"{row['number']:02d}  {row['file']}  |  六方向中性光检查",font=font,fill=(20,25,30))
        if result['status']=='rendered':
            for i,name in enumerate(VIEWS):
                with Image.open(result['views'][name]) as im:
                    rgba=np.asarray(im.convert('RGBA'))
                a=rgba[:,:,3]>=160
                rgb=rgba[:,:,:3]
                gray=np.rint(rgb[:,:,0]*.2126+rgb[:,:,1]*.7152+rgb[:,:,2]*.0722)
                data={'opaque_pixels':int(a.sum()),'median_luma':round(float(np.median(gray[a])),1) if a.any() else None,
                      'dark_fraction':round(float((gray[a]<18).mean()),4) if a.any() else None}
                entry['views'][name]=data
                if not a.any() or (data['median_luma']<18 and data['dark_fraction']>.8):
                    entry['warnings'].append(name+': 中性渲染发黑或为空，需与原素材比较')
                x=(i%3)*330+5;y=(i//3)*330+45
                tile=Image.new('RGBA',(320,285),(255,255,255,255))
                image=Image.fromarray(rgba);image.thumbnail((310,265),Image.Resampling.LANCZOS)
                tile.alpha_composite(image,((320-image.width)//2,(285-image.height)//2))
                board.paste(tile.convert('RGB'),(x,y));draw.text((x+5,y+289),f"{name}  亮度 {data['median_luma']}",font=font,fill=(35,40,45))
        if not entry['source_fidelity'].get('passed'):
            entry['warnings'].append('源视图轮廓、孔洞或细结构未通过；不代表所有其他视角已检查合格')
        if row['number'] in OBSERVATIONS:
            entry['warnings'].append('六视角人工观察：'+OBSERVATIONS[row['number']])
        entry['limitations']=['六视图参考未标定；结构缺陷仍需逐视角人工或有依据的复核',
                              '中性光亮度仅用于发现疑似异常，不能单独证明材质错误']
        board.save(args.out/f"{row['number']:02d}-六视角.jpg",quality=91)
        reports.append(entry)
    (args.out/'audit.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in reports),encoding='utf8')
    listing=''.join(f'<li><a href="{r["number"]:02d}-六视角.jpg">{r["number"]:02d} {html.escape(r["name"])}</a> — {html.escape("；".join(r["warnings"]) or "未发现中性渲染发黑；几何待复核")}</li>' for r in reports)
    (args.out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>39件三维模型六视角检查</title><h1>39件三维模型六视角检查</h1><p>亮度提示不是结构验收；背面和遮挡等无依据部分仍需复核。</p><ol>'+listing+'</ol>',encoding='utf8')
    print(json.dumps({'models':len(reports),'render_failed':sum(r['render_status']!='rendered' for r in reports),
                      'dark_views':sum(sum('发黑' in x for x in r['warnings']) for r in reports),
                      'source_failed':sum(not r['source_fidelity'].get('passed') for r in reports)},ensure_ascii=False))


if __name__=='__main__':main()
