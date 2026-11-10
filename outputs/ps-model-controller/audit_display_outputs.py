"""Independent read-back audit of delivered PNG dimensions, protection and edit bounds."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageOps

from display_batch.common import atomic_json,resolve,file_hash


def main(batches,out):
    records=[]
    for folder in map(Path,batches):
        config=json.loads((folder/'config.json').read_text(encoding='utf8'));cache=resolve(config['cache'])
        for line in (folder/'results.jsonl').read_text(encoding='utf8').splitlines():
            result=json.loads(line)
            if not result.get('image'):continue
            with Image.open(result['background']) as image:original=np.asarray(ImageOps.exif_transpose(image).convert('RGB'))
            with Image.open(result['image']) as image:actual=np.asarray(image.convert('RGB'))
            same_shape=actual.shape==original.shape
            record={'job_id':result['job_id'],'status':result['status'],'dimensions_pass':same_shape}
            if not same_shape:
                records.append(record);continue
            changed=np.any(actual!=original,axis=2);h,w=changed.shape
            analysis=json.loads((cache/'scenes'/result['scene_signature']/'analysis.json').read_text(encoding='utf8'))
            protected=Image.new('1',(w,h));draw=ImageDraw.Draw(protected)
            for poly in analysis['protected']:
                draw.polygon([(round(x*(w-1)),round(y*(h-1))) for x,y in poly],fill=1)
            record['protected_changed_pixels']=int((changed&np.asarray(protected)).sum())
            best_index=result.get('best_iteration')
            if best_index is not None:
                best=next(r for r in result['iterations'] if r['iteration']==best_index)
                x,y,r,b=best['edit_bounds'];outside=changed.copy();outside[y:b,x:r]=False
                record['outside_edit_bounds_changed_pixels']=int(outside.sum())
                record['best_version_copy_matches']=file_hash(result['image'])==best['image_hash']
            else:
                record.update(outside_edit_bounds_changed_pixels=None,best_version_copy_matches=None,
                              unknown_reason='execution_failed_before_final_version_metadata')
            record['passed']=same_shape and record['protected_changed_pixels']==0 and record.get('outside_edit_bounds_changed_pixels')==0 and record.get('best_version_copy_matches') is True
            if result['status']=='accepted_display':assert record['passed'],record
            records.append(record)
    summary={'kind':'independent_png_readback','pngs':len(records),
             'passed':sum(r.get('passed',False) for r in records),
             'unknown':sum('unknown_reason' in r for r in records),'records':records}
    atomic_json(out,summary);print(json.dumps({k:v for k,v in summary.items() if k!='records'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batches',nargs='+',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();main(args.batches,args.out)
