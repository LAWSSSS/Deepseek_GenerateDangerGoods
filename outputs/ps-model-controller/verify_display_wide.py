"""Independently read full PNGs, masks and report links from a wide diagnostic run."""
import argparse
import asyncio
from html.parser import HTMLParser
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from display_batch.ai import Vision
from display_batch.common import atomic_json,file_hash,load_catalog,resolve
from display_batch.schema import BatchConfig
from display_batch.scene import Scenes
from feedback_rollback import placement_masks,ROLLBACK
from validate_display_local_light import frames,WIDE_OUTPUT


class Links(HTMLParser):
    def __init__(self):
        super().__init__();self.images=[];self.links=[]
    def handle_starttag(self,tag,attrs):
        attributes=dict(attrs)
        if tag=='img':self.images.append(attributes['src'])
        if tag=='a':self.links.append(attributes['href'])


async def verify(out):
    start=time.monotonic()
    cases=json.loads((out/'results.json').read_text(encoding='utf8'))
    if len(cases)!=10 or any(c['status']!='review' for c in cases):
        raise ValueError('incomplete_wide_cohort')
    rows={r['number']:r for r in json.loads((resolve(ROLLBACK)/'cases.json').read_text(encoding='utf8'))}
    config=BatchConfig(backgrounds='.',output=str(out),offline=True)
    cache=resolve(config.cache)
    vision=Vision(config,cache/'vision',out/'validation-model-usage.json')
    scenes=Scenes(cache,vision)
    records=[];unique=set()
    try:
        for case in cases:
            row=rows[case['number']];scene=await scenes.prepare(row['background'])
            bounds=row['baseline']['edit_bounds'];x,y,r,b=bounds
            _,foreground,visible,_=placement_masks(scene,row['pose'],row['archived_render'],row['archived_window'],
                                                   row['archived_meta'],bounds)
            permitted=np.zeros(scene['size'][::-1],bool);permitted[y:b,x:r]=visible>0
            with Image.open(case['groups']['pool'][0]['image']) as image:base=np.asarray(image.convert('RGB'))
            background=np.asarray(scene['frame'])
            if np.any(base[scene['protected']]!=background[scene['protected']]) or np.any(
                    base[y:b,x:r][foreground]!=background[y:b,x:r][foreground]):
                raise ValueError('foreground_or_protected_pixels_changed')
            differences={}
            for kind,group in case['groups'].items():
                changes=[]
                for i,item in enumerate(group[1:]):
                    with Image.open(item['image']) as image:rgb=np.asarray(image.convert('RGB'))
                    if rgb.shape!=base.shape:raise ValueError('frame_size_changed')
                    changed=np.any(rgb!=base,axis=2)
                    if changed[~permitted].any():raise ValueError('optical_effect_leaked_outside_visible_object')
                    if kind=='pool':
                        if item['fog_weight']!=0 or item['pool_kernel']!=(16,32,64)[i] or not item['effects']['pool_uniform']:
                            raise ValueError('pool_ladder_is_not_uniform_single_variable')
                    elif item['pool_weight']!=0 or abs(item['fog_extra']-(.3,.6,.85)[i])>1e-6:
                        raise ValueError('fog_ladder_opacity_not_as_recorded')
                    changes.append({'pixels':int(changed.sum()),
                        'mean_rgb_change':float(np.mean(abs(rgb[permitted].astype(float)-base[permitted])))})
                differences[kind]=changes
            for path,item in frames(case).items():
                if file_hash(path)!=item['image_hash']:raise ValueError('image_hash_changed')
                with Image.open(path) as im:
                    if im.size!=(3840,2160):raise ValueError('non_native_report_frame')
                with Image.open(item['preview']) as im:
                    if im.size!=(1600,900):raise ValueError('cropped_report_preview')
                unique.add(path)
            variants=case['lighting_variants']
            for variant in variants.values():
                if not variant['audit']['pass'] or not variant['render']['geometry']['pass']:
                    raise ValueError('lighting_variant_geometry_failed')
            evidence=variants['mean']['light_evidence'];old=evidence['original_bounds'];new=evidence['bounds']
            probe=np.load(case['estimated_probe'],allow_pickle=False)
            import hashlib
            if hashlib.sha256(probe.tobytes()).hexdigest()!=case['light_evidence']['reconstruction']['sha256']:
                raise ValueError('probe_content_changed')
            records.append({'number':case['number'],'differences':differences,
                'geometry':variants['probe']['render']['geometry'],
                'high_fog_total':case['groups']['fog'][-1]['fog_total'],
                'sample_count':evidence['sample_count'],'sample_statistic':evidence['statistic'],
                'sample_area_ratio':((new[2]-new[0])*(new[3]-new[1]))/((old[2]-old[0])*(old[3]-old[1])),
                'exposures':{k:v['calibration']['exposure_ev'] for k,v in variants.items()},
                'reconstruction':case['light_evidence']['reconstruction'],'audit_pass':True})
    finally:
        await vision.close()
    links=Links();links.feed((out/'index.html').read_text(encoding='utf8'))
    for path in links.links:
        if not (out/path).is_file():raise ValueError('missing_report_link: '+path)
    for path in links.images:
        with Image.open(out/path) as image:
            if image.size!=(1600,900):raise ValueError('report_image_not_full_frame')
    if len(unique)!=90:raise ValueError('expected_90_unique_full_frames')
    catalog=load_catalog()
    hashes={k:file_hash(v['mesh']) for k,v in catalog.items() if v['available']}
    expected=json.loads((resolve(ROLLBACK)/'validation.json').read_text(encoding='utf8'))['source_hashes']
    if hashes!=expected:raise ValueError('source_models_changed')
    result={'pass':True,'cases':len(records),'source_models':len(hashes),'unique_full_frames':len(unique),
            'html_image_links':len(links.images),'records':records,'seconds':round(time.monotonic()-start,3)}
    atomic_json(out/'independent-validation.json',result)
    return {k:v for k,v in result.items() if k!='records'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default=WIDE_OUTPUT)
    print(json.dumps(asyncio.run(verify(resolve(parser.parse_args().out))),ensure_ascii=False))
