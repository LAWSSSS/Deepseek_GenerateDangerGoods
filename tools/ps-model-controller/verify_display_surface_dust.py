"""Read back dust-only revisions and keep style confirmation separate from checks."""
import argparse
import asyncio
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from display_batch.ai import Vision
from display_batch.common import atomic_json,file_hash,load_catalog,resolve
from display_batch.scene import Scenes
from display_batch.schema import BatchConfig
from feedback_rollback import ROLLBACK,placement_masks,audit_png
from validate_display_local_light import combined_output,dust_output,frames
from validate_display_scanned import geometry_check
from verify_display_wide import Links


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


async def verify(out,scale):
    start=time.monotonic();cases=read(out/'results.json')
    if len(cases)!=10 or any(c['status']!='review' for c in cases):
        raise ValueError('incomplete_dust_cohort')
    previous={c['number']:c for c in read(resolve(combined_output(3))/'results.json')}
    rows={r['number']:r for r in read(resolve(ROLLBACK)/'cases.json')}
    config=BatchConfig(backgrounds='.',output=str(out),offline=True)
    cache=resolve(config.cache);vision=Vision(config,cache/'vision',out/'validation-model-usage.json')
    scenes=Scenes(cache,vision);records=[];images=set()
    try:
        for case in cases:
            row=rows[case['number']];prior=previous[case['number']]
            scene=await scenes.prepare(row['background'])
            bounds=row['baseline']['edit_bounds'];x,y,r,b=bounds
            alpha,foreground,visible,offset=placement_masks(scene,row['pose'],row['archived_render'],
                row['archived_window'],row['archived_meta'],bounds)
            audit=audit_png(scene,case['composition'],alpha,foreground,visible,row['pose'],offset)
            if not audit['pass']:raise ValueError('dust_pixel_or_geometry_audit_failed')
            with Image.open(case['material']['image']) as image:rgb=np.asarray(image.convert('RGB'))
            with Image.open(case['previous']['image']) as image:old=np.asarray(image.convert('RGB'))
            allowed=np.zeros(scene['size'][::-1],bool)
            allowed[y:b,x:r]=(scene['allowed'][y:b,x:r]|(alpha>0))&~foreground&~scene['protected'][y:b,x:r]
            changed=np.any(rgb!=np.asarray(scene['frame']),axis=2)
            if changed[~allowed].any():raise ValueError('changed_outside_valid_object_and_shadow_region')
            item=case['material'];reference=prior['groups']['combined'][-1]
            if case['previous']['image_hash']!=reference['image_hash']:
                raise ValueError('approved_reference_changed')
            if any(item['effects'][k]!=reference['effects'][k] for k in
                ('pool_kernel','pool_weight_max','fog_extra_max','fog_total_median','fog_color_linear')):
                raise ValueError('confirmed_pooling_or_fog_changed')
            if case['light_evidence']!=prior['light_evidence']:
                raise ValueError('confirmed_lighting_changed')
            variant=case['lighting_variants']['global'];info=variant['render']
            if variant['calibration']!=prior['lighting_variants']['global']['calibration']:
                raise ValueError('exposure_calibration_changed')
            geometry=geometry_check(row['pose'],info['render_meta'],row['archived_meta'],alpha,bounds,
                                    Path(info['render_dir']),offset)
            if not geometry['pass']:raise ValueError('new_render_geometry_failed')
            is_metal=case['asset_id']!='13';surface=case['surface_dust']
            if surface['relative_scale']!=scale or surface['applied']!=is_metal:
                raise ValueError('wrong_surface_dust_scale')
            if is_metal:
                profile=info['render_meta']['matte_profile']
                if profile['dust_scale']!=scale or profile['paint_dust_scale']!=.5:
                    raise ValueError('wrong_rendered_dust_profile')
                if np.array_equal(rgb,old):raise ValueError('metal_dust_change_has_no_effect')
            elif not np.array_equal(rgb,old):raise ValueError('plastic_control_changed')
            for path,item in frames(case).items():
                if file_hash(path)!=item['image_hash']:raise ValueError('saved_frame_changed')
                with Image.open(path) as image:
                    if image.size!=(3840,2160):raise ValueError('wrong_full_frame_size')
                with Image.open(item['preview']) as image:
                    if image.size!=(1600,900):raise ValueError('cropped_report_preview')
                images.add(path)
            records.append({'number':case['number'],'surface_applied':is_metal,'geometry':geometry,'audit':audit,
                'changed_pixels':int(np.any(rgb!=old,axis=2).sum()),'fog_total':reference['fog_total']})
    finally:
        await vision.close()
    links=Links();links.feed((out/'index.html').read_text(encoding='utf8'))
    for link in links.links:
        if not (out/link).is_file():raise ValueError('missing_report_link')
    for link in links.images:
        with Image.open(out/link) as image:
            if image.size!=(1600,900):raise ValueError('non_full_frame_report_image')
    if len(images)!=20 or len(links.images)!=20:raise ValueError('expected_20_full_frame_comparisons')
    hashes={k:file_hash(v['mesh']) for k,v in load_catalog().items() if v['available']}
    if hashes!=read(resolve(ROLLBACK)/'validation.json')['source_hashes']:
        raise ValueError('source_models_changed')
    result={'pass':True,'cases':10,'material_changed':8,'plastic_unchanged':2,'source_models':len(hashes),
            'full_frames':len(images),'records':records,'seconds':round(time.monotonic()-start,3)}
    atomic_json(out/'independent-validation.json',result)
    return {k:v for k,v in result.items() if k!='records'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dust-scale',type=float,default=.5)
    parser.add_argument('--out')
    args=parser.parse_args()
    print(json.dumps(asyncio.run(verify(resolve(args.out or dust_output(args.dust_scale)),args.dust_scale)),ensure_ascii=False))
