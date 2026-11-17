"""Register rust and paint loss from observed view sheets to named mesh parts."""
import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from display_batch.common import atomic_json, file_hash, load_catalog
from display_batch.component_pbr import validate
from display_batch.reference_material import load_library, lookup_reference


def smooth(value, low, high):
    t = np.clip((value - low) / (high - low), 0., 1.)
    return t * t * (3. - 2. * t)


def material_cues(rgb, material):
    """Separate gray abrasion, warm corrosion and original pigment in sRGB."""
    r, g, b = np.moveaxis(rgb, -1, 0)
    peak = rgb.max(axis=-1)
    orange = smooth((r - b) / (peak + .035), .08, .34)
    brown = smooth((g - b) / (peak + .035), .025, .16)
    oxide = orange * brown * smooth(peak, .10, .20)
    chips = np.zeros_like(oxide)
    if material == 'blue':
        original_paint = smooth((b - r) / (peak + .025), .035, .25)
        chips = (1. - original_paint) * smooth(peak, .24, .61)
        oxide *= 1. - original_paint
    elif material == 'red':
        faded = smooth((g + .015) / (r + .015), .40, .72)
        oxide *= faded
        chips = faded * (1. - brown) * smooth(peak, .29, .66)
    elif material not in ('steel', 'dirt'):
        raise ValueError(f'Unknown reference surface class {material}')
    return oxide, chips * (1. - oxide)


def sample(sheet, columns, rows, view, scale):
    rgb = cv2.remap(sheet, np.float32(columns * scale[0]), np.float32(rows * scale[1]),
                    interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(1., 1., 1.))
    support = 1. - smooth(rgb.min(axis=-1), .86, .96)
    for x0, y0, x1, y1 in view.get('exclude', ()):
        distance = np.maximum.reduce([x0 - columns, columns - x1, y0 - rows, rows - y1])
        support *= smooth(distance, 0., 6.)
    return rgb, support


def cylinder_samples(sheet, spec, scale, shape):
    height, width = shape
    angle = ((np.arange(width, dtype=np.float32) + .5) / width - .5) * (2. * np.pi)
    position = (np.arange(height, dtype=np.float32) + .5) / height
    for view in spec['views']:
        delta = angle - np.deg2rad(view['angle'])
        across = np.sin(delta) * (-1. if view.get('flip_across') else 1.)
        # At a grazing angle small image errors can stretch into long streaks.
        facing = smooth(np.cos(delta), .51, .96) ** 1.7
        x0, y0, x1, y1 = view['rect']
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f'Invalid view rectangle {view}')
        vertical = spec['direction'] == 'vertical'
        low, high = (x0,x1) if vertical else (y0,y1)
        profile = view.get('profile', [[0.,low,high], [1.,low,high]])
        sections, lower, upper = np.asarray(profile).T
        low = np.interp(position, sections, lower)[:,None]
        high = np.interp(position, sections, upper)[:,None]
        cross = (low + high) / 2. + (high - low) / 2. * across
        along = position if not view.get('reverse') else 1. - position
        longitudinal = (y0 + (y1-y0)*along) if vertical else (x0 + (x1-x0)*along)
        longitudinal = np.broadcast_to(longitudinal[:,None], (height,width))
        x, y = (cross,longitudinal) if vertical else (longitudinal,cross)
        rgb, support = sample(sheet, x, y, view, scale)
        yield rgb, support * facing[None,:] * view.get('weight', 1.)


def plane_samples(sheet, spec, scale, shape):
    height, width = shape
    view, = spec['views']
    x0, y0, x1, y1 = view['rect']
    x = (np.arange(width, dtype=np.float32) + .5) / width
    y = (np.arange(height, dtype=np.float32) + .5) / height
    xx, yy = np.meshgrid(x0 + (x1-x0)*x, y0 + (y1-y0)*y)
    yield sample(sheet, xx, yy, view, scale)


def bake(sheet, spec, coordinate_size):
    """Blend visible evidence in object coordinates, leaving conflicts and omissions soft."""
    cylinder = spec['mode'] == 'cylinder'
    shape = (512, 1024) if cylinder else (512, 512)
    scale = np.asarray(sheet.shape[1::-1],float) / coordinate_size
    sampler = cylinder_samples if cylinder else plane_samples
    total = np.zeros(shape, np.float32)
    oxide = np.zeros_like(total)
    chips = np.zeros_like(total)
    pigment = np.zeros((*shape, 3), np.float32)
    for rgb, weight in sampler(sheet, spec, scale, shape):
        rust, abrasion = material_cues(rgb, spec['material'])
        total += weight
        oxide += weight * rust
        chips += weight * abrasion
        pigment += (weight * rust)[...,None] * rgb
    support = smooth(total, .015, .12) if cylinder else total
    observed = pigment / np.maximum(1e-6, oxide[...,None])
    # Keep hue, while attenuating low-frequency shading within the rust evidence.
    luma = observed @ np.float32([.2126,.7152,.0722])
    blur = lambda values: cv2.GaussianBlur(values, (0,0), 22.)
    reduced = blur(luma * oxide) / np.maximum(.0001, blur(oxide))
    meaningful = oxide > .04
    if meaningful.any():
        gain = np.clip(np.median(luma[meaningful]) / np.maximum(.05, reduced), .64, 1.42) ** .38
        observed *= gain[...,None]
    observed = np.clip(observed * .86, .023, .68)
    strength = spec.get('strength', 1.)
    values = np.stack([np.clip(strength * oxide / np.maximum(1e-6,total),0.,.98),
                       np.clip(chips / np.maximum(1e-6,total),0.,1.), support],axis=-1)
    return observed, values


def attach_projection(regions, parts, projection):
    """Keep source slot indices and split only pieces sharing a different map."""
    pending = set(parts)
    for region in list(regions):
        selected = [part for part in region['parts'] if part in pending]
        if not selected:
            continue
        if region.get('view_projection'):
            raise ValueError('An observed part may only be registered once')
        updated = copy.deepcopy(region)
        updated.update(parts=selected, view_projection=projection)
        if projection['finish'] == 'oxide' and not updated.get('oxide'):
            updated['oxide'] = 'iron_oxide'
        if selected == region['parts']:
            region.update(updated)
        else:
            region['parts'] = [part for part in region['parts'] if part not in selected]
            regions.append(updated)
        pending.difference_update(selected)
    if pending:
        raise ValueError(f'Unknown or unaudited registered parts {sorted(pending)}')


def run(spec_path, references, source_library, output):
    spec = json.loads(spec_path.read_text(encoding='utf8'))
    if spec.get('coordinate_frame') != 'normalized-render-mesh':
        raise ValueError('Projection axes must use the mesh after display canonicalization')
    source = load_library(source_library)
    catalog = load_catalog()
    output.mkdir(parents=True, exist_ok=False)
    rows, records = {}, []
    for aid, settings in spec['assets'].items():
        digest = catalog[aid]['mesh_hash']
        if digest != settings['mesh_sha256']:
            raise ValueError(f'Source model changed after view registration {aid}')
        sheet_path = references / settings['source_sheet']
        sheet_digest = file_hash(sheet_path)
        if sheet_digest != settings['source_sha256']:
            raise ValueError(f'Source sheet changed after view registration {aid}')
        with Image.open(sheet_path) as image:
            sheet = np.float32(image.convert('RGB')) / 255.
        reference = lookup_reference(source, digest)
        regions = reference['component_regions']
        # Sparse library oxidation fills evidence gaps in this candidate.
        for region in regions:
            region.pop('oxide_spread', None)
        for registration in settings['surfaces']:
            name = registration['name']
            if not name.isidentifier():
                raise ValueError(f'Invalid part projection name {name}')
            color, channels = bake(sheet, registration, np.asarray(spec['coordinate_size'], float))
            folder = output / aid / name
            folder.mkdir(parents=True, exist_ok=False)
            maps = {}
            for key, values in (('albedo',color),('surface',channels)):
                path = folder / (key + '.png')
                Image.fromarray(np.uint8(np.rint(np.clip(values,0.,1.) * 255.))).save(path)
                maps[key] = {'path':str(path.resolve()),'sha256':file_hash(path)}
            record = {key:registration[key] for key in ('mode','axes','origin','limits')}
            record.update(maps=maps, source_sha256=sheet_digest,
                          finish='dirt' if registration['material']=='dirt' else 'oxide')
            attach_projection(regions, registration['parts'], record)
            records.append({'asset':aid,'name':name,'source':str(sheet_path.resolve()),
                            'source_sha256':sheet_digest,'registered_views':[v['view'] for v in registration['views']],
                            'parts':registration['parts'],'mode':registration['mode'],
                            'usable_map_fraction':float((channels[...,2]>.5).mean()),
                            'rust_or_dirt_fraction':float((channels[...,0]>.25).mean())})
        entry = output / f'{aid}-component-materials.json'
        for key in ('binding_signature', 'identity_evidence', 'source_mesh_hash', 'asset_id'):
            reference.pop(key, None)
        reference['recipe_sha256'] = file_hash(spec_path)
        atomic_json(entry, validate(reference))
        rows[digest] = {'asset_id':aid,'reference_path':entry.name,'reference_sha256':file_hash(entry),
                        'evidence':{**source['assets'][digest]['evidence'],'source_sheet':settings['source_sheet'],
                                    'source_sha256':sheet_digest,'local_registration':True}}
    atomic_json(output/'library.json', {'version':'reference-material-library-v1','assets':rows})
    atomic_json(output/'registrations.json', records)
    print(json.dumps({'assets':list(spec['assets']), 'surface_registrations':len(records),
                      'library':str((output/'library.json').resolve())}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--baseline-library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.spec, args.references, args.baseline_library, args.output)


if __name__ == '__main__':
    main()
