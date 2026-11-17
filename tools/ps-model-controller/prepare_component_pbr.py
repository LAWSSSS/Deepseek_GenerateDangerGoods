"""Prepare separate licensed surface packs and a hash-bound component library."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2
import numpy as np
from PIL import Image

from display_batch.common import atomic_json, file_hash, load_catalog
from display_batch.component_pbr import VERSION, number, validate
from display_batch.photography import linear
from display_batch.compose import srgb


def channels(spec, base):
    raw, sources = {}, {}
    for key, value in spec['channels'].items():
        path = (base / value).resolve(strict=True)
        if key == 'albedo':
            with Image.open(path) as im:
                array = np.asarray(im.convert('RGB'), dtype=np.float32) / 255
        else:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f'Unreadable {key} map: {path}')
            array = np.float32(image[..., 0] if image.ndim == 3 else image)
            if np.issubdtype(image.dtype, np.integer):
                array /= np.iinfo(image.dtype).max
        if not np.isfinite(array).all():
            raise ValueError(f'Map contains non-finite data: {path}')
        raw[key] = np.clip(array, 0, 1)
        sources[key] = {'path': str(path), 'sha256': file_hash(path)}
    if set(raw) - {'albedo', 'roughness', 'height', 'metallic'} or not {'albedo', 'roughness', 'height'} <= set(raw):
        raise ValueError('A pack needs base colour, roughness and height, with optional metalness')
    if any(image.shape[:2] != raw['albedo'].shape[:2] for image in raw.values()):
        raise ValueError('Source surface maps must align')
    return raw, sources


def rgb_anchor(values):
    if len(values) != 3 or any(isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 255 for v in values):
        raise ValueError('Anchor colour needs three sRGB code values')
    return linear(np.asarray(values, dtype=np.float32) / 255)


def prepare_package(name, spec, base, directory):
    raw, sources = channels(spec, base)
    role = spec['role']
    height = raw['height']
    hsv = cv2.cvtColor(raw['albedo'], cv2.COLOR_RGB2HSV)
    value = hsv[..., 2]
    luma = linear(raw['albedo']) @ np.float32([.2126, .7152, .0722])
    metallic = raw.get('metallic')
    anchor = rgb_anchor(spec['anchor_srgb'])
    wear = np.zeros_like(value)
    if role == 'paint':
        if metallic is None:
            raise ValueError('Paint exposes steel only with a donor metalness mask')
        wear = np.clip((metallic - .32) / .66, 0, 1)
    elif role == 'oxide':
        # Brown rust in the donor. Pale paint and bright yellow tint cannot
        # become metal damage simply because they are next to rust.
        hue, sat = hsv[..., 0], hsv[..., 1]
        brown = np.clip((hue - 6) / 10, 0, 1) * np.clip((57 - hue) / 17, 0, 1)
        wear = brown * np.clip((sat - .22) / .54, 0, 1) * np.clip((.96 - value) / .33, 0, 1)
        if metallic is not None:
            wear *= 1 - metallic
    elif role in ('steel', 'nonferrous') and metallic is not None and (metallic > .65).mean() < .65:
        raise ValueError(f'Most of {name} is not a suitable metal donor')

    # Reduce large light bands, keep medium and fine donor reflectance variation.
    sigma = max(1.5, value.shape[1] * .015)
    smooth = cv2.GaussianBlur(luma, (0, 0), sigma)
    texture = luma / np.maximum(.004, smooth)
    if role in ('steel', 'nonferrous'):
        # Use a zero-mean reflectance fluctuation around a metal F0 anchor.
        body_colour = np.clip(anchor * np.clip(texture, .58, 1.43)[..., None], .001, 1.)
    elif role == 'oxide':
        brown_rgb = linear(raw['albedo'])
        # Remove the donor paint colour from masked transitions. Retain the
        # sampled chromatic diversity on strong brown texels.
        brown_rgb = brown_rgb * .55 + (brown_rgb @ np.float32([.2126, .7152, .0722]))[..., None] * .45
        accepted = brown_rgb[wear > .25]
        if len(accepted) < .01 * wear.size:
            raise ValueError('Oxide donor has too little usable brown surface')
        scatter = np.clip(brown_rgb / np.maximum(.005, np.median(accepted, axis=0)), .24, 2.5)
        body_colour = anchor * (1 + .62 * (scatter - 1) * np.clip(wear * 2.8, 0, 1)[..., None])
    else:
        detail = np.clip(texture, .7, 1.35)
        # The paint layer stays painted even where the donor has worn through;
        # the same tracked wear mask reveals the separate steel BRDF there.
        if role == 'paint':
            detail = 1 + (detail - 1) * (1 - wear)
        body_colour = anchor * detail[..., None]

    # Normal-height bands are calibrated for the target material. With box
    # projection, using a scalar avoids rotating a tangent-space RGB normal
    # incorrectly at the blended projection boundaries.
    highpass = height - cv2.GaussianBlur(height, (0, 0), max(1.1, height.shape[1] * spec['height_band']))
    height_encoded = .5 + .44 * np.clip(highpass / max(.015, np.quantile(abs(highpass), .98)), -1, 1)
    low, high = (number(v, .08, .95, 'roughness bound') for v in spec['roughness_range'])
    if low >= high:
        raise ValueError('Roughness interval must be increasing')
    rough = raw['roughness']
    p5, p95 = np.quantile(rough, [.05, .95])
    varied = np.clip((rough - p5) / max(.015, p95 - p5), 0, 1)
    roughness = low + (high - low) * varied
    data = np.stack((roughness, height_encoded, wear), axis=-1)
    directory.mkdir(parents=True, exist_ok=True)
    maps = {}
    for key, image in {'albedo': srgb(body_colour), 'surface': data}.items():
        path = directory / (key + '.png')
        Image.fromarray(np.uint8(np.rint(np.clip(image, 0, 1) * 255))).save(path)
        maps[key] = {'path': str(path), 'sha256': file_hash(path)}
    return {'role': role, 'source': spec['source'], 'original_channels': sources, 'maps': maps,
            'tile_m': spec['tile_m'], 'relief_um': spec['relief_um'],
            'anchor_linear': anchor.tolist(), 'albedo_variation': spec['albedo_variation'],
            'albedo_median_luma': float(np.median(body_colour @ np.float32([.2126, .7152, .0722]))),
            'map_encoding': 'surface RGB = roughness, bounded signed height, wear mask',
            'statistics': {'donor_wear_mask_mean': float(wear.mean()), 'strong_wear_fraction': float((wear > .5).mean()),
                           'roughness_p05_p50_p95': np.quantile(roughness, [.05, .5, .95]).tolist()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, help='Root for texture paths, defaults to the spec directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec_file = args.spec.resolve()
    spec = json.loads(spec_file.read_text(encoding='utf8'))
    source_root = args.source_root.resolve() if args.source_root else spec_file.parent
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    catalogue = load_catalog()
    packs = {}
    for name, pack in spec['packages'].items():
        if not name.isidentifier():
            raise ValueError(f'Material pack key is not an identifier: {name}')
        packs[name] = prepare_package(name, pack, source_root, root / name)
    rows = {}
    for aid, config in spec['assets'].items():
        asset = catalogue[aid]
        if asset['mesh_hash'] != config['mesh_sha256']:
            raise ValueError(f'Source model changed since parts were audited: {aid}')
        used = {row[key] for row in config['regions'] for key in ('package', 'substrate', 'oxide') if key in row}
        reference = {'version': VERSION, 'status': 'review', 'packages': {name: packs[name] for name in sorted(used)},
                     'component_regions': config['regions'], 'recipe_sha256': file_hash(spec_file)}
        path = root / f'{aid}-component-materials.json'
        atomic_json(path, validate(reference))
        rows[asset['mesh_hash']] = {'asset_id': aid, 'reference_path': path.name,
                                    'reference_sha256': file_hash(path), 'evidence': config['evidence']}
    atomic_json(root / 'library.json', {'version': 'reference-material-library-v1', 'assets': rows})
    print(json.dumps({'packs': len(packs), 'assets': list(spec['assets']), 'output': str(root)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
