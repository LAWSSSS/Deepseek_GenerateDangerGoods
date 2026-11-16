"""Prepare repeatable numerical maps from the three generated surface tiles."""
from pathlib import Path
import json
import re
import shutil

import cv2
import numpy as np
from PIL import Image

from .common import atomic_json, file_hash, signature
from .surface_maps import prepare_uv_masks, save_image


VERSION = 'generated-pbr-v1'
WEAR_VERSION = 'generated-pbr-v2'
FORM_VERSION = 'generated-pbr-v3'
CYLINDER_VERSION = 'generated-pbr-v4'
HYDRAULIC_VERSION = 'generated-pbr-v5'
TRIPLE_WEAR_VERSION = 'generated-pbr-v6'
HYDRAULIC_VERSIONS = (HYDRAULIC_VERSION, TRIPLE_WEAR_VERSION)
CYLINDER_VERSIONS = (CYLINDER_VERSION, *HYDRAULIC_VERSIONS)
PACK_VERSIONS = (VERSION, WEAR_VERSION, FORM_VERSION, *CYLINDER_VERSIONS)
CHANNELS = {
    'oxide': ('oxidation', 'pitting', 'fine_relief'),
    'paint': ('chips', 'scratches', 'fine_relief'),
    'dust': ('coverage', 'granules', 'fine_relief'),
}
CYLINDER_CHANNELS = ('scuffs', 'exposed_steel', 'rust_nuclei')
REGIONS = ('steel', 'paint', 'oxide', 'nonferrous', 'weld')
LUMA = np.float32([.2126, .7152, .0722])


def smooth_tile(rgb):
    """Fade the wrap onto the continuous center of the same texture."""
    for axis in (0, 1):
        size = rgb.shape[axis]
        position = np.arange(size, dtype=np.float32)
        distance = np.minimum(position, size - 1 - position)
        weight = np.clip(distance / max(1, .055 * size), 0., 1.)
        weight = weight * weight * (3. - 2. * weight)
        weight = weight.reshape((-1, 1, 1) if axis == 0 else (1, -1, 1))
        # Sample the same center row on both sides of the wrap, then move
        # outward together to avoid a discontinuity in fine details.
        center = np.minimum(size - 1, size // 2 + distance.astype(int))
        rgb = weight * rgb + (1. - weight) * np.take(rgb, center, axis=axis)
    return rgb


def levels(field, low, high):
    a, b = np.quantile(field, (low, high))
    return np.clip((field - a) / max(float(b - a), 1e-5), 0., 1.)


def narrow_marks(residual):
    # An oriented opening retains short grooves and rejects isolated grains.
    size = max(7, int(round(residual.shape[0] / 88)) | 1)
    center = size // 2
    marks = np.zeros_like(residual)
    for angle in np.linspace(0., np.pi, 8, endpoint=False):
        delta = np.rint(center * np.array([np.cos(angle), np.sin(angle)])).astype(int)
        kernel = np.zeros((size, size), np.uint8)
        cv2.line(kernel, tuple(center - delta), tuple(center + delta), 1)
        marks = np.maximum(marks, cv2.morphologyEx(residual, cv2.MORPH_OPEN, kernel))
    return marks


def derive_maps(rgb, kind, version=VERSION):
    rgb = smooth_tile(np.float32(rgb) / 255.)
    y = rgb @ LUMA
    n = y.shape[0]
    fine = y - cv2.GaussianBlur(y, (0, 0), n * .0025)
    medium = cv2.GaussianBlur(y, (0, 0), n * .007)
    if kind == 'oxide':
        warm = cv2.GaussianBlur(rgb[..., 0] - rgb[..., 2], (0, 0), n * .007)
        coverage = levels(-medium + .24 * warm, .55, .975)
        pits = levels(-fine, .78, .997)
        channels = (np.maximum(coverage, .58 * pits), pits, levels(fine, .025, .975))
    elif kind == 'paint':
        chips = levels(-medium, .87, .998)
        residual = levels(-fine, .74, .996)
        marks = narrow_marks(residual * np.clip(1. - 1.35 * chips, 0., 1.))
        # Keep the extracted scratch mask sparse, including if a source tile
        # has few elongated features.
        peak = max(.12, float(np.quantile(marks, .995)))
        scratch = np.clip((marks - .04) / peak, 0., 1.)
        if version != VERSION:
            # Widen existing grooves by a few texels so downsampling retains
            # an abrasion mark without painting in additional scratches.
            halo = cv2.dilate(scratch, np.ones((5, 5), np.uint8))
            scratch = np.maximum(scratch, .72 * cv2.GaussianBlur(halo, (0, 0), 1.2))
        channels = (chips, scratch, levels(fine, .035, .965))
    else:
        coverage = levels(medium + .30 * np.maximum(fine, 0.), .26, .98)
        grains = levels(fine, .66, .995)
        channels = (coverage, grains, levels(fine, .04, .96))
    packed = smooth_tile(np.stack(channels, axis=-1))
    return np.uint8(np.round(255 * packed))


def map_statistics(values):
    values = np.float32(values) / 255.
    seam = {}
    for axis, label in enumerate(('vertical', 'horizontal')):
        field = np.moveaxis(values, axis, 0)
        seam[label] = float(np.mean(np.abs(field[0] - field[-1])))
    # A wide autocorrelation peak at half a tile would indicate duplication.
    normal = values[..., 0] - values[..., 0].mean()
    variance = max(1e-6, float(np.mean(normal**2)))
    half_tile = max(float(np.mean(normal * np.roll(normal, len(normal) // 2, axis=axis))) / variance
                    for axis in (0, 1))
    return {'seam_mean': seam, 'half_tile_correlation': half_tile,
            'channel_mean': values.mean(axis=(0, 1)).tolist(),
            'channel_fraction_gt_half': (values > .5).mean(axis=(0, 1)).tolist()}


def atlas_layouts(value):
    if not isinstance(value, dict):
        raise ValueError('expected atlas layouts keyed by source GLB SHA-256')
    for digest, entry in value.items():
        if not re.fullmatch(r'[a-f0-9]{64}', digest) or not isinstance(entry, dict):
            raise ValueError('invalid source atlas identity')
        if not isinstance(entry.get('material'), str) or not entry['material']:
            raise ValueError('source atlas needs its exact material name')
        tiles = entry.get('tiles')
        if entry.get('cylinder') == 'source_uv' and tiles is None:
            label = entry.get('protected_uv')
            if (not isinstance(label, list) or len(label) != 4 or
                    not all(type(x) in (float, int) and 0 <= x <= 1 for x in label) or
                    label[0] >= label[2] or label[1] >= label[3]):
                raise ValueError('invalid protected UV rectangle')
            continue
        if entry.get('cylinder') not in (None, 'normalized_y'):
            raise ValueError('unsupported cylinder coordinates')
        if entry.get('cylinder') == 'normalized_y':
            barrel = entry.get('barrel')
            if (not isinstance(barrel, list) or len(barrel) != 5
                    or not all(type(x) in (float, int) and -.5 <= x <= .5 for x in barrel)
                    or barrel[1] <= barrel[0] or not .01 <= barrel[4] <= .5):
                raise ValueError('invalid normalized barrel bounds, center or radius')
        if not isinstance(tiles, list) or len(tiles) != 16 or any(
                tile is not None and tile not in REGIONS for tile in tiles):
            raise ValueError('source atlas must declare its 4x4 regions; null preserves the source')
    return value


def build_pack(folder, sources, prompts, layouts, *, version=VERSION):
    if version not in PACK_VERSIONS:
        raise ValueError('unsupported generated_pbr manifest version')
    folder = Path(folder).resolve()
    manifest = folder / 'pack.json'
    if manifest.exists():
        raise FileExistsError(manifest)
    layout = atlas_layouts(layouts)
    if version not in CYLINDER_VERSIONS:
        layout = {digest: {k: v for k, v in entry.items() if k not in ('cylinder', 'barrel')}
                  for digest, entry in layout.items() if 'tiles' in entry}
    if version in (VERSION, WEAR_VERSION):
        layout = {digest: {**entry, 'tiles': ['paint' if name == 'weld' else name for name in entry['tiles']]}
                  for digest, entry in layout.items()}
    folder.mkdir(parents=True, exist_ok=True)
    maps = {}
    for kind, channels in CHANNELS.items():
        path = Path(sources[kind])
        prompt = prompts[kind].strip()
        if not prompt:
            raise ValueError(f'missing ImageGen prompt for {kind}')
        with Image.open(path) as source:
            source.load()
            if source.format != 'PNG' or source.width != source.height or not 512 <= source.width <= 2048:
                raise ValueError(f'{kind} requires a square 512–2048 pixel PNG')
            size = source.size
            rgb = np.asarray(source.convert('RGB'))
        original = folder / f'{kind}-imagegen.png'
        shutil.copy2(path, original)
        packed = derive_maps(rgb, kind, version)
        stats = map_statistics(packed)
        if max(stats['seam_mean'].values()) > .045 or stats['half_tile_correlation'] > .48:
            raise ValueError(f'{kind} has a visible seam or repeated large-scale pattern')
        baked = folder / f'{kind}-channels.png'
        save_image(Image.fromarray(packed), baked)
        maps[kind] = {'path': baked.name, 'sha256': file_hash(baked), 'size': list(size),
                      'color_space': 'Non-Color', 'channels': list(channels), 'statistics': stats,
                      'source': {'path': original.name, 'sha256': file_hash(original),
                                 'size': list(size), 'generator': 'ImageGen built-in', 'prompt': prompt}}
    if version in CYLINDER_VERSIONS:
        from .cylinder_wear import cylinder_wear_tile
        for kind in ('cylinder', 'hydraulic') if version in HYDRAULIC_VERSIONS else ('cylinder',):
            layers = 3 if version == TRIPLE_WEAR_VERSION and kind == 'hydraulic' else 1
            packed = cylinder_wear_tile(dense=kind == 'hydraulic', layers=layers)
            baked = folder / f'{kind}-channels.png'
            save_image(Image.fromarray(packed), baked)
            maps[kind] = {'path': baked.name, 'sha256': file_hash(baked), 'size': list(packed.shape[1::-1]),
                          'color_space': 'Non-Color', 'channels': list(CYLINDER_CHANNELS),
                          'statistics': map_statistics(packed), 'source': {'generator': 'deterministic cylinder wear'}}
            if layers > 1:
                maps[kind]['source']['layers'] = layers
    atomic_json(manifest, {'version': version, 'maps': maps, 'atlases': layout})
    return manifest


def load_pack(manifest):
    path = Path(manifest).resolve()
    raw = json.loads(path.read_text(encoding='utf8'))
    version = raw.get('version')
    expected = dict(CHANNELS)
    if version in CYLINDER_VERSIONS:
        expected['cylinder'] = CYLINDER_CHANNELS
    if version in HYDRAULIC_VERSIONS:
        expected['hydraulic'] = CYLINDER_CHANNELS
    if version not in PACK_VERSIONS or set(raw.get('maps', {})) != set(expected):
        raise ValueError('unsupported generated_pbr manifest')
    layouts = atlas_layouts(raw.get('atlases', {}))
    maps = {}
    for kind, channels in expected.items():
        entry = raw['maps'][kind]
        image = (path.parent / entry['path']).resolve()
        if not image.is_relative_to(path.parent) or entry.get('color_space') != 'Non-Color' or (
                entry.get('channels') != list(channels)):
            raise ValueError(f'invalid {kind} map description')
        digest = file_hash(image)
        if digest != entry.get('sha256'):
            raise ValueError(f'{kind} map SHA-256 mismatch')
        with Image.open(image) as tile:
            if tile.mode != 'RGB' or list(tile.size) != entry.get('size'):
                raise ValueError(f'{kind} map dimensions or channels changed')
        maps[kind] = {'path': str(image), 'sha256': digest}
    return {'version': version, 'manifest': str(path), 'maps': maps, 'atlases': layouts,
            'signature': signature(version, {kind: row['sha256'] for kind, row in maps.items()}, layouts)}


def finish_regions(asset, pack, cache):
    """Use authored atlas regions for exact meshes and conservative UV evidence otherwise."""
    layout = pack['atlases'].get(asset['mesh_hash'])
    if layout is None:
        regions, _ = prepare_uv_masks(asset, Path(cache) / 'scanned-uv')
        return regions
    key = signature(VERSION, asset['mesh_hash'], layout)
    folder = Path(cache) / 'generated-pbr-regions' / key
    folder.mkdir(parents=True, exist_ok=True)
    tiles = layout.get('tiles', [])
    weld_tiles = [i for i, tile in enumerate(tiles) if tile == 'weld']
    values = np.zeros((256, 256, 4 if weld_tiles else 3), np.uint8)
    channels = {'steel': 0, 'oxide': 0, 'paint': 1, 'nonferrous': 2}
    for i, tile in enumerate(tiles):
        if tile is not None:
            row, col = 3 - i // 4, i % 4
            region = values[row*64:(row+1)*64, col*64:(col+1)*64]
            if tile == 'weld':
                region[..., 1] = region[..., 3] = 255
            else:
                region[..., channels[tile]] = 255
    if layout.get('cylinder') == 'source_uv':
        l, b, r, t = layout['protected_uv']
        values[..., 1] = 255
        values[int(256 * (1 - t)):int(np.ceil(256 * (1 - b))),
               int(l * 256):int(np.ceil(r * 256)), 1] = 0
    path = folder / 'regions.png'
    if path.is_file():
        with Image.open(path) as old:
            matches = np.array_equal(np.asarray(old), values)
    else:
        matches = False
    if not matches:
        save_image(Image.fromarray(values), path)
    entry = {'coverage': (values > 0).mean(axis=(0, 1)).tolist(),
             'mask': str(path), 'mask_hash': file_hash(path), 'authored_atlas': True,
             'basis': 'authored_tiles_bound_to_exact_glb'}
    if weld_tiles:
        entry['weld_tiles'] = weld_tiles
    if layout.get('cylinder'):
        entry['cylinder'] = layout['cylinder']
        if 'barrel' in layout:
            entry['barrel'] = layout['barrel']
    return {'signature': signature(key, entry['mask_hash']),
            'materials': {layout['material']: entry}}
