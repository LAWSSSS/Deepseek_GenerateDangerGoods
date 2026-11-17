"""Bind licensed PBR channels to the existing reference shader for an offline trial."""
import argparse
import copy
import json
import os
from pathlib import Path

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2
import numpy as np
from PIL import Image

from display_batch.common import atomic_json, file_hash


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def prepare(spec, directory):
    directory.mkdir(parents=True, exist_ok=True)
    names = ('base', 'roughness', 'metallic', 'height', 'normal')
    paths = {key: Path(spec['channels'][key]).resolve(strict=True) for key in names}
    sources = {key: {'path': str(path), 'sha256': file_hash(path)} for key, path in paths.items()}
    with Image.open(paths['base']) as image:
        rgb = np.asarray(image.convert('RGB'), dtype=np.float32) / 255

    def scalar(key):
        image = cv2.imread(str(paths[key]), cv2.IMREAD_UNCHANGED)
        if image is None or image.shape[:2] != rgb.shape[:2]:
            raise ValueError(f'Missing or unaligned {key} channel for {spec["name"]}')
        data = np.float32(image[..., 0] if image.ndim == 3 else image)
        if np.issubdtype(image.dtype, np.integer):
            data /= np.iinfo(image.dtype).max
        if not np.isfinite(data).all():
            raise ValueError(f'Non-finite {key} channel for {spec["name"]}')
        return np.clip(data, 0, 1)

    rough, metal, height = (scalar(key) for key in ('roughness', 'metallic', 'height'))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = (hsv[..., i] for i in range(3))
    # Require brown colour and a non-metal response. Yellow paint and neutral
    # grey metal supply neither a verified oxide colour nor a corrosion mask.
    brown = np.clip((hue - 5) / 11, 0, 1) * np.clip((47 - hue) / 15, 0, 1)
    dark = np.clip((spec['oxide_value_limit'] - value) / .32, 0, 1)
    oxide = brown * dark * np.clip((saturation - .15) / .33, 0, 1) * (1 - metal)
    strength = float(spec.get('oxide_strength', 1.))
    if not 0 <= strength <= 1:
        raise ValueError('oxide_strength must be between 0 and 1')
    oxide *= strength
    # Preserve the scan's broader flakes, but gate bright exposed steel by
    # the supplied metalness map and a local contrast cue.
    fine = value - cv2.GaussianBlur(value, (0, 0), 5.)
    abrasion = np.clip((fine - .015) / .16, 0, 1) * metal * (1 - oxide)
    local_height = height - cv2.GaussianBlur(height, (0, 0), 16.)
    height_range = max(.08, float(np.quantile(np.abs(local_height), .98)))
    microheight = .5 + .42 * np.clip(local_height / height_range, -1, 1)
    # Skip extra deposition in this A/B so it cannot conceal the scan response.
    arrays = {'albedo': rgb, 'roughness': rough, 'oxide': oxide,
              'abrasion': abrasion, 'height': microheight, 'dust': np.zeros_like(value)}
    channels = {}
    for key, data in arrays.items():
        path = directory / (key + '.png')
        Image.fromarray(np.uint8(np.rint(np.clip(data, 0, 1) * 255))).save(path)
        channels[key] = {'path': path.name, 'sha256': file_hash(path)}
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    src = linear[oxide < .15]
    if not len(src):
        src = linear.reshape(-1, 3)
    record = {
        'version': 'reference-pbr-v1', 'status': 'review',
        'scale_m': spec['scale_m'], 'height_amplitude_m': .00008,
        'albedo_mean_linear': np.median(src, axis=0).tolist(),
        'maps': channels, 'provider': spec['provider'], 'name': spec['name'],
        'source_url': spec['source_url'], 'license': 'CC0-1.0',
        'original_channels': sources,
        'adaptation': 'source paint identity; vendor roughness and bounded high-pass height; estimated oxide and abrasion masks',
        'oxide_strength': strength,
        'statistics': {'mean_oxide': float(oxide.mean()), 'strong_oxide_fraction': float((oxide > .5).mean()),
                       'roughness_percentiles': np.percentile(rough, [5, 50, 95]).tolist()},
    }
    manifest = directory / 'reference.json'
    atomic_json(manifest, record)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--baseline-library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    if (root / 'library.json').exists():
        raise ValueError('Use a fresh output directory for a new material trial')
    specs = read(args.spec)
    baseline_path = args.baseline_library.resolve(strict=True)
    baseline = read(baseline_path)
    if baseline.get('version') != 'reference-material-library-v1':
        raise ValueError('Unsupported source material library')
    rows = {}
    for spec in specs:
        manifest = prepare(spec, root / spec['id'])
        matching = [item for item in baseline['assets'].items() if item[1]['asset_id'] in spec['assets']]
        if len(matching) != len(spec['assets']):
            raise ValueError(f'Unaudited source assets in {spec["id"]}')
        for digest, original in matching:
            if digest in rows:
                raise ValueError('Multiple material packs assigned to one asset')
            row = copy.deepcopy(original)
            row.update(reference_path=str(manifest), reference_sha256=file_hash(manifest),
                       surface_package=spec['id'], surface_selection_basis='licensed_vendor_pbr_channels')
            row['evidence']['flat_texture_role'] = 'vendor_pbr_maps_and_estimated_oxide_masks'
            if allowed := spec.get('only_materials', {}).get(row['asset_id']):
                row['evidence']['only_source_materials'] = allowed
            rows[digest] = row
    atomic_json(root / 'library.json', {'version': 'reference-material-library-v1', 'assets': rows})


if __name__ == '__main__':
    main()
