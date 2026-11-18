"""Bind generated-view material evidence to an unchanged source asset.

RGB references supply appearance patterns, not measured BRDF or hidden geometry.
Source-colour anchors remove baked body illumination while retaining paint identity.
"""
import copy
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .common import file_hash, signature
from .photography import linear
from .surface_maps import glb_images

VERSION = 'reference-material-binding-v1'
MAPS = ('albedo', 'roughness', 'oxide', 'abrasion', 'height', 'dust')
SNAPSHOT_VERSION = 'reference-material-snapshot-v1'


class ReferenceInputChanged(RuntimeError):
    """A running batch must stop before mixing different material revisions."""


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def load_library(path):
    path = Path(path).resolve()
    record = read(path)
    if record.get('version') != 'reference-material-library-v1' or not isinstance(record.get('assets'), dict):
        raise ValueError('unsupported reference material library')
    return {**record, 'manifest': str(path), 'manifest_sha256': file_hash(path)}


def validate_reference(reference, base=None):
    from .component_pbr import VERSION as COMPONENT_VERSION, validate as validate_components
    if reference.get('version') == COMPONENT_VERSION:
        result = validate_components(reference, base)
        result.pop('binding_signature', None)
        result['binding_signature'] = signature(VERSION, result, file_hash(__file__))
        return result
    result = copy.deepcopy(reference)
    if result.get('version') != 'reference-pbr-v1' or result.get('status') not in ('ready', 'estimated', 'review'):
        raise ValueError('reference material is missing estimated PBR evidence')
    scale = result.get('scale_m')
    if isinstance(scale, bool) or not isinstance(scale, (float, int)) or not math.isfinite(scale) or not .005 <= scale <= .5:
        raise ValueError('reference texture scale outside .005 to .5 metres')
    amplitude = result.get('height_amplitude_m', .00008)
    if isinstance(amplitude, bool) or not isinstance(amplitude, (float, int)) or not math.isfinite(amplitude) or not 0 <= amplitude <= .00008:
        raise ValueError('estimated microrelief outside 0 to 80 micrometres')
    if set(result.get('maps', {})) != set(MAPS):
        raise ValueError('reference PBR requires six aligned maps')
    shape = None
    for kind, entry in result['maps'].items():
        path = Path(entry['path'])
        if not path.is_absolute():
            if base is None: raise ValueError('relative reference texture has no manifest base')
            path = Path(base) / path
        path = path.resolve()
        if file_hash(path) != entry['sha256']:
            raise ValueError('reference map hash changed: ' + kind)
        with Image.open(path) as im:
            if shape is not None and im.size != shape: raise ValueError('reference maps are not aligned')
            shape = im.size
            if max(shape) > 2048 or min(shape) < 16: raise ValueError('reference texture resolution unsupported')
        entry['path'] = str(path)
        entry['color_space'] = 'sRGB' if kind == 'albedo' else 'Non-Color'
    result.update(estimated=True, height_amplitude_m=amplitude, basecolor_mode='preserve_identity',
                  roughness_encoding='absolute_0_1_estimate', height_encoding='signed_midpoint_0.5')
    result['binding_signature'] = signature(VERSION, result, file_hash(__file__))
    return result


def lookup_reference(library_or_path, mesh_hash):
    library = load_library(library_or_path) if not isinstance(library_or_path, dict) else library_or_path
    if library.get('version') != 'reference-material-library-v1':
        raise ValueError('unsupported reference material library')
    row = library['assets'].get(mesh_hash)
    if row is None:
        raise ValueError('reference material has no binding for this source model')
    path = Path(row['reference_path'])
    if not path.is_absolute(): path = Path(library['manifest']).parent / path
    if file_hash(path) != row['reference_sha256']:
        raise ValueError('reference entry hash changed')
    result = validate_reference(read(path), path.parent)
    result.update(source_mesh_hash=mesh_hash, asset_id=row['asset_id'],
                  identity_evidence=copy.deepcopy(row.get('evidence', {})))
    result['binding_signature'] = signature(result['binding_signature'], mesh_hash, row)
    return result


def freeze_library(path, selected):
    """Validate only this batch's mesh bindings and retain their exact content.

    The snapshot is metadata, not a mutable library lookup. Every use verifies
    its recorded input bytes; changing a file requires a new batch directory.
    """
    library = load_library(path)
    entries = {}
    files = {library['manifest']: {'path':library['manifest'], 'sha256':library['manifest_sha256'], 'role':'library'}}
    for mesh_hash, asset_id in sorted(selected.items()):
        reference = lookup_reference(library, mesh_hash)
        if str(reference['asset_id']) != str(asset_id):
            raise ValueError('reference asset ID differs from selected mesh: ' + str(asset_id))
        entries[mesh_hash] = reference
        row = library['assets'][mesh_hash]
        entry_path = Path(row['reference_path'])
        if not entry_path.is_absolute():entry_path = Path(library['manifest']).parent / entry_path
        entry_path = str(entry_path.resolve())
        records = [(entry_path, row['reference_sha256'], 'entry')]
        if 'packages' in reference:
            from .component_pbr import texture_records
            records += texture_records(reference)
        else:
            records += [(entry['path'], entry['sha256'], kind) for kind, entry in reference['maps'].items()]
        for filename, digest, role in records:
            if filename in files and files[filename]['sha256'] != digest:
                raise ValueError('selected references disagree on shared input bytes')
            files[filename] = {'path':filename, 'sha256':digest, 'role':role}
    if not entries:
        raise ValueError('reference snapshot requires selected assets')
    result = {'version':SNAPSHOT_VERSION, 'library':library['manifest'],
              'library_sha256':library['manifest_sha256'], 'entries':entries,
              'files':[files[key] for key in sorted(files)]}
    result['signature'] = signature(result)
    verify_snapshot(result)
    return result


def verify_snapshot(snapshot):
    if (snapshot.get('version') != SNAPSHOT_VERSION or
            signature({key:value for key,value in snapshot.items() if key!='signature'}) != snapshot.get('signature')):
        raise ReferenceInputChanged('reference snapshot metadata changed')
    for record in snapshot['files']:
        try:actual = file_hash(record['path'])
        except OSError as exc:
            raise ReferenceInputChanged('reference input unavailable: ' + record['path']) from exc
        if actual != record['sha256']:
            raise ReferenceInputChanged('reference input changed during batch: ' + record['path'])


def frozen_reference(snapshot, mesh_hash, manifest):
    if snapshot is None:
        raise ReferenceInputChanged('reference material preflight was not completed')
    if str(Path(manifest).resolve()) != snapshot['library']:
        raise ReferenceInputChanged('reference library changed after preflight')
    verify_snapshot(snapshot)
    if mesh_hash not in snapshot['entries']:
        raise ReferenceInputChanged('asset was not included in reference preflight')
    return copy.deepcopy(snapshot['entries'][mesh_hash])


def compatible_snapshot(current, previous):
    """A pilot may contain fewer assets, but shared asset materials must match."""
    if previous.get('version') != SNAPSHOT_VERSION or current['library_sha256'] != previous.get('library_sha256'):
        return False
    if signature({key:value for key,value in previous.items() if key!='signature'}) != previous.get('signature'):
        return False
    common = set(current['entries']) & set(previous.get('entries', {}))
    return all(current['entries'][key] == previous['entries'][key] for key in common)


def identity_colours(asset, materials):
    """Linear colour anchors per known source region; never include background.

    These are body colour estimates, not measured reflectance. Protected inserts
    still use their original shader; image pixels cannot identify an alloy.
    """
    source, image = glb_images(asset['mesh'])
    output = copy.deepcopy(materials)
    for index, material in enumerate(source):
        name = material.get('name') or f'Material_{index}'
        entry = output.get(name)
        if not entry or not entry.get('eligible') or entry.get('protected'): continue
        pbr = material.get('pbrMetallicRoughness', {})
        factor = np.float32(pbr.get('baseColorFactor', [1, 1, 1, 1])[:3])
        try: base = image(pbr.get('baseColorTexture'))
        except (ValueError, KeyError, IndexError, OSError):
            entry.update(eligible=False, skip_reason='unsupported_source_colour_binding_preserved')
            continue
        rgb = linear(np.float32(base) / 255) * factor if base is not None else factor.reshape(1, 1, 3)
        if entry.get('mask'):
            with Image.open(entry['mask']) as im: mask = np.asarray(im.convert('RGB'))
            mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) / 255.
        else:
            mask = np.broadcast_to(np.float32(entry.get('constant', [0, 0, 0])[:3]), rgb.shape)
        labels = ('metal', 'paint', 'nonferrous') if entry.get('authored_atlas') else ('metal', 'paint', 'existing_oxide')
        colours, counts = {}, {}
        for channel, label in enumerate(labels):
            pixels = rgb[mask[..., channel] > .8]
            if pixels.size:
                # Exclude clipped hotspots/deep seams; preserve hue distribution.
                y = pixels @ np.float32([.2126, .7152, .0722])
                lo, hi = np.quantile(y, [.15, .85])
                selected = pixels[(y >= lo) & (y <= hi)]
                colours[label] = np.clip(np.median(selected, axis=0), .005, .85).tolist()
                counts[label] = int(len(selected))
        entry['source_identity_by_region'] = colours
        entry['source_identity_basis'] = {'type': 'trimmed_source_colour_estimate_not_measured_reflectance',
                                          'samples': counts, 'preserves_protected_shader': True}
        missing = [label for channel, label in enumerate(labels)
                   if entry.get('coverage', [0, 0, 0])[channel] > 0 and label not in colours]
        if missing:
            entry.update(eligible=False, skip_reason='source_colour_estimate_unavailable:' + ','.join(missing))
    return output
