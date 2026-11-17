"""Encode a verified compositor result as isolated Photoshop display layers."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .photoshop import file_hash


def display_object_layer(final, plate, optical_alpha):
    """Convert the camera composite to straight RGBA over its sRGB shadow plate.

    Opacity is the Photoshop display encoding. The original optical coverage
    and geometric alpha are preserved separately for audit.
    """
    final = final.astype(np.float32) / 255
    plate = plate.astype(np.float32) / 255
    difference = final - plate
    needed = np.where(difference > 0, difference / np.maximum(1 - plate, 1e-6),
                      -difference / np.maximum(plate, 1e-6)).max(axis=2)
    if np.any((optical_alpha <= 0) & (np.abs(difference).max(axis=2) > 1.01 / 255)):
        raise ValueError('composite_changed_outside_optical_layer')
    encoded_alpha = np.ceil(np.clip(np.maximum(optical_alpha, needed), 0, 1) * 255 - 1e-5)
    encoded_alpha[optical_alpha <= 0] = 0
    opacity = encoded_alpha.astype(np.float32) / 255
    straight = plate + difference / np.maximum(opacity[..., None], 1e-6)
    straight[opacity == 0] = 0
    rgb = np.rint(np.clip(straight, 0, 1) * 255).astype(np.uint8)
    return np.dstack((rgb, encoded_alpha.astype(np.uint8)))


def save_bundle(folder, scene, output, bounds, plate, optical_alpha, geometry_alpha, foreground, checks):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    x, y, right, bottom = bounds
    window = np.s_[y:bottom, x:right]
    background = np.asarray(scene['frame'].convert('RGB'))
    with Image.open(output) as image:
        final = np.asarray(image.convert('RGB'))
    if final.shape != background.shape:
        raise ValueError('compositor_export_canvas_mismatch')
    plate = np.rint(np.clip(plate, 0, 1) * 255).astype(np.uint8)
    object_local = display_object_layer(final[window], plate, optical_alpha)
    shape = background.shape[:2]
    obj = np.zeros((*shape, 4), dtype=np.uint8)
    obj[window] = object_local
    shadow = np.zeros_like(obj)
    shadow[window] = np.dstack((plate, np.any(plate != background[window], axis=2).astype(np.uint8) * 255))
    fg = np.zeros(shape, dtype=np.uint8)
    fg[window] = foreground.astype(np.uint8) * 255
    files = {}
    for name, pixels in (('background', background), ('object_rgba', obj), ('shadow_rgba', shadow),
                         ('foreground_mask', fg), ('protected_mask', scene['protected'].astype(np.uint8) * 255)):
        path = folder / (name + '.png')
        Image.fromarray(pixels).save(path)
        files[name] = path
    np.savez_compressed(folder / 'alpha-evidence.npz', geometric=geometry_alpha, optical=optical_alpha, bounds=bounds)
    files['expected_png'] = Path(output).resolve()
    hashes = {name: file_hash(path) for name, path in files.items()}
    request = {**{name: str(path.resolve()) for name, path in files.items()}, 'object_id': 'object',
               'geometry_signature': file_hash(folder / 'alpha-evidence.npz'), 'source_hashes': hashes}
    layer_manifest = folder / 'layers.json'
    layer_manifest.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding='utf8')
    proof = folder / 'composition.json'
    proof.write_text(json.dumps({
        'version': 'verified-composition-v1', 'layers': layer_manifest.name,
        'layer_manifest_sha256': file_hash(layer_manifest), 'program_checks': checks,
        'alpha_evidence_sha256': request['geometry_signature'], 'color_encoding': 'sRGB',
        'display_opacity_basis': 'camera output over isolated shadow plate',
    }, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    return proof
