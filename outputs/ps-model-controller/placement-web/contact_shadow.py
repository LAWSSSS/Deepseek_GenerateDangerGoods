"""Bounded screen-space contact shadow for the manual placement preview.

The shadow is an appearance estimate, not a measured 3-D contact shadow.  It
is derived only from the object alpha, the selected support mask and the
foreground mask; it never changes the background outside those regions.
"""
import cv2
import numpy as np
from PIL import Image


def make_contact_shadow(support, foreground, alpha, x, y, side, light_direction=None):
    """Return a local RGBA shadow layer and evidence, or ``(None, evidence)``.

    Parameters are full-frame masks except ``alpha`` (the square object
    alpha).  ``x/y`` are the object's top-left in frame pixels.
    """
    support = np.asarray(support, dtype=bool)
    foreground = np.asarray(foreground, dtype=bool)
    alpha = np.asarray(alpha, dtype=np.uint8)
    if support.shape != foreground.shape or alpha.ndim != 2:
        raise ValueError('contact shadow masks have incompatible shapes')
    height, width = support.shape
    if side <= 0 or x < 0 or y < 0 or x + side > width or y + side > height:
        return None, {'status': 'unavailable', 'reason': '物体超出画面，无法生成接触阴影'}
    object_mask = alpha >= 128
    box = Image.fromarray(object_mask.astype(np.uint8) * 255).getbbox()
    if box is None:
        return None, {'status': 'unavailable', 'reason': '物体透明度为空'}
    bbox_w, bbox_h = box[2] - box[0], box[3] - box[1]
    if bbox_w < 4 or bbox_h < 4:
        return None, {'status': 'unavailable', 'reason': '物体过小，无法生成接触阴影'}
    pad = max(8, round(.28 * side))
    l, t = max(0, x - pad), max(0, y - pad)
    r, b = min(width, x + side + pad), min(height, y + side + pad)
    local_support = support[t:b, l:r]
    local_front = foreground[t:b, l:r]
    local_object = np.zeros((b - t, r - l), dtype=np.uint8)
    ox, oy = x - l, y - t
    local_object[oy:oy + side, ox:ox + side] = object_mask.astype(np.uint8) * 255

    # The bottom band of the object is the only plausible contact region.
    band = max(2, round(.13 * bbox_h))
    contact = np.zeros_like(local_object)
    contact[oy + box[3] - band:oy + box[3], ox + box[0]:ox + box[2]] = \
        local_object[oy + box[3] - band:oy + box[3], ox + box[0]:ox + box[2]]
    if contact.sum() < 12:
        return None, {'status': 'insufficient_samples', 'reason': '接触带像素不足'}

    direction = np.asarray(light_direction if light_direction is not None else [0.0, 1.0], dtype=float)
    if direction.shape != (2,) or not np.all(np.isfinite(direction)) or np.linalg.norm(direction) < 1e-4:
        direction = np.array([0.0, 1.0])
    direction /= np.linalg.norm(direction)
    shift = max(2.0, .055 * side)
    # Shadow falls opposite the estimated screen-space light direction.
    matrix = np.float32([[1, 0, -direction[0] * shift], [0, 1, -direction[1] * shift]])
    shifted = cv2.warpAffine(contact, matrix, (r - l, b - t), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    sigma = max(2.0, .035 * side)
    blurred = cv2.GaussianBlur(shifted.astype(np.float32) / 255.0, (0, 0), sigma)
    shadow = np.clip(blurred * .80, 0, 1)
    shadow *= local_support.astype(np.float32)
    shadow *= (~local_front).astype(np.float32)
    shadow[local_object > 0] = 0
    if shadow.sum() < 1:
        return None, {'status': 'unavailable', 'reason': '承托面内没有可用阴影像素'}
    mask = np.rint(np.clip(shadow, 0, 1) * 255).astype(np.uint8)
    if not np.any(mask):
        return None, {'status': 'unavailable', 'reason': '阴影透明度为空'}
    layer = Image.new('RGBA', (r - l, b - t), (0, 0, 0, 0))
    layer.putalpha(Image.fromarray(mask))
    return layer, {
        'status': 'estimated',
        'method': 'screen_space_contact_from_alpha_and_support',
        'crop': [int(l), int(t), int(r), int(b)],
        'samples': int(np.count_nonzero(mask)),
        'max_alpha': int(mask.max()),
        'shift_px': round(float(shift), 2),
        'direction_screen': [round(float(direction[0]), 3), round(float(direction[1]), 3)],
        'limitation': '屏幕空间近似，不是三维接触阴影或物理支撑证据',
    }
