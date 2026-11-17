"""Render a restrained contact layer from projected near-ground mesh patches."""
import cv2
import numpy as np
from PIL import Image


MIN_SUPPORT = .90


def projected_footprint(polygons, side):
    """Rasterize separate contact faces in the same square as the render."""
    mask = np.zeros((side, side), np.uint8)
    for polygon in polygons:
        coords = np.asarray(polygon, dtype=float)
        cv2.fillConvexPoly(mask, np.rint((coords * side - .5) * 16).astype(np.int32), 255, shift=4)
    return Image.fromarray(mask)


def contact_clusters(footprint):
    count, labels = cv2.connectedComponents(footprint.astype(np.uint8))
    return [np.nonzero(labels == index) for index in range(1, count)]


def contact_coverage(clusters, support):
    return [float(np.mean(support[patch])) for patch in clusters]


def make_contact_shadow(footprint, support, foreground, alpha, major_px):
    """Return a tight contact layer, only on the supported, visible surface."""
    if not (footprint.shape == support.shape == foreground.shape == alpha.shape):
        raise ValueError('contact shadow masks have incompatible shapes')
    contact = footprint.astype(np.float32)
    if not contact.any():
        return None, {'status': 'unavailable', 'reason': '没有可投影的网格接触斑'}
    sigma = float(np.clip(.010 * major_px, .8, 3.5))
    close = cv2.GaussianBlur(contact, (0, 0), sigma)
    soft = cv2.GaussianBlur(contact, (0, 0), sigma * 2.5)
    linear_loss = .55 * close + .16 * soft
    # A black transparent layer acts on display RGB, so encode the linear
    # attenuation as a bounded display-space opacity for preview and PSD.
    opacity = 1 - np.power(1 - np.clip(linear_loss, 0, .63), 1 / 2.2)
    opacity *= support & ~foreground & (alpha < 128)
    mask = Image.fromarray(np.uint8(np.rint(opacity * 255)))
    bounds = mask.getbbox()
    if bounds is None:
        return None, {'status': 'unavailable', 'reason': '接触斑周围没有露出的承托像素'}
    layer = Image.new('RGBA', (bounds[2] - bounds[0], bounds[3] - bounds[1]))
    layer.putalpha(mask.crop(bounds))
    return layer, {
        'status': 'estimated', 'method': 'projected_mesh_contact_on_support',
        'crop': list(bounds), 'samples': int(np.count_nonzero(np.asarray(mask))),
        'max_alpha': mask.getextrema()[1], 'sigma_px': round(sigma, 3),
    }
