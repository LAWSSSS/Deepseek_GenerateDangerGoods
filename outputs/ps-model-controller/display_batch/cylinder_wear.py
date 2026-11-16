"""Procedural scratches and pinpits on a periodic cylinder unwrap."""

import cv2
import numpy as np


def _draw_wrapped(image, polygon, shade):
    """Paint both sides of the angular seam with the same stroke."""
    size = image.shape[0]
    offsets = [0]
    if polygon[:, 1].min() < 0:
        offsets.append(size)
    if polygon[:, 1].max() >= size - 1:
        offsets.append(-size)
    for offset in offsets:
        points = np.rint((polygon + (0, offset)) * 16).astype(np.int32)
        cv2.fillPoly(image, [points], int(shade), cv2.LINE_AA, shift=4)


def _scratch(rng, center, length, angle, width):
    t = np.linspace(-1, 1, 11)
    tangent = np.array([np.cos(angle), np.sin(angle)])
    normal = np.array([-tangent[1], tangent[0]])
    sweep = rng.uniform(-.065, .065) * length
    path = center + t[:, None] * (.5 * length) * tangent + (t*t * sweep)[:, None] * normal
    taper = (1 - t*t)**.7
    radius = .5 * width * taper * rng.uniform(.75, 1.25, len(t))
    left = path + radius[:, None] * normal
    right = path - radius[:, None] * normal
    return np.concatenate((left, right[::-1]))


def cylinder_wear_tile(size=1024):
    """Return uint8 channels for scrape wear, exposed steel, and rust nuclei.

    X follows the axis, and Y wraps once around the circumference. Strokes are
    grouped with related directions and interrupted by sparse clean regions.
    """
    if type(size) is not int or not 1 <= size <= 2048:
        raise ValueError('cylindrical wear tile size must be an integer from 1 to 2048')
    resolution = 1024
    rng = np.random.default_rng(442763)
    abrasion = np.zeros((resolution, resolution), np.uint8)
    exposed = np.zeros_like(abrasion)
    pits = np.zeros_like(abrasion)
    corroded = []

    for _ in range(53):
        origin = rng.uniform(0, resolution, 2)
        heading = rng.normal(0, .25) if rng.random() < .77 else rng.uniform(-1.4, 1.4)
        spread = rng.uniform(17, 68, 2)
        for _ in range(int(rng.integers(11, 25))):
            center = origin + rng.normal(0, spread, 2)
            center[1] %= resolution
            length = float(np.clip(rng.lognormal(3.84, .62), 11, 165))
            deep = rng.random() < .26
            width = rng.uniform(2.2, 4.3) if deep else rng.uniform(.85, 2.6)
            shape = _scratch(rng, center, length, heading + rng.normal(0, .115), width)
            _draw_wrapped(abrasion, shape, rng.integers(110, 231))
            if deep:
                inner = center + (shape-center) * .82
                _draw_wrapped(exposed, inner, rng.integers(126, 241))
                if rng.random() < .29:
                    corroded.append(center)

    centers = [*corroded, *rng.uniform(0, resolution, (51, 2))]
    angles = np.linspace(0, 2*np.pi, 10, endpoint=False)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    for center in centers:
        width = rng.uniform(5, 28, 2)
        for _ in range(int(rng.integers(9, 37))):
            point = center + rng.normal(0, width, 2)
            point[1] %= resolution
            radius = float(np.clip(rng.lognormal(.65, .49), .7, 6))
            outline = point + directions * (radius * rng.uniform(.52, 1.35, (len(angles), 1)))
            _draw_wrapped(pits, outline, rng.integers(86, 238))

    maps = np.stack((abrasion, np.minimum(exposed, abrasion), pits), axis=-1)
    if size != resolution:
        interpolation = cv2.INTER_AREA if size < resolution else cv2.INTER_LINEAR
        maps = cv2.resize(maps, (size, size), interpolation=interpolation)
    return maps
