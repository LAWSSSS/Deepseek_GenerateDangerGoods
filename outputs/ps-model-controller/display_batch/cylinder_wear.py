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


def cylinder_wear_tile(size=1024, *, dense=False, layers=1):
    """Return uint8 channels for scrape wear, exposed steel, and rust nuclei.

    X follows the axis, and Y wraps once around the circumference. Strokes are
    grouped with related directions and interrupted by sparse clean regions.
    Additional layers place independent clusters over the original pattern.
    """
    if type(size) is not int or not 1 <= size <= 2048:
        raise ValueError('cylindrical wear tile size must be an integer from 1 to 2048')
    if type(layers) is not int or not 1 <= layers <= 3:
        raise ValueError('cylindrical wear layers must be an integer from 1 to 3')
    maps = _wear_layer(442763, dense)
    for layer in range(1, layers):
        np.maximum(maps, _wear_layer(442763 + layer, dense), out=maps)
    if size != maps.shape[0]:
        interpolation = cv2.INTER_AREA if size < maps.shape[0] else cv2.INTER_LINEAR
        maps = cv2.resize(maps, (size, size), interpolation=interpolation)
    return maps


def _wear_layer(seed, dense):
    resolution = 1024
    rng = np.random.default_rng(seed)
    abrasion = np.zeros((resolution, resolution), np.uint8)
    exposed = np.zeros_like(abrasion)
    pits = np.zeros_like(abrasion)
    corroded = []

    for _ in range(76 if dense else 53):
        origin = rng.uniform(0, resolution, 2)
        heading = rng.normal(0, .25) if rng.random() < .77 else rng.uniform(-1.4, 1.4)
        spread = rng.uniform(17, 68, 2)
        for _ in range(int(rng.integers(11, 25))):
            center = origin + rng.normal(0, spread, 2)
            center[1] %= resolution
            length = float(np.clip(rng.lognormal(3.84, .62), 11, 165))
            deep = rng.random() < (.34 if dense else .26)
            width = rng.uniform(2.2, 4.3) if deep else rng.uniform(.85, 2.6)
            shape = _scratch(rng, center, length, heading + rng.normal(0, .115), width)
            _draw_wrapped(abrasion, shape, rng.integers(110, 231))
            if deep:
                inner = center + (shape-center) * .82
                _draw_wrapped(exposed, inner, rng.integers(126, 241))
                if rng.random() < .29:
                    corroded.append(center)

    centers = [*corroded, *rng.uniform(0, resolution, (105 if dense else 51, 2))]
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

    return np.stack((abrasion, np.minimum(exposed, abrasion), pits), axis=-1)


def handling_tile():
    """Object-space handling marks wide enough for a monitoring-camera projection."""
    size = 1024
    rng = np.random.default_rng(702533)
    scratches = np.zeros((size, size), np.uint8)
    for _ in range(44):
        origin = rng.uniform(26, size - 26, 2)
        angle = rng.uniform(-1.4, 1.4) if rng.random() < .43 else rng.normal(0., .24)
        for _ in range(int(rng.integers(6, 13))):
            point = origin + rng.normal(0, (62, 29), 2)
            point[1] %= size
            stroke = _scratch(rng, point, rng.uniform(23, 148), angle + rng.normal(0., .16), rng.uniform(3.4, 9.4))
            _draw_wrapped(scratches, stroke, rng.integers(122, 255))

    def periodic_noise(cells):
        grid = np.pad(rng.random((cells, cells), dtype=np.float32), 2, mode='wrap')
        step = size // cells
        large = cv2.resize(grid, ((cells + 4) * step,) * 2, interpolation=cv2.INTER_CUBIC)
        return large[2*step:2*step+size, 2*step:2*step+size]

    oil = .62 * periodic_noise(8) + .26 * periodic_noise(32) + .12 * periodic_noise(128)
    mineral = .45 * periodic_noise(16) + .38 * periodic_noise(64) + .17 * periodic_noise(128)
    grime = np.clip((oil - .48) * 4.8, 0, 1) ** 1.4
    flakes = np.clip((mineral - .52) * 4.4, 0, 1)
    return np.stack((scratches, np.uint8(np.rint(grime * 255)), np.uint8(np.rint(flakes * 255))), axis=-1)
