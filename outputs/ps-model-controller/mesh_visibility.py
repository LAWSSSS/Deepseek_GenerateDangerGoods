"""Conservative per-pixel source visibility using depth and surface provenance."""
import cv2
import numpy as np

VERSION = "visibility-subpixel-v3"


def project(points, camera, size):
    eye = np.asarray(camera["eye"], dtype=float)
    forward = np.asarray(camera.get("target", [0, 0, 0]), dtype=float) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(camera.get("up", [0, 0, 1]), dtype=float))
    if np.linalg.norm(right) < 1e-7:
        raise ValueError("Camera up is parallel to its direction")
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    delta = np.asarray(points, dtype=float) - eye
    z = delta @ forward
    scale = size / float(camera["orthographic_scale"]) if camera.get("orthographic_scale") else \
        .5 * size / np.tan(np.deg2rad(camera.get("fov_deg", 40)) * .5) / np.maximum(z, 1e-8)
    x = size / 2 + scale * (delta @ right) - size * camera.get("shift_x", 0)
    y = size / 2 - scale * (delta @ up) + size * camera.get("shift_y", 0)
    return np.stack((x, y), axis=-1), z


def fragments(triangles, camera, size):
    triangles = np.asarray(triangles, dtype=float)
    xy, depths = project(triangles, camera, size)
    labels = np.full((size, size), -1, np.int32)
    zbuffer = np.full((size, size), np.inf, np.float64)
    weights = np.zeros((size, size, 3), np.float32)
    for i, (p, z) in enumerate(zip(xy, depths)):
        if not np.all(np.isfinite(p)) or np.any(z <= .01):
            continue
        lo = np.maximum(np.floor(p.min(axis=0)).astype(int), 0)
        hi = np.minimum(np.ceil(p.max(axis=0)).astype(int), size - 1)
        if np.any(lo > hi):
            continue
        a, b, c = p
        denom = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
        if abs(denom) < 1e-10:
            continue
        yy, xx = np.mgrid[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
        xx, yy = xx + .5, yy + .5
        w0 = ((b[1]-c[1])*(xx-c[0]) + (c[0]-b[0])*(yy-c[1])) / denom
        w1 = ((c[1]-a[1])*(xx-c[0]) + (a[0]-c[0])*(yy-c[1])) / denom
        w2 = 1 - w0 - w1
        inside = (w0 >= -1e-8) & (w1 >= -1e-8) & (w2 >= -1e-8)
        bary = np.stack((w0, w1, w2), axis=-1)
        if not camera.get("orthographic_scale"):
            bary = bary / z
            bary /= np.maximum(bary.sum(axis=-1, keepdims=True), 1e-20)
        depth = bary @ z
        region = np.s_[lo[1]:hi[1]+1, lo[0]:hi[0]+1]
        replace = inside & (depth < zbuffer[region])
        zbuffer[region][replace] = depth[replace]
        labels[region][replace] = i
        weights[region][replace] = bary[replace]
    return labels, zbuffer, weights


def raster(triangles, ids, camera, size=256):
    labels, _, _ = fragments(triangles, camera, size)
    result = np.zeros(labels.shape, np.int32)
    mask = labels >= 0
    result[mask] = np.asarray(ids, dtype=np.int32)[labels[mask]] + 1
    return result


class SourceProjection:
    """Index every projected triangle, including triangles smaller than a pixel."""

    tile = 8

    def __init__(self, triangles, camera, source_alpha, reference_alpha):
        self.camera = camera
        self.size = max(source_alpha.shape)
        self.xy, self.depth = project(triangles, camera, self.size)
        self.mask = self.alpha_mask(source_alpha) & self.alpha_mask(reference_alpha)
        self.columns = (self.size + self.tile - 1) // self.tile
        self.bins = [[] for _ in range(self.columns**2)]
        self.complete = bool(np.all(np.isfinite(self.xy)) and np.all(self.depth > .01))
        if not self.complete:
            return
        low = np.maximum(np.floor(self.xy.min(axis=1) / self.tile).astype(int), 0)
        high = np.minimum(np.floor(self.xy.max(axis=1) / self.tile).astype(int), self.columns - 1)
        for face, ((left, top), (right, bottom)) in enumerate(zip(low, high)):
            for y in range(top, bottom + 1):
                for x in range(left, right + 1):
                    self.bins[y * self.columns + x].append(face)

    def alpha_mask(self, alpha):
        return cv2.resize(np.asarray(alpha, dtype=np.uint8), (self.size, self.size),
                          interpolation=cv2.INTER_AREA) >= 128

    def observed(self, points):
        coords, depth = project(points, self.camera, self.size)
        pixels = np.floor(coords).astype(int)
        inside = np.flatnonzero(np.all((pixels >= 0) & (pixels < self.size), axis=1))
        visible = np.zeros(len(points), dtype=bool)
        if not self.complete or not inside.size:
            return visible
        x, y = pixels[inside].T
        inside = inside[self.mask[y, x] & (depth[inside] > .01)]
        x, y = pixels[inside].T // self.tile
        tiles = y * self.columns + x
        order = np.argsort(tiles)
        keys, offsets = np.unique(tiles[order], return_index=True)
        for key, group in zip(keys, np.split(inside[order], offsets[1:])):
            faces = self.bins[key]
            if not faces:
                continue
            a, b, c = self.xy[faces].transpose(1, 0, 2)
            z = self.depth[faces]
            denom = (b[:,1]-c[:,1])*(a[:,0]-c[:,0]) + (c[:,0]-b[:,0])*(a[:,1]-c[:,1])
            nondegenerate = np.abs(denom) > 1e-10
            safe_denom = np.where(nondegenerate, denom, 1.)
            # Bound temporary arrays even where many triangles project into one tile.
            chunk = max(1, 200_000 // len(faces))
            for offset in range(0, len(group), chunk):
                rows = group[offset:offset+chunk]
                u, v = coords[rows].T[:, :, None]
                w0 = ((b[:,1]-c[:,1])*(u-c[:,0]) + (c[:,0]-b[:,0])*(v-c[:,1])) / safe_denom
                w1 = ((c[:,1]-a[:,1])*(u-c[:,0]) + (a[:,0]-c[:,0])*(v-c[:,1])) / safe_denom
                w2 = 1 - w0 - w1
                hit = nondegenerate & (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
                if self.camera.get("orthographic_scale"):
                    distance = w0*z[:,0] + w1*z[:,1] + w2*z[:,2]
                else:
                    inv = w0/z[:,0] + w1/z[:,1] + w2/z[:,2]
                    distance = 1 / np.maximum(inv, 1e-20)
                nearest = np.where(hit, distance, np.inf).min(axis=1)
                visible[rows] = np.abs(nearest-depth[rows]) <= 1e-5
        return visible


def _fraction(source, source_ids, posed, posed_ids, projection, posed_camera, dst, bary):
    if not projection.complete or not np.any(dst >= 0):
        return None
    if len(np.unique(source_ids)) != len(source_ids) or len(np.unique(posed_ids)) != len(posed_ids):
        return None
    lookup = {int(value): i for i, value in enumerate(source_ids)}
    mapping = np.array([lookup.get(int(value), -1) for value in posed_ids])
    ys, xs = np.nonzero(dst >= 0)
    target_faces = dst[ys, xs]
    indices = mapping[target_faces]
    safe = np.maximum(indices, 0)
    points = np.einsum('ni,nij->nj', bary[ys, xs], source[safe])
    sn = np.cross(source[safe, 1]-source[safe, 0], source[safe, 2]-source[safe, 0])
    pn = np.cross(posed[target_faces, 1]-posed[target_faces, 0], posed[target_faces, 2]-posed[target_faces, 0])
    source_eye = np.asarray(projection.camera["eye"])
    source_direction = (source_eye-np.asarray(projection.camera.get("target",[0,0,0]))
                        if projection.camera.get("orthographic_scale") else source_eye-points)
    posed_eye = np.asarray(posed_camera["eye"])
    posed_points = np.einsum('ni,nij->nj', bary[ys, xs], posed[target_faces])
    posed_direction = (posed_eye-np.asarray(posed_camera.get("target",[0,0,0]))
                       if posed_camera.get("orthographic_scale") else posed_eye-posed_points)
    same_side = ((sn*source_direction).sum(axis=1) * (pn*posed_direction).sum(axis=1)) > 0
    candidates = np.flatnonzero((indices >= 0) & same_side)
    return float(np.count_nonzero(projection.observed(points[candidates])) / len(ys))


def _coverage(source, source_ids, posed, posed_ids, source_camera, posed_camera,
              source_alpha, posed_alpha, reference_alpha, size, projection=None):
    if projection is None:
        projection = SourceProjection(source, source_camera, source_alpha, reference_alpha)
    src, _, _ = fragments(source, source_camera, size)
    dst, _, bary = fragments(posed, posed_camera, size)
    ious = []
    for labels, alpha in ((src, source_alpha), (dst, posed_alpha)):
        observed = cv2.resize(np.asarray(alpha, dtype=np.uint8), (size, size),
                              interpolation=cv2.INTER_AREA) >= 128
        present = labels >= 0
        union = np.count_nonzero(present | observed)
        ious.append(float(np.count_nonzero(present & observed) / union) if union else 0.)
    if min(ious) < .90:
        return None, ious
    return _fraction(source, source_ids, posed, posed_ids, projection, posed_camera, dst, bary), ious


def coverage(path, source_camera, posed_camera, source_alpha, posed_alpha, reference_alpha=None):
    if reference_alpha is None:
        reference_alpha = source_alpha
    with np.load(path) as data:
        source, source_ids = data['source'], data['source_ids']
        posed, posed_ids = data['posed'], data['posed_ids']
        projection = SourceProjection(source, source_camera, source_alpha, reference_alpha)
        results = [_coverage(source, source_ids, posed, posed_ids, source_camera, posed_camera,
                             source_alpha, posed_alpha, reference_alpha, size, projection) for size in (256, 512)]
    fractions = [value for value, _ in results]
    ious = results[-1][1]
    evidence = {'raster_iou':[round(v,4) for v in ious], 'version':VERSION}
    if any(v is None for v in fractions):
        return {**evidence, 'known':False, 'reason':'visibility raster or provenance is uncertain',
                'source_visible_fraction':None}
    evidence['fractions'] = [round(v,4) for v in fractions]
    if abs(fractions[0] - fractions[1]) > .02:
        return {**evidence, 'known':False, 'reason':'visibility differs across resolutions',
                'source_visible_fraction':None}
    return {**evidence, 'known':True, 'source_visible_fraction':min(fractions)}


def orbit_estimates(path, source_camera, posed_camera, source_alpha, reference_alpha):
    """Rank azimuths using a low-resolution projection of the settled geometry."""
    target = np.asarray(posed_camera['target'])
    delta = np.asarray(posed_camera['eye']) - target
    with np.load(path) as data:
        source, source_ids = data['source'], data['source_ids']
        posed, posed_ids = data['posed'], data['posed_ids']
    projection = SourceProjection(source, source_camera, source_alpha, reference_alpha)
    for degrees in range(30, 360, 30):
        cosine, sine = np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))
        eye = target + np.array([cosine*delta[0]-sine*delta[1], sine*delta[0]+cosine*delta[1], delta[2]])
        camera = {**posed_camera, 'eye':eye.tolist()}
        dst, _, bary = fragments(posed, camera, 128)
        value = _fraction(source, source_ids, posed, posed_ids, projection, camera, dst, bary)
        if value is not None:
            yield degrees, value
