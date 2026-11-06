"""Conservative per-pixel source visibility using depth and surface provenance."""
import cv2
import numpy as np

VERSION = "visibility-zbuffer-v2"


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


def _coverage(source, source_ids, posed, posed_ids, source_camera, posed_camera,
              source_alpha, posed_alpha, reference_alpha, size):
    src, _, _ = fragments(source, source_camera, size)
    dst, _, bary = fragments(posed, posed_camera, size)
    masks = [cv2.resize(np.asarray(a, dtype=np.uint8), (size, size), interpolation=cv2.INTER_AREA) >= 128
             for a in (source_alpha, posed_alpha, reference_alpha)]
    ious = []
    for label, observed in ((src, masks[0]), (dst, masks[1])):
        present = label >= 0
        union = np.count_nonzero(present | observed)
        ious.append(float(np.count_nonzero(present & observed) / union) if union else 0.)
    if min(ious) < .90 or not np.any(dst >= 0):
        return None, ious
    source_ids = np.asarray(source_ids)
    posed_ids = np.asarray(posed_ids)
    if len(np.unique(source_ids)) != len(source_ids) or len(np.unique(posed_ids)) != len(posed_ids):
        return None, ious
    lookup = {int(value): i for i, value in enumerate(source_ids)}
    mapping = np.array([lookup.get(int(value), -1) for value in posed_ids])
    ys, xs = np.nonzero(dst >= 0)
    target_faces = dst[ys, xs]
    indices = mapping[target_faces]
    valid = indices >= 0
    safe = np.maximum(indices, 0)
    points = np.einsum('ni,nij->nj', bary[ys, xs], source[safe])
    coords, point_depth = project(points, source_camera, size)
    sx = np.floor(coords[:, 0]).astype(int)
    sy = np.floor(coords[:, 1]).astype(int)
    valid &= (sx >= 0) & (sx < size) & (sy >= 0) & (sy < size)
    sx = np.clip(sx, 0, size-1); sy = np.clip(sy, 0, size-1)
    valid &= masks[0][sy, sx] & masks[2][sy, sx]
    # Test nearby visible triangles at the exact reprojected location.  A
    # single source pixel cannot identify a small triangle reliably; checking
    # its neighbours avoids treating a whole face as seen or using their
    # depths at the wrong subpixel position.
    found = np.zeros(len(xs), dtype=bool)
    blocked = np.zeros(len(xs), dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            nx, ny = np.clip(sx+dx,0,size-1), np.clip(sy+dy,0,size-1)
            face = src[ny,nx]
            good = face >= 0
            face = np.maximum(face,0)
            corners, depths = project(source[face], source_camera, size)
            a,b,c = corners[:,0],corners[:,1],corners[:,2]
            denom = (b[:,1]-c[:,1])*(a[:,0]-c[:,0])+(c[:,0]-b[:,0])*(a[:,1]-c[:,1])
            good &= np.abs(denom)>1e-10
            denom = np.where(np.abs(denom)>1e-10,denom,1.)
            w0=((b[:,1]-c[:,1])*(coords[:,0]-c[:,0])+(c[:,0]-b[:,0])*(coords[:,1]-c[:,1]))/denom
            w1=((c[:,1]-a[:,1])*(coords[:,0]-c[:,0])+(a[:,0]-c[:,0])*(coords[:,1]-c[:,1]))/denom
            w2=1-w0-w1
            good &= (w0>=-1e-6)&(w1>=-1e-6)&(w2>=-1e-6)
            weights=np.stack((w0,w1,w2),axis=1)
            if not source_camera.get('orthographic_scale'):
                weights=weights/np.maximum(depths,1e-8)
                weights/=np.maximum(weights.sum(axis=1,keepdims=True),1e-20)
            depth=np.sum(weights*depths,axis=1)
            found |= good & (face==indices)
            blocked |= good & (depth < point_depth-1e-5)
    valid &= found & ~blocked
    sn = np.cross(source[safe, 1]-source[safe, 0], source[safe, 2]-source[safe, 0])
    pn = np.cross(posed[target_faces, 1]-posed[target_faces, 0], posed[target_faces, 2]-posed[target_faces, 0])
    source_side = np.einsum('ij,ij->i', sn, np.asarray(source_camera['eye'])-points)
    posed_points = np.einsum('ni,nij->nj', bary[ys, xs], posed[target_faces])
    posed_side = np.einsum('ij,ij->i', pn, np.asarray(posed_camera['eye'])-posed_points)
    valid &= (source_side * posed_side > 0)
    return float(valid.mean()), ious


def coverage(path, source_camera, posed_camera, source_alpha, posed_alpha, reference_alpha=None):
    if reference_alpha is None:
        reference_alpha = source_alpha
    with np.load(path) as data:
        source, source_ids = data['source'], data['source_ids']
        posed, posed_ids = data['posed'], data['posed_ids']
        results = [_coverage(source, source_ids, posed, posed_ids, source_camera, posed_camera,
                             source_alpha, posed_alpha, reference_alpha, size) for size in (256, 512)]
    fractions = [value for value, _ in results]
    ious = results[-1][1]
    if any(v is None for v in fractions):
        return {'known':False, 'reason':'visibility raster or provenance is uncertain',
                'raster_iou':[round(v,4) for v in ious], 'source_visible_fraction':None, 'version':VERSION}
    if abs(fractions[0] - fractions[1]) > .02:
        return {'known':False, 'reason':'visibility differs across resolutions', 'fractions':fractions,
                'raster_iou':[round(v,4) for v in ious], 'source_visible_fraction':None, 'version':VERSION}
    return {'known':True, 'source_visible_fraction':round(min(fractions),4),
            'fractions':[round(v,4) for v in fractions], 'raster_iou':[round(v,4) for v in ious], 'version':VERSION}


def orbit_estimates(path, source_camera, posed_camera, source_alpha, reference_alpha):
    """Fast face-ID estimates only select a view; they can never pass a gate."""
    target = np.asarray(posed_camera['target'])
    delta = np.asarray(posed_camera['eye']) - target
    with np.load(path) as data:
        original = raster(data['source'], data['source_ids'], source_camera, size=source_alpha.shape[0])
        seen = np.unique(original[(np.asarray(source_alpha)>=128) & (np.asarray(reference_alpha)>=128) & (original>0)])
        triangles, ids = data['posed'], data['posed_ids']
    for degrees in range(30, 360, 30):
        cosine, sine = np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))
        eye = target + np.array([cosine*delta[0]-sine*delta[1], sine*delta[0]+cosine*delta[1], delta[2]])
        posed = raster(triangles, ids, {**posed_camera, 'eye':eye.tolist()})
        pixels = posed[posed>0]
        if pixels.size:
            yield degrees, float(np.isin(pixels, seen).mean())
