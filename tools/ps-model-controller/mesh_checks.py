"""Conservative checks for reconstructed objects and proposed 3-D placements."""
import cv2
import numpy as np


def source_fidelity(source_alpha, rendered_alpha):
    """Compare source and rendered silhouettes at the same calibrated camera."""
    source = np.asarray(source_alpha) >= 128
    rendered = np.asarray(rendered_alpha) >= 128
    if source.shape != rendered.shape or not source.any() or not rendered.any():
        raise ValueError("Source and render must be nonempty, same-size alpha masks")
    union = np.logical_or(source, rendered).sum()
    iou = float(np.logical_and(source, rendered).sum()/union)
    # A good overall IoU can still hide a filled bearing eye or a lost hose.
    contours, hierarchy = cv2.findContours(source.astype("uint8"), cv2.RETR_CCOMP,
                                            cv2.CHAIN_APPROX_SIMPLE)
    holes = []
    if hierarchy is not None:
        for i, contour in enumerate(contours):
            if hierarchy[0, i, 3] < 0 or cv2.contourArea(contour) < 16:
                continue
            region = np.zeros(source.shape, "uint8")
            cv2.drawContours(region, contours, i, 1, -1)
            interior = (region > 0) & ~source
            if interior.sum() >= 16:
                holes.append(float(rendered[interior].mean()))
    distance = cv2.distanceTransform(source.astype("uint8"), cv2.DIST_L2, 5)
    # Recall of narrow structures is measured away from the ordinary outer edge.
    from skimage.morphology import skeletonize
    thin = skeletonize(source) & (distance <= 2.2)
    thin_recall = float(rendered[thin].mean()) if thin.sum() >= 12 else 1.0
    filled = sum(value > .20 for value in holes)
    result = {"iou":round(iou, 4), "holes":len(holes), "filled_holes":filled,
              "narrow_recall":round(thin_recall, 4)}
    result["passed"] = iou >= .90 and filled == 0 and thin_recall >= .80
    return result


def face_id_raster(vertices, faces, camera, image_size=256):
    """Rasterize face IDs using the exact mesh and a perspective camera.

    Faces are drawn back-to-front; this conservative low-resolution estimate
    is only used to reject views with too much unobserved surface, not to
    assert that hidden geometry is correct.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    eye = np.asarray(camera["eye"], dtype=np.float64)
    target = np.asarray(camera.get("target", (0, 0, 0)), dtype=np.float64)
    world_up = np.asarray(camera.get("up", (0, 0, 1)), dtype=np.float64)
    forward = target-eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-7:
        raise ValueError("camera up is parallel to view")
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    delta = vertices-eye
    depth = delta @ forward
    focal = .5*image_size/np.tan(np.deg2rad(camera.get("fov_deg", 40))/2)
    xy = np.column_stack((image_size/2+focal*(delta@right)/np.maximum(depth, 1e-6),
                          image_size/2-focal*(delta@up)/np.maximum(depth, 1e-6)))
    tri = xy[faces]
    depths = depth[faces]
    valid = np.all(depths > .01, axis=1) & np.all(np.isfinite(tri), axis=(1, 2))
    order = np.flatnonzero(valid)
    order = order[np.argsort(depths[order].mean(axis=1))[::-1]]
    labels = np.zeros((image_size, image_size), np.int32)
    for i in order:
        points = np.clip(np.rint(tri[i]), -32760, 32760).astype(np.int32)
        cv2.fillConvexPoly(labels, points, int(i)+1)
    return labels


def visible_source_fraction(source_ids, new_ids):
    if source_ids.shape != new_ids.shape or not np.any(new_ids):
        raise ValueError("Face rasters must be compatible and nonempty")
    seen = np.unique(source_ids[source_ids > 0])
    new = new_ids[new_ids > 0]
    return float(np.isin(new, seen).mean())


def placement_gate(evidence):
    """Fail closed when one of the independent geometry checks is absent."""
    issues = []
    fraction = evidence.get("source_visible_fraction")
    if not isinstance(fraction, (int, float)) or fraction < .80:
        issues.append("unseen_surface")
    if not evidence.get("physics_stable", False):
        issues.append("unstable")
    parts = evidence.get("component_support", [])
    if not parts or min(parts) < .90:
        issues.append("unsupported_part")
    if evidence.get("collision") is not False:
        issues.append("collision_or_unknown")
    if evidence.get("occlusion_order") not in ("clear", "verified"):
        issues.append("occlusion_unknown")
    if evidence.get("scene_scale") not in ("calibrated", "consistent_estimate"):
        issues.append("scene_scale_uncertain")
    return issues


def support_screening_gate(evidence):
    """Support-only gate: physics and contacts are required, other evidence stays pending.

    This is deliberately narrower than placement_gate.  It is used when the
    revision has no verified source camera, so source visibility, occlusion and
    scale must be reported as unknown rather than silently treated as passed.
    """
    issues = []
    if not evidence.get("physics_stable", False):
        issues.append("unstable")
    parts = evidence.get("component_support", [])
    if not parts or min(parts) < .90:
        issues.append("unsupported_part")
    return issues
