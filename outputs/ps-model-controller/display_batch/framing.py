"""Render windows on the original square camera frustum."""
import math


def edit_bounds(candidate, size):
    x, y, r, b = candidate['bounds']
    pad = max(12, round(candidate['target_px'] * .12))
    return [max(0, x-pad), max(0, y-pad), min(size[0], r+pad), min(size[1], b+pad)]


def final_resolution(candidate):
    # Four pixels of image-space padding on both sides of the object.
    return min(2048, max(512, 64 * math.ceil(2 * (candidate['target_px'] + 8) / 64)))


def render_windows(candidate, size, resolution, *, full_frame=False):
    if full_frame:
        return {'object': {'frame': resolution, 'rect': [0, 0, resolution, resolution]},
                'shadow': {'frame': 512, 'rect': [0, 0, 512, 512]}}

    full = candidate['full_px']
    offset = candidate['offset']

    def uv(bounds):
        return [max(0., min(1., (coord-offset[i % 2])/full)) for i, coord in enumerate(bounds)]

    x, y, r, b = candidate['bounds']
    box = uv([x-4, y-4, r+4, b+4])
    extent = max(box[2]-box[0], box[3]-box[1])
    if extent <= 0:
        raise ValueError('empty_object_render_window')
    frame = max(resolution, math.floor((resolution-2)/extent))

    def crop(box, frame, alignment, longest=None):
        lo = [math.floor(v*frame) for v in box[:2]]
        hi = [math.ceil(v*frame) for v in box[2:]]
        dims = [max(alignment, alignment*math.ceil((end-start)/alignment))
                for start, end in zip(lo, hi)]
        if longest:
            axis = max(range(2), key=lambda i: hi[i]-lo[i])
            dims[axis] = max(dims[axis], longest)
        dims = [min(frame, n) for n in dims]
        start = [max(0, min(frame-n, (a+b-n)//2)) for a, b, n in zip(lo, hi, dims)]
        return {'frame': frame, 'rect': [*start, *(a+n for a, n in zip(start, dims))]}

    shadow_box=uv(edit_bounds(candidate,size))
    shadow_extent=max(shadow_box[2]-shadow_box[0],shadow_box[3]-shadow_box[1])
    shadow_frame=max(512,math.ceil(512/max(shadow_extent,.001))) if candidate.get('scale',{}).get('calibration',{}).get('projectable') else 512
    return {'object': crop(box, frame, 64, resolution),
            'shadow': crop(shadow_box, shadow_frame, 1)}
