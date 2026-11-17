"""Locate and rectify the visible HPEOK label from the clean item-39 photo."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'outputs/物品高清输入-39件/原貌HD/39-集油器2.png'
OUT = ROOT / 'work/direct-deepseek/39-label-rectified.png'


def main():
    rgb = np.asarray(Image.open(SOURCE).convert('RGB'))
    r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
    yellow = (r > 130) & (g > 110) & (b < 110) & (r > b + 60)
    mask = yellow.astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count < 2:
        raise SystemExit('label not found')
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = (int(v) for v in stats[best])
    print('label bbox', x, y, w, h, 'area', area)
    # Corners of the sticker: use the extreme points of the yellow component in
    # four diagonal directions, then order them.
    ys, xs = np.nonzero(labels == best)
    points = np.stack((xs, ys), axis=1).astype(np.float32)
    centre = points.mean(axis=0)
    sums = points[:, 0] + points[:, 1]
    diffs = points[:, 0] - points[:, 1]
    quad = np.array([points[np.argmin(sums)], points[np.argmax(diffs)],
                     points[np.argmax(sums)], points[np.argmin(diffs)]], np.float32)
    # Make sure the order is TL, TR, BR, BL.
    s = quad.sum(axis=1)
    d = np.diff(quad, axis=1).ravel()
    tl = quad[np.argmin(s)]
    br = quad[np.argmax(s)]
    tr = quad[np.argmin(d)]
    bl = quad[np.argmax(d)]
    src = np.array([tl, tr, br, bl], np.float32)
    print('quad', src.tolist())
    width = int(round((np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2))
    height = int(round((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2))
    width, height = max(width, 32), max(height, 32)
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1],
                    [0, height - 1]], np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    rectified = cv2.warpPerspective(rgb, matrix, (width, height),
                                    flags=cv2.INTER_CUBIC)
    Image.fromarray(rectified).save(OUT)
    print('rectified', width, height, '->', OUT)


if __name__ == '__main__':
    main()
