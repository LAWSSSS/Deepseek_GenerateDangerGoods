"""Measure item 39 source/reference panel proportions for the rebuild config."""
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def panel_bounds(image, index):
    w, h = image.size
    x0, y0 = (index % 3) * w // 3, (index // 3) * h // 2
    return x0, y0, w // 3, h // 2


def report(mask, label):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        print(label, 'empty')
        return None
    print(label, 'bbox', int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()),
          'w', int(xs.max() - xs.min() + 1), 'h', int(ys.max() - ys.min() + 1))
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


source = Image.open(ROOT / 'outputs/物品高清输入-39件/原貌HD/39-集油器2.png')
a = np.asarray(source)
alpha = a[..., 3]
rgb = a[..., :3].astype(int)
report(alpha >= 128, 'source alpha')
teal = (rgb[..., 1] > 55) & (rgb[..., 2] > 55) & (rgb[..., 0] < 110) & (alpha >= 128)
report(teal, 'source teal')
for y in range(60, 1320, 60):
    row = np.nonzero(alpha[y] >= 128)[0]
    if len(row):
        print('row', y, int(row.min()), int(row.max()), int(row.max() - row.min() + 1))

ref = Image.open(ROOT / 'references/parallel-six-view/六视角青绿色压力油罐产品图лириға.png').convert('RGB')
r = np.asarray(ref).astype(int)
h, w, _ = r.shape
print('reference size', w, h)
for idx in range(6):
    x0, y0, pw, ph = panel_bounds(ref, idx)
    p = r[y0:y0 + ph, x0:x0 + pw]
    gray = p.mean(axis=2)
    m = gray < 244
    report(m, f'panel {idx} fg')
    t = (p[..., 1] > 70) & (p[..., 2] > 70) & (p[..., 0] < 100)
    report(t, f'panel {idx} teal')
