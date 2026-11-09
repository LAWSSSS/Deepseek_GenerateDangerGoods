from pathlib import Path
import numpy as np
from PIL import Image

root = Path(__file__).resolve().parents[2]
img = Image.open(root / 'outputs/物品高清输入-39件/生锈HD/39-集油器2.png')
rgb = np.asarray(img.convert('RGB')).astype(float)
alpha = np.asarray(img.getchannel('A')) >= 128
r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
teal = alpha & (g > 55) & (b > 55) & (g > r * 1.25) & (r < 110)
rust = alpha & (r > 62) & (r > g + 15) & (r > b + 22) & (g < 145)
print('teal pixels', int(teal.sum()), 'median', np.median(rgb[teal], 0))
print('rust pixels', int(rust.sum()), 'median', np.median(rgb[rust], 0))
print('rust mean', rgb[rust].mean(0) if rust.any() else None)
# Bright rust-like pixels (paint chips/stains).
rust_like = alpha & (r > g + 8) & (r > b + 12) & (r > 50)
print('rust-like pixels', int(rust_like.sum()), 'median', np.median(rgb[rust_like], 0))
# print a few labeled pixels
for y, x in ((570, 500), (600, 700), (150, 570), (900, 430), (1100, 600)):
    print((x, y), rgb[y, x], alpha[y, x])
