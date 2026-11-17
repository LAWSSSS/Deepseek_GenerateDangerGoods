"""Isolated CPU-only TripoSR worker. The caller enforces time and memory limits."""
import argparse
import json
import os
from pathlib import Path
import sys
import types

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")


def install_cpu_marching_cubes():
    """Provide TripoSR's torchmcubes interface using scikit-image on CPU.

    TripoSR reverses torchmcubes coordinates in its helper; scikit-image
    already returns array-axis order, so reverse them here to compensate.
    """
    import numpy as np
    import torch
    from skimage.measure import marching_cubes

    def cpu_marching_cubes(volume, level):
        vertices, faces, _, _ = marching_cubes(volume.detach().cpu().numpy(), level)
        return (torch.from_numpy(vertices[:, ::-1].copy()),
                torch.from_numpy(faces.astype(np.int64, copy=True)))

    module = types.ModuleType("torchmcubes")
    module.marching_cubes = cpu_marching_cubes
    sys.modules["torchmcubes"] = module


def normalized_input(path, folder):
    import numpy as np
    from PIL import Image

    with Image.open(path) as source:
        image = source.convert("RGBA")
    alpha = image.getchannel("A")
    box = alpha.getbbox()
    if box is None or alpha.getextrema()[0] != 0:
        raise ValueError("Expected an isolated object with transparent background")
    image = image.crop(box)
    image.thumbnail((435, 435), Image.Resampling.LANCZOS)
    rgba = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    rgba.alpha_composite(image, ((512-image.width)//2, (512-image.height)//2))
    rgba.getchannel("A").save(folder/"source-alpha.png")
    array = np.asarray(rgba, dtype=np.float32)/255
    rgb = array[:, :, :3]*array[:, :, 3:4] + .5*(1-array[:, :, 3:4])
    normalized = Image.fromarray(np.rint(rgb*255).astype("uint8"))
    normalized.save(folder/"model-input.png")
    return normalized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=128, choices=[128])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.vendor.resolve(strict=True)))
    install_cpu_marching_cubes()
    import torch
    from tsr.system import TSR

    torch.set_num_threads(4)
    image = normalized_input(args.asset, args.out)
    model = TSR.from_pretrained(str(args.weights.resolve(strict=True)),
                                config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(2048)
    model.to("cpu").eval()
    with torch.inference_mode():
        codes = model([image], device="cpu")
        meshes = model.extract_mesh(codes, has_vertex_color=True, resolution=args.resolution)
    mesh = meshes[0]
    if len(mesh.vertices) < 100 or len(mesh.faces) < 100 or not mesh.is_winding_consistent:
        raise ValueError("Reconstructed mesh has insufficient or inconsistent geometry")
    mesh.export(args.out/"mesh.glb")
    (args.out/"worker-result.json").write_text(json.dumps({
        "vertices":len(mesh.vertices), "faces":len(mesh.faces),
        "bounds":mesh.bounds.tolist(), "resolution":args.resolution,
        "color_source":"TripoSR vertex colors inferred from isolated rust image",
    }, ensure_ascii=False), encoding="utf-8")
    print("mesh reconstructed", flush=True)


if __name__ == "__main__":
    main()
