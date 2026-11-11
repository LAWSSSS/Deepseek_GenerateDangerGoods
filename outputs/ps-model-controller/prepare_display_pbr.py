"""Prepare the pinned local scan library and verify its maps in background Blender."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from urllib.request import Request, urlopen

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2
import numpy as np

from display_batch.common import BLENDER, HERE, atomic_json, file_hash, resolve


CDN = "https://dl.polyhaven.org/file/ph-assets/"
SCANS = {
    "rusty_metal_03": {
        "base": ("diff", "jpg", "e5ec75885884f9440a14eae579035c9f"),
        "roughness": ("rough", "exr", "a8d966ba5aff2a25bd82c17853964a24"),
        "normal": ("nor_gl", "exr", "707e4b4e7b4081689c90389b3a727ff2"),
        "arm": ("arm", "exr", "4c3a7fb21cc8ee0fc5358fd4c8628055"),
    },
    "rusty_painted_metal": {
        "base": ("diff", "jpg", "dd4f975d539100c4d31fa91d41e1ea71"),
        "roughness": ("rough", "exr", "6b449b0e27e0bc49e9be0bee97820c3c"),
        "normal": ("nor_gl", "exr", "4ed4e847cb07573af31e6ea32775da45"),
        "arm": ("arm", "exr", "701ce8b79af40bfe72fa4ba413c2fbd5"),
    },
}
HDRI = "overcast_soil"
LICENSE = "https://polyhaven.com/license"
ADDON_COMMIT = "7240b0d757c0fb83076e12d50559429cb5c85e62"
ADDON_SHA256 = "fa6b274afc9be0ecbcb0dd236e3da1dbaa05ed1e1da528c18f2b2695f0a878a4"
SKILL_COMMIT = "9ebe57d3f45ce8a3dd735f00b438c35559d51aec"


def md5(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "md5").hexdigest()


def fetch(url, target):
    # The native Windows client uses the machine's existing network settings.
    shell = shutil.which("pwsh") if os.name == "nt" else None
    if shell:
        env = {**os.environ, "SCRAP_PBR_URI": url, "SCRAP_PBR_DOWNLOAD": str(target)}
        subprocess.run([shell, "-NoProfile", "-NonInteractive", "-Command",
                        "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; "
                        "Invoke-WebRequest -Uri $env:SCRAP_PBR_URI -OutFile $env:SCRAP_PBR_DOWNLOAD -TimeoutSec 45"],
                       check=True, timeout=60, env=env)
    else:
        request = Request(url, headers={"User-Agent": "scrap-composite-local-preview/1.0"})
        with urlopen(request, timeout=45) as response, target.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def download(root, name, url, checksum, resolution):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    hit = path.is_file() and md5(path) == checksum
    if not hit:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as pending:
            temp = Path(pending.name)
        try:
            fetch(url, temp)
            if md5(temp) != checksum:
                raise ValueError(f"Poly Haven checksum mismatch for {name}")
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
    return {"path": name, "source": url, "sha256": file_hash(path), "provider_md5": checksum,
            "resolution": resolution, "bytes": path.stat().st_size, "cache_hit": hit,
            "license": "CC0-1.0", "license_url": LICENSE}


def low_frequency(image):
    width, height = image.shape[1], image.shape[0]
    small = cv2.resize(image, (width // 8, height // 8), interpolation=cv2.INTER_AREA)
    return cv2.resize(cv2.GaussianBlur(small, (0, 0), 3), (width, height), interpolation=cv2.INTER_LINEAR)


def save_map(root, name, suffix, image, source):
    filename = f"{name}/{name}_{suffix}_2k.png"
    pixels = np.rint(np.clip(image, 0, 1) * 65535).astype(np.uint16)
    if not cv2.imwrite(str(root / filename), pixels):
        raise OSError(f"Cannot save {filename}")
    return {"path": filename, "sha256": file_hash(root / filename), "derived_from": source["sha256"],
            "resolution": "2K", "license": "CC0-1.0"}


def detail_maps(root, name, maps):
    rgb = cv2.imread(str(root / maps["base"]["path"]), cv2.IMREAD_COLOR)
    if rgb is None or rgb.shape != (2048, 2048, 3):
        raise ValueError(f"Invalid Poly Haven base color for {name}")
    gray = cv2.cvtColor(np.float32(rgb) / 255, cv2.COLOR_BGR2GRAY)
    # Use scanned variation without reintroducing the library paint hue or broad shading.
    albedo = .5 + np.clip((gray - low_frequency(gray)) * 1.7, -.5, .5)
    normal = cv2.imread(str(root / maps["normal"]["path"]), cv2.IMREAD_UNCHANGED)
    if (normal is None or normal.ndim != 3 or normal.shape[:2] != rgb.shape[:2]
            or normal.shape[2] not in (3, 4) or not np.isfinite(normal[..., :3]).all()):
        raise ValueError(f"Invalid Poly Haven normal map for {name}")
    xy = normal[..., 1:3]
    delta = np.clip(xy - low_frequency(xy), -.22, .22)
    z = .5 + .5 * np.sqrt(np.maximum(0, 1 - np.square(2 * delta).sum(axis=2)))
    micro_normal = np.dstack((z, .5 + delta))
    return {"detail": save_map(root, name, "detail", albedo, maps["base"]),
            "micro_normal": save_map(root, name, "micro_normal", micro_normal, maps["normal"])}


def prepare(root, addon_scripts, *, verify=True):
    start = time.monotonic()
    requests = {}
    for name, maps in SCANS.items():
        for channel, (suffix, extension, checksum) in maps.items():
            filename = f"{name}_{suffix}_2k.{extension}"
            url = f"{CDN}Textures/{extension}/2k/{name}/{filename}"
            requests[(name, channel)] = (f"{name}/{filename}", url, checksum, "2K")
    filename = f"{HDRI}_1k.exr"
    requests[(HDRI, "environment")] = (filename, f"{CDN}HDRIs/exr/1k/{filename}",
                                         "ac8bf6a989454a6bec7cc536ce94f520", "1K")
    with ThreadPoolExecutor(max_workers=4) as executor:
        pending = {key: executor.submit(download, root, *args) for key, args in requests.items()}
        files = {key: future.result() for key, future in pending.items()}
    materials = {}
    for name in SCANS:
        maps = {channel: files[(name, channel)] for channel in SCANS[name]}
        arm = cv2.imread(str(root / maps["arm"]["path"]), cv2.IMREAD_UNCHANGED)
        if arm is None or arm.shape != (2048, 2048, 3) or not np.isfinite(arm).all():
            raise ValueError(f"Invalid Poly Haven packed map for {name}")
        # Poly Haven ARM uses RGB=AO/Roughness/Metallic. OpenCV loads EXR in BGR.
        maps["metallic"] = {**save_map(root, name, "metallic", arm[..., 0], maps["arm"]), "channel": "blue"}
        maps.update(detail_maps(root, name, maps))
        materials[name] = maps
    addon = addon_scripts / "addons/asset_importer.py"
    if file_hash(addon) != ADDON_SHA256:
        raise ValueError("The pinned Asset Importer file does not match the reviewed source")
    manifest = {"materials": materials, "environment": files[(HDRI, "environment")],
                "addon": {"name": "Asset Importer", "version": "2.0.0", "commit": ADDON_COMMIT,
                          "sha256": ADDON_SHA256, "path": str(addon)},
                "skill": {"name": "using-blender", "version": "1.0.0", "commit": SKILL_COMMIT},
                "download_bytes": sum(f["bytes"] for f in files.values() if not f["cache_hit"]),
                "source_bytes": sum(f["bytes"] for f in files.values())}
    manifest_path = root / "library.json"
    atomic_json(manifest_path, manifest)
    if verify:
        env = {**os.environ, "BLENDER_USER_SCRIPTS": str(addon_scripts)}
        command = [str(BLENDER), "--background", "--factory-startup", "--python-exit-code", "1",
                   "--python", str(HERE / "verify_display_scanned_blender.py"), "--", str(manifest_path)]
        subprocess.run(command, env=env, check=True, timeout=120)
    atomic_json(root / "preparation.json", {"seconds": round(time.monotonic() - start, 3),
                "download_bytes": manifest["download_bytes"], "blender_import_checked": verify})
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="work/render-materials/polyhaven")
    parser.add_argument("--addon-scripts", default="work/blender-addons/scripts")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    print(prepare(resolve(args.out), resolve(args.addon_scripts), verify=not args.download_only))
