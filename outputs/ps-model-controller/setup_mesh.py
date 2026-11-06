"""Install the pinned CPU mesh runtime next to the existing v2 environment."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request, urlopen
import venv
import zipfile


REPO="https://github.com/VAST-AI-Research/TripoSR.git"
COMMIT="107cefdc244c39106fa830359024f6a2f1c78871"
MODEL_REV="5b521936b01fbe1890f6f9baed0254ab6351c04a"
MODEL_SHA="429e2c6b22a0923967459de24d67f05962b235f79cde6b032aa7ed2ffcd970ee"
BLENDER="https://download.blender.org/release/Blender4.5/blender-4.5.3-windows-x64.zip"


def run(*arguments,cwd=None):
    subprocess.run([str(x) for x in arguments],cwd=cwd,check=True)


def download_resume(url,path):
    if path.exists():
        try:
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is None:
                    return
        except zipfile.BadZipFile:
            pass
    for _ in range(10):
        start=path.stat().st_size if path.exists() else 0
        try:
            request=Request(url,headers={"Range":f"bytes={start}-"} if start else {})
            with urlopen(request,timeout=90) as response:
                if start and (response.status!=206 or not response.headers.get("Content-Range","").startswith(f"bytes {start}-")):
                    raise RuntimeError("Server did not honor the download range")
                mode="ab" if start else "wb"
                with path.open(mode) as output:
                    while block:=response.read(1024*1024):
                        output.write(block)
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is None:
                    return
        except (OSError,zipfile.BadZipFile) as issue:
            print("Download retry:",type(issue).__name__,flush=True)
            time.sleep(2)
    raise RuntimeError("Blender archive could not be downloaded completely")


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--work",type=Path,default=Path("work"))
    args=cli.parse_args()
    work=args.work.resolve()
    shared=work/"v2-env"/"Lib"/"site-packages"
    if not shared.is_dir():
        raise RuntimeError("Install the existing v2 runtime before the mesh runtime")
    runtime=work/"triposr-env"
    if not (runtime/"Scripts"/"python.exe").is_file():
        venv.EnvBuilder(with_pip=True).create(runtime)
    paths=[shared,shared/"win32",shared/"win32"/"lib",shared/"pythonwin"]
    (runtime/"Lib"/"site-packages"/"ps-existing-packages.pth").write_text(
        "".join(str(path)+"\n" for path in paths)+"import pywin32_bootstrap\n",encoding="utf8")
    python=runtime/"Scripts"/"python.exe"
    run(python,"-m","pip","install","--disable-pip-version-check","einops==0.8.1",
        "omegaconf==2.3.0","trimesh==4.6.4","psutil==7.1.3","fast-simplification==0.1.12",
        "hf_xet==1.8.0","pymeshlab==2023.12.post3")
    vendor=work/"vendor"/"TripoSR"
    if not (vendor/".git").is_dir():
        vendor.mkdir(parents=True,exist_ok=True)
        run("git","init",vendor)
        run("git","fetch","--depth","1",REPO,COMMIT,cwd=vendor)
        run("git","checkout","--detach","FETCH_HEAD",cwd=vendor)
    actual=subprocess.check_output(["git","rev-parse","HEAD"],cwd=vendor,text=True).strip()
    if actual!=COMMIT:
        raise RuntimeError("TripoSR checkout does not match the pinned revision")
    weights=work/"v2-models"/"TripoSR"
    weights.mkdir(parents=True,exist_ok=True)
    code=("from huggingface_hub import hf_hub_download; "
          f"[hf_hub_download('stabilityai/TripoSR',x,revision='{MODEL_REV}',local_dir={str(weights)!r}) "
          "for x in ('config.yaml','model.ckpt')]; "
          "hf_hub_download('facebook/dino-vitb16','config.json')")
    run(python,"-c",code)
    with (weights/"model.ckpt").open("rb") as file:
        actual=hashlib.file_digest(file,"sha256").hexdigest()
    if actual!=MODEL_SHA:
        raise RuntimeError("TripoSR checkpoint checksum mismatch")
    blender_root=work/"blender45"
    executable=blender_root/"blender-4.5.3-windows-x64"/"blender.exe"
    if not executable.is_file():
        blender_root.mkdir(parents=True,exist_ok=True)
        archive=blender_root/"blender-4.5.3-windows-x64.zip"
        download_resume(BLENDER,archive)
        with zipfile.ZipFile(archive) as data:
            for item in data.infolist():
                if not (blender_root/item.filename).resolve().is_relative_to(blender_root):
                    raise ValueError("Unsafe path in Blender archive")
            data.extractall(blender_root)
    run(executable,"--background","--factory-startup","--python-expr","import bpy; print(bpy.app.version_string)")
    print("Mesh runtime ready. Configure PS_DEEPSEEK_KEY separately; no key is stored here.")


if __name__=="__main__":
    main()
