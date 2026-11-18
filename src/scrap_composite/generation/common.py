"""Shared generation paths and compatibility exports."""
import os
from pathlib import Path
import sys
from functools import lru_cache

from scrap_composite.data.catalog import load_catalog, mesh_input
from scrap_composite.data.files import FileLease, atomic_json, file_hash, signature
from scrap_composite.runtime import BLENDER, CONTROLLER as HERE, ROOT, SOURCE, WEIGHTS, resolve

@lru_cache(maxsize=1)
def implementation_manifest():
    import importlib.metadata
    files=[HERE/'blender_display_worker.py',HERE/'display_collision_worker.py']
    files += [HERE/name for name in ('blender_worker.py','scene_v2.py','run.py')]
    package = Path(__file__).resolve().parents[1]
    core = sorted(path for path in package.rglob('*')
                  if path.relative_to(package).parts[0] not in {'modeling', 'frontend'})
    versions={}
    for name in ('numpy','trimesh','coacd','rtree','scipy','Pillow','opencv-python','pydantic','httpx','torch','transformers','opencolorio','numba','pydantic-graph'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]='unavailable'
    weights={str(p.relative_to(WEIGHTS)):file_hash(p) for p in
             (WEIGHTS/'mobile_sam.pt',WEIGHTS/'depth/model.safetensors',WEIGHTS/'matte/birefnet-general-lite.onnx') if p.is_file()}
    return {'sources':{str(p.relative_to(HERE)):file_hash(p) for p in files},
            'core_sources':{p.relative_to(package).as_posix():file_hash(p) for p in core if p.suffix in ('.py','.jsx','.txt','.json') and p.is_file()},
            'weight_hashes':weights,
            'ocio_config_hash':file_hash(ocio) if (ocio := BLENDER.parent/'4.5/datafiles/colormanagement/config.ocio').is_file() else None,
            'dependencies':versions,'blender':'4.5.3','python':sys.version.split()[0]}


def configure_credential(env_name):
    """Read only the explicitly selected provider credential, never log its value."""
    if not env_name or os.environ.get(env_name):
        return True
    if os.name == "nt":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                value, _ = winreg.QueryValueEx(key, env_name)
                if value:
                    os.environ[env_name] = value
                    return True
        except OSError:
            pass
    if env_name == "PS_DEEPSEEK_KEY":
        credential = Path.home()/".dsh/.credentials.yaml"
        if credential.is_file():
            import yaml
            data = yaml.safe_load(credential.read_text(encoding="utf8")) or {}
            value = data.get("refs", {}).get("DEEPSEEK_API_KEY", "")
            if isinstance(value, str) and value.startswith("sk-"):
                os.environ[env_name] = value
                return True
    return False
