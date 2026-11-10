import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parents[1]
BLENDER = ROOT / "work/blender45/blender-4.5.3-windows-x64/blender.exe"
WEIGHTS = ROOT / "work/v2-models"


def signature(*values):
    return hashlib.sha256(json.dumps(values, sort_keys=True, default=str,
                                     ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str, allow_nan=False)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()


def implementation_manifest():
    import importlib.metadata
    files=sorted((HERE/'display_batch').glob('*.py'))+[HERE/'blender_display_worker.py',HERE/'display_collision_worker.py']
    versions={}
    for name in ('numpy','trimesh','coacd','Pillow','opencv-python','pydantic','httpx'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]='unavailable'
    return {'sources':{str(p.relative_to(HERE)):file_hash(p) for p in files},
            'dependencies':versions,'blender':'4.5.3','python':sys.version.split()[0]}


class FileLease:
    """OS-held lease, automatically released if the owning process crashes."""
    def __init__(self,path):self.path=Path(path);self.handle=None
    def acquire(self):
        if self.handle is not None:return True
        self.path.parent.mkdir(parents=True,exist_ok=True)
        handle=self.path.open('a+b');handle.seek(0);handle.write(b'0');handle.flush();handle.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            handle.close();return False
        self.handle=handle;return True
    def release(self):
        if self.handle:
            if os.name=='nt':
                import msvcrt
                self.handle.seek(0);msvcrt.locking(self.handle.fileno(),msvcrt.LK_UNLCK,1)
            self.handle.close();self.handle=None


def load_catalog():
    sys.path.insert(0, str(HERE/"placement-web"))
    from review_viewer import load_catalog as current_catalog
    rows = current_catalog()
    library = {r["number"]: r for r in (json.loads(x) for x in
               (ROOT/"outputs/三维模型库-圆润复验/results.jsonl").read_text(encoding="utf8").splitlines() if x)}
    for row in rows.values():
        source = library.get(row["number"], {})
        row["original"] = source.get("original")
        row["mesh"] = str(row.get("_mesh", ""))
        row["rust"] = str(row.get("_rust", ""))
        row["board"] = str(row.get("_board", ""))
        row["mesh_hash"] = file_hash(row["mesh"]) if row["available"] else None
    return rows


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
