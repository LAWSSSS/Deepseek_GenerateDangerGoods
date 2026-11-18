"""Content fingerprints and atomic persistence for local inputs."""
import hashlib
import json
import os
from pathlib import Path
import tempfile


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


def read_jsonl(path):
    try:
        with Path(path).open(encoding='utf8') as stream:
            return [json.loads(line) for line in stream if line.strip()]
    except FileNotFoundError:
        return []
