"""Background selection shared by the browser and batch planner."""
from pathlib import Path

IMAGE_SUFFIXES = frozenset({'.jpg', '.jpeg', '.png', '.webp'})


def background_files(directory, selected=()):
    root = Path(directory).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('背景目录不存在')
    if selected:
        files = []
        for name in dict.fromkeys(selected):
            if Path(name).name != name or '/' in name or '\\' in name:
                raise ValueError('背景选择必须是目录内的文件名')
            path = (root / name).resolve()
            if not path.is_relative_to(root) or not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError(f'背景不可用：{name}')
            files.append(path)
        return sorted(files)
    return sorted(path for path in root.iterdir()
                  if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file()
                  and path.resolve().is_relative_to(root))
