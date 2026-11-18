"""Paths shared by application entry points and local worker adapters."""
import os
from pathlib import Path
import sys

ROOT = Path(os.environ.get('SCRAP_COMPOSITE_ROOT', Path(__file__).resolve().parents[2])).resolve()
CONTROLLER = ROOT / 'tools/ps-model-controller'
SOURCE = ROOT / 'src'
BLENDER = ROOT / 'work/blender45/blender-4.5.3-windows-x64/blender.exe'
WEIGHTS = ROOT / 'work/v2-models'


def enable_legacy_tools():
    # The packaged services still invoke the existing perception and Blender workers.
    directory = str(CONTROLLER)
    if directory not in sys.path:
        sys.path.insert(0, directory)


def resolve(path):
    return (ROOT / path).resolve()
