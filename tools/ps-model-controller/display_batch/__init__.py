"""Legacy import name for the packaged generation and data modules."""
from importlib import import_module
from importlib.abc import MetaPathFinder, Loader
from importlib.util import find_spec, spec_from_loader
from pathlib import Path
import sys

source = str(Path(__file__).resolve().parents[3] / 'src')
if source not in sys.path:
    sys.path.insert(0, source)

from scrap_composite.generation import VERSION

_MODULES = {
    'store': 'scrap_composite.data.store',
    'inventory': 'scrap_composite.data.inventory',
    'web': 'scrap_composite.frontend.batches',
}


class _LegacyImports(MetaPathFinder, Loader):
    def find_spec(self, fullname, path, target=None):
        package, _, name = fullname.rpartition('.')
        if package != __name__ or Path(__file__).with_name(name + '.py').is_file():
            return None
        if find_spec(self.destination(name)) is not None:
            return spec_from_loader(fullname, self)
        return None

    @staticmethod
    def destination(name):
        return _MODULES.get(name, 'scrap_composite.generation.' + name)

    def exec_module(self, module):
        name = module.__name__
        sys.modules[name] = import_module(self.destination(name.rsplit('.', 1)[1]))


sys.meta_path.insert(0, _LegacyImports())
