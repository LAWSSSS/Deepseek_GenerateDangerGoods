import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import display_batch  # Locate the shared package in existing local installations.
import scrap_composite.frontend.review_viewer as _module

if __name__ == '__main__':
    raise SystemExit(_module.main())

sys.modules[__name__] = _module
