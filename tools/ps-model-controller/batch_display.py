import sys
import display_batch  # Locate the shared package in existing local installations.
import scrap_composite.generation.cli as _module

if __name__ == '__main__':
    raise SystemExit(_module.main())

sys.modules[__name__] = _module
