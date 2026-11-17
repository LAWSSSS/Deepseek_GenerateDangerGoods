"""Independent, no-model Photoshop export and availability CLI."""
import argparse
import asyncio
import json
from pathlib import Path

from .photoshop import PhotoshopAdapter, availability, load_request


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--availability", action="store_true", help="Inspect only; never launch Photoshop")
    parser.add_argument("--manifest", type=Path, help="Host-authored full-canvas layer manifest")
    parser.add_argument("--out", type=Path, help="New or empty output directory")
    parser.add_argument("--mcp", type=Path)
    args = parser.parse_args(argv)
    if args.availability:
        result = availability(args.mcp)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["available"] else 3
    if not args.manifest or not args.out:
        parser.error("--manifest and --out are required for export")
    try:
        result = asyncio.run(PhotoshopAdapter(mcp=args.mcp).export(load_request(args.manifest), args.out))
    except (OSError, ValueError, RuntimeError, TimeoutError) as error:
        print(json.dumps({"status": "failed", "error_kind": type(error).__name__,
                          "new_model_requests": 0}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
