"""Prepare a local PBR pack from three approved ImageGen PNGs."""
import argparse
import json
from pathlib import Path

from display_batch.common import HERE
from display_batch.generated_material import CHANNELS, VERSION, WEAR_VERSION, build_pack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--version', choices=(VERSION, WEAR_VERSION), default=VERSION)
    for kind in CHANNELS:
        parser.add_argument('--' + kind, required=True, type=Path)
    args = parser.parse_args()
    prompts = json.loads((HERE / 'generated-pbr-prompts.json').read_text(encoding='utf8'))
    layouts = json.loads((HERE / 'generated-pbr-atlases.json').read_text(encoding='utf8'))
    print(build_pack(args.out, vars(args), prompts, layouts, version=args.version))


if __name__ == '__main__':
    main()
