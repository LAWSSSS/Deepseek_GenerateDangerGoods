"""Command line for the local modeling and composition workbench."""
from argparse import ArgumentParser
import sys


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('model', 'generate', 'web'))
    command = parser.parse_args(args[:1]).command
    remainder = args[1:]
    if command == 'model':
        from .modeling.cli import main as run
    elif command == 'generate':
        from .generation.cli import main as run
    else:
        web = ArgumentParser(description='Start the local workbench')
        web.add_argument('--port', type=int, default=8765)
        endpoint = web.parse_args(remainder)
        from .frontend.server import main as serve
        return serve(port=endpoint.port)
    return run(remainder)
