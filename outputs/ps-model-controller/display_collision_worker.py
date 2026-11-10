"""Bounded-process CoACD entry, so a pathological mesh cannot hang the batch."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import sys
from pathlib import Path
from display_batch.assets import make_proxy

if __name__=='__main__':
    make_proxy(Path(sys.argv[1]),Path(sys.argv[2]))
