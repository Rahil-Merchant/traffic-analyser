"""Run DEIMv2's train.py with this folder's extra config ops registered (CCTVPixelAug).

    cd <DEIMv2 repo> && python <this folder>/deim_launch.py -c configs/... --use-amp -t <ckpt>

Arguments go to train.py unchanged. Run from the repo root (train.py and the engine package are found there).
Spawned dataloader workers inherit sys.path, so they can import deim_cctv_aug to unpickle the transform.
"""
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path.cwd()
if not (REPO / 'train.py').exists() or not (REPO / 'engine').is_dir():
    sys.exit(f'run from the DEIMv2 repo root (no train.py/engine in {REPO})')
sys.path[:0] = [str(REPO), str(HERE)]

import deim_cctv_aug  # noqa: E402,F401  registers CCTVPixelAug

if __name__ == '__main__':
    sys.argv = [str(REPO / 'train.py')] + sys.argv[1:]
    runpy.run_path(str(REPO / 'train.py'), run_name='__main__')
