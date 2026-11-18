"""Repeat-factor sampling for the YOLO train split: images holding rare classes are listed several times.

Ultralytics has no per-class loss weights, so rare classes (Bus 1%, Cyclist 0.1%, Motorcycle 0.3%, Person
with wheel barrow 0.04%) are seen a handful of times per epoch. The LVIS repeat-factor rule fixes that
without touching the loss: a class with image frequency f(c) below the threshold t gets repeat factor
r(c) = sqrt(t / f(c)); an image repeats max(r(c)) over the classes it holds (stochastic rounding, seeded).

    python training/oversample_rare.py data/dataset_261110_split --threshold 0.1 --max-repeat 8

Writes train_rfs.txt (one image path per line, with repeats) and data_rfs.yaml pointing train at it.
Train with --data data/dataset_261110_split/data_rfs.yaml; valid/test are unchanged.
"""
import argparse
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('split_dir', help='output folder of resplit_dataset.py (holds data.yaml)')
    ap.add_argument('--threshold', type=float, default=0.1, help='image frequency below which a class is repeated')
    ap.add_argument('--max-repeat', type=float, default=8.0)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(argv)

    root = Path(a.split_dir)
    data = yaml.safe_load(open(root / 'data.yaml'))
    names = data['names']
    img_dir, lbl_dir = root / 'train' / 'images', root / 'train' / 'labels'
    images = sorted(p for p in os.listdir(img_dir) if p.lower().endswith(('.jpg', '.jpeg', '.png')))
    classes_of = {}
    for f in images:
        lbl = lbl_dir / (Path(f).stem + '.txt')
        classes_of[f] = {int(l.split()[0]) for l in open(lbl) if l.strip()} if lbl.exists() else set()
    freq = Counter(c for cs in classes_of.values() for c in cs)
    n = len(images)
    r = {c: min(a.max_repeat, max(1.0, math.sqrt(a.threshold / (freq[c] / n)))) for c in freq}
    rng = random.Random(a.seed)
    lines, per_class = [], defaultdict(int)
    for f in images:
        rf = max((r[c] for c in classes_of[f]), default=1.0)
        k = int(rf) + (1 if rng.random() < rf - int(rf) else 0)
        lines += [str((img_dir / f).resolve())] * k
        for c in classes_of[f]:
            per_class[c] += k
    (root / 'train_rfs.txt').write_text('\n'.join(lines) + '\n')
    out = dict(data)
    out['train'] = 'train_rfs.txt'
    yaml.safe_dump(out, open(root / 'data_rfs.yaml', 'w'), sort_keys=False)
    print(f'{n} train images -> {len(lines)} entries ({len(lines) / n:.2f}x)')
    print('| class | images | freq | repeat | images/epoch after |')
    print('|---|---|---|---|---|')
    for c in sorted(freq):
        print(f'| {names[c]} | {freq[c]} | {freq[c] / n:.4f} | {r[c]:.2f} | {per_class[c]} |')
    print(f'wrote {root / "train_rfs.txt"} and {root / "data_rfs.yaml"}')


if __name__ == '__main__':
    main()
