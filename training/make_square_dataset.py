"""Write a stretched-square copy of a YOLO split: every image resized to SIZE x SIZE without letterboxing.

Why: production letterboxes a 16:9 frame into 640x640, so only 640x360 of the input is picture and the rest is
padding. Resizing the frame to the full 640x640 instead gives 1.78x the vertical resolution at the same model
cost (the engine already runs at 640x640). That is what RF-DETR and DEIMv2 do by default, and small objects
(pedestrians are 14x28 px at the median after the letterbox) are where every model is weakest.

YOLO labels are normalised to the image, so they are unchanged (hard-linked); IoU is invariant under an
axis-aligned stretch, so mAP on the stretched valid/test sets is comparable with the letterboxed numbers.
Ultralytics resizes the long side to imgsz at load time anyway, so storing at SIZE loses nothing.

    python make_square_dataset.py D:/training/data/dataset_261110_split --size 640
    # -> D:/training/data/dataset_261110_split_sq640/{train,valid,test}/{images,labels}, data.yaml, data_rfs.yaml, train_rfs.txt

Deploying a model trained on this needs the pipeline to resize (not letterbox) to 640x640 and to scale boxes
back by (W/640, H/640).
"""
import argparse
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

SPLITS = ('train', 'valid', 'test')


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def convert(src, dst, size, quality):
    if dst.exists():
        return 0
    im = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f'cannot read {src}')
    im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
    tmp = dst.with_name(dst.stem + '.tmp' + dst.suffix)
    if not cv2.imwrite(str(tmp), im, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise RuntimeError(f'cannot write {dst}')
    os.replace(tmp, dst)
    return 1


def rewrite_yaml(src, dst, new_root):
    text = Path(src).read_text()
    text = re.sub(r'(?m)^path:.*$', f'path: {Path(new_root).as_posix()}', text)
    Path(dst).write_text(text)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('split_dir', help='YOLO split from resplit_dataset.py (train/ valid/ test/, data.yaml)')
    ap.add_argument('--size', type=int, default=640)
    ap.add_argument('--out', help='default: <split_dir>_sq<size>')
    ap.add_argument('--quality', type=int, default=95, help='JPEG quality')
    ap.add_argument('--threads', type=int, default=16)
    a = ap.parse_args(argv)

    root = Path(a.split_dir)
    out = Path(a.out) if a.out else root.with_name(f'{root.name}_sq{a.size}')
    total = 0
    with ThreadPoolExecutor(a.threads) as pool:
        for s in SPLITS:
            (out / s / 'images').mkdir(parents=True, exist_ok=True)
            (out / s / 'labels').mkdir(parents=True, exist_ok=True)
            # threaded: per-file overhead (antivirus, USB disks on Windows) makes a serial loop ~10 files/s
            todo = [(lab, out / s / 'labels' / lab.name) for lab in (root / s / 'labels').glob('*.txt')]
            list(pool.map(lambda t: t[1].exists() or link_or_copy(*t), todo))
            ims = sorted((root / s / 'images').iterdir())
            odd = [p.name for p in ims if p.suffix.lower() != '.jpg']
            if odd:   # train_rfs.txt lists file names; a renamed suffix would break it
                raise SystemExit(f'{len(odd)} non-.jpg images in {s}, e.g. {odd[0]}')
            done = sum(pool.map(lambda p: convert(p, out / s / 'images' / p.name, a.size, a.quality), ims))
            print(f'{s}: {len(ims)} images ({done} written)', flush=True)
            total += len(ims)

    for y in ('data.yaml', 'data_rfs.yaml'):
        if (root / y).exists():
            rewrite_yaml(root / y, out / y, out)
    rfs = root / 'train_rfs.txt'
    if rfs.exists():
        lines = []
        for line in rfs.read_text().splitlines():
            if line.strip():
                name = re.split(r'[\\/]', line.strip())[-1]
                lines.append(str(out / 'train' / 'images' / name))
        (out / 'train_rfs.txt').write_text('\n'.join(lines) + '\n')
    (out / 'GEOMETRY.txt').write_text(f'stretched to {a.size}x{a.size} (no letterbox) from {root}\n'
                                      f'inference: resize the full frame to {a.size}x{a.size}, boxes * (W/{a.size}, H/{a.size})\n')
    print(f'wrote {out} ({total} images)')


if __name__ == '__main__':
    main()
