"""Re-split a Roboflow export so that no recording leaks between splits and every class keeps its share.

    python training/recording_groups.py data/dataset_261110 --out data/dataset_261110_analysis
    python training/resplit_dataset.py data/dataset_261110 --out data/dataset_261110_split \
        --groups data/dataset_261110_analysis/groups.csv --group-by scene --ratios 0.8 0.1 0.1

Why not Roboflow's own split: frames of one recording are near-duplicates (same camera, same hour, same
vehicles). Spread across train and valid they make validation mAP a measure of memorisation, so the model
that "wins" is the one that overfits hardest. The export's augmented copies of a frame have the same effect.

Method: group-stratified assignment with random restarts.
  1. Images are grouped by camera scene (recording_groups.py clusters thumbnails; --group-by recording keeps
     only whole videos together). Without --groups the recording is parsed from frame-selector file names.
     Roboflow copies of the same frame always share a group.
  2. Groups are shuffled and assigned greedily, largest first, to the split whose per-class instance counts
     are furthest below target (weighted so a rare class steers the choice as strongly as a common one).
  3. Repeat for --restarts seeds and keep the assignment with the lowest deviation from the targets.
  4. valid/test keep one copy per source frame (originals when recognisable); train keeps every copy unless
     --drop-copies is given.
Writes a YOLO dataset (train/valid/test with images/ labels/ and data.yaml), a COCO twin for RF-DETR and
DEIMv2 (coco/<split>/_annotations.coco.json, ids 0..nc-1), split_manifest.csv and split_report.md.
Images are hard-linked when the output is on the same volume, otherwise copied.
"""
import argparse
import csv
import gzip
import json
import os
import sys
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_common import find_export_root, load_dataset

SPLITS = ('train', 'valid', 'test')


def group_items(items, groups_csv=None, level='scene'):
    """-> {group: [items]}. With a groups.csv from recording_groups.py the group is the image's scene
    (camera view, default) or recording; without one it is the recording parsed from the file name.
    Roboflow copies of the same source frame always share a group."""
    key = {}
    if groups_csv:
        opener = gzip.open if str(groups_csv).endswith('.gz') else open
        with opener(groups_csv, 'rt', newline='') as fh:
            for row in csv.DictReader(fh):
                key[row['stem']] = row[level]
    by_source = defaultdict(list)
    for it in items:
        by_source[it.source].append(it)
    by_group = defaultdict(list)
    for its in by_source.values():
        g = next((key[i.stem] for i in its if i.stem in key), its[0].group)
        by_group[g].extend(its)
    if key:
        matched = sum(1 for it in items if it.stem in key)
        if matched < 0.99 * len(items):
            raise SystemExit(f'{groups_csv} covers only {matched}/{len(items)} images: it was made for a different '
                             'export (Roboflow file names differ). Re-run recording_groups.py on this export.')
    return by_group


def class_vector(items, nc):
    v = np.zeros(nc + 1)
    for it in items:
        for c, *_ in it.boxes:
            if c < nc:
                v[c] += 1
    v[nc] = len(items)                   # image count participates like a class
    return v


IMAGE_WEIGHT = 3.0      # the image-count dimension counts as this many classes in the objective (--image-weight)


def assign(groups, nc, ratios, rng, pinned):
    """Greedy largest-first assignment to the split with the largest weighted shortfall.
    pinned: groups forced into train (too large for a 10% split to stay diverse)."""
    names = list(groups)
    rng.shuffle(names)
    vecs = {g: class_vector(groups[g], nc) for g in names}
    total = sum(vecs.values())
    weight = 1 / np.maximum(total, 1)     # rare classes count as much as common ones
    weight[nc] *= IMAGE_WEIGHT
    names.sort(key=lambda g: -vecs[g][nc])
    cur = {s: np.zeros(nc + 1) for s in SPLITS}
    out = {}
    for g in names:
        if g in pinned:
            out[g] = 'train'
            cur['train'] += vecs[g]
            continue
        best, best_score = None, None
        for s, r in zip(SPLITS, ratios):
            target = total * r
            after = cur[s] + vecs[g]
            # shortfall before adding minus overshoot after adding; rare classes dominate via weight
            score = float((weight * (target - cur[s])).sum() / r) - float((weight * np.maximum(after - target, 0)).sum() / r)
            if best_score is None or score > best_score:
                best, best_score = s, score
        out[g] = best
        cur[best] += vecs[g]
    return out, cur, total


def deviation(cur, total, ratios, nc, n_groups, min_groups=10):
    d = 0.0
    for s, r in zip(SPLITS, ratios):
        frac = cur[s] / np.maximum(total, 1)
        w = np.ones(nc + 1); w[nc] = IMAGE_WEIGHT
        d += float((w * np.abs(frac - r)).sum()) / r ** 0.5
    for s in ('valid', 'test'):
        # heavy penalty for a class absent from valid or test although it exists
        d += 5 * int(((cur[s][:nc] == 0) & (total[:nc] > 0)).sum())
        # and for an evaluation split made of very few cameras
        d += 0.3 * max(0, min_groups - n_groups[s])
    return d


def best_split(groups, nc, ratios, seed, restarts, pinned):
    best = None
    for k in range(restarts):
        rng = random.Random(seed * 1000 + k)
        out, cur, total = assign(groups, nc, ratios, rng, pinned)
        n_groups = Counter(out.values())
        d = deviation(cur, total, ratios, nc, n_groups)
        if best is None or d < best[0]:
            best = (d, out, cur, total)
    return best


def apply_class_config(ds, cfg):
    """Re-index labels to the target class list: remap named classes, drop images that hold classes listed
    under drop_images_with, and drop any remaining box whose class is not in the target list (reported)."""
    target = list(cfg['classes'])
    remap = cfg.get('remap', {}) or {}
    drop_with = set(cfg.get('drop_images_with', []) or [])
    old = ds.classes
    new_index = {}
    for i, name in enumerate(old):
        name = remap.get(name, name)
        new_index[i] = target.index(name) if name in target else None
    kept, dropped_images, dropped_boxes = [], Counter(), Counter()
    for it in ds.items:
        names = {old[c] for c, *_ in it.boxes if c < len(old)}
        hit = names & drop_with
        if hit:
            for n in hit:
                dropped_images[n] += 1
            continue
        boxes = []
        for c, *rest in it.boxes:
            j = new_index.get(c)
            if j is None:
                dropped_boxes[old[c] if c < len(old) else c] += 1
                continue
            boxes.append((j, *rest))
        it.boxes = boxes
        kept.append(it)
    ds.items, ds.classes = kept, target
    print(f'class config: {len(old)} -> {len(target)} classes; remapped {remap}; '
          f'dropped images {dict(dropped_images)}; dropped boxes {dict(dropped_boxes)}')


def dedupe(items):
    """One image per source frame: prefer the copy whose name has no augmentation marker, else the first."""
    by_source = defaultdict(list)
    for it in items:
        by_source[it.source].append(it)
    keep = []
    for v in by_source.values():
        v.sort(key=lambda i: (i.rf_hash != '', i.stem))
        keep.append(v[0])
    return keep


def link_or_copy(src, dst):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_yolo(out, split, items, classes):
    img_dir, lbl_dir = out / split / 'images', out / split / 'labels'
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for it in items:
        link_or_copy(Path(it.path), img_dir / Path(it.path).name)
        with open(lbl_dir / (it.stem + '.txt'), 'w') as fh:
            fh.write(''.join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in it.boxes if c < len(classes)))


def write_coco(out, split, items, classes):
    d = out / 'coco' / split
    d.mkdir(parents=True, exist_ok=True)
    images, anns = [], []
    for i, it in enumerate(items, 1):
        link_or_copy(Path(it.path), d / Path(it.path).name)
        images.append(dict(id=i, file_name=Path(it.path).name, width=it.width, height=it.height))
        for c, cx, cy, w, h in it.boxes:
            if c >= len(classes):
                continue
            x0 = max(0.0, (cx - w / 2) * it.width); y0 = max(0.0, (cy - h / 2) * it.height)
            x1 = min(float(it.width), (cx + w / 2) * it.width); y1 = min(float(it.height), (cy + h / 2) * it.height)
            bw, bh = round(x1 - x0, 2), round(y1 - y0, 2)
            if bw <= 0 or bh <= 0:
                continue
            anns.append(dict(id=len(anns) + 1, image_id=i, category_id=c, bbox=[round(x0, 2), round(y0, 2), bw, bh],
                             area=round(bw * bh, 2), iscrowd=0, segmentation=[]))
    coco = dict(info=dict(description=f'traffic {split}', date_created=datetime.now().isoformat(timespec='seconds')),
                licenses=[], categories=[dict(id=i, name=n, supercategory='traffic') for i, n in enumerate(classes)],
                images=images, annotations=anns)
    (d / '_annotations.coco.json').write_text(json.dumps(coco))


def report(out, assignment, groups, result, classes, ratios, kept, pinned=()):
    nc = len(classes)
    _, _, cur, total = result
    L = [f'# Split report `{out}`', '', f'ratios train/valid/test = {ratios}; {len(groups)} groups assigned as whole units; '
         f'{len(pinned)} groups pinned to train for being larger than a split should be', '',
         '| split | groups | images written | instances | ' + ' | '.join(classes) + ' |', '|---|---|---|---|' + '---|' * nc]
    for s in SPLITS:
        n_groups = sum(1 for g in assignment if assignment[g] == s)
        inst = Counter(c for it in kept[s] for c, *_ in it.boxes)
        L.append(f"| {s} | {n_groups} | {len(kept[s])} | {sum(inst.values())} | " + ' | '.join(str(inst.get(c, 0)) for c in range(nc)) + ' |')
    L += ['', '| split | ' + ' | '.join(f'{c} %' for c in classes) + ' |', '|---|' + '---|' * nc]
    for s in SPLITS:
        inst = Counter(c for it in kept[s] for c, *_ in it.boxes)
        L.append(f"| {s} | " + ' | '.join(f"{100 * inst.get(c, 0) / max(total[c], 1):.1f}" for c in range(nc)) + ' |')
    L += ['', '## Groups per split', '']
    for s in SPLITS:
        gs = sorted((g for g in assignment if assignment[g] == s), key=lambda g: -len(groups[g]))
        L.append(f"**{s}** ({len(gs)}): " + ', '.join(f"{g} ({len(groups[g])})" for g in gs[:60]) + (' …' if len(gs) > 60 else ''))
        L.append('')
    return '\n'.join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root', help='extracted Roboflow export')
    ap.add_argument('--out', required=True)
    ap.add_argument('--ratios', type=float, nargs=3, default=(0.8, 0.1, 0.1), metavar=('TRAIN', 'VALID', 'TEST'))
    ap.add_argument('--groups', default=None, help='groups.csv written by recording_groups.py (strongly recommended)')
    ap.add_argument('--group-by', choices=('scene', 'recording'), default='scene',
                    help='scene = whole camera view in one split (default); recording = one video in one split')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--restarts', type=int, default=200)
    ap.add_argument('--drop-copies', action='store_true', help='also keep one copy per source frame in train')
    ap.add_argument('--drop-empty', action='store_true', help='leave out images without any box (default: keep as negatives)')
    ap.add_argument('--min-val-test', type=int, default=1, help='(reporting only) classes with fewer instances in valid/test are flagged')
    ap.add_argument('--max-group-frac', type=float, default=0.04,
                    help='groups holding more than this fraction of all images are pinned to train, so that valid/test stay multi-camera')
    ap.add_argument('--image-weight', type=float, default=6.0,
                    help='weight of the image-count balance relative to one class (raise to hit the image ratios more exactly)')
    ap.add_argument('--class-config', default=None,
                    help='yaml with the target class list, remap and drop rules (e.g. training/classes_production.yaml)')
    ap.add_argument('--no-coco', action='store_true')
    ap.add_argument('--dry-run', action='store_true', help='choose and report the split, write nothing')
    ap.add_argument('--expect-manifest', default=None,
                    help='split_manifest.csv[.gz] the result must reproduce exactly (e.g. the committed one); exit 1 otherwise')
    a = ap.parse_args(argv)
    if abs(sum(a.ratios) - 1) > 1e-6:
        ap.error('--ratios must sum to 1')

    global IMAGE_WEIGHT
    IMAGE_WEIGHT = a.image_weight
    root = find_export_root(a.root)
    out = Path(a.out)
    ds = load_dataset(root)
    if a.class_config:
        apply_class_config(ds, yaml.safe_load(open(a.class_config)))
    classes = ds.classes
    nc = len(classes)
    items = [it for it in ds.items if it.boxes or not a.drop_empty]
    groups = group_items(items, a.groups, a.group_by)
    print(f'{len(items)} images in {len(groups)} groups ({a.group_by if a.groups else "file name"}), {nc} classes')

    pinned = {g for g, its in groups.items() if len(its) > a.max_group_frac * len(items)}
    if pinned:
        print(f'pinned to train (> {a.max_group_frac:.0%} of images): ' + ', '.join(f'{g} ({len(groups[g])})' for g in sorted(pinned, key=lambda g: -len(groups[g]))))
    result = best_split(groups, nc, a.ratios, a.seed, a.restarts, pinned)
    dev, assignment, cur, total = result
    print(f'best of {a.restarts} restarts: deviation {dev:.3f}')

    kept = {s: [] for s in SPLITS}
    for g, s in assignment.items():
        kept[s].extend(groups[g])
    for s in SPLITS:
        if s != 'train' or a.drop_copies:
            kept[s] = dedupe(kept[s])
        kept[s].sort(key=lambda i: i.path)

    if a.dry_run:
        print(report(out, assignment, groups, result, classes, a.ratios, kept, pinned).split('## Groups')[0])
        return
    out.mkdir(parents=True, exist_ok=True)
    for stale in [out / s for s in SPLITS] + [out / 'coco']:
        if stale.exists():
            shutil.rmtree(stale)          # our own links from a previous run; never the source export
    for s in SPLITS:
        write_yolo(out, s, kept[s], classes)
        if not a.no_coco:
            write_coco(out, s, kept[s], classes)
    yaml.safe_dump(dict(path=str(out.resolve()), train='train/images', val='valid/images', test='test/images',
                        nc=nc, names={i: n for i, n in enumerate(classes)}),
                   open(out / 'data.yaml', 'w'), sort_keys=False)
    with open(out / 'split_manifest.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['image', 'split', 'group', 'source', 'original_split', 'boxes'])
        for s in SPLITS:
            for it in kept[s]:
                w.writerow([Path(it.path).name, s, it.group, it.source, it.split, len(it.boxes)])
    rep = report(out, assignment, groups, result, classes, a.ratios, kept, pinned)
    (out / 'split_report.md').write_text(rep)
    print(rep)
    if a.expect_manifest:
        def pairs(path):
            opener = gzip.open if str(path).endswith('.gz') else open
            with opener(path, 'rt', newline='') as fh:
                return {(r['image'], r['split']) for r in csv.DictReader(fh)}
        want, got = pairs(a.expect_manifest), pairs(out / 'split_manifest.csv')
        if want != got:
            raise SystemExit(f'split differs from {a.expect_manifest}: {len(want - got)} expected assignments missing, '
                             f'{len(got - want)} unexpected')
        print(f'split matches {a.expect_manifest} ({len(got)} images)')
    for s in ('valid', 'test'):
        low = [classes[c] for c in range(nc) if 0 < total[c] and cur[s][c] < a.min_val_test]
        if low:
            print(f'warning: {s} has fewer than {a.min_val_test} instances of: {low}')


if __name__ == '__main__':
    main()
