"""Analyse a Roboflow export (YOLO or COCO) and write a markdown report plus the raw numbers as JSON.

    python training/analyse_dataset.py data/dataset_261110/export --out data/dataset_261110/analysis

Reports: images and instances per split and class, class imbalance, boxes per image, image resolutions,
object sizes in pixels at the training resolution (COCO small/medium/large bands), per-class box size and
aspect ratio, label problems, Roboflow augmentation copies, recordings (groups) and whether the export's
own split leaks a recording across train/valid/test.
"""
import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_common import find_export_root, load_dataset


def letterbox_scale(w, h, size):
    return size / max(w, h)


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float('nan')


def analyse(ds, train_size):
    classes = ds.classes
    nc = len(classes)
    splits = ds.by_split()
    rep = {'root': ds.root, 'layout': ds.layout, 'classes': classes, 'splits': {}, 'train_size': train_size}

    # ---- per split: images, instances per class, boxes per image, empties
    for split, items in splits.items():
        inst = Counter(c for it in items for c, *_ in it.boxes)
        per_img = [len(it.boxes) for it in items]
        rep['splits'][split] = dict(
            images=len(items), instances=sum(per_img), empty_images=sum(1 for n in per_img if n == 0),
            boxes_per_image=dict(mean=float(np.mean(per_img)) if per_img else 0, median=pct(per_img, 50), p95=pct(per_img, 95),
                                 max=max(per_img) if per_img else 0),
            per_class={classes[c]: inst.get(c, 0) for c in range(nc)},
            images_with_class={classes[c]: sum(1 for it in items if any(b[0] == c for b in it.boxes)) for c in range(nc)},
        )

    # ---- whole dataset
    items = ds.items
    total = Counter(c for it in items for c, *_ in it.boxes)
    rep['per_class_total'] = {classes[c]: total.get(c, 0) for c in range(nc)}
    nonzero = [v for v in total.values() if v]
    rep['imbalance_ratio'] = (max(nonzero) / min(nonzero)) if nonzero else None
    rep['unused_class_ids'] = [classes[c] for c in range(nc) if total.get(c, 0) == 0]
    rep['unknown_class_ids'] = sorted({c for c in total if c >= nc})

    # ---- image resolutions
    sizes = Counter((it.width, it.height) for it in items)
    rep['image_sizes'] = [{'w': w, 'h': h, 'images': n} for (w, h), n in sizes.most_common(12)]

    # ---- object sizes in pixels after letterboxing to train_size
    px_w, px_h, px_cls, ar = [], [], [], []
    for it in items:
        s = letterbox_scale(it.width, it.height, train_size)
        for c, cx, cy, w, h in it.boxes:
            pw, ph = w * it.width * s, h * it.height * s
            px_w.append(pw); px_h.append(ph); px_cls.append(c); ar.append(w * it.width / max(h * it.height, 1e-6))
    px_w, px_h, px_cls, ar = map(np.array, (px_w, px_h, px_cls, ar))
    area = np.sqrt(px_w * px_h) if len(px_w) else px_w
    rep['object_size_px'] = dict(
        sqrt_area=dict(p5=pct(area, 5), p25=pct(area, 25), median=pct(area, 50), p75=pct(area, 75), p95=pct(area, 95)),
        small_lt32=int((area < 32).sum()), medium_32_96=int(((area >= 32) & (area < 96)).sum()), large_ge96=int((area >= 96).sum()),
        tiny_lt16=int((area < 16).sum()),
    )
    rep['per_class_size_px'] = {}
    for c in range(nc):
        m = px_cls == c
        if m.any():
            rep['per_class_size_px'][classes[c]] = dict(
                n=int(m.sum()), sqrt_area_median=pct(area[m], 50), sqrt_area_p10=pct(area[m], 10), sqrt_area_p90=pct(area[m], 90),
                width_median=pct(px_w[m], 50), height_median=pct(px_h[m], 50), aspect_median=pct(ar[m], 50),
                frac_small=float((area[m] < 32).mean()))

    # ---- label issues
    issues = Counter()
    for it in items:
        for i in it.issues:
            issues[i.split(':')[-1].strip()] += 1
    rep['label_issues'] = dict(issues)
    rep['images_with_issues'] = sum(1 for it in items if it.issues)

    # ---- Roboflow copies and recordings
    by_source = defaultdict(list)
    for it in items:
        by_source[it.source].append(it)
    copies = {s: len(v) for s, v in by_source.items() if len(v) > 1}
    rep['sources'] = dict(unique=len(by_source), with_multiple_copies=len(copies),
                          extra_copies=sum(n - 1 for n in copies.values()),
                          copies_split_across_splits=sum(1 for v in by_source.values() if len({i.split for i in v}) > 1))

    by_group = defaultdict(list)
    for it in items:
        by_group[it.group].append(it)
    gsizes = sorted((len(v) for v in by_group.values()), reverse=True)
    leaked = {g: Counter(i.split for i in v) for g, v in by_group.items() if len({i.split for i in v}) > 1}
    rep['groups'] = dict(
        count=len(by_group), largest=gsizes[:10], median_size=pct(gsizes, 50),
        singletons=sum(1 for n in gsizes if n == 1),
        frames_with_number=sum(1 for it in items if it.frame >= 0),
        leaked_across_splits=len(leaked),
        leaked_images=sum(sum(c.values()) for c in leaked.values()),
        examples=[{'group': g, **c} for g, c in list(leaked.items())[:8]],
        per_group_classes={g: dict(Counter(ds.classes[b[0]] for i in v for b in i.boxes if b[0] < nc)) for g, v in by_group.items()}
        if len(by_group) <= 400 else None,
    )

    # ---- class co-occurrence of rare classes: how many images contain each class in total
    rep['images_per_class'] = {classes[c]: sum(1 for it in items if any(b[0] == c for b in it.boxes)) for c in range(nc)}
    return rep


def fmt(x, nd=1):
    return '-' if x is None or (isinstance(x, float) and math.isnan(x)) else (f'{x:.{nd}f}' if isinstance(x, float) else str(x))


def markdown(rep):
    L = []
    cls = rep['classes']
    L += [f"# Dataset analysis: `{rep['root']}`", '',
          f"Layout: {rep['layout']} · classes: {len(cls)} · object sizes measured after letterboxing to {rep['train_size']} px", '']
    L += ['## Splits as exported', '', '| split | images | empty | instances | boxes/img mean | median | p95 | max |', '|---|---|---|---|---|---|---|---|']
    for s, d in rep['splits'].items():
        b = d['boxes_per_image']
        L.append(f"| {s} | {d['images']} | {d['empty_images']} | {d['instances']} | {fmt(b['mean'])} | {fmt(b['median'])} | {fmt(b['p95'])} | {b['max']} |")
    L += ['', '## Instances per class', '', '| class | total | % | images | ' + ' | '.join(rep['splits']) + ' |',
          '|---|---|---|---|' + '---|' * len(rep['splits'])]
    tot = sum(rep['per_class_total'].values()) or 1
    for c in cls:
        L.append(f"| {c} | {rep['per_class_total'][c]} | {100 * rep['per_class_total'][c] / tot:.1f} | {rep['images_per_class'][c]} | "
                 + ' | '.join(str(rep['splits'][s]['per_class'][c]) for s in rep['splits']) + ' |')
    L += ['', f"Imbalance ratio (largest / smallest non-empty class): **{fmt(rep['imbalance_ratio'])}**"]
    if rep['unused_class_ids']:
        L.append(f"Classes with no instances: {', '.join(rep['unused_class_ids'])}")
    if rep['unknown_class_ids']:
        L.append(f"Label ids outside the class list: {rep['unknown_class_ids']}")
    L += ['', '## Image sizes', '', '| width | height | images |', '|---|---|---|']
    L += [f"| {d['w']} | {d['h']} | {d['images']} |" for d in rep['image_sizes']]
    o = rep['object_size_px']
    L += ['', f"## Object sizes at {rep['train_size']} px (sqrt of box area)", '',
          f"p5 {fmt(o['sqrt_area']['p5'])} · p25 {fmt(o['sqrt_area']['p25'])} · median {fmt(o['sqrt_area']['median'])} · p75 {fmt(o['sqrt_area']['p75'])} · p95 {fmt(o['sqrt_area']['p95'])} px",
          '', f"tiny (<16 px): {o['tiny_lt16']} · small (<32): {o['small_lt32']} · medium (32–96): {o['medium_32_96']} · large (≥96): {o['large_ge96']}", '',
          '| class | n | sqrt-area p10 | median | p90 | median w×h | aspect | % small |', '|---|---|---|---|---|---|---|---|']
    for c, d in rep['per_class_size_px'].items():
        L.append(f"| {c} | {d['n']} | {fmt(d['sqrt_area_p10'])} | {fmt(d['sqrt_area_median'])} | {fmt(d['sqrt_area_p90'])} | "
                 f"{fmt(d['width_median'], 0)}×{fmt(d['height_median'], 0)} | {fmt(d['aspect_median'], 2)} | {100 * d['frac_small']:.0f} |")
    L += ['', '## Label problems', '', f"images with issues: {rep['images_with_issues']}"]
    L += [f"- {k}: {v}" for k, v in rep['label_issues'].items()] or ['- none']
    s, g = rep['sources'], rep['groups']
    L += ['', '## Duplicates and recordings', '',
          f"- unique source frames: {s['unique']}; frames with several Roboflow copies: {s['with_multiple_copies']} "
          f"(+{s['extra_copies']} extra images); copies of one frame spread over different splits: **{s['copies_split_across_splits']}**",
          f"- recordings (groups) recognised from file names: {g['count']} (singletons {g['singletons']}, median size {fmt(g['median_size'])}, "
          f"largest {g['largest']}); frames with a parsable frame number: {g['frames_with_number']}",
          f"- recordings present in more than one split of the export: **{g['leaked_across_splits']}** ({g['leaked_images']} images)"]
    for e in g['examples']:
        L.append(f"  - {e}")
    return '\n'.join(L) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root', help='extracted Roboflow export (folder with data.yaml or train/_annotations.coco.json)')
    ap.add_argument('--out', default=None, help='output folder (default <root>/../analysis)')
    ap.add_argument('--train-size', type=int, default=640, help='letterbox size used for object-size statistics')
    a = ap.parse_args(argv)
    root = find_export_root(a.root)
    out = Path(a.out or Path(root).parent / 'analysis')
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(root)
    rep = analyse(ds, a.train_size)
    (out / 'analysis.json').write_text(json.dumps(rep, indent=1, default=str))
    (out / 'analysis.md').write_text(markdown(rep))
    print(markdown(rep))
    print(f'written to {out}')


if __name__ == '__main__':
    main()
