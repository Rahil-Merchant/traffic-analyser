"""Where does a detector fail? Breaks one model's test detections (from predict_coco.py) down by cause.

    python error_analysis.py <coco>/test/_annotations.coco.json deimv2.json --conf 0.41 \
        --groups reports/dataset_261110/groups.csv.gz --out deimv2_errors.md

At the given score threshold, detections are matched to ground truth class-agnostically (greedy, highest score
first, IoU >= 0.5), so a box that is found but labelled wrong counts as a confusion, not as a miss plus a false
positive. Reported:
  - mAP class-aware vs class-agnostic: how much of the loss is classification rather than localisation;
  - confusion matrix (rows = true class, columns = predicted class, plus missed);
  - recall / precision by condition: infrared night (grey frames) vs colour, camera resolution, box size at the
    640 px model input, and per test camera (scene from groups.csv.gz).
`--classes` restricts the condition tables to the classes that matter (default: everything except Pedestrian).
"""
import argparse
import contextlib
import csv
import gzip
import io
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from score_coco import iou_matrix


def is_grey(path):
    im = Image.open(path)
    im.draft('RGB', (160, 90))
    a = np.asarray(im.convert('RGB').resize((64, 36)), np.int16)
    return float(np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean()) < 3.0


def coco_map(gt_json, dets, agnostic):
    gt = json.loads(json.dumps(gt_json))
    if agnostic:
        for a in gt['annotations']:
            a['category_id'] = 0
        gt['categories'] = [{'id': 0, 'name': 'object'}]
        dets = [dict(d, category_id=0) for d in dets]
    with contextlib.redirect_stdout(io.StringIO()):
        g = COCO(); g.dataset = gt; g.createIndex()
        ev = COCOeval(g, g.loadRes(dets), 'bbox'); ev.evaluate(); ev.accumulate(); ev.summarize()
    return ev.stats[0], ev.stats[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('gt'); ap.add_argument('dets')
    ap.add_argument('--conf', type=float, required=True, help="operating threshold (the model's best-F1 one)")
    ap.add_argument('--groups', help='groups.csv.gz from recording_groups.py (scene per image stem)')
    ap.add_argument('--classes', default='-Pedestrian', help='comma list, or -Name to exclude (default -Pedestrian)')
    ap.add_argument('--input', type=int, default=640, help='model input size for the box-size buckets')
    ap.add_argument('--out')
    a = ap.parse_args(argv)

    gt_json = json.loads(Path(a.gt).read_text())
    dets_all = json.loads(Path(a.dets).read_text())
    names = {c['id']: c['name'] for c in gt_json['categories']}
    if a.classes.startswith('-'):
        drop = set(a.classes[1:].split(','))
        focus = {c for c, n in names.items() if n not in drop}
    else:
        focus = {c for c, n in names.items() if n in set(a.classes.split(','))}
    imgs = {im['id']: im for im in gt_json['images']}
    split_dir = Path(a.gt).parent

    with ThreadPoolExecutor(16) as pool:
        grey = dict(zip(imgs, pool.map(lambda i: is_grey(split_dir / imgs[i]['file_name']), imgs)))
    scene = {}
    if a.groups:
        by_stem = {r['stem']: r['scene'] for r in csv.DictReader(gzip.open(a.groups, 'rt'))}
        scene = {i: by_stem.get(Path(im['file_name']).stem, '?') for i, im in imgs.items()}

    # class-agnostic greedy matching at the operating point
    gts = defaultdict(list)
    for ann in gt_json['annotations']:
        gts[ann['image_id']].append(ann)
    preds = defaultdict(list)
    for d in dets_all:
        if d['score'] >= a.conf:
            preds[d['image_id']].append(d)
    conf_m = Counter()       # (true, pred) ; pred None = missed ; true None = background FP
    rec = []                 # per GT: image, class, size_px, found(any class), correct class
    fps = []                 # per unmatched det: image, class
    for img_id in imgs:
        g = gts.get(img_id, []); p = sorted(preds.get(img_id, []), key=lambda d: -d['score'])
        ious = iou_matrix([d['bbox'] for d in p], [x['bbox'] for x in g])
        used = np.zeros(len(g), bool); match = {}
        for i, d in enumerate(p):
            if len(g):
                cand = np.where(~used, ious[i], -1); j = int(cand.argmax())
                if cand[j] >= 0.5:
                    used[j] = True; match[j] = d; continue
            conf_m[(None, d['category_id'])] += 1; fps.append((img_id, d['category_id']))
        s = a.input / max(imgs[img_id]['width'], imgs[img_id]['height'])
        for j, x in enumerate(g):
            d = match.get(j)
            conf_m[(x['category_id'], d['category_id'] if d else None)] += 1
            rec.append((img_id, x['category_id'], (x['bbox'][2] * x['bbox'][3]) ** 0.5 * s, d is not None,
                        d is not None and d['category_id'] == x['category_id']))
        for j, d in match.items():
            if d['category_id'] != g[j]['category_id']:
                fps.append((img_id, d['category_id']))   # a wrong label is also a false positive of its class

    m_aw, m50_aw = coco_map(gt_json, dets_all, False)
    m_ag, m50_ag = coco_map(gt_json, dets_all, True)
    cats = sorted(names)
    short = {c: names[c].replace('Light Vehicle', 'LV').replace('Minibus Taxi', 'Taxi').replace(' Truck', ' Trk') for c in cats}

    L = [f'# Error analysis: {Path(a.dets).name} at score >= {a.conf}', '',
         f'{len(imgs)} images, {len(rec)} GT boxes, matched class-agnostically at IoU >= 0.5.', '',
         '| | mAP50-95 | mAP50 |', '|---|---|---|',
         f'| class-aware | {m_aw:.3f} | {m50_aw:.3f} |', f'| class-agnostic (any class counts) | {m_ag:.3f} | {m50_ag:.3f} |',
         '', 'The gap between the two rows is what wrong class labels cost.', '',
         '## Confusion (rows = true class, share of its GT boxes)', '',
         '| true \\ pred | ' + ' | '.join(short[c] for c in cats) + ' | missed | GT |', '|---|' + '---|' * (len(cats) + 2)]
    for t in cats:
        n = sum(v for (tt, _), v in conf_m.items() if tt == t)
        row = [conf_m[(t, p)] / max(1, n) for p in cats] + [conf_m[(t, None)] / max(1, n)]
        L.append(f'| {short[t]} | ' + ' | '.join(('**%.2f**' if i == cats.index(t) else '%.2f') % v if v >= 0.005 else '·'
                                                  for i, v in enumerate(row)) + f' | {n} |')
    L.append('| background FP | ' + ' | '.join(str(conf_m[(None, p)]) for p in cats) + ' | | |')

    def table(title, key_fn, keys=None):
        R = defaultdict(lambda: [0, 0, 0]); P = defaultdict(lambda: [0, 0])  # gt, found, correct ; dets, fp
        for img, c, size, found, ok in rec:
            if c in focus:
                k = key_fn(img, size); R[k][0] += 1; R[k][1] += found; R[k][2] += ok
        for img, c in fps:
            if c in focus:
                P[key_fn(img, None)][1] += 1
        for img, ds in preds.items():
            for d in ds:
                if d['category_id'] in focus:
                    P[key_fn(img, None)][0] += 1
        keys = keys or sorted(R, key=lambda k: -R[k][0])
        out = ['', f'## {title} ({", ".join(short[c] for c in sorted(focus))})', '',
               '| | images | GT boxes | found (any class) | recall (right class) | precision |', '|---|---|---|---|---|---|']
        nimg = Counter(key_fn(i, None) for i in imgs)
        for k in keys:
            g, f, ok = R[k]
            if not g and not P[k][0]:
                continue
            prec = f'{1 - P[k][1] / P[k][0]:.3f}' if P[k][0] else '–'
            out.append(f'| {k} | {nimg.get(k, "")} | {g} | {f / max(1, g):.3f} | {ok / max(1, g):.3f} | {prec} |')
        return out

    L += table('Night infrared (grey frame) vs colour', lambda i, s: 'IR / grey' if grey[i] else 'colour')
    L += table('Camera resolution', lambda i, s: f"{imgs[i]['width']}x{imgs[i]['height']}")
    buckets = ['<8 px', '8-16 px', '16-32 px', '32-64 px', '64-128 px', '>=128 px']

    def bucket(i, s):
        if s is None:
            return 'n/a'
        return buckets[int(np.searchsorted([8, 16, 32, 64, 128], s, side='right'))]
    L += [l for l in table('Box size at the model input (sqrt area, letterbox scale)', bucket, buckets)]
    if scene:
        L += table('Per test camera', lambda i, s: scene[i])
    text = '\n'.join(L) + '\n'
    print(text)
    if a.out:
        Path(a.out).write_text(text)


if __name__ == '__main__':
    main()
