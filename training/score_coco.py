"""Score detection files from `predict_coco.py` against one COCO ground truth, all with the same rules.

    python score_coco.py <coco>/test/_annotations.coco.json yolo=yolo.json rfdetr=rfdetr.json deimv2=deimv2.json \
        --out test_comparison.md

Reports per model: COCO mAP50-95 / mAP50 / mAP75 / AP-small / AR100 (pycocotools, maxDets 100), per-class AP50-95,
and precision / recall per class at the tracker's operating point (score >= --op-conf, default 0.25; greedy
one-to-one matching at IoU >= 0.5 within a class, highest score first). The operating point is what the counter
sees; mAP at conf 0.001 is not.
"""
import argparse
import contextlib
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def iou_matrix(a, b):
    """a: (n,4) xywh, b: (m,4) xywh -> (n,m) IoU."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    a = np.asarray(a, float); b = np.asarray(b, float)
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    return inter / (a[:, 2:].prod(1)[:, None] + b[:, 2:].prod(1)[None] - inter + 1e-9)


def match(gt, dets, conf_min, iou_thr=0.5):
    """Greedy class-wise matching, highest score first, of every detection >= conf_min.

    Greedy matching in score order means the decisions for detections above any threshold t >= conf_min do not
    depend on the ones below it, so one pass gives precision/recall at every threshold.
    Returns (scores, is_tp, classes) arrays and {class: number of GT boxes}.
    """
    gts = defaultdict(list)
    for ann in gt.dataset['annotations']:
        if not ann.get('iscrowd', 0):
            gts[(ann['image_id'], ann['category_id'])].append(ann['bbox'])
    preds = defaultdict(list)
    for d in dets:
        if d['score'] >= conf_min:
            preds[(d['image_id'], d['category_id'])].append((d['score'], d['bbox']))
    npos = defaultdict(int)
    for (img, c), boxes in gts.items():
        npos[c] += len(boxes)
    S, TP, C = [], [], []
    for key, p in preds.items():
        p.sort(key=lambda t: -t[0])
        g = gts.get(key, [])
        ious = iou_matrix([b for _, b in p], g)
        used = np.zeros(len(g), bool)
        for i, (s, _) in enumerate(p):
            hit = False
            if len(g):
                cand = np.where(~used, ious[i], -1)
                j = int(cand.argmax())
                if cand[j] >= iou_thr:
                    used[j] = hit = True
            S.append(s); TP.append(hit); C.append(key[1])
    return np.array(S), np.array(TP, bool), np.array(C), npos


def operating_point(m, cat_ids, conf):
    S, TP, C, npos = m
    keep = S >= conf
    out = {}
    for c in cat_ids:
        sel = keep & (C == c)
        tp = int(TP[sel].sum()); fp = int(sel.sum()) - tp
        out[c] = dict(precision=tp / max(1, tp + fp), recall=tp / max(1, npos[c]), tp=tp, fp=fp, gt=npos[c])
    tp = int(TP[keep].sum()); fp = int(keep.sum()) - tp; n = sum(npos.values())
    out['all'] = dict(precision=tp / max(1, tp + fp), recall=tp / max(1, n), tp=tp, fp=fp, gt=n, conf=conf)
    return out


def best_f1(m, cat_ids, grid=np.round(np.arange(0.05, 0.91, 0.01), 2)):
    best = None
    for t in grid:
        o = operating_point(m, cat_ids, float(t))
        P, R = o['all']['precision'], o['all']['recall']
        f1 = 2 * P * R / max(1e-9, P + R)
        if best is None or f1 > best[0]:
            best = (f1, o)
    best[1]['all']['f1'] = best[0]
    return best[1]


def score(gt, dets, cat_ids):
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dets) if dets else None
        if dt is None:
            raise SystemExit('no detections')
        ev = COCOeval(gt, dt, 'bbox')
        ev.evaluate(); ev.accumulate(); ev.summarize()
    s = ev.stats
    prec = ev.eval['precision']  # [T, R, K, A, M]
    per_class = {}
    for k, c in enumerate(ev.params.catIds):
        p = prec[:, :, k, 0, -1]
        per_class[c] = float(p[p > -1].mean()) if (p > -1).any() else float('nan')
    return dict(map=s[0], map50=s[1], map75=s[2], ap_small=s[3], ap_medium=s[4], ap_large=s[5], ar100=s[8],
                per_class=per_class)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('gt')
    ap.add_argument('runs', nargs='+', help='name=detections.json')
    ap.add_argument('--op-conf', type=float, default=0.25)
    ap.add_argument('--out', help='write a markdown report here (and a .json twin)')
    a = ap.parse_args(argv)

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(a.gt)
    cats = sorted(gt.getCatIds())
    names = {c: gt.cats[c]['name'] for c in cats}
    report = {}
    for spec in a.runs:
        name, path = spec.split('=', 1)
        dets = json.loads(Path(path).read_text())
        r = score(gt, dets, cats)
        m = match(gt, dets, conf_min=min(0.05, a.op_conf))
        r['op'] = operating_point(m, cats, a.op_conf)
        r['f1'] = best_f1(m, cats)
        meta = Path(path).with_suffix('.meta.json')
        if meta.exists():
            r['meta'] = json.loads(meta.read_text())
        report[name] = r

    L = [f"# Test-split comparison ({len(gt.imgs)} images, {len(gt.anns)} boxes)", '',
         f"COCO metrics from pycocotools (maxDets 100); operating point = score >= {a.op_conf}, IoU >= 0.5.", '',
         '| model | mAP50-95 | mAP50 | mAP75 | AP small | AR100 | P@op | R@op | ms/img (model) |',
         '|---|---|---|---|---|---|---|---|---|']
    for n, r in report.items():
        ms = r.get('meta', {}).get('model_ms_per_image', '')
        L.append(f"| {n} | {r['map']:.3f} | {r['map50']:.3f} | {r['map75']:.3f} | {r['ap_small']:.3f} | {r['ar100']:.3f} | "
                 f"{r['op']['all']['precision']:.3f} | {r['op']['all']['recall']:.3f} | {ms} |")
    L += ['', '## Each model at its own best-F1 threshold', '',
          'Score scales differ between model families (DETR scores run lower), so a shared threshold favours one of',
          'them; this is the fairer operating-point comparison. The threshold is the one to give the tracker.', '',
          '| model | threshold | precision | recall | F1 | ' + ' | '.join(f'R {names[c]}' for c in cats) + ' |',
          '|---|---|---|---|---|' + '---|' * len(cats)]
    for n, r in report.items():
        f = r['f1']
        L.append(f"| {n} | {f['all']['conf']:.2f} | {f['all']['precision']:.3f} | {f['all']['recall']:.3f} | "
                 f"{f['all']['f1']:.3f} | " + ' | '.join(f"{f[c]['recall']:.2f}" for c in cats) + ' |')
    L += ['', '## AP50-95 per class', '', '| class | GT boxes | ' + ' | '.join(report) + ' |',
          '|---|---|' + '---|' * len(report)]
    any_r = next(iter(report.values()))
    for c in cats:
        L.append(f"| {names[c]} | {any_r['op'][c]['gt']} | " + ' | '.join(f"{r['per_class'][c]:.3f}" for r in report.values()) + ' |')
    L += ['', f'## Precision / recall per class at score >= {a.op_conf}', '',
          '| class | ' + ' | '.join(f'{n} P | {n} R' for n in report) + ' |', '|---|' + '---|---|' * len(report)]
    for c in cats:
        L.append(f"| {names[c]} | " + ' | '.join(f"{r['op'][c]['precision']:.3f} | {r['op'][c]['recall']:.3f}" for r in report.values()) + ' |')
    text = '\n'.join(L) + '\n'
    print(text)
    if a.out:
        Path(a.out).write_text(text)
        rename = lambda d: {(names[c] if c != 'all' else c): v for c, v in d.items()}
        js = {n: dict(r, per_class=rename(r['per_class']), op=rename(r['op']), f1=rename(r['f1']))
              for n, r in report.items()}
        Path(a.out).with_suffix('.json').write_text(json.dumps(js, indent=2))


if __name__ == '__main__':
    main()
