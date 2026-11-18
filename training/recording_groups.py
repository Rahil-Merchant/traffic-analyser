"""Recover which recording (and which camera scene) every image came from, so the split can keep them together.

    python training/recording_groups.py data/dataset_261110 --out data/dataset_261110_analysis

Three naming families exist in the export:
  <video>_<6 hex>_<7-digit frame>   frame-selector frames: the recording is in the name.
  <number>                           the original dataset, numbered sequentially; consecutive numbers are
                                     consecutive frames of one recording, but the boundaries are not recorded.
  vlcsnap-YYYY-MM-DD-HHhMMmSSsmmm    VLC snapshots taken while a video played; a day holds several videos.
For the last two, images are ordered (by number, or by day and clock time), a 32x18 grey thumbnail is taken
of each, and a recording boundary is placed wherever the correlation between neighbours drops (a static
camera keeps its background from frame to frame; a cut to another camera does not). A gap of more than
--vlc-gap minutes between snapshots also starts a new recording. Recordings are then clustered into scenes
(same camera view across recordings) by thumbnail similarity, for the stricter --group-by scene split.

Writes <out>/groups.csv (stem, family, recording, scene), <out>/signatures.npz (cached thumbnails) and
<out>/recording_boundaries.jpg (last frame before / first frame after each detected cut, for eyeballing).
"""
import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_common import find_export_root, load_dataset  # noqa: E402

THUMB = (32, 18)   # 16:9 grey thumbnail, mean-removed and L2-normalised, so a dot product is a correlation
_VLC = re.compile(r'^vlcsnap-(\d{4})-(\d{2})-(\d{2})-(\d{2})h(\d{2})m(\d{2})s(\d{3})(?:-Copy)?$')
_NUM = re.compile(r'^\d+$')


def family(item):
    if item.frame >= 0 and item.group != item.source:
        return 'selector'
    if _VLC.match(item.source):
        return 'vlcsnap'
    if _NUM.match(item.source):
        return 'numbered'
    return 'other'


def signature(path):
    im = Image.open(path)
    im.draft('L', (THUMB[0] * 4, THUMB[1] * 4))
    a = np.asarray(im.convert('L').resize(THUMB, Image.BILINEAR), dtype=np.float32).ravel()
    a -= a.mean()
    n = np.linalg.norm(a)
    return a / n if n else a


def signatures(items, cache, workers):
    sigs = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        sigs = dict(zip(z['stems'].tolist(), z['sigs']))
    todo = [it for it in items if it.stem not in sigs]
    if todo:
        print(f'thumbnails: {len(todo)} to compute ({len(sigs)} cached)')
        with ThreadPoolExecutor(workers) as ex:
            for i, (it, s) in enumerate(zip(todo, ex.map(lambda it: signature(it.path), todo)), 1):
                sigs[it.stem] = s
                if i % 2000 == 0:
                    print(f'  {i}/{len(todo)}', flush=True)
        np.savez_compressed(cache, stems=np.array(list(sigs)), sigs=np.stack(list(sigs.values())))
    return sigs


def vlc_seconds(source):
    y, mo, d, h, mi, s, ms = (int(x) for x in _VLC.match(source).groups())
    return y, mo, d, h * 3600 + mi * 60 + s + ms / 1000


def fragments(ordered, sigs, cut, gap_fn=None):
    """Conservative over-segmentation of an ordered sequence: a new fragment starts at every resolution
    change, every neighbour correlation below `cut` and wherever gap_fn fires. Fragments of one recording
    that were cut by a passing truck or headlights are re-joined by the scene clustering below."""
    frags, cur = [], [ordered[0]]
    for prev, it in zip(ordered, ordered[1:]):
        corr = float(sigs[prev.stem] @ sigs[it.stem])
        if corr < cut or (prev.width, prev.height) != (it.width, it.height) or (gap_fn and gap_fn(prev, it)):
            frags.append(cur)
            cur = [it]
        else:
            cur.append(it)
    frags.append(cur)
    return frags


def cluster_scenes(frag_sigs, thresh):
    """Average-linkage agglomerative clustering on cosine distance of fragment mean thumbnails.
    scipy if present, else a greedy nearest-centre pass."""
    names = list(frag_sigs)
    X = np.stack([frag_sigs[n] for n in names])
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        labels = fcluster(linkage(X, method='average', metric='cosine'), t=thresh, criterion='distance')
    except ImportError:
        labels = np.zeros(len(names), int)
        centres = []
        for i, x in enumerate(X):
            if centres:
                d = 1 - np.stack(centres) @ x
                j = int(d.argmin())
                if d[j] < thresh:
                    labels[i] = j + 1
                    continue
            centres.append(x)
            labels[i] = len(centres)
    return {n: int(l) for n, l in zip(names, labels)}


def absorb_small_scenes(frag_sigs, scene_of_frag, frag_list, min_images, max_dist):
    """Night infrared frames fragment into many tiny scenes (headlight glare dominates the thumbnail).
    Tiny scenes are folded into the nearest larger scene centroid when close enough; this can only
    merge, never split, so it is safe for leakage and only costs stratification granularity."""
    members = defaultdict(list)
    for i, sc in scene_of_frag.items():
        members[sc].append(i)
    n_images = {sc: sum(len(frag_list[i][2]) for i in idx) for sc, idx in members.items()}
    cent = {}
    for sc, idx in members.items():
        v = np.mean([frag_sigs[i] for i in idx], axis=0)
        cent[sc] = v / max(np.linalg.norm(v), 1e-9)
    big = [sc for sc in members if n_images[sc] >= min_images]
    if not big:
        return scene_of_frag
    B = np.stack([cent[sc] for sc in big])
    merged = 0
    out = dict(scene_of_frag)
    for sc, idx in members.items():
        if n_images[sc] >= min_images:
            continue
        d = 1 - B @ cent[sc]
        j = int(d.argmin())
        if d[j] < max_dist:
            for i in idx:
                out[i] = big[j]
            merged += 1
    print(f'absorbed {merged} scenes smaller than {min_images} images into their nearest scene')
    return out


def runs(frags, scene_of, prefix):
    """Recordings = consecutive fragments of the same scene merged back together."""
    rec, cuts, cur = {}, [], [frags[0]]
    for a, b in zip(frags, frags[1:]):
        if scene_of[id(a)] == scene_of[id(b)]:
            cur.append(b)
        else:
            cuts.append((a[-1], b[0]))
            for f in cur:
                for it in f:
                    rec[it.stem] = f'{prefix}{cur[0][0].source}'
            cur = [b]
    for f in cur:
        for it in f:
            rec[it.stem] = f'{prefix}{cur[0][0].source}'
    return rec, cuts


def montage(cuts, path, max_rows=60):
    if not cuts:
        return
    rows = cuts[:: max(1, len(cuts) // max_rows)][:max_rows]
    W, H = 256, 144
    out = Image.new('RGB', (W * 2 + 8, H * len(rows)), 'black')
    for r, (a, b) in enumerate(rows):
        for c, it in enumerate((a, b)):
            im = Image.open(it.path)
            im.draft('RGB', (W * 2, H * 2))
            out.paste(im.convert('RGB').resize((W, H)), (c * (W + 8), r * H))
    out.save(path, quality=75)


def scene_montage(items, scene_of_stem, path, cols=10):
    """One frame per scene, largest scenes first, with the scene id and image count."""
    from PIL import ImageDraw
    by = defaultdict(list)
    for it in items:
        by[scene_of_stem[it.stem]].append(it)
    order = sorted(by, key=lambda k: -len(by[k]))
    W, H = 192, 108
    rows = (len(order) + cols - 1) // cols
    out = Image.new('RGB', (W * cols, H * rows), 'black')
    draw = ImageDraw.Draw(out)
    for k, sc in enumerate(order):
        it = by[sc][len(by[sc]) // 2]
        im = Image.open(it.path); im.draft('RGB', (W * 2, H * 2))
        x, y = (k % cols) * W, (k // cols) * H
        out.paste(im.convert('RGB').resize((W, H)), (x, y))
        draw.text((x + 3, y + 2), f'{sc} n={len(by[sc])}', fill='yellow')
    out.save(path, quality=75)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root')
    ap.add_argument('--out', required=True)
    ap.add_argument('--cut', type=float, default=0.5, help='neighbour correlation below this starts a new fragment')
    ap.add_argument('--scene-thresh', type=float, default=0.2, help='cosine distance (average linkage) joining fragments into one scene')
    ap.add_argument('--vlc-gap', type=float, default=30, help='minutes between snapshots that start a new fragment')
    ap.add_argument('--absorb-below', type=int, default=30, help='scenes with fewer images are merged into the nearest scene')
    ap.add_argument('--absorb-dist', type=float, default=0.5, help='...if the centroid cosine distance is below this')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(find_export_root(a.root))
    sigs = signatures(ds.items, out / 'signatures.npz', a.workers)

    fams = defaultdict(list)
    for it in ds.items:
        fams[family(it)].append(it)
    print({k: len(v) for k, v in fams.items()})

    # 1. fragments. Named recordings (frame selector, dav-named frames) are one fragment per video.
    named = defaultdict(list)
    for it in fams['selector'] + fams['other']:
        named[it.group].append(it)
    frag_list = [('named', g, its) for g, its in named.items()]
    numbered = sorted(fams['numbered'], key=lambda it: int(it.source))
    if numbered:
        frag_list += [('num', None, f) for f in fragments(numbered, sigs, a.cut)]
    vlc = sorted(fams['vlcsnap'], key=lambda it: vlc_seconds(it.source))
    if vlc:
        def gap(p, q):
            tp, tq = vlc_seconds(p.source), vlc_seconds(q.source)
            return tp[:3] != tq[:3] or tq[3] - tp[3] > a.vlc_gap * 60
        frag_list += [('vlc', None, f) for f in fragments(vlc, sigs, a.cut, gap)]
    counts = defaultdict(int)
    for kind, _, _ in frag_list:
        counts[kind] += 1
    print(f'fragments: {dict(counts)}')

    # 2. scenes: cluster fragment mean thumbnails
    frag_sigs = {i: np.mean([sigs[it.stem] for it in f], axis=0) for i, (_, _, f) in enumerate(frag_list)}
    scene_of_frag = cluster_scenes(frag_sigs, a.scene_thresh)
    scene_of_frag = absorb_small_scenes(frag_sigs, scene_of_frag, frag_list, a.absorb_below, a.absorb_dist)
    scene_of_stem = {it.stem: scene_of_frag[i] for i, (_, _, f) in enumerate(frag_list) for it in f}

    # 3. recordings: named as-is; numbered / vlc fragments re-joined along their sequence when the scene is unchanged
    recording = {}
    all_cuts = []
    for kind, prefix in (('num', 'num_'), ('vlc', 'vlc_')):
        idx = [i for i, (k, _, _) in enumerate(frag_list) if k == kind]
        if not idx:
            continue
        fl = [frag_list[i][2] for i in idx]
        so = {id(f): scene_of_frag[i] for i, f in zip(idx, fl)}
        rec, cuts = runs(fl, so, prefix)
        recording.update(rec)
        all_cuts += cuts
        print(f'{kind}: {sum(len(f) for f in fl)} images, {len(fl)} fragments -> {len(set(rec.values()))} recordings')
    for g, its in named.items():
        for it in its:
            recording[it.stem] = g

    rec_items = defaultdict(list)
    for it in ds.items:
        rec_items[recording[it.stem]].append(it)
    print(f'{len(rec_items)} recordings -> {len(set(scene_of_stem.values()))} scenes')

    with open(out / 'groups.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['stem', 'family', 'recording', 'scene', 'export_split'])
        for it in ds.items:
            w.writerow([it.stem, family(it), recording[it.stem], f'scene_{scene_of_stem[it.stem]:03d}', it.split])
    montage(all_cuts, out / 'recording_boundaries.jpg')
    scene_montage(ds.items, scene_of_stem, out / 'scenes.jpg')
    sizes = sorted((len(v) for v in rec_items.values()), reverse=True)
    print(f'recording sizes: largest {sizes[:8]}, median {sizes[len(sizes) // 2]}, singletons {sum(1 for s in sizes if s == 1)}')
    ssizes = sorted(Counter(scene_of_stem.values()).values(), reverse=True)
    print(f'scene sizes: largest {ssizes[:8]}, singletons {sum(1 for s in ssizes if s == 1)}')
    # how leaky is the export's own split, per recording and per scene
    for level, key in (('recording', lambda it: recording[it.stem]), ('scene', lambda it: scene_of_stem[it.stem])):
        by = defaultdict(set)
        for it in ds.items:
            by[key(it)].add(it.split)
        mixed = sum(1 for v in by.values() if len(v) > 1)
        print(f'export split: {mixed}/{len(by)} {level}s appear in both train and valid')
    print(f'written to {out}')


if __name__ == '__main__':
    main()
