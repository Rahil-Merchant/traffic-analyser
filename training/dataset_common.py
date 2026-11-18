"""Read a Roboflow export (YOLO or COCO layout) into one in-memory table, shared by the analysis and
re-split scripts.

Roboflow names files `<original stem>_<ext>.rf.<hash>.<ext>`; augmented exports hold several copies of
one source frame under different hashes. The frame selector names frames `<video stem>_<6 hex>_<frame>`,
so the video a frame came from is recoverable from the name. Both facts matter for a leak-free split.
"""
import json
import os
import re
import struct
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
_RF = re.compile(r'^(?P<stem>.+?)(?:_(?:jpg|jpeg|png|bmp|webp))?\.rf\.(?P<hash>[0-9a-f]{32})$', re.I)
_FRAME = re.compile(r'^(?P<video>.+?)[_-](?P<frame>\d{4,})$')
_SELECTOR = re.compile(r'^(?P<video>.+_[0-9a-f]{6})_(?P<frame>\d{7})$')


@dataclass
class Item:
    path: str                   # image path
    split: str                  # split in the export: train / valid / test
    source: str                 # original stem before Roboflow's .rf.<hash>
    rf_hash: str                # '' when not a Roboflow name
    group: str                  # recording the frame came from (best effort)
    frame: int                  # frame number parsed from the name, -1 if none
    width: int = 0
    height: int = 0
    boxes: list = field(default_factory=list)   # [(cls, cx, cy, w, h)] normalised
    issues: list = field(default_factory=list)  # label problems found while parsing

    @property
    def stem(self):
        return Path(self.path).stem


@dataclass
class Dataset:
    root: str
    layout: str                 # 'yolo' or 'coco'
    classes: list
    items: list

    def by_split(self):
        out = defaultdict(list)
        for it in self.items:
            out[it.split].append(it)
        return out


def parse_name(stem):
    """-> (source stem, rf hash, group, frame)."""
    m = _RF.match(stem)
    source, rf = (m['stem'], m['hash']) if m else (stem, '')
    m = _SELECTOR.match(source) or _FRAME.match(source)
    if m:
        return source, rf, m['video'], int(m['frame'])
    return source, rf, source, -1


def image_size(path):
    """Width, height from the file header (no decode)."""
    with open(path, 'rb') as fh:
        head = fh.read(32)
        if head[:8] == b'\x89PNG\r\n\x1a\n':
            w, h = struct.unpack('>II', head[16:24])
            return w, h
        if head[:2] == b'\xff\xd8':
            fh.seek(2)
            while True:
                marker = fh.read(2)
                if len(marker) < 2 or marker[0] != 0xFF:
                    break
                if marker[1] in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    fh.read(3)
                    h, w = struct.unpack('>HH', fh.read(4))
                    return w, h
                seg = struct.unpack('>H', fh.read(2))[0]
                fh.seek(seg - 2, 1)
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def _split_dirs(root):
    """-> {split: dir} for the Roboflow splits present."""
    out = {}
    for name, alias in (('train', 'train'), ('valid', 'valid'), ('val', 'valid'), ('test', 'test')):
        d = Path(root) / name
        if d.is_dir():
            out[alias] = d
    return out


def _yolo_boxes(label_path, item):
    boxes = []
    if not os.path.exists(label_path):
        item.issues.append('missing label file')
        return boxes
    for ln, line in enumerate(open(label_path, encoding='utf-8'), 1):
        parts = line.split()
        if not parts:
            continue
        try:
            c = int(float(parts[0]))
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            item.issues.append(f'line {ln}: unparsable')
            continue
        if len(vals) > 4:          # polygon label: use its bounding box
            xs, ys = vals[0::2], vals[1::2]
            cx, cy, w, h = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)
        elif len(vals) == 4:
            cx, cy, w, h = vals
        else:
            item.issues.append(f'line {ln}: {len(vals)} coords')
            continue
        if w <= 0 or h <= 0:
            item.issues.append(f'line {ln}: zero-size box')
            continue
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and w <= 1.001 and h <= 1.001):
            item.issues.append(f'line {ln}: out of bounds')
        boxes.append((c, cx, cy, w, h))
    return boxes


def load_yolo(root, with_sizes=True):
    data = yaml.safe_load(open(Path(root) / 'data.yaml'))
    names = data['names']
    classes = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    items = []
    for split, d in _split_dirs(root).items():
        img_dir, lbl_dir = d / 'images', d / 'labels'
        for f in sorted(os.listdir(img_dir)):
            if not f.lower().endswith(IMG_EXT):
                continue
            stem = Path(f).stem
            source, rf, group, frame = parse_name(stem)
            it = Item(str(img_dir / f), split, source, rf, group, frame)
            it.boxes = _yolo_boxes(str(lbl_dir / (stem + '.txt')), it)
            if with_sizes:
                it.width, it.height = image_size(it.path)
            items.append(it)
    return Dataset(str(root), 'yolo', classes, items)


def load_coco(root):
    items, classes, id_map = [], None, None
    for split, d in _split_dirs(root).items():
        ann = json.load(open(d / '_annotations.coco.json'))
        cats = sorted(ann['categories'], key=lambda c: c['id'])
        # Roboflow COCO exports prepend a supercategory placeholder at id 0 with no annotations.
        used = {a['category_id'] for a in ann['annotations']}
        cats = [c for c in cats if c['id'] in used or c['supercategory'] != 'none' or len(cats) == 1] or cats
        if classes is None:
            classes = [c['name'] for c in cats]
            id_map = {c['id']: i for i, c in enumerate(cats)}
        per_img = defaultdict(list)
        for a in ann['annotations']:
            per_img[a['image_id']].append(a)
        for im in ann['images']:
            stem = Path(im['file_name']).stem
            source, rf, group, frame = parse_name(stem)
            it = Item(str(d / im['file_name']), split, source, rf, group, frame, im['width'], im['height'])
            for a in per_img[im['id']]:
                x, y, w, h = a['bbox']
                if w <= 0 or h <= 0:
                    it.issues.append('zero-size box')
                    continue
                if a['category_id'] not in id_map:
                    it.issues.append(f"unknown category {a['category_id']}")
                    continue
                it.boxes.append((id_map[a['category_id']], (x + w / 2) / im['width'], (y + h / 2) / im['height'],
                                 w / im['width'], h / im['height']))
            items.append(it)
    return Dataset(str(root), 'coco', classes, items)


def load_dataset(root, with_sizes=True):
    root = Path(root)
    if (root / 'data.yaml').exists():
        return load_yolo(root, with_sizes)
    if any((d / '_annotations.coco.json').exists() for d in _split_dirs(root).values()):
        return load_coco(root)
    raise SystemExit(f'{root}: neither data.yaml (YOLO) nor */_annotations.coco.json (COCO) found')


def find_export_root(path):
    """The Roboflow export may be nested one level deep in the extracted zip."""
    path = Path(path)
    for cand in [path] + [p for p in path.iterdir() if p.is_dir()]:
        if (cand / 'data.yaml').exists() or _split_dirs(cand):
            return cand
    return path
