"""Stage 5: write full-resolution frames, YOLO pseudo-labels and the manifest."""
import csv
import os
from datetime import datetime, timedelta

import cv2
import numpy as np

from .detect import MODEL_SIZE, read_frames
from .signals import pseudo_labels

COLOUR = {'fn': (0, 0, 255), 'fp': (0, 165, 255), 'class': (255, 0, 255), 'borderline': (0, 220, 220)}

REASON_TEXT = {'fn': 'missed vehicle', 'fp': 'false positive / hard negative', 'class': 'wrong class', 'borderline': 'low confidence'}


def _yolo_line(cls, box):
    x1, y1, x2, y2 = (np.clip(np.asarray(box, dtype=float), 0, MODEL_SIZE) / MODEL_SIZE)   # squash: normalised coords are the same
    return f"{int(cls)} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"


def export(out_dir, video_path, meta, chosen, tracks):
    """chosen: [(frame, [candidates])]. Returns the manifest rows written."""
    os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'labels'), exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    names = meta['class_names']
    with open(os.path.join(out_dir, 'classes.txt'), 'w') as fh:
        fh.write('\n'.join(names[k] for k in sorted(names)) + '\n')
    fps = meta['original_fps']
    start = datetime.fromisoformat(meta['start_datetime']) if meta.get('start_datetime') else None
    by_frame = dict(chosen)
    rows = []
    for f, img in read_frames(video_path, by_frame, fps):
        name = f"{stem}_{f:07d}"
        cv2.imwrite(os.path.join(out_dir, 'images', name + '.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        labels = pseudo_labels(f, tracks)
        with open(os.path.join(out_dir, 'labels', name + '.txt'), 'w') as fh:
            fh.write('\n'.join(_yolo_line(c, b) for c, b in labels) + ('\n' if labels else ''))
        cands = by_frame[f]
        t = f / fps
        rows.append(dict(image=f"images/{name}.jpg", label=f"labels/{name}.txt", video=os.path.basename(video_path), frame=f,
                         time_s=round(t, 2), clock=(start + timedelta(seconds=t)).strftime('%H:%M:%S') if start else '',
                         reasons='|'.join(dict.fromkeys(c['reason'] for c in cands)), score=round(max(c['score'] for c in cands), 3),
                         tracks='|'.join(str(c['track']) for c in cands if c['track'] is not None),
                         notes=' ; '.join(f"{REASON_TEXT[c['reason']]}: {c['note']}" for c in cands),
                         n_labels=len(labels)))
    return rows


FIELDS = ['image', 'label', 'video', 'frame', 'time_s', 'clock', 'reasons', 'score', 'tracks', 'notes', 'n_labels']


def read_manifest(out_dir):
    path = os.path.join(out_dir, 'manifest.csv')
    if not os.path.exists(path):
        return []
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


def update_manifest(out_dir, video_name, rows):
    """Append this video's rows; an earlier run of the same video is replaced, other videos are kept."""
    old = [r for r in read_manifest(out_dir) if r['video'] != video_name]
    with open(os.path.join(out_dir, 'manifest.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(old + rows)


def _thumb(img, labels, cands, names, width=720):
    H, W = img.shape[:2]
    sx, sy = W / MODEL_SIZE, H / MODEL_SIZE
    vis = img.copy()
    for cls, b in labels:
        x1, y1, x2, y2 = int(b[0] * sx), int(b[1] * sy), int(b[2] * sx), int(b[3] * sy)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (80, 220, 80), 3)
        cv2.putText(vis, names.get(int(cls), str(cls)), (x1, max(28, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 220, 80), 2, cv2.LINE_AA)
    for c in cands:
        b = c['box']
        x1, y1, x2, y2 = int(b[0] * sx), int(b[1] * sy), int(b[2] * sx), int(b[3] * sy)
        cv2.rectangle(vis, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), COLOUR[c['reason']], 4)
    return cv2.resize(vis, (width, int(width * H / W)), interpolation=cv2.INTER_AREA)


def write_thumbs(out_dir, rows, chosen, tracks, meta):
    """Review thumbnails (pseudo-labels in green, flagged boxes in colour) for the frames just exported."""
    thumbs = os.path.join(out_dir, 'review')
    os.makedirs(thumbs, exist_ok=True)
    names = meta['class_names']
    by_frame = dict(chosen)
    for row in rows:
        img = cv2.imread(os.path.join(out_dir, row['image']))
        if img is None:
            continue
        f = row['frame']
        cv2.imwrite(os.path.join(thumbs, os.path.basename(row['image'])),
                    _thumb(img, pseudo_labels(f, tracks), by_frame[f], names), [cv2.IMWRITE_JPEG_QUALITY, 80])
