"""Stage 5: write full-resolution frames, YOLO pseudo-labels and the manifest."""
import csv
import os
from datetime import datetime, timedelta

import cv2
import numpy as np

from .detect import MODEL_SIZE, read_frames
from .signals import pseudo_labels

REASON_TEXT = {'fn': 'missed vehicle', 'fp': 'false positive / hard negative', 'class': 'wrong class', 'borderline': 'low confidence'}


def _yolo_line(cls, box):
    x1, y1, x2, y2 = (np.clip(np.asarray(box, dtype=float), 0, MODEL_SIZE) / MODEL_SIZE)   # squash: normalised coords are the same
    return f"{int(cls)} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"


def export(out_dir, video_path, meta, chosen, tracks, progress=print):
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
        if len(rows) % 25 == 0:
            progress(f"  exported {len(rows)}/{len(by_frame)} frames")
    path = os.path.join(out_dir, 'manifest.csv')
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ['image'])
        if new:
            w.writeheader()
        w.writerows(rows)
    return rows
