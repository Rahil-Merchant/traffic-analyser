"""Stage 5: write full-resolution frames, YOLO pseudo-labels, review thumbnails and the manifest."""
import contextlib
import csv
import os
from datetime import datetime, timedelta

import cv2
import numpy as np

from .detect import MODEL_SIZE, read_frames
from .signals import pseudo_labels

COLOUR = {'fn': (0, 0, 255), 'fp': (0, 165, 255), 'class': (255, 0, 255), 'borderline': (0, 220, 220)}

REASON_TEXT = {'fn': 'miss', 'fp': 'no vehicle', 'class': 'confusion', 'borderline': 'low confidence'}


def _yolo_line(cls, box):
    x1, y1, x2, y2 = (np.clip(np.asarray(box, dtype=float), 0, MODEL_SIZE) / MODEL_SIZE)   # squash: normalised coords are the same
    return f"{int(cls)} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"


def _write_jpg(path, img, quality):
    """cv2.imwrite returns False instead of raising, and can't write to non-ASCII paths on Windows."""
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise OSError(f"could not encode {path}")
    with open(path, 'wb') as fh:
        fh.write(buf.tobytes())


def export(out_dir, video_path, key, meta, chosen, tracks):
    """chosen: [(frame, [candidates])]. Writes images, labels and review thumbnails named after `key`
    (detect.video_key). Returns the manifest rows written."""
    for sub in ('images', 'labels', 'review'):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    names = meta['class_names']
    with open(os.path.join(out_dir, 'classes.txt'), 'w') as fh:
        fh.write('\n'.join(names[k] for k in sorted(names)) + '\n')
    fps = meta['original_fps']
    start = datetime.fromisoformat(meta['start_datetime']) if meta.get('start_datetime') else None
    by_frame = dict(chosen)
    rows = []
    for f, img in read_frames(video_path, by_frame):
        name = f"{key}_{f:07d}"
        cands = by_frame[f]
        _write_jpg(os.path.join(out_dir, 'images', name + '.jpg'), img, 92)
        _write_jpg(os.path.join(out_dir, 'review', name + '.jpg'), _thumb(img, cands[0]), 80)
        labels = pseudo_labels(f, tracks)
        with open(os.path.join(out_dir, 'labels', name + '.txt'), 'w') as fh:
            fh.write('\n'.join(_yolo_line(c, b) for c, b in labels) + ('\n' if labels else ''))
        t = f / fps
        rows.append(dict(image=f"images/{name}.jpg", label=f"labels/{name}.txt", video=os.path.basename(video_path), video_id=key,
                         frame=f, time_s=round(t, 2), clock=(start + timedelta(seconds=t)).strftime('%H:%M:%S') if start else '',
                         reasons='|'.join(dict.fromkeys(c['reason'] for c in cands)), score=round(max(c['score'] for c in cands), 3),
                         tracks='|'.join(str(c['track']) for c in cands if c['track'] is not None),
                         notes=f"{REASON_TEXT[cands[0]['reason']]}: {cands[0]['note']}",       # the vehicle the frame was chosen for
                         n_labels=len(labels)))
    return rows


FIELDS = ['image', 'label', 'video', 'video_id', 'frame', 'time_s', 'clock', 'reasons', 'score', 'tracks', 'notes', 'n_labels']


def read_manifest(out_dir):
    path = os.path.join(out_dir, 'manifest.csv')
    if not os.path.exists(path):
        return []
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


def same_video(row, key, video_name):
    """Rows from before video ids existed are matched by file name."""
    return row.get('video_id') == key if row.get('video_id') else row['video'] == video_name


def row_files(row):
    return (row['image'], row['label'], 'review/' + os.path.basename(row['image']))


def update_manifest(out_dir, key, video_name, rows):
    """Replace this video's rows (from an earlier run) with `rows`, keep other videos' rows, and delete
    the earlier run's images, labels and thumbnails that the new rows don't reuse."""
    old = read_manifest(out_dir)
    keep = [r for r in old if not same_video(r, key, video_name)]
    new_files = {p for r in rows for p in row_files(r)}
    for r in old:
        if same_video(r, key, video_name):
            for p in row_files(r):
                if p not in new_files:
                    with contextlib.suppress(OSError):
                        os.remove(os.path.join(out_dir, p))
    tmp = os.path.join(out_dir, 'manifest.csv.tmp')
    with open(tmp, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(keep + rows)
    os.replace(tmp, os.path.join(out_dir, 'manifest.csv'))


def _thumb(img, cand, width=720):
    """The frame with only the vehicle it was chosen for outlined, and the reason as its label."""
    H, W = img.shape[:2]
    sx, sy = W / MODEL_SIZE, H / MODEL_SIZE
    vis = img.copy()
    b = cand['box']
    x1, y1, x2, y2 = int(b[0] * sx), int(b[1] * sy), int(b[2] * sx), int(b[3] * sy)
    col = COLOUR[cand['reason']]
    cv2.rectangle(vis, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), col, 5)
    text = f"{REASON_TEXT[cand['reason']]}: {cand['note']}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
    ty = y1 - 16 if y1 - 16 - th > 0 else y2 + th + 16
    cv2.rectangle(vis, (x1 - 6, ty - th - 8), (x1 + tw + 6, ty + 8), (0, 0, 0), -1)
    cv2.putText(vis, text, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 2, cv2.LINE_AA)
    return cv2.resize(vis, (width, int(width * H / W)), interpolation=cv2.INTER_AREA)
