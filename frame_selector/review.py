"""A static review page: every chosen frame as a thumbnail with its pseudo-boxes and the flagged box."""
import html
import os

import cv2
import numpy as np

from .detect import MODEL_SIZE
from .export import REASON_TEXT
from .signals import pseudo_labels

COLOUR = {'fn': (0, 0, 255), 'fp': (0, 165, 255), 'class': (255, 0, 255), 'borderline': (0, 220, 220)}


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


def write_review(out_dir, rows_by_video, progress=print):
    """rows_by_video: [(manifest rows, chosen, tracks, meta)] per video processed in this run."""
    thumbs = os.path.join(out_dir, 'review')
    os.makedirs(thumbs, exist_ok=True)
    sections = {r: [] for r in REASON_TEXT}
    for rows, chosen, tracks, meta in rows_by_video:
        names = meta['class_names']
        by_frame = dict(chosen)
        for row in rows:
            f = row['frame']
            img = cv2.imread(os.path.join(out_dir, row['image']))
            if img is None:
                continue
            cands = by_frame[f]
            labels = pseudo_labels(f, tracks)
            tn = os.path.basename(row['image'])
            cv2.imwrite(os.path.join(thumbs, tn), _thumb(img, labels, cands, names), [cv2.IMWRITE_JPEG_QUALITY, 80])
            when = row['clock'] or '%.1fs' % row['time_s']
            sections[cands[0]['reason']].append(
                '<figure data-img="%s"><img src="review/%s" loading="lazy"><figcaption><label><input type="checkbox"> reject</label> '
                '<b>%s</b> frame %d %s<br>%s</figcaption></figure>'
                % (html.escape(row['image']), html.escape(tn), html.escape(row['video']), f, html.escape(when), html.escape(row['notes'])))
    body = ''.join(f'<h2>{html.escape(REASON_TEXT[r])} ({len(figs)})</h2><div class="grid">{"".join(figs)}</div>'
                   for r, figs in sections.items() if figs)
    page = f'''<!doctype html><meta charset="utf-8"><title>Frame selector review</title>
<style>body{{font:14px system-ui;margin:16px;background:#111;color:#ddd}} .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:12px}}
figure{{margin:0;background:#1c1c1c;padding:6px;border-radius:6px}} img{{width:100%;display:block}} figcaption{{padding:6px 2px;line-height:1.4}}
h2{{margin:24px 0 8px}} button{{font:inherit;padding:6px 12px}} .legend span{{display:inline-block;margin-right:14px;padding-left:18px;border-left:12px solid}}</style>
<h1>Frame selector review</h1>
<p class="legend">Green = pseudo-label to correct. Flagged box: <span style="border-color:#f00">missed</span><span style="border-color:#ffa500">false positive</span>
<span style="border-color:#f0f">wrong class</span><span style="border-color:#dcdc00">low confidence</span></p>
<p><button onclick="dl()">Download rejected list</button> (frames ticked "reject" are written to rejected.json for you to delete)</p>
{body}
<script>function dl(){{const r=[...document.querySelectorAll('figure')].filter(f=>f.querySelector('input').checked).map(f=>f.dataset.img);
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(r,null,1)],{{type:'application/json'}}));a.download='rejected.json';a.click();}}</script>'''
    with open(os.path.join(out_dir, 'review.html'), 'w') as fh:
        fh.write(page)
    progress(f"review page: {os.path.join(out_dir, 'review.html')}")
