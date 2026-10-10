"""The whole mining run as one function, shared by the CLI and the GUI.

The GPU detects one video at a time; everything after detection (tracking, flagging, selection,
export, thumbnails) for a finished video runs in a worker process while the next video detects.
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor

from .detect import configured_model, detect_video, load_dump, load_model, video_info
from .export import export, update_manifest, write_thumbs
from .review import write_review_page
from .select import budget_for, select_frames
from .signals import build_tracks, find_candidates


def postprocess(video, dump, out, budget):
    """Everything after detection for one video. Runs in a worker process; returns (rows, summary)."""
    t0 = time.time()
    frames, offs, dets, meta = load_dump(dump)
    fps = meta['original_fps']
    if not meta.get('frame_shape') or meta['frame_shape'] == [853, 640]:   # app dumps store the processing size
        info = video_info(video)
        meta['frame_shape'] = [info['width'], info['height']]
    tracks = build_tracks(frames, offs, dets, fps)
    cands = find_candidates(frames, offs, dets, tracks, fps, meta['class_names'])
    n = budget_for(len(frames) / fps, budget)
    chosen = select_frames(cands, n, fps)
    rows = export(out, video, meta, chosen, tracks)
    write_thumbs(out, rows, chosen, tracks, meta)
    counts = ', '.join(f"{r} {sum(c['reason'] == r for c in cands)}" for r in ('fn', 'fp', 'class', 'borderline'))
    return rows, f"{os.path.basename(video)}: {len(tracks)} tracks; flagged {counts}; selected {len(rows)} frames ({time.time() - t0:.0f}s)"


def run(videos, out, detections=(), budget=100, device=None, progress=print):
    """Mine `videos` into `out` with the model from config.py. `detections[i]` (optional) is an
    existing dump for videos[i], which skips detection for it. Returns the manifest rows of every video."""
    os.makedirs(out, exist_ok=True)
    need_model = any(k >= len(detections) or not detections[k] for k in range(len(videos)))
    model = load_model() if need_model else None
    if model is not None:
        progress(f"model: {configured_model()}")
    all_rows, pending = [], []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for k, video in enumerate(videos):
            t0 = time.time()
            progress(f"== {os.path.basename(video)}")
            if k < len(detections) and detections[k]:
                dump = detections[k]
                progress("using cached detections")
            else:
                dump = os.path.join(out, 'detections_' + os.path.splitext(os.path.basename(video))[0] + '.npz')
                detect_video(video, model, dump, device=device, progress=progress)
                progress(f"detected in {time.time() - t0:.0f}s; tracking, selecting and exporting in the background")
            pending.append((video, pool.submit(postprocess, video, dump, out, budget)))
        for video, fut in pending:                   # in order, so the manifest is updated deterministically
            rows, summary = fut.result()
            update_manifest(out, os.path.basename(video), rows)
            all_rows += rows
            progress(summary)
    write_review_page(out)
    progress(f"review page: {os.path.join(out, 'review.html')}")
    return all_rows
