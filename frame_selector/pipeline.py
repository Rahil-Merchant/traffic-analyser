"""The whole mining run as one function, shared by the CLI and the GUI."""
import os
import time

from .detect import configured_model, detect_video, load_dump, load_model, video_info
from .export import export
from .review import write_review
from .select import budget_for, select_frames
from .signals import build_tracks, find_candidates


def run(videos, out, detections=(), budget=100, device=None, progress=print):
    """Mine `videos` into `out` with the model from config.py. `detections[i]` (optional) is an
    existing dump for videos[i], which skips detection for it. Returns the manifest rows of every video."""
    os.makedirs(out, exist_ok=True)
    need_model = any(k >= len(detections) or not detections[k] for k in range(len(videos)))
    model = load_model() if need_model else None
    if model is not None:
        progress(f"model: {configured_model()}")
    runs, all_rows = [], []
    for k, video in enumerate(videos):
        t0 = time.time()
        progress(f"== {os.path.basename(video)}")
        if k < len(detections) and detections[k]:
            dump = detections[k]
        else:
            dump = os.path.join(out, 'detections_' + os.path.splitext(os.path.basename(video))[0] + '.npz')
            progress("detecting every frame...")
            detect_video(video, model, dump, device=device, progress=progress)
        frames, offs, dets, meta = load_dump(dump)
        fps = meta['original_fps']
        if not meta.get('frame_shape') or meta['frame_shape'] == [853, 640]:   # app dumps store the processing size
            info = video_info(video)
            meta['frame_shape'] = [info['width'], info['height']]
        progress("tracking and flagging...")
        tracks = build_tracks(frames, offs, dets, fps)
        cands = find_candidates(frames, offs, dets, tracks, fps)
        progress(f"{len(tracks)} tracks; candidates: " + ', '.join(
            f"{r} {sum(c['reason'] == r for c in cands)}" for r in ('fn', 'fp', 'class', 'borderline')))
        n = budget_for(len(frames) / fps, budget)
        chosen = select_frames(cands, n, fps)
        progress(f"selected {len(chosen)} frames (budget {n}); exporting...")
        rows = export(out, video, meta, chosen, tracks, progress=progress)
        runs.append((rows, chosen, tracks, meta))
        all_rows += rows
        progress(f"done in {time.time() - t0:.0f}s")
    progress("writing review page...")
    write_review(out, runs, progress=progress)
    return all_rows
