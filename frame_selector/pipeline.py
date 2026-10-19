"""The whole mining run as one function, shared by the CLI and the GUI.

The GPU detects one video at a time, each in its own short-lived process: the GPU decoder leaks
memory with every video, and a long run in one process ran out of GPU memory after ~50 videos.
Everything after detection (tracking, flagging, selection, export, thumbnails) for a finished video
runs in a worker process while the next video detects.
Each video's results go into the manifest as soon as they are ready, and a video that fails is
reported and skipped without stopping the others.
"""
import contextlib
import multiprocessing
import os
import queue
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

from .detect import configured_model, detect_video, dump_status, load_dump, load_model, video_key
from .export import export, update_manifest
from .review import write_review_page
from .select import budget_for, select_frames
from .signals import build_tracks, find_candidates


def postprocess(video, key, dump, out, budget):
    """Everything after detection for one video. Runs in a worker process; returns (rows, summary)."""
    t0 = time.time()
    frames, offs, dets, meta = load_dump(dump)
    fps = meta['original_fps']
    tracks = build_tracks(frames, offs, dets, fps)
    cands = find_candidates(frames, offs, dets, tracks, fps, meta['class_names'])
    n = budget_for(len(frames) / fps, budget)
    chosen = select_frames(cands, n, fps)
    rows = export(out, video, key, meta, chosen, tracks)
    counts = ', '.join(f"{r} {sum(c['reason'] == r for c in cands)}" for r in ('fn', 'fp', 'class', 'borderline'))
    lost = len(chosen) - len(rows)
    return rows, (f"{os.path.basename(video)}: {len(frames)} frames, {len(dets)} detections, {len(tracks)} tracks; "
                  f"flagged {counts}; selected {len(rows)} frames"
                  + (f" ({lost} more could not be read back from the video)" if lost else '')
                  + f" ({time.time() - t0:.0f}s)")


def _detect_child(video, dump, device, messages):
    """Detection for one video, in a process of its own (see the module docstring)."""
    try:
        model = load_model()
        detect_video(video, model, dump, device=device, progress=lambda m: messages.put(('log', m)), model_path=configured_model())
        messages.put(('ok', None))
    except BaseException as e:                       # noqa: BLE001 - handed to the parent
        messages.put(('error', f"{type(e).__name__}: {e}"))


def detect_in_subprocess(video, dump, device=None, progress=print):
    """Run detect_video for one video in a fresh process, relaying its log; raises RuntimeError if it fails."""
    ctx = multiprocessing.get_context('spawn')
    messages = ctx.Queue()
    proc = ctx.Process(target=_detect_child, args=(video, dump, device, messages), daemon=True)
    proc.start()
    result = None
    while result is None:
        try:
            kind, text = messages.get(timeout=1.0 if proc.is_alive() else 5.0)
        except queue.Empty:
            if not proc.is_alive():
                result = ('error', f"the detection process stopped unexpectedly (exit code {proc.exitcode})")
            continue
        if kind == 'log':
            progress(text)
        else:
            result = (kind, text)
    proc.join(timeout=120)
    if result[0] == 'error':
        raise RuntimeError(result[1])


def _remove_old_cache(out, video):
    """Delete this video's cache from frame selector version 1 (named by stem), which is never reused."""
    old = os.path.join(out, 'detections_' + os.path.splitext(os.path.basename(video))[0] + '.npz')
    if os.path.exists(old) and dump_status(old) == "made by an older version of the frame selector":
        with contextlib.suppress(OSError):
            os.remove(old)


def run(videos, out, detections=(), budget=100, device=None, progress=print):
    """Mine `videos` into `out` with the model from config.py. `detections[i]` (optional) is an
    existing dump for videos[i], used when it is still valid. Returns (manifest rows of the videos
    that finished, [(video, error)] for the videos that failed)."""
    os.makedirs(out, exist_ok=True)
    given = list(detections) + [None] * (len(videos) - len(detections))
    model_checked = False
    all_rows, failed, pending = [], [], {}

    def fail(video, e):
        failed.append((video, f"{type(e).__name__}: {e}"))
        progress(f"FAILED {os.path.basename(video)}: {type(e).__name__}: {e}")

    def record(video, key, rows, summary):
        update_manifest(out, key, os.path.basename(video), rows)
        write_review_page(out)
        all_rows.extend(rows)
        progress(summary)

    def inline(video, key, dump):
        """Post-process in this process: the fallback when the worker processes have died."""
        try:
            record(video, key, *postprocess(video, key, dump, out, budget))
        except Exception as e:                       # noqa: BLE001 - reported, the other videos go on
            fail(video, e)

    def finish(fut):
        video, key, dump = pending.pop(fut)
        try:
            rows, summary = fut.result()
        except BrokenProcessPool:
            progress(f"the worker process stopped; finishing {os.path.basename(video)} here")
            inline(video, key, dump)
            return
        except Exception as e:                       # noqa: BLE001 - reported, the other videos go on
            fail(video, e)
            return
        try:
            record(video, key, rows, summary)
        except Exception as e:                       # noqa: BLE001 - e.g. the manifest can't be written
            fail(video, e)

    def submit(video, key, dump):
        try:
            pending[pool.submit(postprocess, video, key, dump, out, budget)] = (video, key, dump)
        except BrokenProcessPool:
            progress(f"worker processes unavailable; finishing {os.path.basename(video)} here")
            inline(video, key, dump)

    workers = 4 if (os.cpu_count() or 1) >= 16 else 2     # export steps through each whole video on the CPU
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for video, dump in zip(videos, given):
            t0 = time.time()
            progress(f"== {os.path.basename(video)}")
            try:
                key = video_key(video)
                why = dump_status(dump, video, match_name=False) if dump else None
                if why:
                    progress(f"not using cached detections {os.path.basename(dump)}: {why}")
                    dump = None
            except Exception as e:                   # noqa: BLE001 - e.g. the video can't be read
                fail(video, e)
                continue
            if dump:
                progress("using cached detections")
            else:
                if not model_checked:                # no model: nothing can run, so this stops the run
                    if not os.path.exists(configured_model()):
                        raise FileNotFoundError(f"model not found: {configured_model()} (set MODEL_PATH / PC in config.py)")
                    progress(f"model: {configured_model()}")
                    model_checked = True
                dump = os.path.join(out, f"detections_{key}.npz")
                try:
                    detect_in_subprocess(video, dump, device, progress)
                except Exception as e:               # noqa: BLE001 - reported, the other videos go on
                    fail(video, e)
                    continue
                _remove_old_cache(out, video)
                progress(f"detected in {time.time() - t0:.0f}s; tracking, selecting and exporting in the background")
            submit(video, key, dump)
            for fut in [f for f in pending if f.done()]:
                finish(fut)
        for fut in as_completed(list(pending)):
            finish(fut)
    write_review_page(out)
    progress(f"review page: {os.path.join(out, 'review.html')}")
    if failed:
        progress(f"{len(failed)} of {len(videos)} videos failed: " + ', '.join(os.path.basename(v) for v, _ in failed))
    return all_rows, failed
