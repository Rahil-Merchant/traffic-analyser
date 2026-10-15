"""Stage 1: run the production model over a video and keep every detection, in the app's dump format.

Decoding is the bottleneck, so frames come from the GPU decoder the app uses (video_reader-rs with
CUDA, resizing on the GPU, 7,600-frame segments) when it is installed, with the next segment decoding
while the current one is being detected. OpenCV decoding is the fallback.
"""
import json
import os
import queue
import threading
import time

import cv2
import numpy as np

MODEL_SIZE = 640
BATCH = 64               # the engine is exported with a dynamic batch of up to 64
SEGMENT = 7600           # frames per GPU decode segment, as in the app


def configured_model():
    """The production model from config.py (MODEL_PATH, chosen per machine by PC_PATHS)."""
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):      # config.py prints a banner on import
        from config import MODEL_PATH
    return MODEL_PATH


def load_model(model_path=None):
    from ultralytics import YOLO
    model_path = model_path or configured_model()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model not found: {model_path} (set MODEL_PATH / PC in config.py)")
    return YOLO(model_path, task='detect')


def video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    info = dict(fps=cap.get(cv2.CAP_PROP_FPS) or 20.0, frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return info


def read_frames(video_path, frame_indices, fps):
    """Yield (frame_index, bgr_frame) for the given indices, in ascending order.
    A seek in these HEVC .dav streams lands a few frames late, but the position reported after a
    read is exact, so seek early and step forward to the frame asked for."""
    cap = cv2.VideoCapture(video_path)
    for idx in sorted(set(int(i) for i in frame_indices)):
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0, idx - 10) / fps * 1000.0)
        for _ in range(40):
            ok, frame = cap.read()
            if not ok:
                break
            got = int(round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 * fps))   # index of the frame just read
            if got >= idx:
                if got == idx:
                    yield idx, frame
                break
    cap.release()


def gpu_frame_count(video_path):
    from video_reader import PyVideoReader
    return int(PyVideoReader(video_path).get_info().get('frame_count', 0))


def _segments_gpu(video_path, total):
    """(first_index, frames RGB (N,640,640,3)) per segment, decoded on the GPU like the app."""
    from video_reader import PyVideoReader
    pvr = PyVideoReader(video_path, device='cuda', resize_shorter_side=MODEL_SIZE, resize_longer_side=MODEL_SIZE)
    for start in range(0, total, SEGMENT):
        end = min(start + SEGMENT, total)
        try:
            seg = pvr.decode(start_frame=start, end_frame=end)
        except Exception as e:                       # noqa: BLE001 - skip a corrupt segment, as the app does
            print(f"  skipping frames {start}-{end}: {e}")
            continue
        if seg is not None and len(seg):
            yield start, np.ascontiguousarray(seg)


def _segments_cv2(video_path, total):
    """Fallback: OpenCV decode on the CPU, in smaller segments, frames converted to RGB."""
    cap = cv2.VideoCapture(video_path)
    idx, buf = 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        buf.append(cv2.cvtColor(cv2.resize(frame, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB))
        idx += 1
        if len(buf) == 512:
            yield idx - len(buf), np.stack(buf)
            buf = []
    if buf:
        yield idx - len(buf), np.stack(buf)
    cap.release()


def _prefetch(gen, depth=2):
    """Run a generator in a thread so decoding overlaps inference."""
    q = queue.Queue(maxsize=depth)
    done = object()

    def pump():
        try:
            for item in gen:
                q.put(item)
        finally:
            q.put(done)
    threading.Thread(target=pump, daemon=True).start()
    while True:
        item = q.get()
        if item is done:
            return
        yield item


def detect_video(video_path, model, out_path, conf=0.05, iou=0.7, device=None, progress=print):
    """Detect on every frame (squashed to 640x640 like production) and save a dump at out_path."""
    import torch
    info = video_info(video_path)
    total = info['frames']
    try:
        import video_reader  # noqa: F401
        total = gpu_frame_count(video_path) or total
        if total <= 0:
            raise RuntimeError(f"no frames reported for {video_path}")
        segments, how = _segments_gpu(video_path, total), 'GPU decode'
    except ImportError:
        segments, how = _segments_cv2(video_path, total), 'CPU decode (video_reader not installed)'
    device = device or ('cuda' if torch.cuda.is_available() else None)
    use_half = device == 'cuda'
    progress(f"  {how}, batch {BATCH}, {info['frames']} frames")
    frames, dets, offs = [], [], [0]
    names = None
    t0 = time.time()
    with torch.no_grad():
        for start, seg in _prefetch(segments):
            for b in range(0, len(seg), BATCH):
                chunk = seg[b:b + BATCH]
                n = len(chunk)
                x = torch.from_numpy(chunk).to(device or 'cpu', non_blocking=True).permute(0, 3, 1, 2)
                x = (x.half() if use_half else x.float()) / 255.0
                results = model(x, imgsz=MODEL_SIZE, conf=conf, iou=iou, verbose=False, half=use_half, device=device)
                for r in results[:n]:
                    if names is None:
                        names = {int(k): str(v) for k, v in r.names.items()}
                    d = r.boxes.data.cpu().numpy()[:, :6].astype(np.float32) if len(r.boxes) else np.zeros((0, 6), np.float32)
                    dets.append(d)
                    offs.append(offs[-1] + len(d))
                frames.extend(range(start + b, start + b + n))
            progress(f"  {len(frames)}/{total} frames, {len(frames) / (time.time() - t0):.0f} fps")
    n_det = offs[-1]
    progress(f"  detected: {len(frames)} frames, {n_det} detections, classes: {', '.join((names or {}).values()) or 'none'}")
    if len(frames) == 0:
        raise RuntimeError(f"no frames could be decoded from {video_path}")
    if n_det == 0:
        progress("  WARNING: the model returned no detections at all on this video")
    meta = dict(video_path=os.path.abspath(video_path), start_datetime=None, original_fps=info['fps'],
                total_frames=len(frames), effective_fps=info['fps'], frame_shape=[info['width'], info['height']],
                model_input_size=MODEL_SIZE, letterbox_params=None, class_names=names or {}, conf=conf)
    np.savez_compressed(out_path, frame=np.array(frames, dtype=np.int64), offs=np.array(offs, dtype=np.int64),
                        dets=np.concatenate(dets) if dets else np.zeros((0, 6), np.float32), meta=np.array(json.dumps(meta)))
    return out_path


def load_dump(path):
    """-> (frames (N,), offs (N+1,), dets (M,6) [x1,y1,x2,y2,conf,cls] in 640x640 model space, meta dict)."""
    z = np.load(path)
    meta = json.loads(str(z['meta']))
    meta['class_names'] = {int(k): v for k, v in meta['class_names'].items()}
    return z['frame'], z['offs'], z['dets'], meta


def dump_is_usable(path):
    """A cache worth reusing: it has frames and at least one detection."""
    try:
        frames, offs, dets, _ = load_dump(path)
        return len(frames) > 0 and len(dets) > 0
    except Exception:                                # noqa: BLE001 - unreadable file
        return False
