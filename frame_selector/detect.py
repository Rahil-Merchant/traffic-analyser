"""Stage 1: run the production model over a video and keep every detection, in the app's dump format.

Decoding is the bottleneck, so frames come from the GPU decoder the app uses (video_reader-rs with
CUDA, resizing on the GPU, 7,600-frame segments) when it is installed, with the next segment decoding
while the current one is being detected. OpenCV decoding is the fallback.
"""
import contextlib
import hashlib
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
# The production app passes no thresholds, so ultralytics' defaults apply there: conf 0.25, iou 0.7
# (config.py's CONF_THRESHOLD / IOU_THRESHOLD never reach the model). Detect exactly as production does.
CONF = 0.25
IOU = 0.7
# Bumped whenever cached detections from earlier versions must not be reused.
# 2: inputs made contiguous (version 1 fed TensorRT scrambled frames), thresholds as production.
DETECTOR_VERSION = 2
LOW_DETECTION_RATE = 0.01    # fewer detections per frame than this gets a warning in the log


def configured_model():
    """The production model from config.py (MODEL_PATH, chosen per machine by PC_PATHS)."""
    import io
    with contextlib.redirect_stdout(io.StringIO()):      # config.py prints a banner on import
        from config import MODEL_PATH
    return MODEL_PATH


def model_id(model_path):
    """What identifies the model a dump was made with: file name, size and modification time.
    None when the model file isn't on this machine."""
    try:
        st = os.stat(model_path)
    except (OSError, TypeError):
        return None
    return dict(name=os.path.basename(model_path), size=st.st_size, mtime=int(st.st_mtime))


def load_model(model_path=None):
    from ultralytics import YOLO
    model_path = model_path or configured_model()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model not found: {model_path} (set MODEL_PATH / PC in config.py)")
    return YOLO(model_path, task='detect')


def video_key(video_path):
    """Short unique name for a video: its stem plus a hash of its size and first MB. Recorders reuse
    file names (06.00.00-08.00.00[R][0@0][0].dav every day), so names alone would let different videos
    share caches, image names and manifest rows."""
    stem = os.path.splitext(os.path.basename(video_path))[0]
    h = hashlib.sha1(str(os.path.getsize(video_path)).encode())
    with open(video_path, 'rb') as fh:
        h.update(fh.read(1 << 20))
    return f"{stem}_{h.hexdigest()[:6]}"


def video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    info = dict(fps=cap.get(cv2.CAP_PROP_FPS) or 20.0, frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return info


def read_frames(video_path, frame_indices):
    """Yield (frame_index, bgr_frame) for the given indices, in ascending order, where an index is the
    frame's position in decode order, as in the detection dumps. The video is stepped through from the
    start: these recorders' timestamps jump (a 1 s gap, 138 ms steps), so any seek or timestamp
    arithmetic lands up to a second away from the frame that was detected."""
    cap = cv2.VideoCapture(video_path)
    try:
        pos = 0                                      # index of the next frame the capture will return
        for idx in sorted(set(int(i) for i in frame_indices)):
            while pos < idx:
                if not cap.grab():                   # decodes without converting: the cheap way to skip
                    return
                pos += 1
            ok, frame = cap.read()
            if not ok:
                return
            pos += 1
            yield idx, frame
    finally:
        cap.release()


def gpu_frame_count(video_path):
    from video_reader import PyVideoReader
    return int(PyVideoReader(video_path).get_info().get('frame_count', 0))


def _segments_gpu(video_path, total, progress=print):
    """(first_index, frames RGB (N,640,640,3)) per segment, decoded on the GPU like the app."""
    from video_reader import PyVideoReader
    pvr = PyVideoReader(video_path, device='cuda', resize_shorter_side=MODEL_SIZE, resize_longer_side=MODEL_SIZE)
    for start in range(0, total, SEGMENT):
        end = min(start + SEGMENT, total)
        try:
            seg = pvr.decode(start_frame=start, end_frame=end)
        except Exception as e:                       # noqa: BLE001 - skip a corrupt segment, as the app does
            progress(f"  WARNING: could not decode frames {start}-{end}, skipped: {e}")
            continue
        if seg is None or not len(seg):
            progress(f"  WARNING: no frames decoded for {start}-{end}, skipped")
            continue
        if len(seg) != end - start:
            progress(f"  WARNING: frames {start}-{end}: the decoder returned {len(seg)} of {end - start}")
        yield start, np.ascontiguousarray(seg)


def _segments_cv2(video_path, total, progress=print):
    """Fallback: OpenCV decode on the CPU, in smaller segments, frames converted to RGB."""
    cap = cv2.VideoCapture(video_path)
    idx, buf = 0, []
    try:
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
    finally:
        cap.release()


def _prefetch(gen, depth=1):
    """Run a generator in a thread so decoding overlaps inference (a GPU segment is ~9 GB of RAM, so
    only one waits). A decoder error is re-raised in the caller's thread. Closing this generator,
    including on an error in the caller, stops the thread and closes `gen`, releasing the video file."""
    q = queue.Queue(maxsize=depth)
    done = object()
    stop = threading.Event()

    def put(item):
        while not stop.is_set():
            try:
                q.put(item, timeout=0.5)
                return True
            except queue.Full:
                pass
        return False

    def pump():
        try:
            for item in gen:
                if not put(item):
                    return
            put(done)
        except BaseException as e:                   # noqa: BLE001 - handed to the consumer
            put(e)
        finally:
            gen.close()

    t = threading.Thread(target=pump, daemon=True)
    t.start()
    try:
        while True:
            item = q.get()
            if item is done:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        with contextlib.suppress(queue.Empty):
            while True:
                q.get_nowait()
        t.join(timeout=120)


def detect_video(video_path, model, out_path, device=None, progress=print, model_path=None):
    """Detect on every frame (squashed to 640x640 like production) and save a dump at out_path."""
    import torch
    info = video_info(video_path)
    total = info['frames']
    temp = None
    try:
        try:
            import video_reader  # noqa: F401
            source = video_path
            if video_path.lower().endswith('.dav'):  # the GPU decoder needs a container: same conversion as the app
                from core.converters import convert_dav_to_mp4_internal
                progress("  converting .dav to .mp4 for the GPU decoder (stream copy)...")
                temp = convert_dav_to_mp4_internal(video_path)
                if not temp:
                    raise RuntimeError(f"could not convert {video_path} to mp4 (is ffmpeg on PATH?)")
                source = temp
            total = gpu_frame_count(source) or total
            if total <= 0:
                raise RuntimeError(f"no frames reported for {video_path}")
            segments, how = _segments_gpu(source, total, progress), 'GPU decode'
        except ImportError:
            segments, how = _segments_cv2(video_path, total, progress), 'CPU decode (video_reader not installed)'
        device = device or ('cuda' if torch.cuda.is_available() else None)
        use_half = device == 'cuda'
        progress(f"  {how}, batch {BATCH}, {total} frames")
        frames, dets, offs = [], [], [0]
        names = None
        t0 = time.time()
        with torch.no_grad(), contextlib.closing(_prefetch(segments)) as batches:
            for start, seg in batches:
                for b in range(0, len(seg), BATCH):
                    chunk = seg[b:b + BATCH]
                    n = len(chunk)
                    # NHWC -> NCHW. contiguous() matters: the TensorRT backend reads the tensor's raw memory,
                    # so a permuted view reaches the engine as scrambled pixels (it then detects almost nothing).
                    x = torch.from_numpy(chunk).to(device or 'cpu', non_blocking=True).permute(0, 3, 1, 2).contiguous()
                    x = (x.half() if use_half else x.float()) / 255.0
                    assert x.is_contiguous()
                    results = model(x, imgsz=MODEL_SIZE, conf=CONF, iou=IOU, verbose=False, half=use_half, device=device)
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
        if len(frames) < 0.99 * total:
            progress(f"  WARNING: only {len(frames)} of {total} frames were decoded")
        if n_det < LOW_DETECTION_RATE * len(frames):
            progress(f"  WARNING: only {n_det} detections in {len(frames)} frames. Unless the road really is empty, "
                     f"check that the model works on this video")
        meta = dict(video_path=os.path.abspath(video_path), video_name=os.path.basename(video_path),
                    video_size=os.path.getsize(video_path), start_datetime=None, original_fps=info['fps'],
                    total_frames=len(frames), effective_fps=info['fps'], frame_shape=[info['width'], info['height']],
                    model_input_size=MODEL_SIZE, letterbox_params=None, class_names=names or {},
                    detector=dict(version=DETECTOR_VERSION, model=model_id(model_path or configured_model()), conf=CONF, iou=IOU))
        np.savez_compressed(out_path, frame=np.array(frames, dtype=np.int64), offs=np.array(offs, dtype=np.int64),
                            dets=np.concatenate(dets) if dets else np.zeros((0, 6), np.float32), meta=np.array(json.dumps(meta)))
        return out_path
    finally:
        if temp:
            try:
                os.remove(temp)
            except OSError as e:
                progress(f"  WARNING: could not delete the temporary file {temp}: {e}")


def load_dump(path):
    """-> (frames (N,), offs (N+1,), dets (M,6) [x1,y1,x2,y2,conf,cls] in 640x640 model space, meta dict)."""
    z = np.load(path)
    meta = json.loads(str(z['meta']))
    meta['class_names'] = {int(k): v for k, v in meta['class_names'].items()}
    return z['frame'], z['offs'], z['dets'], meta


def dump_status(path, video_path=None, match_name=True):
    """None when the dump at `path` can be reused for `video_path`, otherwise why not.
    Frame selector dumps must come from this detector version, the current model and the same video
    file. Production app dumps (named after the app's temp file, '<stem>_conv_xxxx') must have
    detections and a matching frame count. match_name=False skips the name checks, for dumps a user
    paired with a video explicitly."""
    try:
        frames, offs, dets, meta = load_dump(path)
    except Exception as e:                           # noqa: BLE001 - unreadable file
        return f"unreadable ({type(e).__name__})"
    if len(frames) == 0:
        return "no frames"
    det = meta.get('detector')
    if det is not None:                              # made by the frame selector
        if det.get('version') != DETECTOR_VERSION or det.get('conf') != CONF or det.get('iou') != IOU:
            return "made by an older version of the frame selector"
        if frames[0] != 0 or frames[-1] != len(frames) - 1:
            return "incomplete: part of the video could not be decoded"
        current = model_id(configured_model())
        if current is not None and det.get('model') != current:
            return f"made with a different model ({(det.get('model') or {}).get('name')})"
        if video_path is not None:
            if os.path.getsize(video_path) != meta.get('video_size'):
                return "made from a different video"
            if match_name and meta.get('video_name') != os.path.basename(video_path):
                return "made from a different video"
        return None
    if 'conf' in meta:                               # frame selector version 1: scrambled engine input
        return "made by an older version of the frame selector"
    if len(dets) == 0:                               # production app dump
        return "no detections"
    if video_path is not None:
        if match_name:
            stem = os.path.splitext(os.path.basename(video_path))[0]
            src = os.path.splitext(meta.get('video_path', '').replace('\\', '/').rsplit('/', 1)[-1])[0]   # a Windows path on any OS
            if src != stem and not src.startswith(stem + '_conv_'):
                return "made from a different video"
        n = video_info(video_path)['frames']
        if n > 0 and abs(len(frames) - n) > 0.02 * n:
            return f"has {len(frames)} frames but the video has {n}"
    return None


def dump_is_usable(path, video_path=None, match_name=True):
    return dump_status(path, video_path, match_name) is None
