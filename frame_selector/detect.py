"""Stage 1: run the production model over a video and keep every detection, in the app's dump format."""
import json
import os
import time

import cv2
import numpy as np

MODEL_SIZE = 640


def load_model(model_path):
    from ultralytics import YOLO
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


def detect_video(video_path, model, out_path, conf=0.05, iou=0.7, batch=32, device=None, progress=print):
    """Detect on every frame (squashed to 640x640 like production) and save a dump at out_path."""
    info = video_info(video_path)
    cap = cv2.VideoCapture(video_path)
    frames, dets, offs = [], [], [0]
    pending, pending_idx = [], []
    names = None
    t0 = time.time()
    idx = 0

    def flush():
        nonlocal names
        results = model.predict(pending, imgsz=MODEL_SIZE, conf=conf, iou=iou, device=device, verbose=False, half=True)
        for r in results:
            if names is None:
                names = {int(k): str(v) for k, v in r.names.items()}
            d = r.boxes.data.cpu().numpy()[:, :6].astype(np.float32) if len(r.boxes) else np.zeros((0, 6), np.float32)
            dets.append(d)
            offs.append(offs[-1] + len(d))
        frames.extend(pending_idx)
        pending.clear()
        pending_idx.clear()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pending.append(cv2.resize(frame, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_LINEAR))
        pending_idx.append(idx)
        idx += 1
        if len(pending) == batch:
            flush()
            if idx % (batch * 50) == 0:
                progress(f"  {idx}/{info['frames']} frames, {idx / (time.time() - t0):.0f} fps")
    if pending:
        flush()
    cap.release()
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
