"""Run one trained detector over a COCO split and write its detections in COCO results format.

All three model families are scored by the same `score_coco.py` from these files, so the numbers are directly
comparable (the training tools each compute mAP their own way). Each family runs in its own environment:

    envs/yolo26/python  predict_coco.py yolo   --weights runs/yolo26/<run>/weights/best.pt      --split <coco>/test --out yolo.json
    envs/rfdetr/python  predict_coco.py rfdetr --weights runs/rfdetr/<run>/checkpoint_best_total.pth --split <coco>/test --out rfdetr.json
    envs/deimv2/python  predict_coco.py deimv2 --weights runs/deimv2/best_stg1.pth --config <DEIMv2>/configs/deimv2/<cfg>.yml \
                        --deim-repo <DEIMv2> --split <coco>/test --out deimv2.json

The app's own TensorRT engines are scored the same way, in the app's environment (TensorRT, repo root on the path):

    <app python>        predict_coco.py engine --weights models/weights/<name>.engine --split <coco>/test --out engine.json [--stretch]

A DEIM engine runs through models/trt_detector.py exactly as in the app; a YOLO engine through Ultralytics (pass
--stretch for a model trained on make_square_dataset.py output, which is how the app feeds every model).

Each model gets the preprocessing it was trained with: YOLO letterboxes to 640 (Ultralytics), RF-DETR and DEIMv2
resize the frame to a 640x640 square. Detections are kept down to `--conf` (default 0.001, the mAP setting), at
most `--max-det` per image. No DataLoader workers, so it runs over ssh on Windows; images load in threads.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image


def load_split(split):
    split = Path(split)
    gt = json.loads((split / '_annotations.coco.json').read_text())
    return split, gt['images']


def batches(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def threaded_images(split, images, batch, threads=8):
    """Yield (image records, PIL images) per batch, decoding the next batch while the GPU works."""
    pool = ThreadPoolExecutor(threads)
    load = lambda rec: Image.open(split / rec['file_name']).convert('RGB')
    chunks = list(batches(images, batch))
    pending = pool.map(load, chunks[0]) if chunks else None
    for i, chunk in enumerate(chunks):
        ims = list(pending)
        if i + 1 < len(chunks):
            pending = pool.map(load, chunks[i + 1])
        yield chunk, ims
    pool.shutdown()


def add(results, image_id, boxes_xyxy, scores, labels, conf, max_det):
    order = sorted(range(len(scores)), key=lambda k: -float(scores[k]))
    kept = 0
    for k in order:
        s = float(scores[k])
        if s < conf or kept >= max_det:
            break
        x1, y1, x2, y2 = (float(v) for v in boxes_xyxy[k])
        results.append(dict(image_id=image_id, category_id=int(labels[k]), score=round(s, 5),
                            bbox=[round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]))
        kept += 1


def run_yolo(a, split, images):
    import cv2
    import numpy as np
    from ultralytics import YOLO
    model = YOLO(a.weights)
    results, t_model = [], 0.0
    for recs, ims in threaded_images(split, images, a.batch):
        scale = [(1.0, 1.0)] * len(ims)
        if a.stretch:   # model trained by make_square_dataset.py: full frame resized to imgsz x imgsz, no letterbox
            scale = [(im.width / a.imgsz, im.height / a.imgsz) for im in ims]
            ims = [cv2.resize(np.asarray(im)[:, :, ::-1], (a.imgsz, a.imgsz), interpolation=cv2.INTER_AREA) for im in ims]
        t = time.perf_counter()
        preds = model.predict(ims, imgsz=a.imgsz, conf=a.conf, iou=0.7, max_det=a.max_det, verbose=False,
                              batch=len(ims), **({'half': True} if a.half else {}))
        t_model += time.perf_counter() - t
        for rec, p, (sx, sy) in zip(recs, preds, scale):
            b = p.boxes
            xyxy = (b.xyxy.cpu() * b.xyxy.new_tensor([sx, sy, sx, sy]).cpu()).tolist()
            add(results, rec['id'], xyxy, b.conf.cpu().tolist(), b.cls.cpu().tolist(), a.conf, a.max_det)
    return results, t_model


def run_rfdetr(a, split, images):
    import rfdetr
    cls = getattr(rfdetr, {'small': 'RFDETRSmall', 'medium': 'RFDETRMedium', 'large': 'RFDETRLarge'}[a.rfdetr_size])
    model = cls(pretrain_weights=a.weights, resolution=a.imgsz, num_classes=a.num_classes)
    results, t_model = [], 0.0
    for recs, ims in threaded_images(split, images, a.batch):
        t = time.perf_counter()
        dets = model.predict(ims, threshold=a.conf)
        t_model += time.perf_counter() - t
        dets = dets if isinstance(dets, list) else [dets]
        for rec, d in zip(recs, dets):
            add(results, rec['id'], d.xyxy.tolist(), d.confidence.tolist(), d.class_id.tolist(), a.conf, a.max_det)
    return results, t_model


def run_deimv2(a, split, images):
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    sys.path.insert(0, str(Path(a.deim_repo).resolve()))
    from engine.core import YAMLConfig

    cfg = YAMLConfig(a.config, resume=a.weights)
    ckpt = torch.load(a.weights, map_location='cpu')
    cfg.model.load_state_dict(ckpt['ema']['module'] if 'ema' in ckpt else ckpt['model'])

    class Deployed(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.post = cfg.postprocessor.deploy()

        def forward(self, x, sizes):
            return self.post(self.model(x), sizes)

    model = Deployed().cuda().eval()
    size = cfg.yaml_cfg['eval_spatial_size']
    tf = T.Compose([T.Resize(size), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    results, t_model = [], 0.0
    with torch.inference_mode():
        for recs, ims in threaded_images(split, images, a.batch):
            x = torch.stack([tf(im) for im in ims]).cuda()
            sizes = torch.tensor([[im.width, im.height] for im in ims]).cuda()
            torch.cuda.synchronize()
            t = time.perf_counter()
            with torch.autocast('cuda', dtype=torch.float16, enabled=a.half):
                labels, boxes, scores = model(x, sizes)
            torch.cuda.synchronize()
            t_model += time.perf_counter() - t
            for i, rec in enumerate(recs):
                add(results, rec['id'], boxes[i].float().cpu().tolist(), scores[i].float().cpu().tolist(),
                    labels[i].cpu().tolist(), a.conf, a.max_det)
    return results, t_model


def run_engine(a, split, images):
    """A TensorRT engine built by utility_scripts/onnx2tensorrt.py, run the way the app runs it."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from models.trt_detector import TensorRTDetector, engine_preprocessing, read_engine_metadata
    meta = read_engine_metadata(a.weights)
    if meta.get('model_type') != 'deim':
        return run_yolo(a, split, images)
    import torch
    import torchvision.transforms as T
    from models.preprocessing import normalize
    model = TensorRTDetector(a.weights, conf=a.conf)
    _, height, width = model.input_shape
    tf = T.Compose([T.Resize((height, width)), T.ToTensor()])
    results, t_model = [], 0.0
    for recs, ims in threaded_images(split, images, a.batch):
        x = normalize(torch.stack([tf(im) for im in ims]).cuda(), engine_preprocessing(meta))
        torch.cuda.synchronize()
        t = time.perf_counter()
        dets = model(x)
        torch.cuda.synchronize()
        t_model += time.perf_counter() - t
        for rec, im, d in zip(recs, ims, dets):
            d = d.boxes.data.cpu()
            xyxy = d[:, :4] * d.new_tensor([im.width / width, im.height / height] * 2)
            add(results, rec['id'], xyxy.tolist(), d[:, 4].tolist(), d[:, 5].tolist(), a.conf, a.max_det)
    return results, t_model


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('kind', choices=['yolo', 'rfdetr', 'deimv2', 'engine'])
    ap.add_argument('--weights', required=True)
    ap.add_argument('--split', required=True, help='COCO split folder holding _annotations.coco.json and the images')
    ap.add_argument('--out', required=True, help='results JSON to write')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.001)
    ap.add_argument('--max-det', type=int, default=300)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--half', action='store_true', help='FP16 inference (closer to the TensorRT engine)')
    ap.add_argument('--limit', type=int, default=0, help='only the first N images (smoke test)')
    ap.add_argument('--stretch', action='store_true', help='YOLO trained on make_square_dataset.py output')
    ap.add_argument('--rfdetr-size', default='medium', choices=['small', 'medium', 'large'])
    ap.add_argument('--num-classes', type=int, default=9, help='RF-DETR head size (production: 9)')
    ap.add_argument('--config', help='DEIMv2 config the model was trained with')
    ap.add_argument('--deim-repo', help='DEIMv2 clone (for its engine package)')
    a = ap.parse_args(argv)
    if a.kind == 'deimv2' and not (a.config and a.deim_repo):
        ap.error('deimv2 needs --config and --deim-repo')

    split, images = load_split(a.split)
    if a.limit:
        images = images[:a.limit]
    t0 = time.perf_counter()
    results, t_model = {'yolo': run_yolo, 'rfdetr': run_rfdetr, 'deimv2': run_deimv2, 'engine': run_engine}[a.kind](a, split, images)
    wall = time.perf_counter() - t0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results))
    meta = dict(kind=a.kind, weights=str(a.weights), split=str(split), images=len(images), detections=len(results),
                conf=a.conf, max_det=a.max_det, imgsz=a.imgsz, half=a.half, batch=a.batch,
                model_ms_per_image=round(1000 * t_model / max(1, len(images)), 2), wall_s=round(wall, 1))
    Path(a.out).with_suffix('.meta.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
