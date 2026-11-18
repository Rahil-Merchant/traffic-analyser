"""Export a trained detector to the ONNX file that utility_scripts/onnx2tensorrt.py turns into the app's engine.

    # DEIMv2, in the DEIMv2 training environment
    python utility_scripts/export_onnx.py deim --weights runs/deimv2-l-cctv/best_stg2.pth \
        --config <DEIMv2>/configs/deimv2/deimv2_dinov3_l_traffic_deimv2-l-cctv.yml --deim-repo <DEIMv2> \
        --out models/weights/deimv2_l_cctv.onnx

    # YOLO26 trained with the one-to-many head only (training/cfg/yolo26-o2m*.yaml), in the YOLO26 environment
    python utility_scripts/export_onnx.py yolo --weights runs/yolo26/<run>/weights/best.pt --out models/weights/yolo26l_o2m.onnx

Two steps and two environments because the training code needs newer packages than the app runs with, and
the engine has to be built by the app's own TensorRT. Both ONNX files take `images` (batch, 3, 640, 640) with a
dynamic batch and carry the class names and the preprocessing they need in their metadata:

  deim  input: RGB 0..1, then ImageNet mean/std (MODEL_PREPROCESSING 'deim').
        output `detections` (batch, 300, 6): x1, y1, x2, y2, score, class in input pixels. The model's own
        postprocessor is part of the graph and the boxes are final: no NMS.
        The file is mixed precision, traced under PyTorch autocast: FP16, with the operations that overflow
        FP16 (attention scores, softmax, layer norm, pow, log) in FP32. A fully FP16 engine of this model
        returns NaN.
  yolo  input: RGB 0..1 (MODEL_PREPROCESSING 'yolo').
        output `output0` (batch, 4 + classes, anchors), the raw one-to-many head: Ultralytics runs NMS on it in
        the app, as for every YOLO engine before it. A model with the end-to-end head is refused.
"""
import argparse
import contextlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

DEFAULT_CLASSES = HERE.parent / 'training' / 'classes_production.yaml'


def stamp(onnx_path, **meta):
    """Add metadata to the ONNX file, the way Ultralytics does (values stored as strings)."""
    import onnx
    model = onnx.load(str(onnx_path))
    have = {p.key: p for p in model.metadata_props}
    for k, v in meta.items():
        prop = have.get(k) or model.metadata_props.add()
        prop.key, prop.value = k, str(v)
    onnx.save(model, str(onnx_path))


def class_names(path):
    import yaml
    data = yaml.safe_load(Path(path).read_text())
    names = data.get('classes', data.get('names'))
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    if not names:
        raise SystemExit(f"{path}: no 'classes' or 'names' list")
    return {i: str(n) for i, n in enumerate(names)}


@contextlib.contextmanager
def fp32_attention():
    """While active, attention is traced in FP32 where a TensorRT FP16 engine of DEIMv2 returns NaN.

    Two places, both found by exposing the engine's intermediate layers on the 261110 test images:

    - Scaled dot-product attention (the DINOv3 backbone blocks) computes its scores, q @ k and the softmax, in
      FP32. In the first block every score exceeds what FP16 can hold (they reach 2.5e6; FP16 ends at 65504).
      PyTorch and ONNX Runtime survive a softmax over a row of inf; TensorRT returns NaN from there on. The
      values (attention @ v) and the linear layers around it stay FP16.
    - nn.MultiheadAttention modules (the encoder's one transformer layer, the decoder's self-attention) run
      entirely in FP32. TensorRT fuses the encoder layer into one block that returns NaN in FP16 at batch 64
      (not at batch 4). These modules see 400 tokens and 300 queries, so FP32 costs nothing measurable.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    sdpa, mha_forward = F.scaled_dot_product_attention, nn.MultiheadAttention.forward

    def as_fp32(v):
        return v.float() if torch.is_tensor(v) and v.is_floating_point() else v

    def attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
        if is_causal or kwargs.get('enable_gqa'):
            raise NotImplementedError('causal / grouped-query attention is not handled by this export')
        with torch.autocast(query.device.type, enabled=False):
            scale_ = query.shape[-1] ** -0.5 if scale is None else scale
            scores = (query.float() * scale_) @ key.float().transpose(-2, -1)
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, float('-inf')) if attn_mask.dtype == torch.bool else scores + attn_mask.float()
            return scores.softmax(dim=-1).to(value.dtype) @ value

    def forward(self, query, key, value, *args, **kwargs):
        with torch.autocast(query.device.type, enabled=False):
            return mha_forward(self, query.float(), key.float(), value.float(),
                               *[as_fp32(v) for v in args], **{k: as_fp32(v) for k, v in kwargs.items()})

    F.scaled_dot_product_attention, nn.MultiheadAttention.forward = attention, forward
    try:
        yield
    finally:
        F.scaled_dot_product_attention, nn.MultiheadAttention.forward = sdpa, mha_forward


def export_deim(a):
    import torch
    import torch.nn as nn
    sys.path.insert(0, str(Path(a.deim_repo).resolve()))
    from engine.core import YAMLConfig

    names = class_names(a.classes)
    cfg = YAMLConfig(a.config, resume=a.weights)
    ckpt = torch.load(a.weights, map_location='cpu')
    cfg.model.load_state_dict(ckpt['ema']['module'] if 'ema' in ckpt else ckpt['model'])   # the EMA weights are the evaluated ones
    num_classes = cfg.yaml_cfg['num_classes']
    if num_classes != len(names):
        raise SystemExit(f"the model has {num_classes} classes but {a.classes} lists {len(names)}")
    height, width = cfg.yaml_cfg['eval_spatial_size']

    mixed = not a.fp32
    if mixed and not torch.cuda.is_available():
        raise SystemExit('the mixed-precision export traces the model on the GPU; no CUDA here (or pass --fp32)')
    device = 'cuda' if mixed else 'cpu'

    class Deployed(nn.Module):
        """Model + its postprocessor, boxes scaled to the input size, as one (batch, queries, 6) tensor."""
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.post = cfg.postprocessor.deploy()
            self.register_buffer('size', torch.tensor([[float(width), float(height)]]))

        def forward(self, images):
            # Mixed precision is PyTorch autocast, the mode the model was trained in: FP16 for the convolutions and
            # linear layers that take the time, FP32 where FP16 (largest value 65504) overflows: softmax, layer
            # norm, pow, log, plus the attention parts in fp32_attention(). The tracer records those casts,
            # so the ONNX file carries a type for every layer.
            with torch.autocast(device, dtype=torch.float16, enabled=mixed):
                out = self.model(images)
            with torch.autocast(device, enabled=False):          # scores and pixel boxes in full precision
                out = {k: v.float() for k, v in out.items()}
                labels, boxes, scores = self.post(out, self.size.expand(images.shape[0], -1))
                return torch.cat([boxes, scores.unsqueeze(-1), labels.unsqueeze(-1).to(boxes.dtype)], dim=-1)

    model = Deployed().eval().to(device)
    dummy = torch.rand(2, 3, height, width, device=device)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad(), (fp32_attention() if mixed else contextlib.nullcontext()):
        ref = model(dummy)
        torch.onnx.export(model, dummy, str(out), input_names=['images'], output_names=['detections'],
                          dynamic_axes={'images': {0: 'batch'}, 'detections': {0: 'batch'}},
                          opset_version=a.opset, do_constant_folding=True)
    precision = 'mixed' if mixed else 'fp32'
    stamp(out, model_type='deim', preprocessing='deim', names=names, imgsz=[height, width], task='detect',
          precision=precision,
          description=f"DEIMv2 {Path(a.config).stem}, weights {Path(a.weights).name}",
          output='detections (batch, queries, 6): x1, y1, x2, y2, score, class in input pixels; no NMS needed')
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB, {precision} precision): images (batch, 3, {height}, {width}) -> "
          f"detections {tuple(ref.shape[1:])} per image, {len(names)} classes")
    check_onnx(out, dummy.cpu(), ref.cpu())


def check_onnx(out, dummy, ref):
    """Compare ONNX Runtime with PyTorch on a random input, at a batch size other than the traced one."""
    try:
        import onnxruntime as ort
    except ImportError:
        print('onnxruntime is not installed here: ONNX output not compared with PyTorch (the engine is checked later)')
        return
    sess = ort.InferenceSession(str(out), providers=['CPUExecutionProvider'])
    got = sess.run(None, {'images': dummy[:1].numpy()})[0]
    keep = ref[0, :, 4] > 0.05
    box = float(abs(got[0, keep.numpy(), :4] - ref[0, keep, :4].numpy()).max()) if keep.any() else 0.0
    score = float(abs(got[0, :, 4] - ref[0, :, 4].numpy()).max())
    print(f"ONNX Runtime vs PyTorch: largest score difference {score:.5f}, largest box difference {box:.3f} px")


def export_yolo(a):
    from ultralytics import YOLO
    model = YOLO(a.weights)
    if getattr(model.model, 'end2end', False):
        # A dual-head YOLO26 exports its one-to-one head, whose (batch, 300, 6) output the app's NMS cannot read,
        # and its one-to-many head is under-trained (the loss weight moves away from it during training).
        raise SystemExit(f"{a.weights} has the end-to-end head. Train with training/cfg/yolo26-o2m*.yaml "
                         f"(end2end: False), which trains the one-to-many head the app uses.")
    made = Path(model.export(format='onnx', imgsz=a.imgsz, dynamic=True, simplify=True, nms=False))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if made.resolve() != out.resolve():
        made.replace(out)
    stamp(out, preprocessing='yolo')
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB): images (batch, 3, {a.imgsz}, {a.imgsz}) -> raw head output, "
          f"{len(model.names)} classes: {', '.join(model.names.values())}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='kind', required=True)
    d = sub.add_parser('deim', help='DEIMv2 checkpoint (.pth)')
    d.add_argument('--weights', required=True)
    d.add_argument('--config', required=True, help='the config the model was trained with, inside the DEIMv2 clone')
    d.add_argument('--deim-repo', required=True, help='DEIMv2 clone (for its engine package)')
    d.add_argument('--classes', default=str(DEFAULT_CLASSES), help='yaml with the class list in model order')
    d.add_argument('--opset', type=int, default=17)
    d.add_argument('--fp32', action='store_true', help='export in FP32 on the CPU (the engine is about 4x slower than mixed precision)')
    d.add_argument('--out', required=True)
    d.set_defaults(run=export_deim)
    y = sub.add_parser('yolo', help='Ultralytics YOLO checkpoint (.pt)')
    y.add_argument('--weights', required=True)
    y.add_argument('--imgsz', type=int, default=640)
    y.add_argument('--out', required=True)
    y.set_defaults(run=export_yolo)
    a = ap.parse_args(argv)
    a.run(a)


if __name__ == '__main__':
    main()
