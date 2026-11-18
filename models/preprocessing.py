"""Pixel preprocessing per model family, chosen by `MODEL_PREPROCESSING` in config.py.

Frames reach the model as RGB, already resized to MODEL_INPUT_SIZE x MODEL_INPUT_SIZE (the GPU decoder
stretches the full frame to a square) and scaled to 0..1. What differs between families is the
normalisation applied after that:

    'yolo' : nothing more                        (Ultralytics YOLO)
    'deim' : subtract the ImageNet mean, divide by its std, per channel   (DEIMv2 with a DINOv3 backbone, RF-DETR)

A model fed the other family's input still returns boxes, only far fewer and worse ones, so engines built by
utility_scripts/onnx2tensorrt.py record which one they need and the loaders refuse a mismatch.
"""

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# mode -> (mean, std) per RGB channel, or None for no normalisation
MODES = {
    'yolo': None,
    'deim': (IMAGENET_MEAN, IMAGENET_STD),
}


def check_mode(mode):
    if mode not in MODES:
        raise ValueError(f"MODEL_PREPROCESSING is {mode!r}; it must be one of {sorted(MODES)}")
    return mode


def normalize(batch, mode):
    """batch: float tensor (N, 3, H, W), RGB in 0..1. Returns the tensor the model expects (same dtype)."""
    stats = MODES[check_mode(mode)]
    if stats is None:
        return batch
    mean = batch.new_tensor(stats[0]).view(1, 3, 1, 1)
    std = batch.new_tensor(stats[1]).view(1, 3, 1, 1)
    return (batch - mean) / std
