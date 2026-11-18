"""Fine-tune DEIMv2 (L by default: DINOv3-S backbone, 32M params, the size class of YOLO26-L).

DEIMv2 is a config-driven repo, not a pip package. This script writes a dataset+schedule yml next to the
repo's own configs and launches its train.py with torchrun, tuning from the COCO checkpoint.

One-time setup on the server:
    git clone https://github.com/Intellindust-AI-Lab/DEIMv2 && cd DEIMv2
    conda create -n deimv2 python=3.11 -y && conda activate deimv2 && pip install -r requirements.txt
    pip install comet_ml huggingface_hub safetensors
    python training/fetch_deimv2_ckpt.py --size l --out <DEIMv2>/ckpts/deimv2_dinov3_l_coco.pth
    # The COCO checkpoint carries the whole backbone, so the DINOv3 / ViT-Tiny backbone files are only
    # needed when training without -t (a missing weights_path is skipped by DEIMv2).

    python training/train_deimv2.py --repo /opt/DEIMv2 --dataset /data/dataset_261110_split/coco \
        --size l --coco-ckpt /opt/DEIMv2/ckpts/deimv2_dinov3_l_coco.pth --gpus 0,1 --total-batch 16

Single GPU (and always on Windows) runs `python train.py` without torchrun. Comet: on by default
(--no-comet), project `vehicle-detection`; the run's log.txt is forwarded each epoch.
Fetch the COCO checkpoint with training/fetch_deimv2_ckpt.py (Hugging Face, no DINOv3 licence form needed).

Schedule rules follow the repo's own conventions: flat_epoch = 4 + epochs // 2, no-aug tail of 8 epochs,
mosaic/mixup/copyblend active from epoch 4 until the no-aug tail, matcher switch at 3/4 of training.
Batch-size scaling follows the README: lr and backbone lr linear in total batch (reference 32), EMA decay
1 - (1 - 0.9999) * (batch / 32), warmup iterations capped to ~10% of the run.
"""
import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comet_utils  # noqa: E402

SIZES = {
    # config name, reference base lr / backbone lr / wd at total batch 32, default epochs
    'm': dict(cfg='deimv2_dinov3_m_coco.yml', lr=0.0005, lr_backbone=0.000025, weight_decay=0.0001),
    'l': dict(cfg='deimv2_dinov3_l_coco.yml', lr=0.0005, lr_backbone=0.0000125, weight_decay=0.000125),
    'x': dict(cfg='deimv2_dinov3_x_coco.yml', lr=0.0005, lr_backbone=0.0000125, weight_decay=0.000125),
}
REF_BATCH = 32


def count_images(ann_file):
    return len(json.load(open(ann_file))['images']), \
        [c['name'] for c in sorted(json.load(open(ann_file))['categories'], key=lambda c: c['id'])]


def write_rfs_annotations(ann_file, rfs_list, out_file):
    """COCO twin of oversample_rare.py's train_rfs.txt: an image listed k times appears k times (new ids)."""
    gt = json.load(open(ann_file))
    by_stem = {Path(im['file_name']).stem: im for im in gt['images']}
    anns = {}
    for ann in gt['annotations']:
        anns.setdefault(ann['image_id'], []).append(ann)
    images, annotations = [], []
    for line in Path(rfs_list).read_text().splitlines():
        if not line.strip():
            continue
        stem = Path(line.strip().replace('\\', '/')).stem
        im = by_stem.get(stem)
        if im is None:
            raise SystemExit(f'{rfs_list}: {stem} is not in {ann_file}')
        new_id = len(images) + 1
        images.append(dict(im, id=new_id))
        for ann in anns.get(im['id'], []):
            annotations.append(dict(ann, id=len(annotations) + 1, image_id=new_id))
    Path(out_file).write_text(json.dumps(dict(gt, images=images, annotations=annotations)))
    return len(images)


# --aug cctv: the stock DEIMv2 recipe is tuned for COCO. On this data (62% of boxes < 32 px at 640, 30% of
# vehicles < 16 px, fixed upright cameras) three parts of it work against us:
#  - Mosaic pastes 4 frames into a 2x2 canvas, RandomAffine keeps the canvas size and the final Resize squashes
#    it to 640: every object ends at half size (x the 0.5-1.5 affine scale). Scale range 1.5-2.5 crops the
#    centre instead (objects at 0.75-1.25x of a normal frame, like Ultralytics); tiles at 640 px, not 320,
#    so the crop is not an upsampled blur. No rotation (upright cameras; rotated boxes are loose).
#  - RandomZoomOut (side 1-4, p 0.5) shrinks objects up to 4x more: limited to 1-1.5x at p 0.3.
#  - RandomPhotometricDistort permutes colour channels; colour separates taxis from buses. Replaced by the
#    shared CCTV pixel set (blur, noise, sub-stream downscale, JPEG, gamma, IR-grey, weather) + mild HSV.
def cctv_transforms(a, flat, aug_stop):
    return f"""    transforms:
      ops:
        - {{type: Mosaic, output_size: 640, rotation_range: 0, translation_range: [0.1, 0.1], scaling_range: [1.5, 2.5],
           probability: 1.0, fill_value: 0, use_cache: True, max_cached_images: 50, random_pop: True}}
        - {{type: RandomZoomOut, fill: 0, side_range: [1.0, 1.5], p: 0.3}}
        - {{type: RandomIoUCrop, p: 0.8}}
        - {{type: SanitizeBoundingBoxes, min_size: 1}}
        - {{type: RandomHorizontalFlip}}
        - {{type: CCTVPixelAug, level: cctv, strength: {a.alb_strength}, colour_jitter: True}}
        - {{type: Resize, size: [640, 640], }}
        - {{type: SanitizeBoundingBoxes, min_size: 1}}
        - {{type: ConvertPILImage, dtype: 'float32', scale: True}}
        - {{type: Normalize, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]}}
        - {{type: ConvertBoxes, fmt: 'cxcywh', normalize: True}}
      policy:
        name: stop_epoch
        epoch: [4, {flat}, {aug_stop}]
        ops: ['Mosaic', 'RandomZoomOut', 'RandomIoUCrop', 'CCTVPixelAug']
      mosaic_prob: {a.mosaic_prob}
"""


def make_config(a, n_train, classes):
    s = SIZES[a.size]
    scale = a.total_batch / REF_BATCH
    e = a.epochs
    flat = 4 + e // 2
    no_aug = max(4, round(e * 8 / 68))          # 8 of 68 epochs in the reference schedule
    aug_stop = e - no_aug
    iters_per_epoch = max(1, n_train // a.total_batch)
    warmup = min(2000, max(200, iters_per_epoch * e // 10))
    ema_decay = round(1 - (1 - 0.9999) * scale, 6)
    lr = s['lr'] * scale
    lr_bb = s['lr_backbone'] * scale
    if a.aug == 'cctv':
        transforms = cctv_transforms(a, flat, aug_stop)
    else:
        transforms = f"""    transforms:
      policy:
        epoch: [4, {flat}, {aug_stop}]
      mosaic_prob: {a.mosaic_prob}
"""
    cfg = f"""# generated by training/train_deimv2.py on {datetime.now():%Y-%m-%d %H:%M} (aug {a.aug})
__include__: [
  './{s['cfg']}',
]

output_dir: {a.output}

# ---- dataset (COCO folder from resplit_dataset.py; ids are 0..nc-1, no remapping) ----
num_classes: {len(classes)}
remap_mscoco_category: False

train_dataloader:
  dataset:
    img_folder: {a.dataset}/train
    ann_file: {a.train_ann}
{transforms}  collate_fn:
    mixup_prob: {a.mixup_prob}
    mixup_epochs: [4, {flat}]
    stop_epoch: {aug_stop}
    copyblend_prob: {a.copyblend_prob}
    copyblend_epochs: [4, {aug_stop}]
  total_batch_size: {a.total_batch}
  num_workers: {a.workers}

val_dataloader:
  dataset:
    img_folder: {a.dataset}/valid
    ann_file: {a.dataset}/valid/_annotations.coco.json
  total_batch_size: {max(a.total_batch, 16)}
  num_workers: {a.workers}

# ---- schedule ----
epoches: {e}
flat_epoch: {flat}
no_aug_epoch: {no_aug}
warmup_iter: {warmup}
checkpoint_freq: 5

# ---- optimizer, scaled linearly from the reference total batch of {REF_BATCH} ----
optimizer:
  type: AdamW
  params:
    -
      params: '^(?=.*.dinov3)(?!.*(?:norm|bn|bias)).*$'
      lr: {lr_bb:.8f}
    -
      params: '^(?=.*.dinov3)(?=.*(?:norm|bn|bias)).*$'
      lr: {lr_bb:.8f}
      weight_decay: 0.
    -
      params: '^(?=.*(?:sta|encoder|decoder))(?=.*(?:norm|bn|bias)).*$'
      weight_decay: 0.
  lr: {lr:.8f}
  betas: [0.9, 0.999]
  weight_decay: {s['weight_decay']}

ema:
  decay: {ema_decay}
  warmups: {warmup}

DEIMCriterion:
  matcher:
    matcher_change_epoch: {round(e * 0.75)}
"""
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repo', required=True, help='path to the cloned DEIMv2 repository')
    ap.add_argument('--dataset', required=True, help='COCO folder from resplit_dataset.py (train/ valid/ test/)')
    ap.add_argument('--size', choices=sorted(SIZES), default='l')
    ap.add_argument('--coco-ckpt', default=None, help='COCO-pretrained DEIMv2 checkpoint to tune from (-t). Strongly recommended.')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--mosaic-prob', type=float, default=0.5, help='dense-O2O mosaic probability (repo default 0.5)')
    ap.add_argument('--mixup-prob', type=float, default=0.5, help='dense-O2O mixup probability (repo default 0.5)')
    ap.add_argument('--copyblend-prob', type=float, default=0.5, help='dense-O2O copy-blend probability (repo default 0.5)')
    ap.add_argument('--aug', choices=['stock', 'cctv'], default='stock',
                    help='stock = the repo recipe; cctv = scale-preserving mosaic, mild zoom-out, no rotation or '
                         'channel permutation, CCTV pixel augmentations (see cctv_transforms)')
    ap.add_argument('--alb-strength', type=float, default=1.0, help='scales the CCTV pixel probabilities (--aug cctv)')
    ap.add_argument('--rfs-list', default=None,
                    help="train_rfs.txt from oversample_rare.py: repeat rare-class images like the YOLO runs")
    ap.add_argument('--total-batch', type=int, default=16, help='total batch across all GPUs (reference 32)')
    ap.add_argument('--gpus', default='0', help='CUDA_VISIBLE_DEVICES, e.g. 0,1')
    ap.add_argument('--workers', type=int, default=8, help='dataloader workers per process')
    ap.add_argument('--output', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--resume', default=None, help='checkpoint to resume (-r)')
    ap.add_argument('--port', type=int, default=7777)
    ap.add_argument('--dry-run', action='store_true', help='write the config and print the command only')
    ap.add_argument('--comet', action=argparse.BooleanOptionalAction, default=True, help='log to Comet (default on)')
    ap.add_argument('--comet-project', default=comet_utils.DEFAULT_PROJECT)
    a = ap.parse_args(argv)

    a.dataset = str(Path(a.dataset).resolve())
    a.output = a.output or f"./outputs/deimv2_{a.size}_traffic_{datetime.now():%y%m%d_%H%M}"
    base_ann = Path(a.dataset) / 'train' / '_annotations.coco.json'
    if a.rfs_list:
        rfs_ann = Path(a.dataset) / 'train' / '_annotations_rfs.coco.json'
        n = write_rfs_annotations(base_ann, a.rfs_list, rfs_ann)
        print(f'wrote {rfs_ann} ({n} image entries from {a.rfs_list})')
        a.train_ann = rfs_ann.as_posix()
    else:
        a.train_ann = base_ann.as_posix()
    n_train, classes = count_images(a.train_ann)
    n_gpu = len(a.gpus.split(','))
    if a.total_batch % n_gpu:
        ap.error('--total-batch must be divisible by the number of GPUs')

    # one config per run (named after the output folder), so a new run never rewrites an earlier run's config
    run_tag = Path(a.output).name
    cfg_path = Path(a.repo) / 'configs' / 'deimv2' / f'deimv2_dinov3_{a.size}_traffic_{run_tag}.yml'
    cfg_path.write_text(make_config(a, n_train, classes))
    print(f'wrote {cfg_path}  ({n_train} train images, {len(classes)} classes, {n_train // a.total_batch} it/epoch)')

    # torchrun only for multi-GPU on Linux: NCCL does not exist on Windows, and with one GPU DEIMv2's
    # setup_distributed falls back to plain single-process training when launched without torchrun.
    if n_gpu > 1 and platform.system() != 'Windows':
        launcher = [sys.executable, '-m', 'torch.distributed.run', f'--master_port={a.port}', f'--nproc_per_node={n_gpu}']
    else:
        if n_gpu > 1:
            ap.error('multi-GPU DEIMv2 training needs Linux (NCCL); use one GPU on Windows')
        launcher = [sys.executable]
    # deim_launch.py registers this folder's config ops (CCTVPixelAug), then runs the repo's train.py unchanged
    cmd = launcher + [str(Path(__file__).resolve().parent / 'deim_launch.py'), '-c', str(cfg_path.relative_to(a.repo)),
                      '--use-amp', f'--seed={a.seed}']
    if a.coco_ckpt:
        cmd += ['-t', a.coco_ckpt]
    else:
        print('warning: no --coco-ckpt; training the detector head from scratch on a small dataset converges far worse')
    if a.resume:
        cmd += ['-r', a.resume]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpus)
    print('cd', a.repo, '&& CUDA_VISIBLE_DEVICES=' + a.gpus, ' '.join(shlex.quote(c) for c in cmd))
    if a.dry_run:
        return
    out_dir = (Path(a.repo) / a.output).resolve() if not Path(a.output).is_absolute() else Path(a.output)
    exp = comet_utils.start(a.comet_project, out_dir.name, tags=['deimv2', f'deimv2-{a.size}', 'img640', f'aug-{a.aug}',
                                                                 *(['rfs'] if a.rfs_list else []), Path(a.dataset).parent.name],
                            params=dict(size=a.size, epochs=a.epochs, total_batch=a.total_batch, coco_ckpt=a.coco_ckpt,
                                        mosaic_prob=a.mosaic_prob, mixup_prob=a.mixup_prob, copyblend_prob=a.copyblend_prob,
                                        aug=a.aug, alb_strength=a.alb_strength, rfs_list=a.rfs_list,
                                        seed=a.seed, config=str(cfg_path)), enabled=a.comet)
    # the config is uploaded at the end with log.txt: an upload right at start, minutes after a YOLO run's burst
    # of plot uploads, hit Comet's file-upload throttle and the client's retry loop killed the logging (2026-11-16)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / cfg_path.name).write_text(cfg_path.read_text())   # keep the config with the run
    tail = comet_utils.MetricTail(exp, out_dir / 'log.txt', 'jsonl')
    tail.start()
    try:
        code = subprocess.call(cmd, cwd=a.repo, env=env)
    finally:
        tail.stop()
        comet_utils.finish(exp, [out_dir / 'log.txt', out_dir / cfg_path.name])
    sys.exit(code)


if __name__ == '__main__':
    main()
