"""Fine-tune RF-DETR (Medium by default, the size class of YOLO26-L) on the re-split dataset.

RF-DETR reads the COCO folder written by resplit_dataset.py (train/ valid/ test/, each holding images and
_annotations.coco.json). Category ids are 0..nc-1 with no supercategory placeholder.

    pip install "rfdetr[train,loggers]>=1.12"            # Python >= 3.10
    python training/train_rfdetr.py --dataset /data/dataset_261110_split/coco --size medium
    python training/train_rfdetr.py --dataset ... --size large --batch 8 --grad-accum 2

Sizes (rfdetr 1.12):
    medium  DINOv2-S backbone, 4 decoder layers, native 576 px   -> closest to YOLO26-L (~25M params)
    large   DINOv2-S backbone, 4 decoder layers, native 704 px   -> same weights class, more compute
    small   DINOv2-S backbone, 3 decoder layers, native 512 px
Resolution must be divisible by patch_size * num_windows (= 32 for these sizes); 640 matches the pipeline.

Comet: on by default (--no-comet), project `vehicle-detection`; the run's metrics.csv is forwarded as it grows.

Memory guide for the nominal batch of 16 (batch x grad_accum): A100/H100 16x1, RTX 4090/3090 8x2, T4 4x4.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comet_utils  # noqa: E402

SIZES = {
    'small': 'RFDETRSmall',
    'medium': 'RFDETRMedium',
    'large': 'RFDETRLarge',
}

# Fine-tuning recipe. Defaults of the package are kept where they are already the documented
# fine-tuning values (lr 1e-4, lr_encoder 1.5e-4, weight decay 1e-4, EMA on, per-batch multi-scale).
# Changes from the defaults and why:
#   epochs 80, no early stopping: with a cosine schedule the gains come in the low-lr tail; the YOLO26 baseline
#       (2026-11-11) plateaued at high lr and patience stopped it before it annealed. --epochs-patience N turns it on.
#   lr_scheduler cosine: smoother than the single 10x step for runs this short.
#   warmup_epochs 1: DETR heads with a fresh class projection are unstable in the first few hundred steps.
#   checkpoint_interval 5, run_test True: evaluate the best checkpoint on the held-out test split.
#   num_workers 8: the default of 2 starves the GPU on JPEG decoding at 640 px.
RECIPE = dict(
    epochs=80, lr=1e-4, lr_encoder=1.5e-4, weight_decay=1e-4, warmup_epochs=1.0,
    lr_scheduler='cosine', lr_scheduler_kwargs={'min_factor': 0.01},
    use_ema=True, ema_decay=0.993, ema_tau=100,
    multi_scale='per-batch', expanded_scales=True, scale_jitter=True,
    early_stopping=False, early_stopping_patience=15, early_stopping_min_delta=0.001, early_stopping_use_ema=True,
    checkpoint_interval=5, run_test=True, log_per_class_metrics=True,
    tensorboard=True, wandb=False, num_workers=8, clip_max_norm=0.1,
)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, help='COCO folder from resplit_dataset.py (train/ valid/ test/)')
    ap.add_argument('--size', choices=sorted(SIZES), default='medium')
    ap.add_argument('--resolution', type=int, default=640, help='square input size, multiple of 32 (default 640)')
    ap.add_argument('--batch', default='auto', help='per-GPU batch; "auto" probes the GPU (CUDA only)')
    ap.add_argument('--grad-accum', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--lr-encoder', type=float, default=None)
    ap.add_argument('--devices', default='1', help='number of GPUs for DDP (PyTorch Lightning), default 1')
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--output', default='runs/rfdetr')
    ap.add_argument('--name', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pretrain-weights', default=None,
                    help='where the COCO weights live or are downloaded to (default: package default name in the cwd)')
    ap.add_argument('--gradient-checkpointing', action='store_true', help='~20%% slower, much less VRAM')
    ap.add_argument('--resume', default=None, help='.ckpt to continue (restores optimizer and scheduler)')
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--epochs-patience', type=int, default=None, dest='patience', help='turn on early stopping with this patience (default off)')
    ap.add_argument('--weight-decay', type=float, default=None)
    ap.add_argument('--warmup-epochs', type=float, default=None)
    ap.add_argument('--multi-scale', choices=('per-batch', 'per-sample', 'off'), default=None, help='default per-batch')
    ap.add_argument('--expanded-scales', action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument('--scale-jitter', action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument('--no-ema', action='store_true')
    ap.add_argument('--comet', action=argparse.BooleanOptionalAction, default=True, help='log to Comet (default on)')
    ap.add_argument('--comet-project', default=comet_utils.DEFAULT_PROJECT)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)

    if a.resolution % 32:
        ap.error('--resolution must be a multiple of 32')
    classes = [c['name'] for c in sorted(
        json.load(open(Path(a.dataset) / 'train' / '_annotations.coco.json'))['categories'], key=lambda c: c['id'])]
    name = a.name or f"rfdetr_{a.size}_{a.resolution}_{datetime.now():%y%m%d_%H%M}"
    train_args = dict(RECIPE)
    train_args.update(dict(
        dataset_dir=a.dataset, dataset_file='roboflow', class_names=classes,
        output_dir=str(Path(a.output) / name), run=name, project='traffic-rfdetr' if a.wandb else None,
        batch_size='auto' if a.batch == 'auto' else int(a.batch), grad_accum_steps=a.grad_accum,
        devices=int(a.devices) if a.devices.isdigit() else a.devices, seed=a.seed, resume=a.resume,
        wandb=a.wandb,
    ))
    for k, v in (('epochs', a.epochs), ('lr', a.lr), ('lr_encoder', a.lr_encoder), ('num_workers', a.workers),
                 ('early_stopping_patience', a.patience), ('weight_decay', a.weight_decay), ('warmup_epochs', a.warmup_epochs),
                 ('multi_scale', a.multi_scale), ('expanded_scales', a.expanded_scales), ('scale_jitter', a.scale_jitter)):
        if v is not None:
            train_args[k] = v
    if a.patience is not None:
        train_args['early_stopping'] = True
    if a.no_ema:
        train_args['use_ema'] = False
    if a.batch == 'auto' and a.grad_accum != 1:
        print('note: batch_size="auto" chooses its own accumulation; --grad-accum is ignored')
    model_kwargs = dict(resolution=a.resolution, gradient_checkpointing=a.gradient_checkpointing)
    if a.pretrain_weights:
        model_kwargs['pretrain_weights'] = a.pretrain_weights

    print(f'model: {SIZES[a.size]}({model_kwargs})')
    print(json.dumps(train_args, indent=2, default=str))
    if a.dry_run:
        return

    exp = comet_utils.start(a.comet_project, name, tags=['rfdetr', f'rfdetr-{a.size}', f'img{a.resolution}', Path(a.dataset).parent.name],
                            params={**train_args, **model_kwargs, 'model': SIZES[a.size]}, enabled=a.comet)
    tail = comet_utils.MetricTail(exp, Path(train_args['output_dir']) / 'metrics.csv', 'csv')
    tail.start()

    import rfdetr
    model_cls = getattr(rfdetr, SIZES[a.size])
    model = model_cls(**model_kwargs)
    try:
        model.train(**train_args)
    finally:
        tail.stop()

    # The best checkpoint lives in output_dir as checkpoint_best_ema.pth (or checkpoint_best_regular.pth).
    out = Path(train_args['output_dir'])
    best = next((p for p in (out / 'checkpoint_best_ema.pth', out / 'checkpoint_best_regular.pth') if p.exists()), None)
    print(f'best checkpoint: {best}')
    if best:
        # ONNX for TensorRT: RF-DETR exports a fixed-resolution graph (boxes + logits, no NMS needed).
        model_cls(pretrain_weights=str(best), resolution=a.resolution).export(output_dir=str(out / 'export'))
    comet_utils.finish(exp, [out / 'metrics.csv', out / 'training_config.json'])


if __name__ == '__main__':
    main()
