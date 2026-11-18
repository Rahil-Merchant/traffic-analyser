"""Train YOLO26 (L by default) for the production pipeline: 9 production classes, 640 px letterbox input,
one-to-many head with external NMS, with every hyperparameter and augmentation exposed on the command line.

    pip install -r training/requirements-train.txt
    python training/train_yolo26.py --data /data/dataset_261110_split/data_rfs.yaml                 # baseline
    python training/train_yolo26.py --data ... --preset p2                                             # P2-P5 head
    python training/train_yolo26.py --data ... --preset hires --device 0,1                              # 960 px
    python training/train_yolo26.py --data ... --mixup 0.2 --scale 0.7 --alb light --epochs 150         # any override
    python training/train_yolo26.py --data ... --weights runs/yolo26/x/weights/best.pt --eval-only      # test + export
    python training/train_yolo26.py --data ... --dry-run                                                # show resolved args

Head choice (--head, --dual-head):
    The shipped yolo26*.pt carry two heads. The one-to-one head is detached from the backbone and the
    ProgLoss schedule moves the loss weight from the one-to-many head (0.8) to the one-to-one head (0.9)
    over training, so by default most of the gradient that reaches the backbone late in training is for
    the head we do not deploy. Default here: build the model from training/cfg/yolo26-o2m*.yaml
    (end2end: False), load the pretrained weights into it, and train the one-to-many head alone with the
    plain detection loss. --dual-head keeps the stock two-head model instead (val/export still default to
    the one-to-many head; nms=False would pick the NMS-free head).
    --head p2p5 adds the stride-4 P2 level (yolo26-p2): 34% of our boxes are under 16 px at 640, which
    stride 8 resolves poorly; costs about +25% FLOPs and 4x the NMS candidates.

Augmentation: geometric and colour jitter through Ultralytics flags (preset below), plus a targeted
Albumentations stage (--alb cctv|light|off, --alb-strength): motion blur, sensor/ISO noise, downscale,
JPEG/stream compression, gamma and contrast, grayscale (night infrared), CLAHE, fog, drizzle, tree shadows
and headlight-like flares. Pixel-level only, so boxes are untouched; serialised into the checkpoint and
DDP-safe (ultralytics >= 8.4 `augmentations` argument).

Comet: on by default through the Ultralytics integration, project `vehicle-detection` (--comet-project),
experiment named after the run and tagged with model, preset, head, size and augmentation; --no-comet
disables it. Credentials come from COMET_API_KEY or ~/.comet.config (see comet_utils.py).
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from comet_utils import DEFAULT_PROJECT, name_ultralytics_experiment  # noqa: E402

PRODUCTION = yaml.safe_load(open(HERE / 'classes_production.yaml'))['classes']

# Every entry is a CLI flag (--imgsz, --hsv-v, --dual-head/--no-dual-head ...); None means "ultralytics default".
BASE = dict(
    # model
    size='l', head='p3p5', dual_head=False, pretrained='auto', freeze=0,
    # schedule
    imgsz=640, epochs=80, patience=0, batch=32.0, nbs=64, optimizer='auto', lr0=None, lrf=None, momentum=None,
    weight_decay=None, warmup_epochs=3.0, cos_lr=True, close_mosaic=15,
    # loss
    box=7.5, cls=0.75, dfl=1.5,
    # geometric / colour augmentation (ultralytics). Fixed upright CCTV: no rotation, shear, perspective or
    # vertical flip. Mirrored traffic is still traffic. Scale 0.5 = objects between 0.5x and 1.5x; the data is
    # already 62% small objects, so no harder down-scaling. HSV value/saturation above the COCO recipe for
    # dawn, dusk, overcast and headlights. Mosaic until the last 15 epochs; a little mixup as a regulariser.
    hsv_h=0.015, hsv_s=0.6, hsv_v=0.5, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.10, cutmix=0.0, copy_paste=0.0, multi_scale=0.0,
    # targeted pixel-level augmentation
    alb='cctv', alb_strength=1.0,
    # runtime
    device='0', workers=8, cache='', seed=0, project='runs/yolo26', name='', resume=False,
    eval_conf=0.25, export=True, allow_any_classes=False, comet=True, comet_project=DEFAULT_PROJECT,
)
PRESETS = {
    'default': {},
    # exact MuSGD values of the yolo26l.pt COCO run (Ultralytics YOLO26 recipe, batch 128; lr0 scaled to --batch)
    'musgd': dict(optimizer='MuSGD', lr0=0.00038, lrf=0.882, momentum=0.948, weight_decay=0.00027, warmup_epochs=1.0,
                  cos_lr=False, box=9.83, cls=0.65, dfl=0.96),
    'p2': dict(head='p2p5', batch=16.0),
    'hires': dict(imgsz=960, batch=12.0),
    'dual': dict(dual_head=True),
    'noalb': dict(alb='off'),
}
RECIPE_BATCH = 128
UL_KEYS = {'imgsz', 'epochs', 'patience', 'batch', 'nbs', 'optimizer', 'lr0', 'lrf', 'momentum', 'weight_decay',
           'warmup_epochs', 'cos_lr', 'close_mosaic', 'box', 'cls', 'dfl', 'hsv_h', 'hsv_s', 'hsv_v', 'degrees',
           'translate', 'scale', 'shear', 'perspective', 'flipud', 'fliplr', 'bgr', 'mosaic', 'mixup', 'cutmix',
           'copy_paste', 'multi_scale', 'device', 'workers', 'seed', 'project', 'freeze'}


def build_albumentations(level, strength):
    """Targeted pixel-level augmentations for roadside CCTV (shared with DEIMv2, see cctv_aug.py)."""
    import cctv_aug
    return cctv_aug.build(level, strength)


def resolve(a):
    cfg = dict(BASE)
    cfg.update(PRESETS[a.preset])
    cfg.update({k: v for k, v in vars(a).items() if k in BASE and v is not None})
    if cfg['optimizer'] == 'MuSGD' and a.lr0 is None and cfg['lr0'] and cfg['batch'] >= 1:
        cfg['lr0'] = round(cfg['lr0'] * cfg['batch'] / RECIPE_BATCH, 7)
    return cfg


def model_spec(cfg):
    """-> (model to build, weights to load). One-to-many-only models come from training/cfg yamls."""
    size, p2 = cfg['size'], cfg['head'] == 'p2p5'
    weights = f'yolo26{size}.pt' if cfg['pretrained'] == 'auto' else cfg['pretrained']
    if cfg['dual_head']:
        return (weights if not p2 else f'yolo26{size}-p2.yaml'), (None if not p2 else weights)
    return str(HERE / 'cfg' / f'yolo26{size}-o2m{"-p2" if p2 else ""}.yaml'), weights


def check_classes(data_yaml, allow):
    names = yaml.safe_load(open(data_yaml))['names']
    names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    if names != PRODUCTION:
        msg = f'{data_yaml} classes {names} differ from the production set {PRODUCTION}'
        if not allow:
            sys.exit(msg + '\nRe-split with --class-config training/classes_production.yaml or pass --allow-any-classes')
        print('warning: ' + msg)
    return names


def train_args(cfg, a, run_name):
    args = {k: cfg[k] for k in UL_KEYS if cfg[k] is not None}
    args['batch'] = int(cfg['batch']) if cfg['batch'] >= 1 else cfg['batch']    # -1 or 0.7 = auto-batch
    if cfg['cache']:
        args['cache'] = cfg['cache']
    args.update(data=a.data, name=run_name, exist_ok=False, pretrained=True, amp=True, plots=True, save=True,
                val=True, rect=False, deterministic=False, resume=cfg['resume'])
    if not cfg['freeze']:
        args.pop('freeze', None)
    return args


def evaluate_and_export(model, a, cfg, run_dir):
    """Test-split numbers at the mAP conf and at the production operating point, then ONNX for onnx2trt."""
    common = dict(data=a.data, split='test', imgsz=cfg['imgsz'], batch=16, device=cfg['device'], iou=0.6,
                  rect=False, plots=True, project=str(run_dir), exist_ok=True)
    model.val(conf=0.001, name='test_map', **common)
    model.val(conf=cfg['eval_conf'], name=f'test_conf{cfg["eval_conf"]}', **common)
    if cfg['export']:
        # one-to-many head, no embedded NMS: (N, 4 + nc, anchors), the layout utility_scripts/onnx2trt expects
        path = model.export(format='onnx', imgsz=cfg['imgsz'], quantize=16, dynamic=True, simplify=True, device=cfg['device'])
        print(f'exported {path}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='data.yaml (or data_rfs.yaml) from resplit_dataset.py')
    ap.add_argument('--preset', choices=sorted(PRESETS), default='default', help='bundle of defaults; flags override')
    ap.add_argument('--weights', default=None, help='existing .pt to evaluate/export instead of training')
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    g = ap.add_argument_group('model')
    g.add_argument('--size', choices=list('nsmlx'))
    g.add_argument('--head', choices=('p3p5', 'p2p5'), help='p2p5 adds the stride-4 level for small objects')
    g.add_argument('--dual-head', action=argparse.BooleanOptionalAction, default=None,
                   help='keep the stock two-head model (default: one-to-many head only)')
    g.add_argument('--pretrained', help="weights to initialise from ('auto' = yolo26<size>.pt, or a path)")
    g.add_argument('--freeze', type=int, help='freeze the first N layers (10 = backbone)')
    for group, keys in (('schedule', ('imgsz', 'epochs', 'patience', 'batch', 'nbs', 'optimizer', 'lr0', 'lrf', 'momentum',
                                      'weight_decay', 'warmup_epochs', 'cos_lr', 'close_mosaic')),
                        ('loss', ('box', 'cls', 'dfl')),
                        ('augmentation', ('hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale', 'shear', 'perspective',
                                          'flipud', 'fliplr', 'bgr', 'mosaic', 'mixup', 'cutmix', 'copy_paste', 'multi_scale',
                                          'alb', 'alb_strength')),
                        ('runtime', ('device', 'workers', 'cache', 'seed', 'project', 'name', 'resume', 'eval_conf',
                                     'export', 'allow_any_classes', 'comet', 'comet_project'))):
        g = ap.add_argument_group(group)
        for k in keys:
            flag = '--' + k.replace('_', '-')
            default = BASE[k]
            if isinstance(default, bool):
                g.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
            elif k == 'alb':
                g.add_argument(flag, choices=('cctv', 'light', 'off'), default=None)
            else:
                t = type(default) if default is not None else float
                g.add_argument(flag, type=t if t is not type(None) else float, default=None,
                               help=f'default {default if default is not None else "ultralytics"}')
    a = ap.parse_args(argv)

    cfg = resolve(a)
    names = check_classes(a.data, cfg['allow_any_classes'])
    model_src, weights = model_spec(cfg)
    run_name = cfg['name'] or f"yolo26{cfg['size']}_{a.preset}_{cfg['head']}{'_dual' if cfg['dual_head'] else ''}_{cfg['imgsz']}_{datetime.now():%y%m%d_%H%M}"
    args = train_args(cfg, a, run_name)
    if weights:
        args['pretrained'] = weights
    try:
        alb = build_albumentations(cfg['alb'], cfg['alb_strength'])
    except ImportError as e:
        if not a.dry_run:
            sys.exit(f'{e}: pip install "albumentations>=2.0" opencv-python, or pass --alb off')
        alb, missing = None, e
    print(f'model: {model_src}  weights: {weights}  classes: {len(names)}')
    print(json.dumps(args, indent=2, default=str))
    print(f"albumentations ({cfg['alb']}, strength {cfg['alb_strength']}): "
          + (', '.join(t.__class__.__name__ for t in alb) if alb else ('none' if cfg['alb'] == 'off' else f'not importable here ({missing})')))
    print(f"comet: {'project ' + cfg['comet_project'] if cfg['comet'] else 'off'}")
    if a.dry_run:
        return

    if cfg['comet']:
        os.environ.setdefault('COMET_PROJECT_NAME', cfg['comet_project'])
        os.environ.setdefault('COMET_EVAL_LOG_CONFUSION_MATRIX', 'true')
    else:
        sys.modules['comet_ml'] = None          # Ultralytics enables its Comet callbacks only if this imports
    from ultralytics import YOLO, __version__ as ul_version
    if tuple(int(x) for x in ul_version.split('.')[:2]) < (8, 4):
        sys.exit(f'ultralytics {ul_version} predates YOLO26; install ultralytics>=8.4')

    if a.eval_only:
        if not a.weights:
            sys.exit('--eval-only needs --weights')
        evaluate_and_export(YOLO(a.weights), a, cfg, Path(a.weights).resolve().parent.parent)
        return

    model = YOLO(model_src)
    if cfg['comet']:
        tags = ['yolo26', f"yolo26{cfg['size']}", a.preset, cfg['head'], 'dual-head' if cfg['dual_head'] else 'o2m',
                f"img{cfg['imgsz']}", f"alb-{cfg['alb']}", Path(a.data).parent.name]
        model.add_callback('on_pretrain_routine_end', name_ultralytics_experiment(run_name, tags))
    if alb:
        args['augmentations'] = alb
    results = model.train(**args)
    run_dir = Path(results.save_dir) if results is not None else Path(cfg['project']) / run_name
    best = run_dir / 'weights' / 'best.pt'
    print(f'best weights: {best}')
    evaluate_and_export(YOLO(str(best)), a, cfg, run_dir)


if __name__ == '__main__':
    main()
