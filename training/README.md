# Training: dataset 261110, YOLO26-L / RF-DETR / DEIMv2

Tools and recipes for the Roboflow export `full-res-labelled v5 (no aug, 2026-11-10)`. Everything in `data/`
is git-ignored; the analysis and the chosen split are committed under [reports/dataset_261110/](reports/dataset_261110/)
so the split can be rebuilt bit-for-bit on the training server.

| step | script | output |
|---|---|---|
| 1. analyse the export | `analyse_dataset.py` | `analysis.md` / `analysis.json` |
| 2. recover recordings and camera scenes | `recording_groups.py` | `groups.csv`, `scenes.jpg` |
| 3. leak-free stratified re-split, production class set | `resplit_dataset.py --class-config classes_production.yaml` | YOLO dataset + COCO twin, `split_report.md`, `split_manifest.csv` |
| 4. rare-class oversampling (YOLO) | `oversample_rare.py` | `train_rfs.txt`, `data_rfs.yaml` |
| 5. train | `train_yolo26.py`, `train_rfdetr.py`, `train_deimv2.py` | runs/, ONNX exports |

## 1. What is in the dataset

37,676 images, 182,529 boxes, 11 classes, three camera resolutions (2560×1440: 45%, 1600×900: 29%,
1920×1080: 27%). 1,345 images (3.6%) have no box. Median 2 boxes per image, p95 18, max 59.
Full tables: [reports/dataset_261110/analysis.md](reports/dataset_261110/analysis.md).

**Class balance is extreme.** Light Vehicle is 59% of boxes, Minibus Taxi 15%, Pedestrian 13%, Long Truck
6%, Short Truck 4%; then Medium Truck 1.8%, Bus 1.0%, Motorcycle 0.26%, Cyclist 0.13%, Person with wheel
barrow 57 boxes, Animal drawn vehicle 21 boxes (ratio largest/smallest 5,145). The report classes the
pipeline is judged on (trucks, buses, taxis) are the minority; this drives the loss weights and the
oversampling below. Animal drawn vehicle (21 boxes in 20 images, 4 cameras) cannot be learned or evaluated
from this data; it stays in the class list so the deployed model keeps its 11 outputs, and its mAP is noise.

**Objects are small.** After letterboxing to 640 px the median box is 23.7 px on a side; 62% of boxes are
COCO-small (< 32 px) and 34% are under 16 px. Pedestrians are 14×28 px at the median (82% small), light
vehicles 26×17 px (66% small), minibus taxis 30×22 px. Only the trucks and buses are medium-sized.
At 640 px a 14 px pedestrian is below two cells of the stride-8 feature map, which is the practical floor
for all three detectors. This is the single biggest lever: training and running at 960 px (or the
existing SAHI/patchwise mode) will do more for pedestrians, cyclists and motorcycles than any optimiser setting.

**Labels are clean.** No out-of-range, zero-size or malformed boxes; no Roboflow augmentation copies.

## 2. Why the export's split had to be replaced

Three naming families make up the export: 6,301 frame-selector frames (`<video>_<hash>_<frame>`, 68 videos),
20,878 sequentially numbered frames from the original dataset, and 10,497 VLC snapshots (10 days).
Consecutive numbers and same-day snapshots are frames of one camera; the Roboflow split scattered them
67/33 at random, so **121 of the 147 camera scenes appear in both train and valid** (and 25 of the 68 named
videos). Validation mAP on that split measures how well the model memorises backgrounds it has seen,
and it will prefer the model that overfits hardest.

`recording_groups.py` recovers the structure without metadata: a 32×18 grey thumbnail per image, the
numbered and VLC sequences cut at resolution changes, neighbour-correlation drops and 30-minute gaps, the
fragments clustered into **147 camera scenes** by average-linkage on thumbnail similarity, tiny night-IR
fragments absorbed into their nearest scene. Checked against the frame-selector videos (the only ground
truth): none of the 68 videos is split across scenes. See `scenes.jpg` in the analysis folder for one
frame per scene.

## 3. The new split (production class set)

The deployed engine has **nine classes in this order**: Bus, Cyclist, Light Vehicle, Long Truck, Medium Truck,
Minibus Taxi, Motorcycle, Pedestrian, Short Truck (read from a production detection dump; see
[classes_production.yaml](classes_production.yaml)). The split uses exactly that list: *Person with wheel
barrow* (57 boxes) is relabelled Pedestrian, and the 20 images holding an *Animal drawn vehicle* (21 boxes)
are left out, so no unlabelled object teaches the model that it is background. Production input is a 640 px
letterbox of the full frame; training uses the same geometry (`imgsz 640`, letterbox, no stretch).

`resplit_dataset.py --group-by scene` assigns whole scenes to train/valid/test (greedy, largest first, 400
random restarts, objective = per-class instance fractions with rare classes weighted like common ones, plus
the image count at weight 6, plus a penalty for a class missing from valid/test or a split with fewer than
10 cameras). The six scenes with more than 4% of all images (5,671 … 1,739 images) are pinned to train so
that valid and test are not dominated by one camera.

| split | cameras | images | boxes | Bus | Cyclist | Light Veh. | Long Trk | Med. Trk | Taxi | Motorcycle | Pedestrian | Short Trk |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train | 118 | 29,281 | 146,189 | 1,567 | 182 | 83,233 | 8,921 | 2,768 | 22,886 | 383 | 20,083 | 6,166 |
| valid | 18 | 3,821 | 19,984 | 184 | 23 | 13,670 | 993 | 350 | 2,038 | 47 | 1,865 | 814 |
| test | 11 | 4,554 | 16,021 | 81 | 26 | 10,985 | 511 | 248 | 1,696 | 39 | 1,840 | 595 |

Valid and test cameras are never seen in training, so their numbers estimate performance at a new survey
site, which is what deployment looks like. Consequences to keep in mind:

- Per-class mAP for Cyclist, Motorcycle and Bus rests on 23–184 boxes per split: compare models on
  mAP50-95 overall and on the five report classes (Light Vehicle, Taxi, Long/Short/Medium Truck), and read
  the rare classes as recall at the production threshold, not as mAP.
- `--image-weight` trades image balance against box balance, `--ratios` moves the targets, and
  `--group-by recording` is the weaker alternative (one video per split, same camera allowed in several
  splits); use it only if the in-domain number is what you need.
- Night infrared scenes are the least reliable part of the clustering; a few night cameras may be split
  across train and valid.

Rebuild on the server (no thumbnails needed, `groups.csv.gz` carries the scene of every image):

```bash
unzip roboflow.zip -d /data/dataset_261110
python training/resplit_dataset.py /data/dataset_261110 --out /data/dataset_261110_split \
    --groups training/reports/dataset_261110/groups.csv.gz --group-by scene \
    --class-config training/classes_production.yaml --seed 0 --restarts 400
python training/oversample_rare.py /data/dataset_261110_split
```

Outputs: `train|valid|test/{images,labels}` + `data.yaml` (Ultralytics), `coco/{train,valid,test}/_annotations.coco.json`
(RF-DETR, DEIMv2; category ids 0..8 in production order), `data_rfs.yaml` (train list with rare-class
repeats: Cyclist ×4.3, Motorcycle ×2.8, Bus ×1.4; 1.07× epoch length). Images are hard-linked, so the split
costs no disk space on the same volume.

## 4. Models and recipes

All three are the ~25–35M parameter class, all read the nine production classes from the split, and all are
trained at 640 px to match `MODEL_INPUT_SIZE`; the `hires` variants exist because of the object sizes above.

| model | backbone | params | FLOPs @640 | COCO mAP | training entry |
|---|---|---|---|---|---|
| YOLO26-L (one-to-many head + NMS) | CSP; P3–P5 or P2–P5 | 24.8M / 25.8M | 86.8G / ~107G | 55.0 | `train_yolo26.py [--head p2p5]` |
| RF-DETR Medium (1.12, default) / Large-2026 | DINOv2-S, 4 dec. layers | ~33M | 576 / 704 px native | 54.7 / 56+ | `train_rfdetr.py --size medium|large` |
| DEIMv2-L (default) / M | DINOv3-S / distilled ViT-Tiny+ | 32.2M / 18.1M | 144.5G / 84.1G | 56.0 / 53.0 | `train_deimv2.py --size l|m` |

### YOLO26-L (`train_yolo26.py`)
Every setting is a flag (`--help` lists them grouped: model, schedule, loss, augmentation, runtime); a
`--preset` only changes defaults. `--dry-run` prints the resolved arguments. The script refuses a data.yaml
whose class list is not the production list unless `--allow-any-classes` is given.

**Which head to train.** The shipped `yolo26l.pt` has two heads. The one-to-one (NMS-free) head is
*detached* from the backbone, and the ProgLoss schedule moves the loss weight from the one-to-many head
(0.8) to the one-to-one head (0.9) over training. We deploy the one-to-many head with the pipeline's own
NMS, so by default the stock model would spend most of its late-training gradient on a head we never run.
Default here: the model is built from [cfg/yolo26-o2m.yaml](cfg/yolo26-o2m.yaml) (`end2end: False`), the
COCO weights are loaded into it, and the one-to-many head trains alone with the plain detection loss.
`--dual-head` keeps the stock behaviour for comparison. Val, predict and export use the one-to-many head
either way (`nms=False` would select the NMS-free head).

**P2 / P6 heads.** Box size at 640 px: 13% under 8 px, 21% at 8–16, 28% at 16–32, 22% at 32–64, 13% at
64–128, 3% over 128, 0.1% over 256. P6 (stride 64) has nothing to do here. Dropping P5 for a P2–P4 model
would hurt the trucks (26% of Long Trucks are over 128 px). The P2 level (stride 4) targets the 34% of boxes
under 16 px, mostly pedestrians, light vehicles and taxis far from the camera: `--head p2p5` builds
[cfg/yolo26-o2m-p2.yaml](cfg/yolo26-o2m-p2.yaml) (25.8M params, 115 vs 94 GFLOPs unfused, 34k instead of
8.4k NMS candidates, batch 16 on 24 GB). It cannot recover detail that the 4× downscale from 2560 px
already removed, so run it against `--preset hires` (960 px, 2.25× cost): if hires wins clearly, resolution
is the bottleneck, not the head.

> **Why no early stopping** (2026-11-11 baseline): with a cosine schedule the validation mAP plateaus while the
> learning rate is still high; most of the remaining gain usually comes in the low-lr tail and the last 15 no-mosaic epochs. The first
> YOLO26-L run (120 epochs, patience 30) peaked at epoch 43 and was stopped at epoch 73 with the lr still at a
> third of its peak, so it never annealed. Run the full schedule and shorten `--epochs` instead.

**Preset `default`** (start here): optimizer auto, 80 epochs, no early stopping (patience 0), batch 32, imgsz 640, cosine LR,
warmup 3, `box 7.5 / cls 0.75 / dfl 1.5` (cls raised from 0.5 because the minority classes carry the
report), trained on `data_rfs.yaml`.
**`musgd`**: the values the `yolo26l.pt` COCO run used (`lr0 0.00038 @ batch 128`, scaled linearly to the batch
used, `lrf 0.882`, `momentum 0.948`, `wd 0.00027`, `box 9.83 / cls 0.65 / dfl 0.96`). **`p2`**, **`hires`**,
**`dual`**, **`noalb`** as above.

**Augmentation.** Ultralytics stage, tuned for fixed upright CCTV: `degrees 0`, `shear 0`, `perspective 0`,
`flipud 0`, `fliplr 0.5` (mirrored traffic is still traffic), `scale 0.5`, `translate 0.1` (objects are
already 62% small, so no harder down-scaling), `hsv_s 0.6 / hsv_v 0.5` (dawn, dusk, overcast, headlights),
`mosaic 1.0` until `close_mosaic 15`, `mixup 0.10`, `cutmix 0`, `copy_paste 0` (needs masks).
Targeted Albumentations stage (`--alb cctv`, pixel-level only so boxes stay exact, DDP-safe through the
`augmentations` argument), probabilities per image: motion blur 0.12 and defocus 0.04 (20 fps of fast
vehicles, night shutter); Gaussian 0.10 and ISO noise 0.08 (night gain); downscale 0.15 (sub-stream
1600×900 cameras) and JPEG 40–90 at 0.15 (H.264/.dav artefacts); gamma 0.15, brightness/contrast 0.10,
CLAHE 0.05, grayscale 0.08 (infrared night mode); fog 0.05, drizzle 0.04, tree shadows 0.08, headlight-like
flare 0.03. `--alb light` halves every probability, `--alb-strength` scales them, `--alb off` removes the
stage. Class balance is handled by repeat-factor sampling (`oversample_rare.py`), not by augmentation.

Expect roughly 4–6 min/epoch on a 24 GB card at 640 (≈ 8–12 h), about 1.3× for P2 and 2.3× at 960.

### RF-DETR (Medium at 640)
Package defaults are already the fine-tuning recipe (`lr 1e-4`, `lr_encoder 1.5e-4`, `wd 1e-4`, EMA,
per-batch multi-scale, group-DETR 13). Changed: 80 epochs without early stopping (`--epochs-patience N` turns it on),
cosine LR to 1%, 1 warmup epoch, `checkpoint_interval 5`, `run_test` on the held-out split, 8 workers,
resolution 640 (must be a multiple of 32; native is 576). Nominal batch 16: `batch_size=auto` on CUDA,
or `--batch 8 --grad-accum 2` on 24 GB. `--size large` is the same backbone at 704 px with more
attention heads; it is the "hires" variant here. `--gradient-checkpointing` if memory is short.
Expect roughly 10–14 min/epoch at 640 on a 24 GB card (≈ 12–18 h with early stopping).

### DEIMv2-L
Config-driven repo (clone it, Python 3.11, torch 2.5.1). The script writes
`configs/deimv2/deimv2_dinov3_l_traffic.yml` that includes the stock L config and overrides dataset,
classes, schedule and optimiser, then launches `torchrun … -t <COCO checkpoint>` (tuning keeps every
weight whose shape matches, so the 11-class head is re-initialised while backbone, encoder and decoder
start from COCO). Schedule for 60 epochs follows the repo's own rules: `flat_epoch 34`, no-aug tail 7,
dense-O2O augmentations (mosaic/mixup/copy-blend) from epoch 4 to 53, matcher switch at epoch 45, warmup
2,000 iterations. LR scaled linearly from the reference batch 32 (`lr 2.5e-4`, backbone `6.25e-6` at batch
16), EMA decay `1-(1-0.9999)·batch/32`. Needs Meta's `dinov3_vits16` weights (licence form) in `ckpts/`;
`--size m` uses the freely downloadable distilled ViT-Tiny+ instead and is 40% cheaper.
Expect roughly 12–16 min/epoch at 640 on two 24 GB cards (measured: ~20 min/epoch on one RTX 3090, batch 16).

**`--aug cctv`** (2026-11-15). The stock recipe is COCO's. Its mosaic pastes four frames into a 2×2 canvas and
squashes the canvas to 640, so every object trains at half size, and RandomZoomOut shrinks up to 4× more.
Measured over 400 training samples (median box side at the model input / share under 8 px): real frames
34 px / 5%; stock mosaic phase 18 px / 21%; `--aug cctv` 29 px / 6%. With 30% of test vehicles under 16 px,
and recall there at 12–60%, the stock recipe trains on objects that do not exist at test time. `--aug cctv`:
mosaic tiles at 640 with affine scale 1.5–2.5 (a centre crop, as Ultralytics does), no rotation; zoom-out
1–1.5× at p 0.3; RandomPhotometricDistort (which permutes colour channels) replaced by the shared CCTV pixel
set from `cctv_aug.py` plus a mild HSV jitter. The op is registered by `deim_cctv_aug.py`. `deim_launch.py`
runs the repo's `train.py`, so the clone stays unmodified. `--rfs-list train_rfs.txt` gives DEIMv2 the same
rare-class repeats as YOLO (Bus is called Taxi 22% of the time). The config is now written per run
(`deimv2_dinov3_l_traffic_<output name>.yml`) and copied into the run folder. Pass a new `--output` for every
run, because the launcher's default is `runs\deimv2`.

## 5. Suggested order of runs

1. `train_yolo26.py --data …/data_rfs.yaml` → baseline (one-to-many head only, cctv augmentation), ONNX,
   TensorRT; compare on the test split against the deployed `premier_station` engine with
   `utility_scripts/replay_detections.py`. The script also reports the test split at `--eval-conf 0.25`,
   the tracker's `det_thresh`, which is the operating point that matters for counts.
1b. `--preset dual` and `--preset noalb` are the two ablations worth one run each: they tell you whether the
   head choice and the targeted augmentation paid off on unseen cameras.
2. `train_rfdetr.py --size medium` and `train_deimv2.py --size l` with the same split → the architecture
   decision. Judge on test mAP50-95, per-class recall at the OC-SORT `det_thresh` (0.25), and T4/3090
   TensorRT latency at batch 64, since the pipeline is throughput-bound.
3. `--preset p2` and `--preset hires` (or RF-DETR `--size large`) if pedestrians/cyclists/motorcycles are the
   weak classes in step 1, which the object-size numbers predict. P2 is the cheaper of the two at inference;
   hires needs `MODEL_INPUT_SIZE = 960` and halves pipeline throughput.
4. Only then tune: `--cls` 0.5–1.0, `--mixup` 0–0.2, `--close-mosaic` 10–20, `--scale` 0.5–0.7,
   `--alb-strength` 0.5–1.5, `--epochs 150 --patience 40`. Changes below ±0.5 mAP on this test set are noise
   given the 11–18 cameras per split.

### Results so far (test split, one scorer for all, 2026-11-14)

`predict_coco.py` writes each model's test detections in COCO format (each model with its own preprocessing),
`score_coco.py` scores them all with pycocotools plus precision/recall per class, at 0.25 and at each model's
own best-F1 threshold. Full tables: [reports/dataset_261110/model_comparison/](reports/dataset_261110/model_comparison/).

| model (run) | mAP50-95 | mAP50 | best-F1 thr | P | R | F1 | Pedestrian AP / R |
|---|---|---|---|---|---|---|---|
| YOLO26-L o2m, letterbox (`yolo26l-o2m-base`, best ep 43, un-annealed) | 0.457 | 0.678 | 0.34 | 0.82 | 0.68 | 0.743 | 0.24 / 0.32 |
| RF-DETR-M (`rfdetr-m-base`, best ep 17 of 60) | 0.479 | 0.733 | 0.37 | 0.79 | 0.71 | 0.748 | 0.24 / 0.38 |
| DEIMv2-L (`best_stg1`, ep 32 of 60) | **0.499** | **0.754** | 0.41 | 0.80 | 0.73 | **0.763** | 0.26 / 0.42 |

Ultralytics' own scorer reports 0.479 for the same YOLO model, RF-DETR's 0.483, so the trainers' numbers are
not comparable with each other; use `score_coco.py`. RF-DETR overfits after ~epoch 17 (valid 0.441 → 0.418 at
60), so ~25 epochs is enough. DEIMv2's no-aug stage did not beat its stage-1 best. Pedestrians stay the weakest
class for all three. Note the input geometry: YOLO letterboxes a 16:9 frame into 640x360 of real pixels, the
DETRs resize it to the full 640x640. `make_square_dataset.py` builds a stretched YOLO split to test whether that
explains part of the gap (same engine cost; deploying it means resize instead of letterbox in the pipeline).

### Round 2 (test split, 2026-11-18)

Full tables: `test_comparison_round2.md` and `errors_*.md` in the same folder. "Vehicle AP" is the mean
AP50-95 over every class except Pedestrian.

| model | mAP50-95 | AP small | best-F1 (thr) | vehicle AP | notes |
|---|---|---|---|---|---|
| YOLO26-L letterbox, base | 0.457 | 0.051 | 0.743 (0.34) | 0.484 | |
| YOLO26-L letterbox, 80 ep full schedule | 0.455 | 0.056 | 0.741 (0.33) | 0.482 | valid +0.6, test 0: annealing did not transfer |
| **YOLO26-L stretched 640×640** (`yolo26l-o2m-sq640`) | **0.474** | 0.069 | 0.750 (0.30) | **0.502** | every vehicle class up; needs resize instead of letterbox |
| DEIMv2-L stock | 0.499 | 0.076 | 0.763 (0.41) | 0.529 | |
| **DEIMv2-L `--aug cctv --rfs-list`** (`best_stg2`) | **0.503** | **0.095** | **0.765** (0.42) | **0.534** | <8 px recall 0.155 → 0.214, Bus→Taxi 22% → 15% |

The DEIMv2 gain is small overall but lands where the change was aimed: tiny vehicles, Bus, and Medium and
Short Truck. The cost is training time: the 640 px mosaic tiles make epochs ~37 min instead of ~20 on swest.

## 6. Other things that help, in order of expected value

- **Operating-point evaluation.** mAP at conf 0.001 is not what the counter sees. Judge on recall and
  precision at `det_thresh 0.25` (the script's second test run) and on counts via the replay tool.
- **Evaluate the TensorRT FP16 engine, not the .pt.** Export, build the engine, run `--eval-only` on the
  ONNX/engine: FP16 and the batched decode occasionally shift small-object scores.
- **Train the way we decode.** Production decodes on the GPU straight to 640x640, which stretches the
  frame (`CROP_MODE = 'letterbox'` then has nothing left to pad; a detection dump's `letterbox_params`
  are `[0, 0, 640, 640]`). So models trained on stretched frames (`make_square_dataset.py`, and both DETRs)
  see in production what they saw in training; a letterbox-trained YOLO does not.
- **More frames of the rare classes, not more augmentation.** Cyclist (231 boxes) and Motorcycle (469)
  will not become reliable from repeats alone; the frame selector's class-instability flag is the cheapest
  way to mine them from existing footage.
- **Masks would unlock copy-paste.** Ultralytics `copy_paste` needs segmentation labels; a SAM pass over
  the rare-class boxes would allow pasting buses and cyclists into other scenes. Worth it only if step 1
  shows those classes limiting the result.
- **Hard negatives.** The 1,345 empty frames stay in training; add more from cameras with false positives
  (the inflated-counts video is one).
- **Checkpoint choice.** Ultralytics picks `best.pt` by a 0.1·mAP50 + 0.9·mAP50-95 fitness on valid
  (unseen cameras), which is the right selector here; do not select on train loss.

## 7. Training on the Windows boxes (emaan-box, swest-box)

Everything lives in one folder outside the production repo (the repo on each box is not touched):
`<root>\code\training` (this folder), `envs\{yolo26,rfdetr,deimv2}` (private Python 3.11 via uv),
`DEIMv2\` (pinned commit), `weights\`, `data\dataset_261110[_split]`, `runs\`, and three launchers.

```bash
# from the Mac: code, plus the export zip (or set ROBOFLOW_API_KEY on the box to download it there)
training/remote/push_to_box.sh emaan-box D:/training data/dataset_261110/roboflow.zip
ssh emaan-box 'powershell -NoProfile -ExecutionPolicy Bypass -File D:\training\code\training\remote\setup_box.ps1 -Root D:\training'
```

The setup builds three environments (DEIMv2 needs torch 2.5.1, the others torch 2.7.1 cu126), downloads
`yolo26l.pt`, `rf-detr-medium.pth` and the DEIMv2-L COCO checkpoint, extracts the export, rebuilds the split
and refuses to continue unless it matches `reports/dataset_261110/split_manifest.csv.gz` exactly, checks the
Comet key against the Comet API, and dry-runs all three scripts. It never starts training.

Start a run (any flag of the training script can be appended):

```powershell
D:\training\train_yolo26.ps1                     # baseline
D:\training\train_yolo26.ps1 --preset p2 --batch 16
D:\training\train_rfdetr.ps1 --batch 8 --grad-accum 2
D:\training\train_deimv2.ps1 --total-batch 8
```

Windows specifics: DEIMv2 runs single-process without torchrun (NCCL does not exist on Windows), so
multi-GPU DEIMv2 needs Linux. Comet reads `COMET_API_KEY` (user environment variable) or `~\.comet.config`;
all three scripts log to the `vehicle-detection` project, and `--no-comet` turns it off. Runs started over
ssh die with the session, so start long runs from a local or RDP terminal on the box.

## 8. Deploying a model to the app

Two steps, in two environments: the training code needs newer packages than the app runs with
(Ultralytics 8.3 cannot load a YOLO26 checkpoint), and an engine only loads with the TensorRT version and GPU
model it was built on, so the app's own environment has to build it.

```powershell
# 1. checkpoint -> ONNX, in the model's training environment
cd D:\training\DEIMv2
D:\training\envs\deimv2\Scripts\python.exe <repo>\utility_scripts\export_onnx.py deim --weights D:\training\runs\deimv2-l-cctv\best_stg2.pth `
    --config configs\deimv2\deimv2_dinov3_l_traffic_deimv2-l-cctv.yml --deim-repo D:\training\DEIMv2 --out D:\training\runs\export\deimv2_l_cctv.onnx
D:\training\envs\yolo26\Scripts\python.exe <repo>\utility_scripts\export_onnx.py yolo --weights D:\training\runs\yolo26\yolo26l-o2m-sq640\weights\best.pt `
    --out D:\training\runs\export\yolo26l_o2m_sq640.onnx

# 2. ONNX -> engine, in the app's environment (traffic_env_py310), on the machine that runs the app
python <repo>\utility_scripts\onnx2tensorrt.py D:\training\runs\export\deimv2_l_cctv.onnx
python <repo>\utility_scripts\onnx2tensorrt.py D:\training\runs\export\yolo26l_o2m_sq640.onnx

# 3. score the engine through the app's runtime before using it (compare with the PyTorch numbers)
python <repo>\training\predict_coco.py engine --weights D:\training\runs\export\deimv2_l_cctv.engine --split <coco>\test --out deim-trt.json --batch 64
```

Then copy the engine into `models\weights\` and set `model_path` and `model_preprocessing` for the machine in
`config.py` (`'deim'` for DEIMv2, `'yolo'` for YOLO). The engine file carries its class names and the
preprocessing it needs in a header; the app and the frame selector refuse to start on a mismatch, because a
model fed the other family's normalisation still returns boxes, only far worse ones.

What differs between the two families in the app:

| | YOLO26 (one-to-many head) | DEIMv2 |
|---|---|---|
| input | RGB 0..1 | RGB 0..1, then ImageNet mean/std |
| output | raw head `(batch, 13, 8400)`; Ultralytics runs NMS | final boxes `(batch, 300, 6)`; no NMS |
| loaded by | Ultralytics (`YOLO(engine)`), as before | `models/trt_detector.py` |
| score floor | Ultralytics default 0.25 | 0.25 (same, so the tracker sees the same range) |
| precision | FP16 | FP16 with the attention scores, softmax, layer norms and decoder attention in FP32 |

**DEIMv2 cannot be a plain FP16 engine.** It returns NaN for every box, for two reasons found by exposing the
engine's intermediate layers: the attention scores of the first backbone block reach 2.5e6 (FP16 ends at
65504), and TensorRT's fused FP16 version of the encoder's transformer layer returns NaN at batch 64.
`export_onnx.py` therefore traces the model under PyTorch autocast with attention in FP32
(`fp32_attention()`), and `onnx2tensorrt.py` builds that file strongly typed. `models/trt_detector.py` raises
if an engine ever returns NaN scores, so this cannot pass as "no vehicles".

### Engines on swest (2026-11-18, TensorRT 10.10, RTX 3090)

Test split, same scorer (`reports/dataset_261110/model_comparison/test_comparison_engines.md`):

| engine | mAP50-95 (PyTorch) | best-F1 (thr) | full video, frame selector | file |
|---|---|---|---|---|
| `deimv2_l_cctv.engine` | 0.504 (0.503) | 0.766 (0.42) | 120 frames/s | `D:\training\runs\export\` |
| `yolo26l_o2m_sq640.engine` | 0.473 (0.474) | 0.751 (0.30) | 259 frames/s | `D:\training\runs\export\` |

The app with the previous model ran at about 240 frames/s on the same video, so DEIMv2 roughly halves
throughput (a two-hour video takes about 20 minutes to detect instead of 10); the YOLO engine costs nothing.

Counts on the hand-counted video (`inflated_counts_261002.dav`, 144,002 frames; detections through the frame
selector's GPU decode, then `replay_detections.py` with the zones of the production dump). The hand count is
905 vehicles and is known to undercount by about 3%:

| model | vehicles | S>N | N>S | Light | Taxi | Bus | Short / Medium / Long Truck | pedestrian and cyclist rows |
|---|---|---|---|---|---|---|---|---|
| `premier_station` (production dump) | 946 | 724 | 222 | 677 | 156 | 47 | 43 / 9 / 13 | 63 |
| DEIMv2 engine | 933 | 717 | 216 | 725 | 188 | 6 | 9 / 5 / 0 | 0 |
| DEIMv2 engine, tracker `det_thresh` 0.42 | 943 | 728 | 215 | 731 | 190 | 6 | 11 / 5 / 0 | 0 |
| YOLO26 sq640 engine | 949 | 735 | 214 | 736 | 195 | 2 | 11 / 5 / 0 | 0 |

Totals agree within 1.5%. The class mix does not, and nobody has checked it against the video yet:

- The new models call far fewer vehicles Bus, Short Truck and Long Truck and more of them Light Vehicle and
  Minibus Taxi. The production model was known to read bakkies as Short Truck and some minibuses as Bus, so
  part of this is a correction, but zero Long Trucks in two hours needs a look.
- The new models output almost no Pedestrian, Cyclist or Motorcycle boxes on this camera (DEIMv2: 15
  Pedestrian boxes in 144,002 frames; the production model: 6,881), although DEIMv2's pedestrian recall on the
  test split is 0.69 at the same threshold. Unexplained. Candidates: this camera (2560x1440, not in the
  dataset) or the decoder's 4x downscale to 640, which differs from the resize used for training images.
