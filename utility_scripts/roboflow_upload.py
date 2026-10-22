"""Upload images with their YOLO labels to a Roboflow project.

The API key is read from the ROBOFLOW_API_KEY environment variable; never write it into this file.
Set it once on a machine (then open a new terminal):   setx ROBOFLOW_API_KEY "<key>"

    python utility_scripts/roboflow_upload.py frames_for_annotation --batch 261023_frame_selector --dry-run
    python utility_scripts/roboflow_upload.py frames_for_annotation --batch 261023_frame_selector

FOLDER holds images/ and labels/ (YOLO .txt with the same names) and, optionally, classes.txt, as the
frame selector writes them. Images with an empty label file are uploaded without annotations, so they
arrive unannotated and ready to label. Uploaded images are logged in FOLDER/roboflow_uploaded_<batch>.txt,
so a re-run after an interruption skips them.
"""
import argparse
import glob
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_LABELS = ['Bus', 'Cyclist', 'Light Vehicle', 'Long Truck', 'Medium Truck', 'Minibus Taxi', 'Motorcycle',
                  'Pedestrian', 'Short Truck']
IMG_EXT = ('.jpg', '.jpeg', '.png')


def find_pairs(folder):
    """-> [(image, label file or None if empty)], [images without a label file]."""
    pairs, missing = [], []
    for img in sorted(p for p in glob.glob(os.path.join(folder, 'images', '*')) if p.lower().endswith(IMG_EXT)):
        ann = os.path.join(folder, 'labels', os.path.splitext(os.path.basename(img))[0] + '.txt')
        if not os.path.exists(ann):
            missing.append(img)
        else:
            pairs.append((img, ann if os.path.getsize(ann) > 0 else None))
    return pairs, missing


def label_map(folder):
    path = os.path.join(folder, 'classes.txt')
    names = [l.strip() for l in open(path, encoding='utf-8') if l.strip()] if os.path.exists(path) else DEFAULT_LABELS
    return dict(enumerate(names))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folder', help='folder with images/, labels/ and classes.txt')
    ap.add_argument('--project', default='full-res-labelled', help='Roboflow project id (default: full-res-labelled)')
    ap.add_argument('--batch', required=True, help='Roboflow batch name, e.g. 261023_frame_selector')
    ap.add_argument('--tag', default=None, help='tag for every image (default: the batch name)')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--dry-run', action='store_true', help='list what would be uploaded, upload nothing')
    a = ap.parse_args(argv)
    tag = a.tag or a.batch

    pairs, missing = find_pairs(a.folder)
    labels = label_map(a.folder)
    log_path = os.path.join(a.folder, f'roboflow_uploaded_{a.batch}.txt')
    done = set(open(log_path, encoding='utf-8').read().split('\n')) if os.path.exists(log_path) else set()
    todo = [(i, l) for i, l in pairs if os.path.basename(i) not in done]
    size_gb = sum(os.path.getsize(i) for i, _ in todo) / 1e9
    print(f"{len(pairs)} images with label files ({sum(1 for _, l in pairs if l is None)} with no labels: uploaded unannotated); "
          f"{len(missing)} images without a label file are skipped")
    print(f"already uploaded in this batch: {len(pairs) - len(todo)}; to upload now: {len(todo)} images, {size_gb:.1f} GB")
    print(f"project {a.project} | batch {a.batch} | tag {tag} | classes {labels}")

    key = os.environ.get('ROBOFLOW_API_KEY')
    if not key:
        msg = "ROBOFLOW_API_KEY is not set. Set it with:  setx ROBOFLOW_API_KEY \"<key>\"  and open a new terminal."
        if a.dry_run:
            print("note: " + msg)
            return 0
        sys.exit(msg)
    from roboflow import Roboflow
    project = Roboflow(api_key=key).workspace().project(a.project)
    print(f"Roboflow project found: {project.name} ({project.type})")
    if a.dry_run or not todo:
        return 0

    lock = threading.Lock()
    failed = []

    def upload_one(img, ann):
        for attempt in range(3):
            try:
                kw = dict(annotation_path=ann, annotation_labelmap=labels) if ann else {}
                project.single_upload(image_path=img, batch_name=a.batch, tag_names=[tag], **kw)
                with lock, open(log_path, 'a', encoding='utf-8') as fh:
                    fh.write(os.path.basename(img) + '\n')
                return
            except Exception:                        # noqa: BLE001 - retried, then reported
                if attempt == 2:
                    raise
                time.sleep(5 * (attempt + 1))

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futures = {ex.submit(upload_one, i, l): i for i, l in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                fut.result()
            except Exception as e:                   # noqa: BLE001 - listed at the end
                failed.append((futures[fut], e))
                print(f"FAILED: {os.path.basename(futures[fut])}: {e}", flush=True)
            if n % 100 == 0 or n == len(todo):
                print(f"{n}/{len(todo)} done, {len(failed)} failed, {n / (time.time() - t0):.1f} images/s", flush=True)
    print(f"uploaded {len(todo) - len(failed)} of {len(todo)}; log: {log_path}")
    if failed:
        print(f"{len(failed)} failed; run the same command again to retry only those")
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
