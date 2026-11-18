"""Comet ML logging shared by the three training scripts.

Credentials are never stored in this repo. comet_ml finds them on its own, in this order: the COMET_API_KEY
environment variable (set once per machine: `setx COMET_API_KEY <key>` on Windows), then ~/.comet.config.
COMET_WORKSPACE is optional. Without a key everything here is a no-op and training runs unlogged.

YOLO26 uses the built-in Ultralytics integration (project from COMET_PROJECT_NAME, named and tagged by
name_ultralytics_experiment). RF-DETR and DEIMv2 have no Comet logger, so MetricTail follows the metrics
file each one writes anyway (RF-DETR metrics.csv, DEIMv2 log.txt) and forwards new rows every poll.
"""
import configparser
import csv
import json
import math
import os
import threading
import time
from pathlib import Path

DEFAULT_PROJECT = 'vehicle-detection'      # the project the earlier YOLOv8/RT-DETR runs logged to


def has_credentials():
    if os.environ.get('COMET_API_KEY'):
        return True
    cfg = Path.home() / '.comet.config'
    if cfg.exists():
        p = configparser.ConfigParser()
        p.read(cfg)
        return bool(p.get('comet', 'api_key', fallback=''))
    return False


def start(project, name, tags=(), params=None, enabled=True):
    """-> a running comet_ml experiment, or None (disabled, no key, package missing, or offline failure)."""
    if not enabled:
        return None
    if not has_credentials():
        print('comet: no COMET_API_KEY or ~/.comet.config; not logging this run')
        return None
    try:
        import comet_ml
        exp = comet_ml.start(project_name=project, experiment_config=comet_ml.ExperimentConfig(
            name=name, tags=list(tags), auto_metric_logging=False, auto_param_logging=False,
            auto_histogram_weight_logging=False, log_code=False))
        if params:
            exp.log_parameters({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in params.items()})
        print(f'comet: logging to {exp.url}')
        return exp
    except Exception as e:                       # never let logging stop a training run
        print(f'comet: not logging this run ({e})')
        return None


def name_ultralytics_experiment(name, tags):
    """Ultralytics callback factory: sets name and tags on the experiment its integration created."""
    def cb(trainer):
        try:
            import comet_ml
            exp = comet_ml.get_running_experiment()
            if exp is not None:
                exp.set_name(name)
                exp.add_tags(list(tags))
        except Exception as e:
            print(f'comet: could not name experiment ({e})')
    return cb


def _num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _rows_csv(path):
    """PyTorch Lightning CSVLogger file: sparse rows with epoch/step columns."""
    with open(path, newline='') as fh:
        for row in csv.DictReader(fh):
            step, epoch = _num(row.pop('step', None)), _num(row.pop('epoch', None))
            vals = {k: x for k, v in row.items() if (x := _num(v)) is not None}
            if vals:
                yield (int(step) if step is not None else None), (int(epoch) if epoch is not None else None), vals


COCO_STATS = ('mAP50-95', 'mAP50', 'mAP75', 'mAP_small', 'mAP_medium', 'mAP_large',
              'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large')


def _rows_jsonl(path):
    """DEIMv2 log.txt: one JSON object per epoch; COCO stats arrive as a 12-number list."""
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                return                              # half-written last line; picked up next poll
            epoch = d.pop('epoch', None)
            vals = {}
            for k, v in d.items():
                if isinstance(v, list) and k.endswith('coco_eval_bbox'):
                    vals.update({f'val/{s}': x for s, x in zip(COCO_STATS, v) if _num(x) is not None})
                elif (x := _num(v)) is not None and k != 'n_parameters':
                    vals[k] = x
            yield epoch, epoch, vals


class MetricTail(threading.Thread):
    """Forward rows appended to a metrics file to a Comet experiment. stop() flushes the remainder."""

    def __init__(self, exp, path, fmt, interval=30):
        super().__init__(daemon=True)
        self.exp, self.path, self.interval = exp, Path(path), interval
        self.reader = _rows_csv if fmt == 'csv' else _rows_jsonl
        self.done = 0
        self._stop = threading.Event()

    def poll(self):
        if self.exp is None or not self.path.exists():
            return
        try:
            rows = list(self.reader(self.path))
        except (OSError, csv.Error):
            return
        if len(rows) < self.done:                 # file was rewritten from scratch (new run in same folder)
            self.done = 0
        for step, epoch, vals in rows[self.done:]:
            self.exp.log_metrics(vals, step=step, epoch=epoch)
        self.done = len(rows)

    def run(self):
        while not self._stop.wait(self.interval):
            self.poll()

    def stop(self):
        self._stop.set()
        self.poll()


def finish(exp, assets=()):
    if exp is None:
        return
    for a in assets:
        if a and Path(a).exists():
            try:
                exp.log_asset(str(a))
            except Exception as e:
                print(f'comet: could not upload {a} ({e})')
    exp.end()
