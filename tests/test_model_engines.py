"""Engine file header, preprocessing choice and the result shape the pipeline reads (no GPU or TensorRT needed)."""
import numpy as np
import pytest

from models import preprocessing
from models import trt_detector as td


def test_engine_header_round_trips_and_headerless_engines_read_as_yolo(tmp_path):
    meta = {'model_type': 'deim', 'preprocessing': 'deim', 'names': {'0': 'Bus', '2': 'Light Vehicle'}, 'imgsz': [640, 640]}
    path = tmp_path / 'm.engine'
    td.write_engine(path, meta, b'\x00\x01ENGINE-BYTES')
    assert td.read_engine_metadata(path) == meta
    with open(path, 'rb') as f:
        got, offset = td._read_header(f)
        assert got == meta and f.read() == b'\x00\x01ENGINE-BYTES'
    raw = tmp_path / 'trtexec.engine'                       # no header: built by trtexec / trtutils
    raw.write_bytes(b'ftrt' + bytes(range(256)) * 4)
    assert td.read_engine_metadata(raw) == {}
    assert td.engine_preprocessing({}) == 'yolo'
    with open(raw, 'rb') as f:
        assert td._read_header(f) == ({}, 0) and f.read(4) == b'ftrt'


def test_a_mismatched_preprocessing_is_refused(tmp_path):
    deim = tmp_path / 'deim.engine'
    td.write_engine(deim, {'model_type': 'deim', 'preprocessing': 'deim', 'names': {}}, b'x')
    yolo = tmp_path / 'yolo.engine'
    td.write_engine(yolo, {'task': 'detect', 'names': {}}, b'x')        # an Ultralytics export says nothing
    assert td.check_engine_preprocessing(deim, 'deim')['model_type'] == 'deim'
    assert td.check_engine_preprocessing(yolo, 'yolo') == {'task': 'detect', 'names': {}}
    with pytest.raises(ValueError, match="needs MODEL_PREPROCESSING = 'deim'"):
        td.check_engine_preprocessing(deim, 'yolo')
    with pytest.raises(ValueError, match="needs MODEL_PREPROCESSING = 'yolo'"):
        td.check_engine_preprocessing(yolo, 'deim')
    with pytest.raises(ValueError, match='must be one of'):
        preprocessing.check_mode('imagenet')


def test_normalisation_per_mode():
    torch = pytest.importorskip('torch')
    x = torch.rand(2, 3, 4, 5)
    assert preprocessing.normalize(x, 'yolo') is x
    for dtype in (torch.float32, torch.float16):
        out = preprocessing.normalize(x.to(dtype), 'deim')
        assert out.dtype == dtype and out.shape == x.shape
        want = (x.numpy() - np.array(preprocessing.IMAGENET_MEAN).reshape(1, 3, 1, 1)) / np.array(preprocessing.IMAGENET_STD).reshape(1, 3, 1, 1)
        assert np.abs(out.float().numpy() - want).max() < (1e-5 if dtype == torch.float32 else 2e-2)


def test_detections_look_like_the_ultralytics_result_the_pipeline_reads():
    torch = pytest.importorskip('torch')
    names = {0: 'Bus'}
    full = td.Detections(torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.9, 0.0]]), names)
    empty = td.Detections(torch.zeros((0, 6)), names)
    assert len(full.boxes) == 1 and full.boxes and full.boxes.data.shape == (1, 6) and full.names is names
    assert len(empty.boxes) == 0 and not empty.boxes and empty                 # `if r and r.boxes` in the app
