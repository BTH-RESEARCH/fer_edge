"""Kiểm tra nhanh: pytest -q  (cần ít nhất một .ferpack trong models/ hoặc cache)."""
import numpy as np
import pytest

from fer_edge.constants import EMOTION_COLORS, hex_to_bgr
from fer_edge.evaluate import cls_metrics
from fer_edge.model import softmax
from fer_edge.registry import cached_models
from fer_edge.runners import gt_from_path, source_kind


def test_softmax():
    p = softmax(np.array([[1.0, 2.0, 3.0]]))
    assert np.isclose(p.sum(), 1.0) and p.argmax() == 2


def test_metrics_perfect():
    y = np.array([0, 1, 2, 0])
    m = cls_metrics(y, y, 3)
    assert m["acc"] == 100 and m["mca"] == 100 and m["f1"] == 100


def test_source_kind(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "v.mp4").write_bytes(b"x")
    assert source_kind(0) == "webcam" and source_kind("1") == "webcam"
    assert source_kind(tmp_path) == "folder"
    assert source_kind(tmp_path / "a.jpg") == "image"
    assert source_kind(tmp_path / "v.mp4") == "video"


def test_gt_from_path(tmp_path):
    classes = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]
    assert gt_from_path(tmp_path / "happy" / "a.png", tmp_path, classes) == "happiness"
    assert gt_from_path(tmp_path / "suprise" / "a.png", tmp_path, classes) == "surprise"
    assert gt_from_path(tmp_path / "2" / "a.png", tmp_path, classes) == "happiness"
    assert gt_from_path(tmp_path / "xxx" / "a.png", tmp_path, classes) is None


def test_colors():
    assert hex_to_bgr(EMOTION_COLORS["neutral"]) == (214, 120, 42)


@pytest.mark.skipif(not cached_models(), reason="chưa tải mô hình nào")
def test_model_runs():
    from fer_edge.model import FERModel
    m = FERModel(next(iter(cached_models())))
    p = m.predict_faces([np.zeros((64, 64, 3), np.uint8)])
    assert p.shape == (1, len(m.classes)) and np.isclose(p.sum(), 1.0, atol=1e-4)
