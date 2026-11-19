"""Phát hiện khuôn mặt. Mặc định YuNet (OpenCV, nhẹ, tự tải, chạy được cả trên Pi).
YOLO-face chỉ dùng khi cài thêm `pip install fer-edge[yolo]` và truyền --detector yolo."""
import math
from dataclasses import dataclass

import cv2
import numpy as np

from .registry import cache_dir, download, yunet_path

YOLO_FACE_URL = "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov11n-face.pt"


@dataclass
class Face:
    box: tuple            # x1, y1, x2, y2 trên ảnh gốc
    score: float
    eyes: tuple = None    # ((x,y) mắt phải, (x,y) mắt trái) nếu bộ phát hiện cho biết

    @property
    def area(self):
        return max(0.0, self.box[2] - self.box[0]) * max(0.0, self.box[3] - self.box[1])


class YuNetDetector:
    """YuNet của OpenCV: ~0,1 MB, không cần PyTorch, cho cả khung mặt lẫn 5 điểm mốc."""
    name = "yunet"

    def __init__(self, conf=0.6, det_width=320, min_face=40):
        self.weights = str(yunet_path())
        self.conf, self.det_width, self.min_face = conf, det_width, min_face
        self.det = cv2.FaceDetectorYN.create(self.weights, "", (det_width, det_width), conf, 0.3, 50)
        self._size = None

    def detect(self, bgr):
        H, W = bgr.shape[:2]
        dw = min(self.det_width, W)
        dh = max(1, int(round(H * dw / W)))
        if self._size != (dw, dh):
            self.det.setInputSize((dw, dh))
            self._size = (dw, dh)
        small = cv2.resize(bgr, (dw, dh), interpolation=cv2.INTER_LINEAR) if (dw, dh) != (W, H) else bgr
        _, faces = self.det.detect(small)
        if faces is None:
            return []
        sx, sy = W / dw, H / dh
        out = []
        for f in faces:
            x, y, w, h = f[0] * sx, f[1] * sy, f[2] * sx, f[3] * sy
            eyes = ((f[4] * sx, f[5] * sy), (f[6] * sx, f[7] * sy))
            out.append(Face((float(x), float(y), float(x + w), float(y + h)), float(f[14]), eyes))
        return out


class YoloFaceDetector:
    """YOLOv11n-face (ultralytics). Chính xác hơn YuNet ở ảnh khó, nhưng nặng và cần PyTorch."""
    name = "yolo"

    def __init__(self, conf=0.5, imgsz=640, min_face=40, weights=None):
        from pathlib import Path

        from ultralytics import YOLO
        w = Path(weights) if weights else cache_dir() / "detectors" / "yolov11n-face.pt"
        if not w.is_file():
            download(YOLO_FACE_URL, w, desc="YOLOv11n-face")
        self.model = YOLO(str(w))
        self.weights, self.conf, self.imgsz, self.min_face = w.name, conf, imgsz, min_face

    def detect(self, bgr):
        r = self.model.predict(bgr, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        return [Face(tuple(map(float, b)), float(s))
                for b, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]


class WholeImage:
    """Không phát hiện: coi cả ảnh là khuôn mặt (ảnh FERPlus 48x48 / RAF-DB đã căn sẵn)."""
    name = "none"
    weights = "-"

    def __init__(self, **kw):
        self.min_face = 0

    def detect(self, bgr):
        h, w = bgr.shape[:2]
        return [Face((0.0, 0.0, float(w), float(h)), 1.0)]


def build_detector(kind="yunet", conf=0.6, det_width=320, min_face=40, imgsz=640):
    if kind == "none":
        return WholeImage()
    if kind == "yolo":
        return YoloFaceDetector(conf=conf, imgsz=imgsz, min_face=min_face)
    return YuNetDetector(conf=conf, det_width=det_width, min_face=min_face)


# ----------------------------------------------------------------------------- cắt khuôn mặt
def crop_face(bgr, face, margin=0.15, align=False):
    """Khung vuông quanh mặt, nới `margin` mỗi phía. align=True thì xoay cho hai mắt ngang nhau."""
    x1, y1, x2, y2 = face.box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + 2 * margin)
    img = bgr
    if align and face.eyes:
        (rx, ry), (lx, ly) = face.eyes
        angle = math.degrees(math.atan2(ly - ry, lx - rx))
        big = side * 1.5
        H, W = img.shape[:2]
        x0, y0 = int(max(0, cx - big / 2)), int(max(0, cy - big / 2))
        x3, y3 = int(min(W, cx + big / 2)), int(min(H, cy + big / 2))
        roi = img[y0:y3, x0:x3]
        if roi.size:
            M = cv2.getRotationMatrix2D((cx - x0, cy - y0), angle, 1.0)
            img = cv2.warpAffine(roi, M, (roi.shape[1], roi.shape[0]), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
            cx, cy = cx - x0, cy - y0
    H, W = img.shape[:2]
    a, b = int(max(0, round(cx - side / 2))), int(max(0, round(cy - side / 2)))
    c, d = int(min(W, round(cx + side / 2))), int(min(H, round(cy + side / 2)))
    crop = img[b:d, a:c]
    return crop if crop.size else None


def as_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def biggest(faces, n=1):
    return sorted(faces, key=lambda f: f.area, reverse=True)[:n]


def to_numpy_boxes(faces):
    return np.array([f.box for f in faces], dtype=np.float32) if faces else np.zeros((0, 4), np.float32)
