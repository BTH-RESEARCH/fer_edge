"""Một khung hình: phát hiện mặt -> cắt -> mô hình -> kết quả, kèm thời gian từng khâu.
Thêm bộ làm mượt theo thời gian (video / webcam) và hàm vẽ kết quả lên ảnh."""
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from .constants import EMOTION_COLORS, hex_to_bgr
from .detect import as_rgb, biggest, crop_face


@dataclass
class FaceResult:
    box: tuple
    det_score: float
    probs: np.ndarray
    label: str
    conf: float


@dataclass
class FrameResult:
    faces: list = field(default_factory=list)      # mặt lớn nhất đứng đầu ("mặt chính")
    timings: dict = field(default_factory=dict)    # detect_ms, preprocess_ms, classify_ms, total_ms

    @property
    def primary(self):
        return self.faces[0] if self.faces else None


class FERPipeline:
    def __init__(self, model, detector, margin=0.15, min_face=40, max_faces=5, align=False):
        self.model, self.detector = model, detector
        self.margin, self.min_face, self.max_faces, self.align = margin, min_face, max_faces, align
        self.classes = model.classes
        self.whole_image = detector.name == "none"

    def process(self, bgr):
        t0 = time.perf_counter()
        faces = self.detector.detect(bgr)
        if not self.whole_image:
            faces = [f for f in faces if min(f.box[2] - f.box[0], f.box[3] - f.box[1]) >= self.min_face]
        faces = biggest(faces, self.max_faces)
        t1 = time.perf_counter()

        crops, kept = [], []
        for f in faces:
            c = bgr if self.whole_image else crop_face(bgr, f, self.margin, self.align)
            if c is not None and c.size:
                crops.append(as_rgb(c))
                kept.append(f)

        out, t2, t3 = FrameResult(), t1, t1
        if crops:
            x = self.model.preprocess(crops)
            t2 = time.perf_counter()
            probs = self.model.predict(x)
            t3 = time.perf_counter()
            for f, p in zip(kept, probs):
                k = int(np.argmax(p))
                out.faces.append(FaceResult(f.box, f.score, p, self.classes[k], float(p[k])))
        out.timings = {"detect_ms": 1000 * (t1 - t0),
                       "preprocess_ms": 1000 * (t2 - t1) if crops else np.nan,
                       "classify_ms": 1000 * (t3 - t2) if crops else np.nan,
                       "total_ms": 1000 * (time.perf_counter() - t0)}
        return out


class TemporalSmoother:
    """Làm mượt xác suất của mặt chính qua các khung hình: EMA hoặc bỏ phiếu đa số."""

    def __init__(self, method="ema", alpha=0.3, window=15, reset_after=10, n_classes=8):
        self.method, self.alpha, self.window, self.reset_after, self.k = method, alpha, window, reset_after, n_classes
        self.reset()

    def reset(self):
        self.ema, self.votes, self.missing = None, deque(maxlen=self.window), 0

    def update(self, probs):
        if probs is None:
            self.missing += 1
            if self.missing > self.reset_after:
                self.reset()
            return None
        self.missing = 0
        if self.method == "none":
            return probs
        if self.method == "ema":
            self.ema = probs.copy() if self.ema is None else self.alpha * probs + (1 - self.alpha) * self.ema
            return self.ema
        self.votes.append(int(np.argmax(probs)))
        hist = np.bincount(np.asarray(self.votes), minlength=self.k).astype(float)
        return hist / hist.sum()


# ----------------------------------------------------------------------------- vẽ
def _text(img, s, org, color, scale=0.55, thick=1):
    (w, h), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    cv2.rectangle(img, (x, y - h - base - 4), (x + w + 6, y), color, -1)
    cv2.putText(img, s, (x + 3, y - base - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)


def draw(img, fr, classes, smoothed=None, fps=None, show_bars=True):
    """Khung + nhãn cho mọi mặt; thanh xác suất cho mặt chính; FPS ở góc phải."""
    H, W = img.shape[:2]
    scale = max(0.45, min(1.0, W / 1100))
    for i, f in enumerate(fr.faces):
        x1, y1, x2, y2 = map(int, f.box)
        label, conf = f.label, f.conf
        if i == 0 and smoothed is not None:
            k = int(np.argmax(smoothed))
            label, conf = classes[k], float(smoothed[k])
        col = hex_to_bgr(EMOTION_COLORS.get(label, "#6b6b6b"))
        cv2.rectangle(img, (x1, y1), (x2, y2), col, max(2, int(3 * scale)))
        _text(img, f"{label} {conf:.0%}", (x1, max(int(22 * scale) + 8, y1)), col, 0.6 * scale + 0.1, 1)
    p = fr.primary
    if show_bars and p is not None:
        probs = smoothed if smoothed is not None else p.probs
        bw, bh, x0, y0 = int(150 * scale + 40), int(16 * scale + 4), 10, 10
        overlay = img.copy()
        cv2.rectangle(overlay, (x0 - 6, y0 - 6), (x0 + bw + 110, y0 + len(classes) * (bh + 4) + 2), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
        for j, c in enumerate(classes):
            y = y0 + j * (bh + 4)
            cv2.rectangle(img, (x0, y), (x0 + int(bw * float(probs[j])), y + bh),
                          hex_to_bgr(EMOTION_COLORS.get(c, "#6b6b6b")), -1)
            cv2.putText(img, f"{c} {probs[j]:.2f}", (x0 + bw + 6, y + bh - 3), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4 * scale + 0.08, (240, 240, 240), 1, cv2.LINE_AA)
    if fps is not None:
        s = f"{fps:.1f} FPS"
        (w, _), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        _text(img, s, (W - w - 16, 34), (40, 40, 40), 0.7, 2)
    return img


def upscale(img, fr, side=192):
    """Ảnh nhỏ (48x48): phóng to để đọc được chữ; khung mặt nhân theo cùng tỉ lệ."""
    import copy
    if min(img.shape[:2]) >= side:
        return img.copy(), fr
    s = side / min(img.shape[:2])
    big = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_NEAREST)
    fr2 = copy.deepcopy(fr)
    for f in fr2.faces:
        f.box = tuple(v * s for v in f.box)
    return big, fr2
