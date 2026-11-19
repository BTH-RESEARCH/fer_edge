"""Chạy mô hình trên mọi loại nguồn: ảnh, thư mục ảnh, video, webcam.

    predict("anh.jpg")        predict("thu_muc/")        predict("video.mp4")        predict(0)

Mỗi lần chạy tạo một thư mục kết quả: outputs/<kịch bản>_<tên>_<thời gian>/
    frames.csv  faces.csv  run.json  annotated/ | annotated.mp4 | recording.mp4+snapshots/  report/
"""
import platform
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:                       # thanh tiến trình là tùy chọn
    def tqdm(it=None, total=None, **kw):
        class _Bar:
            def __iter__(self):
                return iter(it or ())

            def update(self, n=1):
                pass

            def close(self):
                pass
        return _Bar() if it is None else it

from .constants import IMG_EXT, LABEL_ALIASES, VIDEO_EXT
from .detect import build_detector
from .model import FERModel
from .pipeline import FERPipeline, TemporalSmoother, draw, upscale
from .sysinfo import env_info, write_json


# ----------------------------------------------------------------------------- tiện ích
def new_run_dir(out_dir, scenario, name=None):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run = Path(out_dir) / f"{scenario}_{name + '_' if name else ''}{stamp}"
    run.mkdir(parents=True, exist_ok=True)
    return run


def face_rows(key, fr, classes):
    rows = []
    for i, f in enumerate(fr.faces):
        x1, y1, x2, y2 = f.box
        rows.append({**key, "face_idx": i, "x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1),
                     "y2": round(y2, 1), "face_px": round(min(x2 - x1, y2 - y1), 1),
                     "det_score": round(f.det_score, 4), "pred": f.label, "conf": round(f.conf, 4),
                     **{f"p_{c}": round(float(v), 4) for c, v in zip(classes, f.probs)}})
    return rows


def frame_row(key, fr, classes, smoothed=None, extra=None):
    p = fr.primary
    row = {**key, "n_faces": len(fr.faces), **{k: round(v, 3) for k, v in fr.timings.items()},
           "pred": p.label if p else "", "conf": round(p.conf, 4) if p else np.nan}
    if p is not None:
        row.update({f"p_{c}": round(float(v), 4) for c, v in zip(classes, p.probs)})
    if smoothed is not None:
        k = int(np.argmax(smoothed))
        row.update(smooth_pred=classes[k], smooth_conf=round(float(smoothed[k]), 4),
                   **{f"sp_{c}": round(float(v), 4) for c, v in zip(classes, smoothed)})
    row.update(extra or {})
    return row


def run_meta(scenario, source, pipe, smoothing, extra):
    m = pipe.model
    d = pipe.detector
    return {"scenario": scenario, "source": str(source), "created": datetime.now().isoformat(timespec="seconds"),
            "model": {**m.info(), "pack": m.name},
            "detector": {"type": d.name, "weights": Path(str(getattr(d, "weights", "-"))).name,
                         "conf": getattr(d, "conf", None), "imgsz": getattr(d, "det_width", getattr(d, "imgsz", None)),
                         "margin": pipe.margin, "align": pipe.align},
            "smoothing": {"method": smoothing.method, "alpha": smoothing.alpha, "window": smoothing.window},
            "env": env_info(), **extra}


def build_pipeline(model="resnet18", detector="yunet", threads=0, margin=0.15, min_face=40, max_faces=5,
                   align=False, det_conf=0.6, det_width=320, models_dir=None):
    m = model if isinstance(model, FERModel) else FERModel(model, threads=threads, models_dir=models_dir)
    d = detector if hasattr(detector, "detect") else build_detector(detector, det_conf, det_width, min_face)
    return FERPipeline(m, d, margin=margin, min_face=min_face, max_faces=max_faces, align=align)


def gt_from_path(path, root, classes):
    """Nhãn thật từ tên thư mục cha: tên lớp (happy, sad, suprise...) hoặc số 1..K. Không có -> None."""
    try:
        parts = Path(path).relative_to(root).parts[:-1]
    except ValueError:
        return None
    for name in reversed(parts):
        n = LABEL_ALIASES.get(name.lower(), name.lower())
        if n in classes:
            return n
        if name.isdigit() and 1 <= int(name) <= len(classes):
            return classes[int(name) - 1]
    return None


def source_kind(source):
    """'image' | 'folder' | 'video' | 'webcam'."""
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        return "webcam"
    p = Path(str(source))
    if p.is_dir():
        return "folder"
    if p.suffix.lower() in VIDEO_EXT:
        return "video"
    if p.suffix.lower() in IMG_EXT:
        return "image"
    if p.is_file():
        return "video"                    # đuôi lạ: thử mở như video
    raise FileNotFoundError(f"Không thấy nguồn '{source}' (ảnh, thư mục, video, hoặc số hiệu webcam)")


# ----------------------------------------------------------------------------- ảnh và thư mục ảnh
def run_images(pipe, source, out_dir="outputs", save_max=200, labels=True, name=None, smoothing=None):
    source = Path(source)
    files = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.suffix.lower() in IMG_EXT)
    if not files:
        raise SystemExit(f"Không thấy ảnh nào trong {source}")
    root = source if source.is_dir() else source.parent
    run = new_run_dir(out_dir, "image", name or (source.stem if source.is_file() else source.name))
    (run / "annotated").mkdir()
    classes = pipe.classes
    frames, faces, t0 = [], [], time.perf_counter()
    for i, f in enumerate(tqdm(files, desc="ảnh", unit="ảnh")):
        img = cv2.imread(str(f))
        if img is None:
            print(f"  bỏ qua (không đọc được): {f}")
            continue
        fr = pipe.process(img)
        key = {"frame": i, "file": str(f.relative_to(root)), "width": img.shape[1], "height": img.shape[0]}
        gt = gt_from_path(f, root, classes) if labels else None
        frames.append(frame_row(key, fr, classes, extra={"gt": gt or ""}))
        faces += face_rows(key, fr, classes)
        if i < save_max:
            big, fr2 = upscale(img, fr, 192)
            cv2.imwrite(str(run / "annotated" / f"{i:05d}_{f.stem}.jpg"),
                        draw(big, fr2, classes, show_bars=min(big.shape[:2]) >= 300))
    wall = time.perf_counter() - t0
    pd.DataFrame(frames).to_csv(run / "frames.csv", index=False)
    pd.DataFrame(faces).to_csv(run / "faces.csv", index=False)
    write_json(run / "run.json", run_meta("image", source, pipe, smoothing or TemporalSmoother(), {
        "n_items": len(frames), "wall_s": round(wall, 2), "labeled": any(r["gt"] for r in frames),
        "annotated_saved": min(len(frames), save_max)}))
    return run


# ----------------------------------------------------------------------------- video
def run_video(pipe, source, out_dir="outputs", smoothing=None, save_video=True, max_frames=0, show=False, name=None):
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise SystemExit(f"Không mở được video {source}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    run = new_run_dir(out_dir, "video", name or Path(source).stem)
    writer = (cv2.VideoWriter(str(run / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (W, H))
              if save_video else None)
    sm = smoothing or TemporalSmoother(n_classes=len(pipe.classes))
    classes = pipe.classes
    frames, faces, i, fps_ema = [], [], 0, None
    bar = tqdm(total=(min(total, max_frames) if max_frames else total) or None, desc="video", unit="khung")
    t0 = time.perf_counter()
    while True:
        ok, img = cap.read()
        if not ok or (max_frames and i >= max_frames):
            break
        fr = pipe.process(img)
        s = sm.update(fr.primary.probs if fr.primary else None)
        inst = 1000 / max(fr.timings["total_ms"], 1e-6)
        fps_ema = inst if fps_ema is None else 0.9 * fps_ema + 0.1 * inst
        key = {"frame": i, "time_s": round(i / src_fps, 3)}
        frames.append(frame_row(key, fr, classes, s))
        faces += face_rows(key, fr, classes)
        if writer is not None or show:
            out = draw(img, fr, classes, s, fps=fps_ema)
            if writer is not None:
                writer.write(out)
            if show:
                cv2.imshow("fer-edge (q: quit)", out)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
        i += 1
        bar.update(1)
    bar.close()
    wall = time.perf_counter() - t0
    cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    pd.DataFrame(frames).to_csv(run / "frames.csv", index=False)
    pd.DataFrame(faces).to_csv(run / "faces.csv", index=False)
    write_json(run / "run.json", run_meta("video", source, pipe, sm, {
        "n_items": i, "wall_s": round(wall, 2), "src_fps": round(src_fps, 2), "resolution": f"{W}x{H}",
        "duration_s": round(i / src_fps, 2), "annotated_video": "annotated.mp4" if save_video else None}))
    return run


# ----------------------------------------------------------------------------- webcam
def open_camera(index, width, height):
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def run_webcam(pipe, cam=0, out_dir="outputs", smoothing=None, width=1280, height=720, seconds=0,
               record=False, show=True, name=None):
    cap = open_camera(int(cam), width, height)
    if not cap.isOpened():
        raise SystemExit(f"Không mở được camera {cam}. Thử --source 1, hoặc kiểm tra quyền truy cập camera.")
    ok, img = cap.read()
    if not ok:
        raise SystemExit("Camera mở được nhưng không đọc được khung hình.")
    H, W = img.shape[:2]
    run = new_run_dir(out_dir, "webcam", name)
    (run / "snapshots").mkdir()
    sm = smoothing or TemporalSmoother(n_classes=len(pipe.classes))
    classes = pipe.classes
    writer, show_bars, frames, faces, i, fps_ema = None, True, [], [], 0, None
    if show:
        print("Phím: q/ESC = thoát | s = chụp ảnh | r = bật/tắt ghi video | b = bật/tắt thanh xác suất")
    t_start = t_prev = time.perf_counter()
    while True:
        t_cap = time.perf_counter()
        if i > 0:
            ok, img = cap.read()
            if not ok:
                break
        cap_ms = 1000 * (time.perf_counter() - t_cap)
        fr = pipe.process(img)
        s = sm.update(fr.primary.probs if fr.primary else None)
        now = time.perf_counter()
        loop_fps = 1 / max(now - t_prev, 1e-6)
        t_prev = now
        fps_ema = loop_fps if fps_ema is None else 0.9 * fps_ema + 0.1 * loop_fps
        out = draw(img, fr, classes, s, fps=fps_ema, show_bars=show_bars)
        if writer is not None:
            writer.write(out)
            cv2.circle(out, (W - 24, H - 24), 9, (0, 0, 255), -1)
        key = 255
        if show:
            cv2.imshow("fer-edge webcam (q: quit)", out)
            key = cv2.waitKey(1) & 0xFF
        disp_ms = 1000 * (time.perf_counter() - now)
        k = {"frame": i, "time_s": round(now - t_start, 3)}
        frames.append(frame_row(k, fr, classes, s, extra={"capture_ms": round(cap_ms, 3),
                                                          "display_ms": round(disp_ms, 3),
                                                          "loop_fps": round(loop_fps, 2),
                                                          "recording": writer is not None}))
        faces += face_rows(k, fr, classes)
        i += 1
        if key in (ord("q"), 27) or (seconds and now - t_start >= seconds):
            break
        if key == ord("s"):
            p = run / "snapshots" / f"snap_{i:05d}.jpg"
            cv2.imwrite(str(p), out)
            print(f"  đã chụp {p.name}")
        if key == ord("b"):
            show_bars = not show_bars
        if key == ord("r") or (record and writer is None and i == 1):
            if writer is None:
                fps_guess = max(5.0, min(30.0, fps_ema or 15.0))
                writer = cv2.VideoWriter(str(run / "recording.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps_guess, (W, H))
                print(f"  bắt đầu ghi recording.mp4 ({fps_guess:.0f} FPS)")
            else:
                writer.release()
                writer = None
                print("  dừng ghi")
    wall = time.perf_counter() - t_start
    cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    pd.DataFrame(frames).to_csv(run / "frames.csv", index=False)
    pd.DataFrame(faces).to_csv(run / "faces.csv", index=False)
    write_json(run / "run.json", run_meta("webcam", f"camera {cam}", pipe, sm, {
        "n_items": i, "wall_s": round(wall, 2), "resolution": f"{W}x{H}", "duration_s": round(wall, 2)}))
    return run


# ----------------------------------------------------------------------------- cửa chính
def predict(source, model="resnet18", detector="yunet", out_dir="outputs", report=True, **kw):
    """Chạy trên bất kỳ nguồn nào và trả về thư mục kết quả."""
    kind = source_kind(source)
    pipe_kw = {k: kw.pop(k) for k in ("threads", "margin", "min_face", "max_faces", "align", "det_conf",
                                      "det_width", "models_dir") if k in kw}
    smooth_kw = {k: kw.pop(k) for k in ("method", "alpha", "window", "reset_after") if k in kw}
    pipe = build_pipeline(model, detector, **pipe_kw)
    sm = TemporalSmoother(n_classes=len(pipe.classes), **smooth_kw)
    if kind in ("image", "folder"):
        run = run_images(pipe, source, out_dir, smoothing=sm, **kw)
    elif kind == "video":
        run = run_video(pipe, source, out_dir, smoothing=sm, **kw)
    else:
        run = run_webcam(pipe, int(source), out_dir, smoothing=sm, **kw)
    if report:
        from .report import build_report
        build_report(run)
    return run
