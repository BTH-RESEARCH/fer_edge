"""Dòng lệnh fer-edge."""
import argparse
import sys
from pathlib import Path


def _utf8():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# ----------------------------------------------------------------------------- lệnh
def cmd_doctor(a):
    from .registry import cache_dir, cached_models, load_registry, local_models_dir
    from .sysinfo import system_info
    print("Môi trường")
    for k, v in system_info().items():
        print(f"  {k:16s} {v}")
    print(f"\nThư mục cache   {cache_dir()}")
    print(f"Thư mục mô hình {local_models_dir()}")
    have = cached_models()
    print(f"\nMô hình có sẵn trên máy ({len(have)}):")
    for n, p in have.items():
        print(f"  {n:30s} {p.stat().st_size / 1e6:8.1f} MB  {p}")
    if not have:
        reg = load_registry()
        print("  (chưa có) — tải bằng:  fer-edge download " + (next(iter(reg["models"]), "<tên>")))
        return
    name = a.model or next(iter(have))
    print(f"\nThử nạp và chạy '{name}' ...")
    import numpy as np

    from .model import FERModel
    m = FERModel(name, threads=a.threads)
    info = m.info()
    for k, v in info.items():
        print(f"  {k:16s} {v}")
    img = (np.random.default_rng(0).random((200, 200, 3)) * 255).astype("uint8")
    import time
    t = time.perf_counter()
    for _ in range(10):
        p = m.predict_faces([img])
    ms = 1000 * (time.perf_counter() - t) / 10
    print(f"  suy luận 1 khuôn mặt: ~{ms:.1f} ms ({1000 / ms:.1f} FPS) | nhãn thử: {m.classes[int(p[0].argmax())]}")
    print("\nOK — sẵn sàng chạy predict / bench / eval.")


def cmd_models(a):
    from .registry import cached_models, load_registry
    reg = load_registry()
    have = cached_models()
    print(f"{'tên':30s} {'trên máy':10s} {'kích thước':>12s}  nguồn")
    for name, e in reg["models"].items():
        p = have.get(name)
        size = f"{p.stat().st_size / 1e6:.1f} MB" if p else (f"{e['size_mb']} MB" if e.get("size_mb") else "?")
        print(f"{name:30s} {'có' if p else '—':10s} {size:>12s}  {'cache/local' if p else e['url']}")
    for name, p in have.items():
        if name not in reg["models"]:
            print(f"{name:30s} {'có':10s} {p.stat().st_size / 1e6:10.1f} MB  {p}")
    if reg.get("aliases"):
        print("\nTên tắt:", ", ".join(f"{k} -> {v}" for k, v in reg["aliases"].items()))


def cmd_download(a):
    from .pack import open_pack
    for name in a.models:
        pack = open_pack(name, a.models_dir)
        print(f"  {pack.name}: sẵn sàng ({pack.dir})")


def cmd_predict(a):
    from .runners import predict
    kw = dict(model=a.model, detector=a.detector, out_dir=a.out_dir, report=not a.no_report,
              threads=a.threads, margin=a.margin, min_face=a.min_face, max_faces=a.max_faces,
              align=a.align, det_conf=a.det_conf, det_width=a.det_width, models_dir=a.models_dir,
              method=a.smoothing, name=a.name)
    if a.alpha is not None:
        kw["alpha"] = a.alpha
    if a.window is not None:
        kw["window"] = a.window
    from .runners import source_kind
    kind = source_kind(a.source)
    if kind in ("image", "folder"):
        kw.update(save_max=a.save_max, labels=not a.no_labels)
    elif kind == "video":
        kw.update(save_video=not a.no_save_video, max_frames=a.max_frames, show=a.show)
    else:
        kw.update(width=a.width, height=a.height, seconds=a.seconds, record=a.record, show=not a.no_show)
    run = predict(a.source, **kw)
    print(f"\nKết quả: {run}")
    if not a.no_report:
        print(f"Báo cáo: {run / 'report' / 'report.md'}")


def cmd_bench(a):
    from .bench import bench
    bench(a.models, threads=a.threads, warmup=a.warmup, runs=a.runs, cool_to=a.cool_to, pause=a.pause,
          soak=a.soak, window=a.window, out_dir=a.results_dir, models_dir=a.models_dir)


def cmd_eval(a):
    from .evaluate import evaluate
    evaluate(a.models, threads=a.threads, out_dir=a.results_dir, models_dir=a.models_dir, limit=a.limit)


def cmd_report(a):
    from .report import build_report, compare
    for r in a.runs:
        build_report(r)
    if len(a.runs) > 1:
        compare(a.runs, Path(a.runs[0]).parent)


def cmd_summarize(a):
    from .summarize import summarize
    summarize(a.models, results_dir=a.results_dir, threads=a.threads, rt_fps=a.rt_fps, models_dir=a.models_dir)


def cmd_pack(a):
    import json

    from .pack import convert_export_dir, registry_entry
    from .registry import load_registry, save_registry
    out = Path(a.out)
    made = []
    for d in a.dirs:
        d = Path(d)
        for meta in sorted(d.rglob("meta.json")) or []:
            print(f"{meta.parent}:")
            made += convert_export_dir(meta.parent, out, include_testset=not a.no_testset)
    if not made:
        raise SystemExit(f"Không thấy meta.json nào trong {a.dirs}")
    reg = load_registry()
    base = a.base_url or reg.get("release_base_url")
    entries = dict(registry_entry(f, base) for f in made)
    print("\nMục cho models.json:\n" + json.dumps(entries, indent=2))
    if a.update_registry:
        reg["models"].update(entries)
        save_registry(reg)
        print(f"\nĐã ghi vào {Path(__file__).with_name('models.json')}")
    print(f"\nTiếp theo: tải {len(made)} file .ferpack trong {out} lên GitHub Releases "
          f"(Releases -> Draft a new release -> kéo thả file).")


# ----------------------------------------------------------------------------- parser
def build_parser():
    ap = argparse.ArgumentParser(prog="fer-edge", description="Nhận dạng biểu cảm khuôn mặt trên thiết bị biên")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, model_default="resnet18"):
        p.add_argument("--model", default=model_default, help="tên mô hình hoặc đường dẫn .ferpack")
        p.add_argument("--threads", type=int, default=0, help="số luồng CPU (0 = mặc định của ONNX Runtime)")
        p.add_argument("--models-dir", help="thư mục chứa .ferpack cục bộ")

    p = sub.add_parser("doctor", help="kiểm tra môi trường và chạy thử")
    common(p, None)
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("models", help="liệt kê mô hình trong registry và trên máy")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("download", help="tải mô hình về máy")
    p.add_argument("models", nargs="+")
    p.add_argument("--models-dir")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("predict", help="chạy trên ảnh / thư mục ảnh / video / webcam")
    common(p)
    p.add_argument("--source", required=True, help="ảnh, thư mục, video, hoặc số hiệu webcam (0)")
    p.add_argument("--detector", default="yunet", choices=["yunet", "yolo", "none"],
                   help="none = ảnh đã là khuôn mặt cắt sẵn (FERPlus 48x48, RAF-DB)")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--name", help="thêm tên vào thư mục kết quả")
    p.add_argument("--margin", type=float, default=0.15, help="nới khung mặt mỗi phía")
    p.add_argument("--min-face", type=int, default=40)
    p.add_argument("--max-faces", type=int, default=5)
    p.add_argument("--align", action="store_true", help="xoay cho hai mắt ngang nhau trước khi cắt")
    p.add_argument("--det-conf", type=float, default=0.6)
    p.add_argument("--det-width", type=int, default=320)
    p.add_argument("--smoothing", default="ema", choices=["ema", "majority", "none"])
    p.add_argument("--alpha", type=float)
    p.add_argument("--window", type=int)
    p.add_argument("--save-max", type=int, default=200, help="[ảnh] số ảnh có chú thích tối đa được lưu")
    p.add_argument("--no-labels", action="store_true", help="[ảnh] không lấy nhãn thật từ tên thư mục")
    p.add_argument("--max-frames", type=int, default=0, help="[video]")
    p.add_argument("--no-save-video", action="store_true", help="[video]")
    p.add_argument("--show", action="store_true", help="[video] hiện cửa sổ khi đang xử lý")
    p.add_argument("--no-show", action="store_true", help="[webcam] không hiện cửa sổ (chạy qua SSH)")
    p.add_argument("--width", type=int, default=1280, help="[webcam]")
    p.add_argument("--height", type=int, default=720, help="[webcam]")
    p.add_argument("--seconds", type=float, default=0, help="[webcam] tự dừng sau N giây")
    p.add_argument("--record", action="store_true", help="[webcam] ghi video ngay từ đầu")
    p.add_argument("--no-report", action="store_true")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("bench", help="đo độ trễ theo giao thức cố định (Raspberry Pi)")
    p.add_argument("models", nargs="*", help="mặc định: mọi mô hình đang có trên máy")
    p.add_argument("--threads", nargs="*", type=int, default=[4])
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--runs", type=int, default=500)
    p.add_argument("--cool-to", type=float, default=50.0)
    p.add_argument("--pause", type=float, default=5.0)
    p.add_argument("--soak", type=float, default=0, help="> 0: chạy liên tục N giây để đo ảnh hưởng nhiệt")
    p.add_argument("--window", type=float, default=10.0)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--models-dir")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("eval", help="kiểm tra lại độ chính xác bằng testset kèm trong gói")
    p.add_argument("models", nargs="*")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="chỉ chạy N ảnh đầu (chạy thử nhanh)")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--models-dir")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("report", help="tạo lại báo cáo cho thư mục kết quả; nhiều thư mục -> thêm bảng so sánh")
    p.add_argument("runs", nargs="+")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("summarize", help="bảng chính + đồ thị Pareto cho bài báo")
    p.add_argument("models", nargs="*")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--threads", type=int, default=4, help="lấy số đo độ trễ ở cấu hình số luồng này")
    p.add_argument("--rt-fps", type=float, default=10, help="ngưỡng thời gian thực (đặt TRƯỚC khi đo)")
    p.add_argument("--models-dir")
    p.set_defaults(func=cmd_summarize)

    p = sub.add_parser("pack", help="đóng gói thư mục export của Kaggle thành .ferpack")
    p.add_argument("dirs", nargs="+", help="thư mục chứa meta.json + model_*.onnx")
    p.add_argument("--out", default="packs", help="thư mục ghi file .ferpack")
    p.add_argument("--base-url", help="URL gốc của GitHub Release để sinh mục registry")
    p.add_argument("--update-registry", action="store_true", help="ghi thẳng vào src/fer_edge/models.json")
    p.add_argument("--no-testset", action="store_true", help="không kèm testset (gói nhẹ hơn)")
    p.set_defaults(func=cmd_pack)
    return ap


def main(argv=None):
    _utf8()
    a = build_parser().parse_args(argv)
    try:
        a.func(a)
    except KeyboardInterrupt:
        print("\nĐã dừng.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
