"""Đo độ trễ suy luận theo một giao thức cố định (dùng cho Raspberry Pi 5, chạy được cả trên laptop).

Giao thức (ghi được thẳng vào bài báo):
  - batch 1, ảnh thật lấy từ testset trong gói, ONNX Runtime CPU, ORT_ENABLE_ALL;
  - mỗi cấu hình chạy trong MỘT TIẾN TRÌNH RIÊNG (đo RAM sạch, không ảnh hưởng lẫn nhau);
  - chờ SoC nguội dưới --cool-to °C; làm nóng --warmup lần rồi đo --runs lần;
  - báo cáo trung vị và p95 (ms), FPS = 1000 / trung vị; ghi nhiệt độ, cờ giảm xung, tần số CPU.
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .pack import open_pack
from .registry import cached_models
from .sysinfo import append_csv, cpu_freq_mhz, pctl, read_temp, read_throttled, system_info, wait_cool


def sample_input(pack):
    """Một ảnh thật trong testset của gói; không có thì dùng nhiễu."""
    from PIL import Image

    from .model import preprocess_pil
    ts = pack.testset
    files = sorted((ts / "images").glob("*")) if ts else []
    if files:
        return preprocess_pil(Image.open(files[0]), pack.meta)
    s = pack.img_size
    return np.random.default_rng(0).standard_normal((1, 3, s, s)).astype(np.float32)


# ----------------------------------------------------------------- tiến trình con: đo một cấu hình
def _worker(a):
    import resource

    from .model import make_session
    pack = open_pack(a.pack)
    x = sample_input(pack)
    rss_base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024        # MB trên Linux
    t0 = time.perf_counter()
    sess = make_session(pack.onnx, a.threads)
    load_s = time.perf_counter() - t0
    feed = {sess.get_inputs()[0].name: x}

    temp0, thr0 = read_temp(), read_throttled()
    for _ in range(a.warmup):
        sess.run(None, feed)

    if a.soak > 0:
        return _soak(a, sess, feed, temp0, thr0)

    lat = np.empty(a.runs)
    for i in range(a.runs):
        t = time.perf_counter_ns()
        sess.run(None, feed)
        lat[i] = (time.perf_counter_ns() - t) / 1e6
    out = {"load_s": round(load_s, 3), "lat_median_ms": round(pctl(lat, 50), 3),
           "lat_p95_ms": round(pctl(lat, 95), 3), "lat_mean_ms": round(float(lat.mean()), 3),
           "lat_std_ms": round(float(lat.std()), 3), "lat_min_ms": round(float(lat.min()), 3),
           "fps_median": round(1000 / pctl(lat, 50), 2), "rss_base_mb": round(rss_base, 1),
           "rss_peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
           "temp_start": temp0, "temp_end": read_temp(), "throttled_start": thr0,
           "throttled_end": read_throttled(), "freq_end_mhz": cpu_freq_mhz()}
    out["model_mem_mb"] = round(out["rss_peak_mb"] - rss_base, 1)
    if a.raw_out:
        np.save(a.raw_out, lat)
    print("RESULT " + json.dumps(out))


def _soak(a, sess, feed, temp0, thr0):
    """Chạy liên tục a.soak giây; mỗi a.window giây ghi FPS, nhiệt độ, tần số, cờ giảm xung."""
    import pandas as pd
    rows, t_start = [], time.time()
    while time.time() - t_start < a.soak:
        n, w0 = 0, time.perf_counter()
        while time.perf_counter() - w0 < a.window:
            sess.run(None, feed)
            n += 1
        rows.append({"t_s": round(time.time() - t_start, 1), "fps": round(n / (time.perf_counter() - w0), 2),
                     "temp": read_temp(), "freq_mhz": cpu_freq_mhz(), "throttled": read_throttled()})
        r = rows[-1]
        print(f"  t={r['t_s']:6.0f}s  fps {r['fps']:7.2f}  {r['temp']:.1f}°C  {r['freq_mhz']:.0f} MHz  "
              f"throttled {r['throttled']}", file=sys.stderr, flush=True)
    pd.DataFrame(rows).to_csv(a.raw_out, index=False)
    fps = [r["fps"] for r in rows]
    k = max(1, len(fps) // 10)
    out = {"soak_s": a.soak, "fps_first": round(float(np.mean(fps[:k])), 2),
           "fps_last": round(float(np.mean(fps[-k:])), 2), "temp_start": temp0,
           "temp_max": max(r["temp"] for r in rows), "throttled_start": thr0, "throttled_end": read_throttled(),
           "freq_min_mhz": min(r["freq_mhz"] for r in rows)}
    out["fps_drop_pct"] = round(100 * (out["fps_first"] - out["fps_last"]) / max(out["fps_first"], 1e-9), 2)
    print("RESULT " + json.dumps(out))


# ----------------------------------------------------------------- tiến trình cha
def _run_one(pack_name, pack_dir, threads, a, out_root):
    tag = f"{pack_name}_t{threads}"
    raw_dir = Path(out_root) / ("soak" if a.soak else "raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw = raw_dir / (f"{tag}.csv" if a.soak else f"{tag}.npy")
    cmd = [sys.executable, "-m", "fer_edge.bench", "--worker", "--pack", str(pack_dir), "--threads", str(threads),
           "--warmup", str(a.warmup), "--runs", str(a.runs), "--raw-out", str(raw),
           "--soak", str(a.soak), "--window", str(a.window)]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    lines = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    if p.returncode != 0 or not lines:
        print(f"  LỖI khi đo {tag} (mã {p.returncode})")
        print("\n".join(p.stdout.splitlines()[-5:]))
        return None
    return json.loads(lines[-1][7:])


def bench(models=None, threads=(4,), warmup=50, runs=500, cool_to=50.0, pause=5.0, soak=0, window=10.0,
          out_dir="results", models_dir=None):
    """Đo một hoặc nhiều gói mô hình. models=None -> mọi gói đang có sẵn trên máy."""
    names = list(models) if models else sorted(cached_models())
    if not names:
        raise SystemExit("Chưa có gói mô hình nào trên máy. Chạy: fer-edge download resnet18-ferplus-fp32 ...")
    packs = [open_pack(n, models_dir) for n in names]
    sysinfo = system_info()
    print("Hệ thống:", {k: sysinfo[k] for k in ("board", "os_name", "onnxruntime", "ram_gb", "governor") if k in sysinfo})
    if sysinfo.get("governor") not in ("performance", "n/a"):
        print(f"  Lưu ý: CPU governor đang là '{sysinfo['governor']}'. Trên Pi nên đặt 'performance' để số đo ổn định:\n"
              f"  echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")
    out_root = Path(out_dir)
    out_csv = out_root / ("soak.csv" if soak else "latency.csv")
    configs = [(p, t) for p in packs for t in threads]
    print(f"{len(configs)} cấu hình -> {out_csv}\n")

    for i, (pack, th) in enumerate(configs, 1):
        wait_cool(cool_to)
        print(f"[{i}/{len(configs)}] {pack.name:28s} {th} luồng ...", flush=True)
        a = argparse.Namespace(warmup=warmup, runs=runs, soak=soak, window=window)
        res = _run_one(pack.name, pack.dir, th, a, out_root)
        if res is None:
            continue
        info = pack.info()
        row = {"time": datetime.now().isoformat(timespec="seconds"), "pack": pack.name, "model": info["model"],
               "dataset": info["dataset"], "variant": info["variant"], "threads": th, "img_size": info["img_size"],
               "params_M": info["params_M"], "size_mb": info["size_mb"], "warmup": warmup, "runs": runs, **res,
               **{f"sys_{k}": v for k, v in sysinfo.items()}}
        append_csv(out_csv, [row])
        if soak:
            print(f"    FPS {res['fps_first']} -> {res['fps_last']} (giảm {res['fps_drop_pct']}%), "
                  f"nhiệt tối đa {res['temp_max']}°C, throttled {res['throttled_end']}")
        else:
            print(f"    trung vị {res['lat_median_ms']:.2f} ms | p95 {res['lat_p95_ms']:.2f} ms | "
                  f"{res['fps_median']:.1f} FPS | RAM mô hình ~{res['model_mem_mb']:.0f} MB | "
                  f"{res['temp_start']}->{res['temp_end']}°C | throttled {res['throttled_end']}")
        time.sleep(pause)
    print("\nXong. Kết quả:", out_csv)
    return out_csv


if __name__ == "__main__":                      # tiến trình con do _run_one gọi
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--pack")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--runs", type=int, default=500)
    ap.add_argument("--raw-out")
    ap.add_argument("--soak", type=float, default=0)
    ap.add_argument("--window", type=float, default=10.0)
    _worker(ap.parse_args())
