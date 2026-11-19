"""Gộp mọi kết quả thành bảng chính + đồ thị Pareto (độ chính xác – độ trễ) cho bài báo.

Nguồn: độ chính xác trên TOÀN BỘ tập test lấy từ gói mô hình (đo lúc export);
       độ trễ lấy từ results/latency.csv; kiểm tra lại trên thiết bị lấy từ results/accuracy_device.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import VARIANT_SUFFIX
from .pack import open_pack
from .registry import cached_models

INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e3"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = {"fp32": "o", "int8_dyn": "s", "int8_static": "^"}
VARIANT_LABEL = {"fp32": "FP32", "int8_dyn": "INT8 động", "int8_static": "INT8 tĩnh"}
MODEL_LABEL = {"resnet18": "ResNet-18", "apvit": "APViT", "vit_b16": "ViT-B/16",
               "posterv2": "POSTER V2", "clip_lora": "CLIP+LoRA"}


def pack_accuracy(models=None, models_dir=None):
    """Độ chính xác trên toàn bộ tập test, đọc từ fer.json của từng gói."""
    rows = []
    for name in (models or sorted(cached_models())):
        try:
            p = open_pack(name, models_dir)
        except Exception as e:
            print(f"  bỏ qua {name}: {type(e).__name__}: {e}")
            continue
        m, info = p.meta, p.info()
        met = m.get("metrics") or {}
        ref = m.get("metrics_torch_fp32") or {}
        rows.append({"pack": p.name, "model": info["model"], "dataset": info["dataset"], "variant": info["variant"],
                     "params_M": info["params_M"], "size_mb": info["size_mb"], "img_size": info["img_size"],
                     "n_test": m.get("n_test"), "acc": met.get("acc"), "mca": met.get("mca"), "f1": met.get("f1"),
                     "acc_torch": ref.get("acc"),
                     "acc_drop_vs_torch": (round(ref["acc"] - met["acc"], 2)
                                           if met.get("acc") is not None and ref.get("acc") is not None else None),
                     "agree_torch": m.get("agree_torch")})
    return pd.DataFrame(rows)


def latency_table(path, threads):
    lat = pd.read_csv(path)
    if "threads" in lat:
        sub = lat[lat.threads == threads]
        if sub.empty:
            print(f"  Cảnh báo: latency.csv không có số đo với --threads {threads} "
                  f"(có: {sorted(lat.threads.unique())}); dùng toàn bộ.")
        else:
            lat = sub
    lat = lat.sort_values("time").groupby("pack", as_index=False).last()
    keep = [c for c in ("pack", "lat_median_ms", "lat_p95_ms", "fps_median", "model_mem_mb", "temp_end",
                        "throttled_end") if c in lat]
    return lat[keep]


def pareto_mask(lat, acc):
    keep = np.ones(len(lat), bool)
    for i in range(len(lat)):
        dom = (lat <= lat[i]) & (acc >= acc[i]) & ((lat < lat[i]) | (acc > acc[i]))
        keep[i] = not dom.any()
    return keep


def plot_pareto(df, out, rt_fps=10, title_extra=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRID,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10})
    d = df[df.lat_median_ms.notna() & df.acc.notna()].copy()
    if d.empty:
        return None
    models = list(dict.fromkeys(d.model))
    color = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    if rt_fps:
        ax.axvline(1000 / rt_fps, color=MUTED, ls="--", lw=1)
        ax.text(1000 / rt_fps, 0.02, f" ngưỡng {rt_fps:g} FPS ({1000 / rt_fps:g} ms)",
                transform=ax.get_xaxis_transform(), color=MUTED, fontsize=9, va="bottom", ha="left")
    for m in models:
        dm = d[d.model == m].sort_values("lat_median_ms")
        ax.plot(dm.lat_median_ms, dm.acc, color=color[m], lw=1, alpha=0.5)
        for _, r in dm.iterrows():
            ax.scatter(r.lat_median_ms, r.acc, s=64, marker=MARKERS.get(r.variant, "o"), color=color[m],
                       edgecolor="white", linewidth=1.5, zorder=3)
        best = dm.loc[dm.acc.idxmax()]
        ax.annotate(MODEL_LABEL.get(m, m), (best.lat_median_ms, best.acc), xytext=(6, 6),
                    textcoords="offset points", fontsize=9, color=INK)
    front = d[pareto_mask(d.lat_median_ms.to_numpy(), d.acc.to_numpy())].sort_values("lat_median_ms")
    ax.step(front.lat_median_ms, front.acc, where="post", color=INK, lw=1, ls=":", zorder=2)
    ax.set_xscale("log")
    from matplotlib.ticker import FuncFormatter, LogLocator
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_xlabel("độ trễ trung vị, batch 1 (ms, thang log)")
    ax.set_ylabel("accuracy trên toàn bộ tập test (%)")
    handles = [plt.Line2D([], [], marker=MARKERS[v], ls="", color=MUTED, markersize=8, label=VARIANT_LABEL[v])
               for v in MARKERS if v in set(d.variant)]
    handles += [plt.Line2D([], [], color=color[m], lw=3, label=MODEL_LABEL.get(m, m)) for m in models]
    handles.append(plt.Line2D([], [], color=INK, lw=1, ls=":", label="biên Pareto"))
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title(f"Đánh đổi độ chính xác – độ trễ{title_extra}", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def summarize(models=None, results_dir="results", threads=4, rt_fps=10, models_dir=None):
    res = Path(results_dir)
    df = pack_accuracy(models, models_dir)
    if df.empty:
        raise SystemExit("Không đọc được gói mô hình nào. Chạy `fer-edge download <tên>` trước.")
    lat_path = res / "latency.csv"
    if lat_path.is_file():
        df = df.merge(latency_table(lat_path, threads), on="pack", how="left")
    else:
        print(f"Chưa có {lat_path} -> bảng chỉ có độ chính xác. Chạy `fer-edge bench` trước.")
        df["lat_median_ms"] = np.nan
    dev_path = res / "accuracy_device.csv"
    if dev_path.is_file():
        dev = pd.read_csv(dev_path).sort_values("time").groupby("pack", as_index=False).last()
        keep = [c for c in ("pack", "acc", "agree_export_same_variant") if c in dev]
        df = df.merge(dev[keep].rename(columns={"acc": "acc_device_subset"}), on="pack", how="left")
    if "fps_median" in df:
        df["realtime"] = df.fps_median >= rt_fps
    order = {v: i for i, v in enumerate(MARKERS)}
    df["_v"] = df.variant.map(order).fillna(9)
    df = df.sort_values(["model", "_v"]).drop(columns="_v")
    res.mkdir(parents=True, exist_ok=True)
    df.to_csv(res / "table_main.csv", index=False)
    try:
        (res / "table_main.md").write_text(df.to_markdown(index=False, floatfmt=".2f"), encoding="utf-8")
    except ImportError:
        pass
    with pd.option_context("display.width", 220, "display.max_columns", 40):
        print(df.to_string(index=False))
    for ds in sorted(set(df.dataset.dropna())):
        out = plot_pareto(df[df.dataset == ds], res / f"pareto_{ds}.png", rt_fps, f" ({ds})")
        if out:
            print("Đồ thị:", out)
    print("Bảng:", res / "table_main.md")
    return df
