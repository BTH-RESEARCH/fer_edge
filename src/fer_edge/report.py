"""Tổng kết và trực quan hóa một lần chạy (ảnh / video / webcam), và bảng so sánh nhiều lần chạy.

Mỗi lần chạy -> <run>/report/: report.md (đọc trực tiếp hoặc chép vào báo cáo), summary.json, các hình .png.
Nhiều lần chạy -> outputs/summary_<thời gian>.md + .csv (một dòng mỗi lần chạy)."""
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .constants import EMOTION_COLORS

INK, MUTED, GRID, BLUE = "#222222", "#6b6b6b", "#e6e6e3", "#2a78d6"
STAGES = [("detect_ms", "phát hiện mặt"), ("preprocess_ms", "cắt + tiền xử lý"), ("classify_ms", "ResNet-18")]


# ----------------------------------------------------------------------------- tính toán
def _stats(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {"median": np.nan, "p95": np.nan, "mean": np.nan}
    return {"median": float(s.median()), "p95": float(s.quantile(0.95)), "mean": float(s.mean())}


def cls_metrics(y, p, classes):
    k = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    yi = np.array([idx[v] for v in y])
    pi = np.array([idx[v] for v in p])
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (yi, pi), 1)
    tp = np.diag(cm).astype(float)
    rec = tp / np.maximum(cm.sum(1), 1)
    prec = tp / np.maximum(cm.sum(0), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    present = cm.sum(1) > 0                                  # chỉ tính trung bình trên các lớp có trong dữ liệu
    return {"acc": 100 * float((yi == pi).mean()), "mca": 100 * float(rec[present].mean()),
            "f1": 100 * float(f1[present].mean()), "cm": cm, "recall": rec, "support": cm.sum(1)}


def switches(labels, times):
    """Số lần đổi nhãn / giây trên các khung có mặt (thước đo 'giật nhãn')."""
    lab = [l for l in labels if isinstance(l, str) and l]
    if len(lab) < 2:
        return np.nan, np.nan
    n = sum(a != b for a, b in zip(lab, lab[1:]))
    dur = max(float(times.max() - times.min()), 1e-6)
    return n / dur, (dur / (n + 1))


def summarize(run):
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    fr = pd.read_csv(run / "frames.csv")
    classes = meta["model"]["classes"]
    has = fr["n_faces"] > 0
    S = {"run": run.name, "scenario": meta["scenario"], "source": meta["source"], "device": meta["env"].get("device", "CPU"),
         "detector": meta["detector"]["type"], "n_items": int(len(fr)), "wall_s": meta.get("wall_s"),
         "detection_rate_pct": round(100 * float(has.mean()), 2) if len(fr) else np.nan,
         "faces_per_item": round(float(fr["n_faces"].mean()), 2) if len(fr) else np.nan}
    lat = {}
    for col, _ in STAGES + [("total_ms", "tổng")]:
        lat[col] = _stats(fr.loc[has, col] if col in ("preprocess_ms", "classify_ms") else fr[col])
    S["latency"] = {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in lat.items()}
    S["proc_fps"] = round(1000 / lat["total_ms"]["median"], 2) if lat["total_ms"]["median"] > 0 else np.nan
    if "loop_fps" in fr:
        S["loop_fps_median"] = round(float(fr["loop_fps"].iloc[1:].median()), 2) if len(fr) > 1 else np.nan
        S["capture_ms_median"] = round(float(fr["capture_ms"].iloc[1:].median()), 2) if len(fr) > 1 else np.nan
    pred_col = "smooth_pred" if "smooth_pred" in fr and meta["scenario"] != "image" else "pred"
    preds = fr.loc[has, pred_col].dropna()
    dist = preds.value_counts(normalize=True).reindex(classes).fillna(0) * 100
    S["emotion_pct"] = {c: round(float(v), 2) for c, v in dist.items()}
    S["emotion_source"] = "đã làm mượt" if pred_col == "smooth_pred" else "từng ảnh"
    S["top_emotion"] = dist.idxmax() if len(preds) else ""
    conf = fr.loc[has, "conf"]
    S["conf_mean"] = round(float(conf.mean()), 3) if len(conf) else np.nan
    S["low_conf_pct"] = round(100 * float((conf < 0.5).mean()), 2) if len(conf) else np.nan
    faces_csv = run / "faces.csv"
    if faces_csv.exists() and faces_csv.stat().st_size > 5:
        fc = pd.read_csv(faces_csv)
        if "face_px" in fc and len(fc):
            S["face_px_median"] = round(float(fc["face_px"].median()), 1)

    if meta["scenario"] == "image" and "gt" in fr and fr["gt"].fillna("").astype(str).str.len().gt(0).any():
        lab = fr[fr["gt"].fillna("").astype(str).str.len() > 0]
        det = lab[lab["n_faces"] > 0]
        S["n_labeled"] = int(len(lab))
        S["labeled_detected_pct"] = round(100 * len(det) / len(lab), 2)
        if len(det):
            m = cls_metrics(det["gt"].tolist(), det["pred"].tolist(), classes)
            S["acc"], S["mca"], S["f1"] = round(m["acc"], 2), round(m["mca"], 2), round(m["f1"], 2)
            S["acc_end2end"] = round(100 * float((lab["pred"] == lab["gt"]).mean()), 2)   # không thấy mặt = sai
            S["per_class_recall"] = {c: round(100 * float(r), 2) for c, r, n in zip(classes, m["recall"], m["support"]) if n}
            S["_cm"] = m["cm"]

    if meta["scenario"] in ("video", "webcam") and "time_s" in fr:
        f = fr[has]
        S["duration_s"] = round(float(fr["time_s"].max()), 2) if len(fr) else np.nan
        if len(f) > 1:
            r_raw, seg_raw = switches(f["pred"].tolist(), f["time_s"])
            S["switch_per_s_raw"], S["segment_s_raw"] = round(r_raw, 3), round(seg_raw, 3)
            if "smooth_pred" in f:
                r_sm, seg_sm = switches(f["smooth_pred"].tolist(), f["time_s"])
                S["switch_per_s_smooth"], S["segment_s_smooth"] = round(r_sm, 3), round(seg_sm, 3)
    return S, meta, fr


# ----------------------------------------------------------------------------- hình
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRID,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
                         "axes.titlesize": 11, "axes.titlelocation": "left", "axes.titlecolor": INK,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    return plt


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path.name


def fig_latency(S, out):
    plt = _plt()
    rows = [(lbl, S["latency"][c]) for c, lbl in STAGES if not math.isnan(S["latency"][c]["median"])]
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(rows) + 1.3))
    y = np.arange(len(rows))[::-1]
    med = [r[1]["median"] for r in rows]
    p95 = [r[1]["p95"] for r in rows]
    ax.barh(y, med, height=0.5, color=BLUE)
    ax.scatter(p95, y, marker="|", s=260, color=INK, linewidths=1.5, zorder=3, label="p95")
    for yi, m, p in zip(y, med, p95):
        ax.text(max(m, p) + max(p95) * 0.02, yi, f"{m:.1f} ms  (p95 {p:.1f})", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(0, max(p95) * 1.45)
    ax.set_xlabel("ms / khung hình (thanh = trung vị, vạch = p95)")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    t = S["latency"]["total_ms"]
    ax.set_title(f"Thời gian xử lý từng khâu — tổng trung vị {t['median']:.1f} ms ≈ {S['proc_fps']:.1f} FPS ({S['device']})")
    return _save(fig, out / "fig_latency.png")


def fig_emotions(S, out):
    plt = _plt()
    d = S["emotion_pct"]
    names = list(d)
    vals = [d[c] for c in names]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(names))
    ax.bar(x, vals, width=0.6, color=[EMOTION_COLORS.get(c, MUTED) for c in names], edgecolor="white", linewidth=2)
    for xi, v in zip(x, vals):
        ax.text(xi, v + max(vals) * 0.02, f"{v:.1f}%", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, names, rotation=0)
    ax.set_ylabel("% " + ("thời gian có mặt" if S["scenario"] != "image" else "ảnh có mặt"))
    ax.set_ylim(0, max(vals) * 1.18 + 1)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.set_title(f"Phân bố cảm xúc dự đoán ({S['emotion_source']}, mặt chính)")
    return _save(fig, out / "fig_emotions.png")


def fig_confidence(fr, S, out):
    plt = _plt()
    c = fr.loc[fr["n_faces"] > 0, "conf"].dropna()
    if c.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.hist(c, bins=np.linspace(0, 1, 21), color=BLUE, edgecolor="white", linewidth=1.5)
    ax.axvline(c.median(), color=INK, lw=1, ls="--")
    ax.text(c.median(), ax.get_ylim()[1] * 0.95, f" trung vị {c.median():.2f}", color=INK, fontsize=9, va="top")
    ax.set_xlabel("độ tin cậy của nhãn dự đoán (xác suất lớn nhất, từng khung/ảnh)")
    ax.set_ylabel("số lượng")
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.set_title(f"Độ tin cậy — {S['low_conf_pct']:.1f}% dưới 0,5")
    return _save(fig, out / "fig_confidence.png")


def fig_confusion(S, classes, out):
    plt = _plt()
    cm = S["_cm"]
    keep = [i for i in range(len(classes)) if cm[i].sum() > 0]
    cmn = cm[keep] / np.maximum(cm[keep].sum(1, keepdims=True), 1) * 100
    fig, ax = plt.subplots(figsize=(0.75 * len(classes) + 2.2, 0.62 * len(keep) + 1.8))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(len(keep)):
        for j in range(len(classes)):
            v = cmn[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color="white" if v > 55 else INK)
    ax.set_xticks(range(len(classes)), classes, rotation=40, ha="right")
    ax.set_yticks(range(len(keep)), [f"{classes[i]} (n={cm[i].sum()})" for i in keep])
    ax.set_xlabel("dự đoán")
    ax.set_ylabel("nhãn thật")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Ma trận nhầm lẫn (% theo hàng) — acc {S['acc']:.1f}% | MCA {S['mca']:.1f}% | F1 {S['f1']:.1f}%")
    return _save(fig, out / "fig_confusion.png")


def fig_timeline(fr, S, classes, out):
    plt = _plt()
    from matplotlib.patches import Patch
    t = fr["time_s"].to_numpy()
    dt = np.median(np.diff(t)) if len(t) > 1 else 0.04
    fig, axes = plt.subplots(3, 1, figsize=(10, 4.6), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1.6]})
    for ax, col, title in ((axes[0], "pred", "từng khung (thô)"), (axes[1], "smooth_pred", "đã làm mượt")):
        if col not in fr:
            ax.set_visible(False)
            continue
        labs = fr[col].fillna("").astype(str).to_numpy()
        start = 0
        for i in range(1, len(labs) + 1):                       # gộp các khung liên tiếp cùng nhãn thành 1 đoạn
            if i == len(labs) or labs[i] != labs[start]:
                if labs[start]:
                    ax.barh(0, t[i - 1] - t[start] + dt, left=t[start], height=0.8,
                            color=EMOTION_COLORS.get(labs[start], MUTED), linewidth=0)
                start = i
        ax.set_yticks([0], [title])
        ax.set_ylim(-0.5, 0.5)
        ax.tick_params(axis="y", length=0)
        for s in ("left", "bottom"):
            ax.spines[s].set_visible(False)
    ax = axes[2]
    conf_col = "smooth_conf" if "smooth_conf" in fr else "conf"
    ax.plot(t, fr[conf_col], color=INK, lw=1.2, label="độ tin cậy (đã làm mượt)" if conf_col == "smooth_conf" else "độ tin cậy")
    no_face = fr["n_faces"].to_numpy() == 0
    if no_face.any():
        ax.fill_between(t, 0, 1, where=no_face, color=GRID, step="mid", label="không thấy mặt")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("độ tin cậy")
    ax.set_xlabel("thời gian (s)")
    ax.grid(axis="y", color=GRID)
    present = [c for c in classes if (fr.get("pred", pd.Series(dtype=str)) == c).any() or
               (fr.get("smooth_pred", pd.Series(dtype=str)) == c).any()]
    handles = [Patch(color=EMOTION_COLORS.get(c, MUTED), label=c) for c in present]
    h2, l2 = ax.get_legend_handles_labels()
    axes[0].legend(handles=handles, ncol=min(8, len(handles)), frameon=False, fontsize=8,
                   loc="lower left", bbox_to_anchor=(0, 0.95), borderaxespad=0)
    ax.legend(h2, l2, frameon=False, fontsize=8, loc="lower right")
    sw = S.get("switch_per_s_raw", np.nan)
    sws = S.get("switch_per_s_smooth", np.nan)
    fig.suptitle(f"Diễn biến cảm xúc của mặt chính — đổi nhãn {sw:.2f} lần/s (thô) → {sws:.2f} lần/s (làm mượt)",
                 x=0.01, ha="left", fontsize=11, color=INK, y=1.0)
    return _save(fig, out / "fig_timeline.png")


def fig_speed_time(fr, S, out):
    plt = _plt()
    t = fr["time_s"].to_numpy()
    y = fr["total_ms"].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.plot(t, y, color=BLUE, lw=0.8, alpha=0.35, label="từng khung")
    w = max(5, len(y) // 50)
    ax.plot(t, pd.Series(y).rolling(w, center=True, min_periods=1).median(), color=BLUE, lw=2, label=f"trung vị trượt ({w} khung)")
    ax.set_ylabel("ms / khung")
    ax.set_xlabel("thời gian (s)")
    ax.set_ylim(0, np.nanpercentile(y, 99) * 1.2)
    ax.grid(axis="y", color=GRID)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("Thời gian xử lý theo thời gian (phát hiện + phân loại)")
    return _save(fig, out / "fig_speed_time.png")


def _thumb(img, w=320):
    h = int(img.shape[0] * w / img.shape[1])
    return cv2.cvtColor(cv2.resize(img, (w, h)), cv2.COLOR_BGR2RGB)


def gallery_images(run, fr, S, out, n=12):
    files = sorted((run / "annotated").glob("*.jpg"))
    if not files:
        return None
    by_idx = {int(f.name.split("_")[0]): f for f in files}
    sub = fr[fr["frame"].isin(by_idx)].copy()
    pick = []
    if "gt" in sub and sub["gt"].fillna("").astype(str).str.len().gt(0).any():
        wrong = sub[(sub["gt"].fillna("") != "") & (sub["pred"] != sub["gt"])]
        right = sub[(sub["gt"].fillna("") != "") & (sub["pred"] == sub["gt"])]
        pick = right.groupby("pred").head(1)["frame"].tolist()[: n - n // 3] + wrong["frame"].head(n // 3).tolist()
    else:
        pick = sub[sub["n_faces"] > 0].groupby("pred").head(2)["frame"].tolist()
    pick = (pick + [i for i in sub["frame"] if i not in pick])[:n]
    return _grid(run, [(by_idx[i], sub.set_index("frame").loc[i]) for i in pick], out / "fig_gallery.png",
                 "Ảnh minh họa (khung + nhãn dự đoán)" + (" — hàng cuối: một số ảnh dự đoán sai" if "gt" in sub and S.get("acc") else ""))


def gallery_video(run, meta, out, n=8):
    src = run / (meta.get("annotated_video") or "recording.mp4")
    items = []
    if src.exists():
        cap = cv2.VideoCapture(str(src))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        for k in np.linspace(0, max(total - 1, 0), n).astype(int) if total else []:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
            ok, img = cap.read()
            if ok:
                items.append((img, f"t = {k / fps:.1f} s"))
        cap.release()
    else:
        for p in sorted((run / "snapshots").glob("*.jpg"))[:n]:
            items.append((cv2.imread(str(p)), p.stem))
    if not items:
        return None
    plt = _plt()
    cols = 4
    rows = math.ceil(len(items) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.0 * rows + 0.4))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, (img, title) in zip(np.atleast_1d(axes).ravel(), items):
        ax.imshow(_thumb(img))
        ax.set_title(title, fontsize=8, color=MUTED)
    fig.suptitle("Khung hình minh họa từ video kết quả", x=0.01, ha="left", fontsize=11, color=INK)
    return _save(fig, out / "fig_frames.png")


def _grid(run, items, path, title, cols=4):
    plt = _plt()
    rows = math.ceil(len(items) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.4 * rows + 0.3))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, (f, r) in zip(np.atleast_1d(axes).ravel(), items):
        ax.imshow(_thumb(cv2.imread(str(f)), 300))
        gt = str(r.get("gt", "") or "")
        txt = f"{r['pred'] or '(không thấy mặt)'}" + (f" | thật: {gt}" if gt else "")
        ok = (not gt) or gt == r["pred"]
        ax.set_title(txt, fontsize=8, color=INK if ok else "#c0392b")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    return _save(fig, path)


# ----------------------------------------------------------------------------- markdown
def _fmt(v, nd=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def build_report(run):
    run = Path(run)
    S, meta, fr = summarize(run)
    classes = meta["model"]["classes"]
    out = run / "report"
    out.mkdir(exist_ok=True)
    figs = {"latency": fig_latency(S, out), "emotions": fig_emotions(S, out), "confidence": fig_confidence(fr, S, out)}
    if "_cm" in S:
        figs["confusion"] = fig_confusion(S, classes, out)
    if S["scenario"] == "image":
        figs["gallery"] = gallery_images(run, fr, S, out)
    else:
        figs["timeline"] = fig_timeline(fr, S, classes, out)
        figs["speed"] = fig_speed_time(fr, S, out)
        figs["frames"] = gallery_video(run, meta, out)

    sc = {"image": "Ảnh", "video": "Video", "webcam": "Webcam thời gian thực"}.get(S["scenario"], S["scenario"])
    L = [f"# Kết quả FER — kịch bản: {sc}", "",
         f"- **Nguồn**: `{S['source']}`  ",
         f"- **Mô hình**: `{meta['model'].get('pack', '?')}` — {meta['model'].get('arch', '?')}"
         f"{f' ({meta["model"]["params_M"]}M tham số)' if meta['model'].get('params_M') else ''}, "
         f"{len(classes)} lớp ({meta['model'].get('dataset', '?')}), "
         f"ảnh {meta['model']['img_size']}×{meta['model']['img_size']}, {meta['model'].get('size_mb', '?')} MB  ",
         f"- **Tiền xử lý khuôn mặt**: cắt vuông → "
         f"{'ảnh xám → ' if meta['model'].get('grayscale') else ''}"
         f"{str(meta['model']['train_res']) + '×' + str(meta['model']['train_res']) + ' → ' if meta['model'].get('train_res') else ''}"
         f"{meta['model']['img_size']}×{meta['model']['img_size']}, chuẩn hóa ImageNet (giống lúc huấn luyện)  ",
         (f"- **Phát hiện mặt**: không (ảnh đã là khuôn mặt cắt sẵn)  "
          if meta['detector']['type'] == 'none' else
          f"- **Phát hiện mặt**: {meta['detector']['type']} (`{meta['detector']['weights']}`, "
          f"conf ≥ {meta['detector']['conf']}, nới khung {meta['detector']['margin']})  "),
         f"- **Thiết bị**: {S['device']} | {meta['env'].get('board') or meta['env'].get('os', '')} | "
         f"onnxruntime {meta['env'].get('onnxruntime')}  ",
         f"- **Thời điểm**: {meta['created']}", "", "## Tóm tắt", "",
         "| Chỉ số | Giá trị |", "|---|---|",
         f"| Số {'ảnh' if S['scenario'] == 'image' else 'khung hình'} | {S['n_items']} |",
         f"| Tỉ lệ phát hiện được mặt | {_fmt(S['detection_rate_pct'], 1)}% |",
         f"| Số mặt trung bình / {'ảnh' if S['scenario'] == 'image' else 'khung'} | {_fmt(S['faces_per_item'])} |",
         f"| Thời gian xử lý trung vị (phát hiện + phân loại) | {_fmt(S['latency']['total_ms']['median'], 1)} ms ≈ **{_fmt(S['proc_fps'], 1)} FPS** |"]
    if "loop_fps_median" in S:
        L.append(f"| FPS thực tế của vòng lặp (gồm đọc camera + hiển thị) | {_fmt(S['loop_fps_median'], 1)} |")
    if "duration_s" in S:
        L.append(f"| Thời lượng | {_fmt(S['duration_s'], 1)} s |")
    L += [f"| Cảm xúc chiếm nhiều nhất | {S['top_emotion']} ({_fmt(S['emotion_pct'].get(S['top_emotion'], np.nan), 1)}%) |",
          f"| Độ tin cậy trung bình / % dưới 0,5 | {_fmt(S['conf_mean'], 3)} / {_fmt(S['low_conf_pct'], 1)}% |"]
    if "face_px_median" in S:
        L.append(f"| Cạnh khuôn mặt trung vị | {_fmt(S['face_px_median'], 0)} px |")
    if "acc" in S:
        L += [f"| Accuracy / MCA / Macro-F1 (trên ảnh có nhãn và thấy mặt) | **{S['acc']:.2f}% / {S['mca']:.2f}% / {S['f1']:.2f}%** |",
              f"| Accuracy đầu-cuối (không thấy mặt tính là sai) | {S['acc_end2end']:.2f}% ({S['labeled_detected_pct']:.1f}% ảnh có nhãn được phát hiện mặt) |"]
    if "switch_per_s_raw" in S:
        L.append(f"| Đổi nhãn mỗi giây: thô → làm mượt | {_fmt(S['switch_per_s_raw'])} → {_fmt(S.get('switch_per_s_smooth'))} |")

    L += ["", "## Tốc độ", "", "| Khâu | Trung vị (ms) | p95 (ms) | Trung bình (ms) |", "|---|---|---|---|"]
    for col, lbl in STAGES + [("total_ms", "**tổng**")]:
        st = S["latency"][col]
        L.append(f"| {lbl} | {_fmt(st['median'])} | {_fmt(st['p95'])} | {_fmt(st['mean'])} |")
    L += ["", f"![Thời gian từng khâu]({figs['latency']})"]
    if figs.get("speed"):
        L += ["", f"![Thời gian theo thời gian]({figs['speed']})"]

    L += ["", "## Cảm xúc dự đoán", "", f"![Phân bố cảm xúc]({figs['emotions']})"]
    if figs.get("confidence"):
        L += ["", f"![Độ tin cậy]({figs['confidence']})"]
    if figs.get("timeline"):
        L += ["", "## Diễn biến theo thời gian và độ ổn định nhãn", "",
              "| | Thô (từng khung) | Đã làm mượt |", "|---|---|---|",
              f"| Số lần đổi nhãn / giây | {_fmt(S.get('switch_per_s_raw'))} | {_fmt(S.get('switch_per_s_smooth'))} |",
              f"| Độ dài trung bình một đoạn cùng nhãn (s) | {_fmt(S.get('segment_s_raw'))} | {_fmt(S.get('segment_s_smooth'))} |",
              "", f"Làm mượt: `{meta['smoothing']['method']}` (alpha {meta['smoothing']['alpha']}, cửa sổ {meta['smoothing']['window']}).",
              "", f"![Diễn biến]({figs['timeline']})"]
    if figs.get("confusion"):
        L += ["", "## Độ chính xác so với nhãn", "", "| Lớp | Recall (%) |", "|---|---|"]
        L += [f"| {c} | {v:.1f} |" for c, v in S["per_class_recall"].items()]
        L += ["", f"![Ma trận nhầm lẫn]({figs['confusion']})"]
    if figs.get("gallery"):
        L += ["", "## Minh họa", "", f"![Ảnh minh họa]({figs['gallery']})"]
    if figs.get("frames"):
        L += ["", "## Minh họa", "", f"![Khung hình]({figs['frames']})"]
    L += ["", "---", "*Ghi chú*: thời gian ở trên đo trên chính máy đã chạy lệnh này. "
          "'Mặt chính' là khuôn mặt lớn nhất trong khung hình. Độ tin cậy là xác suất softmax của lớp dự đoán, "
          "không phải xác suất đúng thực sự."]
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    S_json = {k: v for k, v in S.items() if not k.startswith("_")}
    (out / "summary.json").write_text(json.dumps(S_json, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"[report] {out / 'report.md'}")
    return S_json


def compare(runs, out_dir):
    rows = []
    for r in runs:
        p = Path(r) / "report" / "summary.json"
        S = json.loads(p.read_text(encoding="utf-8")) if p.exists() else build_report(r)
        rows.append({"lần chạy": S["run"], "kịch bản": S["scenario"], "thiết bị": S["device"], "số mục": S["n_items"],
                     "phát hiện mặt (%)": S["detection_rate_pct"],
                     "phát hiện (ms)": S["latency"]["detect_ms"]["median"],
                     "mô hình (ms)": S["latency"]["classify_ms"]["median"],
                     "tổng (ms)": S["latency"]["total_ms"]["median"], "tổng p95 (ms)": S["latency"]["total_ms"]["p95"],
                     "FPS xử lý": S["proc_fps"], "FPS vòng lặp": S.get("loop_fps_median", np.nan),
                     "cảm xúc chính": S["top_emotion"], "tin cậy TB": S["conf_mean"],
                     "acc (%)": S.get("acc", np.nan), "MCA (%)": S.get("mca", np.nan), "F1 (%)": S.get("f1", np.nan),
                     "đổi nhãn/s thô": S.get("switch_per_s_raw", np.nan),
                     "đổi nhãn/s mượt": S.get("switch_per_s_smooth", np.nan)})
    df = pd.DataFrame(rows)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(out_dir)
    df.to_csv(out_dir / f"summary_{stamp}.csv", index=False)
    md = df.dropna(axis=1, how="all").to_markdown(index=False, floatfmt=".2f")
    (out_dir / f"summary_{stamp}.md").write_text("# So sánh các lần chạy\n\n" + md + "\n", encoding="utf-8")
    print(md)
    print(f"\n[compare] {out_dir / f'summary_{stamp}.md'}")
    return df
