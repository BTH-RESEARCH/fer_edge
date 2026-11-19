"""Kiểm tra lại độ chính xác NGAY TRÊN THIẾT BỊ bằng testset đóng kèm trong gói mô hình.

Với mỗi gói: Acc / MCA / macro-F1 trên testset, và tỉ lệ dự đoán TRÙNG với kết quả tham chiếu tính lúc export
(trên Kaggle, x86). Bản fp32 phải trùng gần 100%; lệch nhiều nghĩa là tiền xử lý hoặc phiên bản ONNX Runtime khác.
"""
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .model import FERModel, preprocess_pil
from .pack import open_pack
from .registry import cached_models
from .sysinfo import append_csv


def cls_metrics(y, p, k):
    """Acc, mean class accuracy, macro-F1 (%), recall từng lớp — cùng định nghĩa với notebook huấn luyện."""
    y, p = np.asarray(y), np.asarray(p)
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y, p), 1)
    tp = np.diag(cm).astype(float)
    rec = tp / np.maximum(cm.sum(1), 1)
    prec = tp / np.maximum(cm.sum(0), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    present = cm.sum(1) > 0
    return {"acc": 100 * float((y == p).mean()), "mca": 100 * float(rec[present].mean()),
            "f1": 100 * float(f1[present].mean()), "per_class": [round(100 * float(r), 2) for r in rec]}


def evaluate(models=None, threads=0, out_dir="results", models_dir=None, limit=0):
    names = list(models) if models else sorted(cached_models())
    if not names:
        raise SystemExit("Chưa có gói mô hình nào trên máy. Chạy: fer-edge download <tên mô hình>")
    out_csv = Path(out_dir) / "accuracy_device.csv"
    rows = []
    for name in names:
        pack = open_pack(name, models_dir)
        ts = pack.testset
        if ts is None:
            print(f"  {pack.name}: gói không kèm testset, bỏ qua")
            continue
        labels = pd.read_csv(ts / "labels.csv")
        if limit:
            labels = labels.head(limit)
        y = labels["label"].to_numpy()
        ref = np.load(ts / "ref_logits.npz") if (ts / "ref_logits.npz").is_file() else {}
        m = FERModel(pack, threads=threads)
        t0 = time.perf_counter()
        logits = np.concatenate([m.logits(preprocess_pil(Image.open(ts / "images" / f), pack.meta))
                                 for f in labels["file"]])
        dt = time.perf_counter() - t0
        pred = logits.argmax(1)
        met = cls_metrics(y, pred, len(pack.classes))
        row = {"time": datetime.now().isoformat(timespec="seconds"), "pack": pack.name,
               "model": pack.meta.get("model"), "dataset": pack.meta.get("dataset"),
               "variant": pack.meta.get("variant"), "n": len(y), "acc": round(met["acc"], 2),
               "mca": round(met["mca"], 2), "f1": round(met["f1"], 2), "sec_per_img": round(dt / len(y), 4),
               "per_class": json.dumps(dict(zip(pack.classes, met["per_class"])), ensure_ascii=False)}
        key = pack.meta.get("variant", "fp32")
        for refname, col in ((key, "agree_export_same_variant"), ("torch_fp32", "agree_torch_fp32")):
            if refname in ref:
                r = ref[refname][: len(y)]
                row[col] = round(100 * float((pred == r.argmax(1)).mean()), 2)
                if refname == key:
                    row["max_abs_diff_vs_export"] = round(float(np.abs(logits - r).max()), 5)
        rows.append(row)
        print(f"{pack.name:28s} acc {row['acc']:6.2f} | mca {row['mca']:6.2f} | f1 {row['f1']:6.2f}"
              + (f" | trùng bản export {row['agree_export_same_variant']:6.2f}%"
                 if "agree_export_same_variant" in row else ""))
        if row.get("agree_export_same_variant", 100) < 99.5 and key == "fp32":
            print("   CẢNH BÁO: bản fp32 không khớp kết quả lúc export -> kiểm tra phiên bản onnxruntime.")
    if rows:
        append_csv(out_csv, rows)
        print("\nKết quả:", out_csv)
    return out_csv
