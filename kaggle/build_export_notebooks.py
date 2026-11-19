"""Tạo notebook export ONNX cho ResNet-18 và APViT từ chính các notebook huấn luyện (thư mục training/).
Chạy lại khi sửa notebook huấn luyện:  python kaggle/build_export_notebooks.py

Các ô cấu hình, đọc dữ liệu và định nghĩa mô hình được COPY NGUYÊN từ notebook huấn luyện tương ứng
(để kiến trúc và tiền xử lý khớp tuyệt đối với lúc huấn luyện); bỏ các ô huấn luyện/trực quan hóa;
thêm các ô export ONNX, lượng tử hóa INT8, kiểm tra tương đương và đóng gói cho Raspberry Pi 5.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
SOURCES = {
    # key = MODEL_NAME lúc huấn luyện: (notebook huấn luyện, tên hiển thị, lệnh tạo mô hình trong make_model)
    "resnet18": (HERE / "training" / "fer_resnet_kaggle.ipynb", "ResNet-18", "return build_model(len(classes))"),
    "apvit": (HERE / "training" / "fer_apvit_kaggle.ipynb", "APViT", "return build_model(len(classes))"),
}

NOTES = {
    "resnet18": "",
    "apvit": "\n- **APViT**: bước chọn token (top-k) được export thành toán tử `TopK` với k cố định (160 patch, rồi giảm 10% "
             "ở 4 lớp cuối), nên số token trên Pi giống hệt lúc huấn luyện. Ảnh đầu vào 112×112.",
}


def src(cell):
    s = cell["source"]
    return s if isinstance(s, str) else "".join(s)


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")}


def intro(name, key):
    size = 112 if key == "apvit" else 224
    return md(f"""
# Export ONNX: {name} → Raspberry Pi 5

Notebook này **không huấn luyện**. Nó lấy trọng số tốt nhất (`*_best.pt`) mà notebook huấn luyện {name} đã lưu, rồi:

1. Dựng lại **đúng** kiến trúc và tiền xử lý (các ô 0–4 được copy nguyên từ notebook huấn luyện).
2. Export sang **ONNX FP32** (batch 1, ảnh {size}×{size}).
3. Lượng tử hóa **INT8** bằng ONNX Runtime: động (`int8_dyn`) và tĩnh có hiệu chỉnh (`int8_static`).
4. **Kiểm tra tương đương** trên tập test: PyTorch FP32 so với từng bản ONNX (tỉ lệ dự đoán trùng, sai khác logits, Acc/MCA/F1, độ chính xác từng lớp).
5. Đóng gói cho Pi: file `.onnx`, `meta.json` (tên lớp, kích thước ảnh, mean/std, kết quả kiểm tra) và một **tập con test** (ảnh + logits tham chiếu) để kiểm tra lại độ chính xác ngay trên Pi.

**Thiết lập Kaggle**
- *Settings → Accelerator*: GPU T4 x1 (chỉ dùng để tính logits tham chiếu PyTorch; ONNX Runtime chạy trên CPU).
- *Settings → Internet*: **On**.
- *Add Input*: **checkpoint đã huấn luyện**: dataset `fer-checkpoints` gồm các thư mục `<MODEL_NAME>_sqrt/` (hoặc Output của notebook huấn luyện {name}). Ô **5** tự tìm thư mục đó trong `/kaggle/input`, kể cả khi Kaggle đã **tự giải nén file `.pt` thành thư mục** (code tự đóng gói lại).
- Mặc định chỉ export mô hình **FERPlus** (ô **1b**). Chỉ cần Add Input **RAF-DB** nếu thêm `"rafdb"` vào `EXPORT_DATASETS`.
- **FER2013 không bắt buộc Add Input**: nếu không thấy `fer2013.csv` trong `/kaggle/input`, ô **3** tự tải từ Kaggle bằng `kagglehub` (dataset `deadskull7/fer2013`, dự phòng `ashishpatel26/facial-expression-recognitionferchallenge`); số dòng được kiểm tra đủ 35.887 ảnh như lúc huấn luyện.

**Cách dùng**: kiểm tra ô **1. Cấu hình** giống hệt lúc huấn luyện (cùng `MODEL_NAME`, `SAMPLING`), chọn dataset ở ô **1b** → *Save Version → Save & Run All (Commit)*. Kết quả: các file **`.ferpack`** trong `/kaggle/working/packs/` (tải ở tab *Output*) — đây là thứ duy nhất cần mang đi.
{NOTES[key]}
""")


EXPORT_CELLS = [
    md("""
## 5. Cấu hình export và tìm checkpoint
Checkpoint được tìm theo tên thư mục `<MODEL_NAME>_<SAMPLING>` (ví dụ `resnet18_sqrt`) trong `/kaggle/input`. Nếu tự tìm không được, điền đường dẫn vào `CKPT_DIR` (xem kết quả ô 0).
"""),
    code(r'''
import gc, inspect, shutil, subprocess, sys, zipfile
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime", "onnxscript"], check=False)
import onnx
import onnxruntime as ort
from torch.nn.utils import parametrize as _parametrize

# ---------------- Export ----------------
EXPORT_SEED = 0              # seed của checkpoint cần export (phải có trong SEEDS lúc huấn luyện)
CKPT_DIR = "auto"            # "auto" = tự tìm thư mục <MODEL_NAME>_<SAMPLING> trong /kaggle/input
EXPORT_DIR = Path("/kaggle/working/export") / MODEL_NAME
OPSET = 17
QUANT_MODES = ["int8_dyn", "int8_static"]   # bỏ bớt nếu không cần
N_CALIB = 200                # số ảnh train (không augmentation) để hiệu chỉnh INT8 tĩnh
N_PI_SUBSET = 500            # số ảnh test (phân tầng theo lớp) mang sang Pi để kiểm tra lại độ chính xác
PARITY_MAX_IMAGES = None     # None = toàn bộ tập test (dùng cho bài báo); đặt 300 để chạy thử nhanh
ORT_THREADS = os.cpu_count()
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Tắt các đường tắt có thể làm lệch số học giữa PyTorch và ONNX
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
try:
    torch.backends.mha.set_fastpath_enabled(False)      # nn.MultiheadAttention: dùng đường export được
except Exception:
    pass


def find_ckpt_dir():
    if CKPT_DIR != "auto":
        return Path(CKPT_DIR)
    want = f"{MODEL_NAME}_{SAMPLING}"
    hits = sorted({Path(dp) for dp, dirs, files in os.walk(INPUT_ROOT, followlinks=True)   # Kaggle gắn input qua liên kết tắt
                   if Path(dp).name == want and (any(f.endswith("_best.pt") for f in files)
                                                 or any(d.endswith("_best") for d in dirs))})   # .pt bị Kaggle giải nén
    if not hits:
        raise FileNotFoundError(
            f"Không tìm thấy thư mục '{want}' chứa *_best.pt trong {INPUT_ROOT}. Hãy Add Input Output của notebook "
            f"huấn luyện (phiên bản đã Save & Run All), hoặc điền CKPT_DIR. Thư mục input hiện có: "
            f"{sorted(p.name for p in INPUT_ROOT.iterdir())}")
    if len(hits) > 1:
        print("Cảnh báo: thấy nhiều thư mục checkpoint, dùng thư mục đầu tiên:", *hits, sep="\n  ")
    return hits[0]


CKPT = find_ckpt_dir()
print("checkpoint:", CKPT)
for p in sorted(CKPT.iterdir()):
    if p.is_file():
        print(f"  {p.name:40s} {p.stat().st_size / 1e6:8.1f} MB")
    else:
        print(f"  {p.name + '/':40s} (thư mục: file .pt đã bị giải nén, sẽ được đóng gói lại)")
REF = pd.read_csv(CKPT / "results.csv") if (CKPT / "results.csv").exists() else None
if REF is not None:
    display(REF[["model", "dataset", "seed", "test_acc", "test_mca", "test_f1", "best_epoch"]])
print("torch", torch.__version__, "| onnx", onnx.__version__, "| onnxruntime", ort.__version__, "| CPU threads", ORT_THREADS)
'''),
    md("""
## 6. Hàm export, lượng tử hóa và kiểm tra
- **Export**: thử bộ export TorchScript (`dynamo=False`, opset 17) trước, nếu lỗi thì thử bộ export mới (`dynamo=True`, opset ≥ 18). Đầu vào cố định `1×3×IMG_SIZE×IMG_SIZE`, tên `input`; đầu ra `logits`.
- **INT8 động** (`quantize_dynamic`): trọng số INT8, activation lượng tử hóa lúc chạy; không cần dữ liệu hiệu chỉnh.
- **INT8 tĩnh** (`quantize_static`, định dạng QDQ, trọng số theo kênh, MinMax): hiệu chỉnh bằng `N_CALIB` ảnh train qua đúng tiền xử lý đánh giá.
- **Tương đương**: logits ONNX Runtime (CPU, batch 1) so với PyTorch FP32 trên cùng ảnh; báo tỉ lệ dự đoán trùng, sai khác logits lớn nhất và Acc/MCA/macro-F1/độ chính xác từng lớp.
"""),
    code(r'''
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quantize_dynamic, quantize_static)
try:
    from onnxruntime.quantization.shape_inference import quant_pre_process
except Exception:
    quant_pre_process = None


def make_model(classes):
    __MAKE_MODEL__


def resolve_ckpt(name, seed):
    """Trả về file .pt. Nếu Kaggle đã giải nén .pt thành thư mục <name>_s<seed>_best/, đóng gói lại vào /kaggle/working."""
    import zipfile
    stem = f"{name}_s{seed}_best"
    if (CKPT / f"{stem}.pt").is_file():
        return CKPT / f"{stem}.pt"
    src = CKPT / stem
    pkl = None
    if src.is_dir():
        for dp, _, fs in os.walk(src, followlinks=True):
            if "data.pkl" in fs:
                pkl = Path(dp) / "data.pkl"
                break
    if pkl is None:
        raise FileNotFoundError(f"Không thấy {stem}.pt hoặc thư mục {stem}/ chứa data.pkl trong {CKPT}")
    dst = Path("/kaggle/working/ckpt_fixed") / MODEL_NAME / f"{stem}.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)
    root = pkl.parent
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as z:
        for dp, _, fs in os.walk(root, followlinks=True):
            for f in sorted(fs):
                fp = Path(dp) / f
                z.write(fp, f"{stem}/{fp.relative_to(root).as_posix()}")
    print(f"  {stem}/ là file .pt đã bị giải nén -> đóng gói lại thành {dst}")
    return dst


def load_finetuned(name, seed):
    classes = DATA[name][3]
    path = resolve_ckpt(name, seed)
    model = make_model(classes)
    sd = torch.load(path, map_location="cpu")
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        raise RuntimeError(f"Checkpoint {path.name} không khớp kiến trúc. Kiểm tra ô Cấu hình có giống lúc huấn luyện "
                           f"không (MODEL_NAME, PEFT_METHOD, phiên bản thư viện).\n{e}") from e
    merged = 0                       # gộp mọi parametrization (LoRA của CLIP) vào trọng số thường
    for mod in list(model.modules()):
        if _parametrize.is_parametrized(mod):
            for pname in list(mod.parametrizations.keys()):
                _parametrize.remove_parametrizations(mod, pname, leave_parametrized=True)
                merged += 1
    model = model.float().eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    return model, classes, path, merged, n_params


def cls_metrics(y, p, k):
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y, p), 1)
    tp = np.diag(cm).astype(float)
    rec = tp / np.maximum(cm.sum(1), 1)
    prec = tp / np.maximum(cm.sum(0), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    return {"acc": 100 * float((y == p).mean()), "mca": 100 * float(rec.mean()), "f1": 100 * float(f1.mean()),
            "per_class": [round(100 * float(r), 2) for r in rec]}


def loader_of(df, bs=32):
    return DataLoader(FERDataset(df, eval_tf), batch_size=bs, shuffle=False, num_workers=NUM_WORKERS)


@torch.no_grad()
def torch_logits(model, df):
    model.to(DEVICE)
    out = [model(x.to(DEVICE)).float().cpu().numpy() for x, _ in loader_of(df, BATCH_SIZE)]
    model.cpu()
    torch.cuda.empty_cache()
    return np.concatenate(out)


def ort_session(path, threads=ORT_THREADS):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def ort_logits(path, df):
    sess = ort_session(path)
    iname = sess.get_inputs()[0].name
    out = []
    for x, _ in loader_of(df):
        for xi in x.numpy():                                   # chạy từng ảnh: batch 1 như trên robot
            out.append(sess.run(None, {iname: xi[None]})[0][0])
    return np.stack(out).astype(np.float32)


def export_onnx(model, path):
    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    has_dynamo = "dynamo" in inspect.signature(torch.onnx.export).parameters
    errors = []
    for use_dynamo in ([False, True] if has_dynamo else [False]):
        try:
            kw = dict(input_names=["input"], output_names=["logits"],
                      opset_version=max(OPSET, 18) if use_dynamo else OPSET)
            if has_dynamo:
                kw["dynamo"] = use_dynamo
            if not use_dynamo:
                kw["do_constant_folding"] = True
            with torch.no_grad():
                torch.onnx.export(model, (x,), str(path), **kw)
            m = onnx.load(str(path))                            # gộp external data (nếu có) vào 1 file
            onnx.checker.check_model(m)
            onnx.save(m, str(path))
            for extra in path.parent.glob(path.name + ".data"):
                extra.unlink()
            return ("dynamo" if use_dynamo else "torchscript"), int(m.opset_import[0].version)
        except Exception as e:
            errors.append(f"{'dynamo' if use_dynamo else 'torchscript'}: {type(e).__name__}: {str(e)[:800]}")
            print("  export lỗi ->", errors[-1][:300])
    raise RuntimeError("Không export được ONNX:\n" + "\n".join(errors))


class CalibReader(CalibrationDataReader):
    def __init__(self, df, iname):
        self.items = iter([{iname: x.numpy()[None]} for x, _ in FERDataset(df, eval_tf)])

    def get_next(self):
        return next(self.items, None)


def calib_subset(name):
    tr = DATA[name][0]
    return tr.sample(n=min(N_CALIB, len(tr)), random_state=0).reset_index(drop=True)


def preprocess_for_quant(fp32):
    pre = fp32.with_name("_prep.onnx")
    if quant_pre_process is None:
        return fp32
    for kw in ({}, {"skip_symbolic_shape": True}, {"skip_symbolic_shape": True, "skip_optimization": True}):
        try:
            quant_pre_process(str(fp32), str(pre), **kw)
            return pre
        except Exception as e:
            print(f"  quant_pre_process{kw} lỗi: {type(e).__name__}: {str(e)[:200]}")
    return fp32


def quantize(fp32, pre, mode, name):
    out = fp32.with_name(f"model_{mode}.onnx")
    if mode == "int8_dyn":
        quantize_dynamic(str(pre), str(out), weight_type=QuantType.QInt8)
    elif mode == "int8_static":
        iname = ort_session(fp32, 1).get_inputs()[0].name
        quantize_static(str(pre), str(out), CalibReader(calib_subset(name), iname),
                        quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                        calibrate_method=CalibrationMethod.MinMax,
                        extra_options={"CalibMaxIntermediateOutputs": 32})
    else:
        raise ValueError(mode)
    onnx.checker.check_model(str(out))
    return out


def stratified(df, n, seed=0):
    if n >= len(df):
        return df.reset_index(drop=True)
    frac = n / len(df)
    parts = [d.sample(n=max(1, round(len(d) * frac)), random_state=seed) for _, d in df.groupby("label")]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
'''.replace("__MAKE_MODEL__", "{MAKE_MODEL}")),
    md("""
## 7. Chạy export cho từng dataset
Mỗi dataset tạo một thư mục `export/<MODEL_NAME>/<dataset>/` gồm:
`model_fp32.onnx`, `model_int8_dyn.onnx`, `model_int8_static.onnx`, `meta.json`, `pi_subset/` (ảnh + `labels.csv` + `ref_logits.npz`).
"""),
    code(r'''
VERSIONS = {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__, "python": sys.version.split()[0]}
SUMMARY = []

for name in DATASETS_TO_RUN:
    print(f"\n########## {MODEL_NAME} | {name} ##########")
    out_dir = EXPORT_DIR / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    model, classes, ckpt, merged, n_params = load_finetuned(name, EXPORT_SEED)
    k = len(classes)
    print(f"  nạp {ckpt.name}: {n_params:.2f}M tham số" + (f", đã gộp {merged} LoRA" if merged else ""))

    te = DATA[name][2]
    te_eval = te if PARITY_MAX_IMAGES is None else stratified(te, PARITY_MAX_IMAGES)
    y = te_eval["label"].to_numpy()
    pi_df = stratified(te, N_PI_SUBSET)

    ref = torch_logits(model, te_eval)                          # PyTorch FP32 (không AMP)
    ref_pi = torch_logits(model, pi_df)
    m_torch = cls_metrics(y, ref.argmax(1), k)
    print(f"  PyTorch FP32  acc {m_torch['acc']:.2f} | mca {m_torch['mca']:.2f} | f1 {m_torch['f1']:.2f}  ({len(y)} ảnh test)")

    files, errors = {}, {}
    fp32 = out_dir / "model_fp32.onnx"
    exporter, opset = export_onnx(model, fp32)
    files["fp32"] = fp32
    print(f"  export: {exporter}, opset {opset}, {fp32.stat().st_size / 1e6:.1f} MB")
    del model
    gc.collect()

    if QUANT_MODES:
        pre = preprocess_for_quant(fp32)
        for mode in QUANT_MODES:
            t0 = time.time()
            try:
                files[mode] = quantize(fp32, pre, mode, name)
                print(f"  {mode}: {files[mode].stat().st_size / 1e6:.1f} MB ({time.time() - t0:.0f}s)")
            except Exception as e:
                errors[mode] = f"{type(e).__name__}: {str(e)[:500]}"
                print(f"  {mode} LỖI: {errors[mode][:300]}")
        if pre != fp32 and pre.exists():
            pre.unlink()

    # ---- kiểm tra tương đương trên tập test + logits tham chiếu cho tập con của Pi
    results, ref_npz = {}, {"torch_fp32": ref_pi}
    for variant, path in files.items():
        t0 = time.time()
        lg = ort_logits(path, te_eval)
        m = cls_metrics(y, lg.argmax(1), k)
        m.update(agree_torch=100 * float((lg.argmax(1) == ref.argmax(1)).mean()),
                 max_abs_logit_diff=float(np.abs(lg - ref).max()),
                 size_mb=round(path.stat().st_size / 1e6, 2), file=path.name)
        results[variant] = m
        ref_npz[variant] = ort_logits(path, pi_df)
        print(f"  ONNX {variant:12s} acc {m['acc']:.2f} | mca {m['mca']:.2f} | f1 {m['f1']:.2f} | "
              f"trùng PyTorch {m['agree_torch']:.2f}% | max|Δlogit| {m['max_abs_logit_diff']:.4f}  ({time.time() - t0:.0f}s)")
        SUMMARY.append({"model": MODEL_NAME, "dataset": name, "variant": variant, "n_test": len(y),
                        **{c: round(m[c], 2) for c in ("acc", "mca", "f1", "agree_torch")},
                        "acc_drop_vs_torch": round(m_torch["acc"] - m["acc"], 2),
                        "f1_drop_vs_torch": round(m_torch["f1"] - m["f1"], 2),
                        "max_abs_logit_diff": round(m["max_abs_logit_diff"], 4), "size_mb": m["size_mb"],
                        "per_class": json.dumps(dict(zip(classes, m["per_class"])))})

    # ---- tập con test cho Pi
    sub = out_dir / "pi_subset"
    (sub / "images").mkdir(parents=True)
    rows = []
    for i, r in pi_df.iterrows():
        dst = f"{i:04d}{Path(r['path']).suffix.lower()}"
        shutil.copy(r["path"], sub / "images" / dst)
        rows.append({"file": dst, "label": int(r["label"]), "source": Path(r["path"]).name})
    pd.DataFrame(rows).to_csv(sub / "labels.csv", index=False)
    np.savez_compressed(sub / "ref_logits.npz", **ref_npz)

    ref_row = None
    if REF is not None:
        rr = REF[(REF.dataset == name) & (REF.seed == EXPORT_SEED)]
        if len(rr):
            ref_row = {c: float(rr.iloc[0][c]) for c in ("test_acc", "test_mca", "test_f1")}
    meta = {
        "model": MODEL_NAME, "dataset": name, "seed": EXPORT_SEED, "classes": list(classes), "num_classes": k,
        "img_size": IMG_SIZE, "mean": list(MEAN), "std": list(STD), "color": "RGB", "layout": "NCHW float32",
        "grayscale": name == "ferplus", "train_res": 48 if name == "ferplus" else None,   # camera: xám -> 48x48 -> IMG_SIZE
        "resize": "PIL Image.resize((S, S), BILINEAR) rồi /255, (x-mean)/std (giống torchvision Resize + ToTensor + Normalize)",
        "input_name": "input", "output_name": "logits", "batch": 1, "exporter": exporter, "opset": opset,
        "params_M": round(n_params, 2), "checkpoint": str(ckpt), "merged_parametrizations": merged,
        "files": {v: p.name for v, p in files.items()}, "quant_errors": errors,
        "test_eval": {"n": int(len(y)), "torch_fp32": m_torch, **results},
        "results_csv_ref": ref_row, "pi_subset": {"n": len(pi_df), "dir": "pi_subset"},
        "quant": {"int8_dyn": "quantize_dynamic, weight QInt8, per-tensor",
                  "int8_static": f"quantize_static QDQ, per-channel, act QUInt8 / weight QInt8, MinMax, {N_CALIB} ảnh train"},
        "versions": VERSIONS,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print("  đã lưu", out_dir)
'''),
    md("""
## 8. Tổng hợp và đóng gói
`acc_drop_vs_torch` = độ chính xác PyTorch FP32 trừ độ chính xác của bản ONNX. Bản **FP32** phải có `agree_torch` ≥ 99,5% và `acc_drop` ≈ 0; nếu không, export có vấn đề và **không dùng số đo trên Pi** của mô hình đó. Với INT8, so sánh thêm cột `per_class` để xem lớp thiểu số (fear, disgust, contempt) có bị ảnh hưởng nhiều hơn không.
"""),
    code(r'''
summary = pd.DataFrame(SUMMARY)
summary.to_csv(EXPORT_DIR / "export_summary.csv", index=False)
display(summary.drop(columns=["per_class"]))

bad = summary[(summary.variant == "fp32") & ((summary.agree_torch < 99.5) | (summary.acc_drop_vs_torch.abs() > 0.2))]
if len(bad):
    print("CẢNH BÁO: bản FP32 không tương đương PyTorch ở:", bad[["dataset", "agree_torch", "acc_drop_vs_torch"]].to_dict("records"))
else:
    print("OK: mọi bản FP32 tương đương PyTorch.")

print("\nĐộ chính xác từng lớp (%):")
for _, r in summary.iterrows():
    print(f"  {r.dataset:8s} {r.variant:12s}", json.loads(r.per_class))

'''),
    md("""
## 9. Đóng gói thành .ferpack cho fer-edge
Mỗi biến thể thành **một file `.ferpack`** (zip gồm `fer.json` + `model.onnx` + `testset/`). Đây là thứ duy nhất cần mang sang máy khác: tải ở tab **Output**, rồi đưa lên **GitHub Releases** để `fer-edge` tự tải về.
"""),
    code(r'''
import hashlib

ARCH_LABEL = {"resnet18": "ResNet-18", "apvit": "APViT", "vit_b16": "ViT-B/16",
              "posterv2": "POSTER V2", "clip_lora": "CLIP ViT-B/16 + LoRA"}
VARIANT_SUFFIX = {"fp32": "fp32", "int8_dyn": "int8d", "int8_static": "int8s"}
PACK_DIR = Path("/kaggle/working/packs")
PACK_DIR.mkdir(parents=True, exist_ok=True)
entries = {}

for name in DATASETS_TO_RUN:
    d = EXPORT_DIR / name
    meta_all = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    te = meta_all.get("test_eval", {})
    base = {k: v for k, v in meta_all.items()
            if k not in ("files", "test_eval", "quant_errors", "pi_subset", "results_csv_ref")}
    for variant, fname in meta_all["files"].items():
        onnx_file = d / fname
        if not onnx_file.is_file():
            continue
        keep = ("acc", "mca", "f1", "per_class")
        meta = {**base, "name": f"{MODEL_NAME}-{name}-{VARIANT_SUFFIX.get(variant, variant)}",
                "variant": variant, "onnx": "model.onnx", "arch": ARCH_LABEL.get(MODEL_NAME, MODEL_NAME),
                "metrics": {k: v for k, v in (te.get(variant) or {}).items() if k in keep},
                "metrics_torch_fp32": {k: v for k, v in (te.get("torch_fp32") or {}).items() if k in keep},
                "n_test": te.get("n"), "agree_torch": (te.get(variant) or {}).get("agree_torch"),
                "max_abs_logit_diff": (te.get(variant) or {}).get("max_abs_logit_diff")}
        out = PACK_DIR / f"{meta['name']}.ferpack"
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            z.writestr("fer.json", json.dumps(meta, indent=1, ensure_ascii=False))
            z.write(onnx_file, "model.onnx")
            sub = d / "pi_subset"
            for f in sorted(sub.rglob("*")) if sub.is_dir() else []:
                if f.is_file():
                    z.write(f, f"testset/{f.relative_to(sub).as_posix()}")
        h = hashlib.sha256(out.read_bytes()).hexdigest()
        entries[meta["name"]] = {"url": f"<URL release>/{out.name}", "sha256": h,
                                 "size_mb": round(out.stat().st_size / 1e6, 1)}
        print(f"  {out.name:34s} {out.stat().st_size / 1e6:7.1f} MB")

print("\nMục cho src/fer_edge/models.json (thay <URL release> bằng link GitHub Release):")
print(json.dumps(entries, indent=2))
print(f"\nTải {len(entries)} file trong {PACK_DIR} ở tab Output.")
'''),
    md("""
## Ghi chú cho bài báo
- Export ONNX opset 17 (bộ export TorchScript của PyTorch), batch 1, đầu vào float32 `1×3×S×S`; suy luận bằng ONNX Runtime trên CPU.
- INT8 động: `quantize_dynamic` (trọng số INT8 theo tensor). INT8 tĩnh: `quantize_static`, định dạng QDQ, trọng số INT8 theo kênh, activation UINT8, hiệu chỉnh MinMax trên 200 ảnh train ngẫu nhiên (seed 0) qua tiền xử lý đánh giá.
- Báo cáo độ chính xác của mỗi bản trên **toàn bộ tập test** (bảng `export_summary.csv`) và tỉ lệ dự đoán trùng với PyTorch FP32.
- Độ chính xác trong `results.csv` được đo với AMP (FP16) lúc huấn luyện; `torch_fp32` ở đây đo FP32 nên có thể lệch vài phần trăm của một điểm.
"""),
]


OLD_FIND_CSV = '''def find_csv(path, names, search_root):
    """Trả về path nếu tồn tại, nếu không thì tìm file có tên trong `names` dưới search_root."""
    if path != "auto" and Path(path).exists():
        return Path(path)
    for dp, _, files in os.walk(search_root):
        for n in names:
            if n in files:
                return Path(dp) / n
    raise FileNotFoundError(f"Không tìm thấy {names} (đường dẫn khai báo: {path}). "
                            f"Hãy Add Input dataset FER2013 có file fer2013.csv.")'''

NEW_FIND_CSV = '''# Nguồn tự tải fer2013.csv khi notebook chưa Add Input dataset FER2013 (thử lần lượt; số dòng được kiểm tra = 35887)
FER2013_KAGGLE = ["deadskull7/fer2013", "ashishpatel26/facial-expression-recognitionferchallenge"]
FERPLUS_LABELS_URL = "https://raw.githubusercontent.com/microsoft/FERPlus/master/fer2013new.csv"


def _walk_find(root, names):
    for dp, _, files in os.walk(root, followlinks=True):     # followlinks: Kaggle có thể gắn input qua liên kết tắt
        for n in names:
            if n in files:
                return Path(dp) / n
    return None


def find_csv(path, names, search_root):
    """Trả về path nếu tồn tại; nếu không thì tìm dưới search_root rồi /kaggle/input;
    nếu vẫn không có: fer2013.csv được tự tải từ Kaggle (kagglehub), fer2013new.csv tự tải từ GitHub của Microsoft."""
    if path != "auto" and Path(path).exists():
        return Path(path)
    for root in dict.fromkeys([Path(search_root), INPUT_ROOT]):
        hit = _walk_find(root, names)
        if hit:
            return hit
    if "fer2013new.csv" in names:
        import urllib.request
        dst = Path("/kaggle/working/fer2013new.csv")
        print(f"  tự tải nhãn FERPlus -> {dst}")
        urllib.request.urlretrieve(FERPLUS_LABELS_URL, dst)
        return dst
    if "fer2013.csv" in names:
        try:
            import kagglehub
        except ImportError:
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kagglehub"], check=False)
            import kagglehub
        for handle in FER2013_KAGGLE:
            try:
                print(f"  chưa có fer2013.csv trong input -> tự tải dataset Kaggle '{handle}' (cần bật Internet) ...")
                hit = _walk_find(kagglehub.dataset_download(handle), names)
                if hit:
                    return hit
                print(f"  '{handle}' không chứa {names}")
            except Exception as e:
                print(f"  tải '{handle}' lỗi: {type(e).__name__}: {str(e)[:200]}")
    raise FileNotFoundError(f"Không tìm thấy {names} (đường dẫn khai báo: {path}) và không tự tải được. "
                            f"Hãy bật Internet hoặc Add Input dataset FER2013 có file fer2013.csv.")'''


def build(key):
    path, name, make = SOURCES[key]
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    end = next(i for i, c in enumerate(cells) if c["cell_type"] == "markdown" and src(c).startswith("## 5."))
    keep, skip_next = [], False
    for c in cells[1:end]:
        if skip_next:
            skip_next = False
            continue
        if c["cell_type"] == "markdown" and src(c).startswith("### Kiểm tra nhãn bằng mắt"):
            skip_next = True                 # bỏ cả ô vẽ lưới ảnh mẫu đi kèm
            continue
        keep.append({k: v for k, v in c.items() if k != "id"} | ({"outputs": [], "execution_count": None}
                                                                  if c["cell_type"] == "code" else {}))
    patched = 0                              # thay find_csv: theo cả liên kết tắt + tự tải nếu chưa Add Input
    for c in keep:
        s = src(c)
        if c["cell_type"] == "code" and OLD_FIND_CSV in s:
            c["source"] = s.replace(OLD_FIND_CSV, NEW_FIND_CSV)
            patched += 1
    assert patched == 1, f"{path.name}: không tìm thấy find_csv để thay ({patched})"
    extra = []
    for c in EXPORT_CELLS:
        c = json.loads(json.dumps(c))
        c["source"] = c["source"].replace("{MAKE_MODEL}", make)
        extra.append(c)
    cfg_i = next(i for i, c in enumerate(keep) if c["cell_type"] == "code" and "DATASETS_TO_RUN" in src(c)
                 and "MODEL_NAME" in src(c))
    keep[cfg_i + 1:cfg_i + 1] = [md("### 1b. Dataset cần export"), code('''
EXPORT_DATASETS = ["ferplus"]          # thêm "rafdb" nếu cần (khi đó phải Add Input dataset RAF-DB)
DATASETS_TO_RUN = [d for d in DATASETS_TO_RUN if d in EXPORT_DATASETS]
assert MODEL_NAME == "__KEY__", f"Notebook này dành cho __KEY__, nhưng MODEL_NAME = {MODEL_NAME}"
print("export:", MODEL_NAME, DATASETS_TO_RUN)
'''.replace("__KEY__", key))]
    out_cells = [intro(name, key)] + keep + extra
    for i, c in enumerate(out_cells):
        c["id"] = f"e{i:02d}"
    out = {"cells": out_cells, "metadata": nb["metadata"], "nbformat": 4, "nbformat_minor": 5}
    dst = HERE / "export" / f"fer_export_{key}_kaggle.ipynb"
    dst.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return dst


if __name__ == "__main__":
    for key in SOURCES:
        print("wrote", build(key))
