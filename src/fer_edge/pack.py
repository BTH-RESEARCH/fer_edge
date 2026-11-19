"""Đọc và tạo file .ferpack.

Một .ferpack là file zip chứa MỌI thứ cần để chạy mô hình, nên code suy luận không cần biết
mô hình được huấn luyện trên bộ dữ liệu nào:

    fer.json            tên lớp, kích thước ảnh, mean/std, quy tắc tiền xử lý, kết quả đo lúc export
    model.onnx          mô hình
    testset/            (tùy chọn) ảnh test + nhãn + logits tham chiếu, để kiểm tra lại trên thiết bị
"""
import json
import shutil
import zipfile
from pathlib import Path

from .registry import cache_dir, find_pack, sha256

MANIFEST = "fer.json"


class Pack:
    """Một gói mô hình đã sẵn sàng dùng (đã giải nén ra đĩa)."""

    def __init__(self, directory, name=None):
        self.dir = Path(directory)
        self.meta = json.loads((self.dir / MANIFEST).read_text(encoding="utf-8"))
        self.name = name or self.meta.get("name") or self.dir.name

    # ---- thuộc tính hay dùng
    @property
    def onnx(self):
        return self.dir / self.meta.get("onnx", "model.onnx")

    @property
    def classes(self):
        return list(self.meta["classes"])

    @property
    def img_size(self):
        return int(self.meta["img_size"])

    @property
    def testset(self):
        d = self.dir / "testset"
        return d if (d / "labels.csv").is_file() else None

    def info(self):
        m = self.meta
        return {"name": self.name, "arch": m.get("arch", m.get("model", "?")), "model": m.get("model"),
                "dataset": m.get("dataset"), "variant": m.get("variant", "fp32"), "classes": self.classes,
                "img_size": self.img_size, "params_M": m.get("params_M"),
                "grayscale": bool(m.get("grayscale")), "train_res": m.get("train_res"),
                "size_mb": round(self.onnx.stat().st_size / 1e6, 2)}

    def __repr__(self):
        i = self.info()
        return f"<Pack {self.name}: {i['arch']} {i['img_size']}px, {len(i['classes'])} lớp, {i['size_mb']} MB>"


def open_pack(name, models_dir=None):
    """Tên mô hình / đường dẫn -> Pack đã giải nén trong cache."""
    src, full = find_pack(name, models_dir)
    if src.is_dir():
        return Pack(src, full)
    out = cache_dir() / "unpacked" / full
    stamp = out / ".ok"
    if not stamp.is_file() or stamp.read_text().strip() != str(src.stat().st_size):
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(src) as z:
            bad = [n for n in z.namelist() if n.startswith(("/", "\\")) or ".." in Path(n).parts]
            if bad:
                raise ValueError(f"{src} chứa đường dẫn không hợp lệ: {bad[:3]}")
            z.extractall(out)
        if not (out / MANIFEST).is_file():
            raise ValueError(f"{src} không phải .ferpack hợp lệ (thiếu {MANIFEST})")
        stamp.write_text(str(src.stat().st_size))
    return Pack(out, full)


def build_pack(src_dir, out_file, meta_extra=None, include_testset=True):
    """Đóng gói thư mục export (model_*.onnx + meta.json [+ pi_subset/]) thành .ferpack.

    Dùng sau khi tải kết quả export từ Kaggle. Mỗi biến thể (fp32 / int8_dyn / int8_static) thành một .ferpack.
    """
    src_dir, out_file = Path(src_dir), Path(out_file)
    meta_path = src_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Không thấy meta.json trong {src_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(meta_extra or {})
    onnx = src_dir / meta["onnx"]
    if not onnx.is_file():
        raise FileNotFoundError(f"Không thấy {onnx}")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_suffix(".part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        meta["onnx"] = "model.onnx"
        z.writestr(MANIFEST, json.dumps(meta, indent=1, ensure_ascii=False))
        z.write(onnx, "model.onnx")
        sub = src_dir / "testset"
        if not sub.is_dir():
            sub = src_dir / "pi_subset"              # tên cũ của notebook export
        if include_testset and sub.is_dir():
            for f in sorted(sub.rglob("*")):
                if f.is_file():
                    z.write(f, f"testset/{f.relative_to(sub).as_posix()}")
    tmp.replace(out_file)
    return out_file


ARCH_LABEL = {"resnet18": "ResNet-18", "apvit": "APViT", "vit_b16": "ViT-B/16",
              "posterv2": "POSTER V2", "clip_lora": "CLIP ViT-B/16 + LoRA"}


def convert_export_dir(src_dir, out_dir, include_testset=True):
    """Thư mục export của Kaggle (meta.json + model_*.onnx [+ pi_subset/]) -> một .ferpack cho MỖI biến thể."""
    from .constants import VARIANT_SUFFIX
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    meta = json.loads((src_dir / "meta.json").read_text(encoding="utf-8"))
    if "files" not in meta:                                   # đã là gói một biến thể
        name = meta.get("name") or src_dir.name
        return [build_pack(src_dir, out_dir / f"{name}.ferpack", include_testset=include_testset)]
    te = meta.get("test_eval", {})
    made = []
    for variant, fname in meta["files"].items():
        if not (src_dir / fname).is_file():
            print(f"  bỏ qua {variant}: không thấy {fname}")
            continue
        met = {k: v for k, v in (te.get(variant) or {}).items() if k in ("acc", "mca", "f1", "per_class")}
        ref = {k: v for k, v in (te.get("torch_fp32") or {}).items() if k in ("acc", "mca", "f1", "per_class")}
        name = f"{meta['model']}-{meta['dataset']}-{VARIANT_SUFFIX.get(variant, variant)}"
        extra = {
            "name": name, "variant": variant, "onnx": fname,
            "arch": ARCH_LABEL.get(meta["model"], meta["model"]),
            "metrics": met, "metrics_torch_fp32": ref, "n_test": te.get("n"),
            "agree_torch": (te.get(variant) or {}).get("agree_torch"),
            "max_abs_logit_diff": (te.get(variant) or {}).get("max_abs_logit_diff"),
        }
        for k in ("files", "test_eval", "quant_errors", "pi_subset", "results_csv_ref"):
            meta.pop(k, None)
        made.append(build_pack(src_dir, out_dir / f"{name}.ferpack", {**meta, **extra}, include_testset))
        print(f"  {made[-1].name}  ({made[-1].stat().st_size / 1e6:.1f} MB)")
        meta = json.loads((src_dir / "meta.json").read_text(encoding="utf-8"))   # nạp lại cho biến thể sau
    return made


def registry_entry(pack_file, base_url=None):
    """Mục JSON để dán vào src/fer_edge/models.json sau khi tải .ferpack lên GitHub Releases."""
    pack_file = Path(pack_file)
    url = f"{base_url.rstrip('/')}/{pack_file.name}" if base_url else f"<URL tải {pack_file.name}>"
    return pack_file.stem, {"url": url, "sha256": sha256(pack_file),
                            "size_mb": round(pack_file.stat().st_size / 1e6, 1)}
