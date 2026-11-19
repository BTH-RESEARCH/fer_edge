"""Phân giải tên mô hình -> file .ferpack trên đĩa, tự tải về và lưu cache nếu cần.

Thứ tự tìm một tên (ví dụ "resnet18-ferplus-fp32"):
  1. Chính nó là đường dẫn tới file .ferpack hoặc thư mục đã giải nén  -> dùng luôn
  2. <thư mục mô hình cục bộ>/<tên>.ferpack   (FER_EDGE_MODELS hoặc ./models)
  3. cache: ~/.cache/fer-edge/packs/<tên>.ferpack
  4. models.json -> tải về cache, kiểm tra SHA-256

Biến môi trường:
  FER_EDGE_CACHE   thư mục cache (mặc định ~/.cache/fer-edge)
  FER_EDGE_MODELS  thư mục chứa .ferpack cục bộ (mặc định ./models nếu tồn tại)
"""
import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
REGISTRY_FILE = PKG_DIR / "models.json"

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
YUNET_SHA = None          # OpenCV Zoo có thể cập nhật file; không ép kiểm tra băm


def cache_dir():
    d = Path(os.environ.get("FER_EDGE_CACHE", Path.home() / ".cache" / "fer-edge"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def local_models_dir():
    d = os.environ.get("FER_EDGE_MODELS")
    if d:
        return Path(d)
    return Path.cwd() / "models"


def load_registry():
    try:
        return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"models": {}, "aliases": {}}


def save_registry(reg):
    REGISTRY_FILE.write_text(json.dumps(reg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def resolve_name(name):
    """Mở rộng tên rút gọn: 'resnet18' -> alias; 'resnet18-ferplus' -> thêm '-fp32'."""
    reg = load_registry()
    name = reg.get("aliases", {}).get(name, name)
    if name in reg["models"]:
        return name
    if f"{name}-fp32" in reg["models"]:
        return f"{name}-fp32"
    return name


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _progress(blocks, bsize, total):
    if total <= 0:
        return
    done = min(blocks * bsize, total)
    pct = 100 * done / total
    bar = "█" * int(pct / 2.5) + "·" * (40 - int(pct / 2.5))
    print(f"\r  [{bar}] {pct:5.1f}%  {done / 1e6:.0f}/{total / 1e6:.0f} MB", end="", flush=True)


def download(url, dst, expect_sha=None, desc=""):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"[fer-edge] tải {desc or dst.name}\n           {url}")
    try:
        urllib.request.urlretrieve(url, tmp, _progress)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Tải thất bại: {url}\n  {type(e).__name__}: {e}\n"
                           f"  Kiểm tra mạng, hoặc tải tay rồi đặt vào {dst}") from e
    print()
    if expect_sha:
        got = sha256(tmp)
        if got != expect_sha:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"File tải về sai SHA-256 (hỏng hoặc sai phiên bản).\n"
                               f"  mong đợi {expect_sha}\n  nhận được {got}")
    tmp.replace(dst)
    return dst


def find_pack(name, models_dir=None):
    """Trả về (đường dẫn .ferpack hoặc thư mục đã giải nén, tên chuẩn hóa). Tự tải nếu cần."""
    p = Path(name)
    if p.suffix == ".ferpack" and p.is_file():
        return p, p.stem
    if p.is_dir() and (p / "fer.json").exists():
        return p, p.name

    full = resolve_name(name)
    for d in [models_dir, local_models_dir(), cache_dir() / "packs"]:
        if d and (Path(d) / f"{full}.ferpack").is_file():
            return Path(d) / f"{full}.ferpack", full
        if d and (Path(d) / full / "fer.json").is_file():
            return Path(d) / full, full

    reg = load_registry()
    entry = reg["models"].get(full)
    if entry is None:
        raise FileNotFoundError(
            f"Không biết mô hình '{name}'.\n"
            f"  Có trong registry: {', '.join(sorted(reg['models'])) or '(trống)'}\n"
            f"  Hoặc đưa thẳng đường dẫn tới file .ferpack, hoặc đặt file vào {local_models_dir()}")
    dst = cache_dir() / "packs" / f"{full}.ferpack"
    download(entry["url"], dst, entry.get("sha256"), desc=f"{full} ({entry.get('size_mb', '?')} MB)")
    return dst, full


def yunet_path():
    """Mô hình phát hiện mặt YuNet (OpenCV), tự tải về cache."""
    dst = cache_dir() / "detectors" / "face_detection_yunet_2023mar.onnx"
    if dst.is_file() and dst.stat().st_size > 100_000:
        return dst
    return download(YUNET_URL, dst, YUNET_SHA, desc="YuNet (phát hiện khuôn mặt)")


def cached_models():
    out = {}
    for d in (cache_dir() / "packs", local_models_dir()):
        if d.is_dir():
            for f in sorted(d.glob("*.ferpack")):
                out.setdefault(f.stem, f)
    return out


def clear_cache():
    shutil.rmtree(cache_dir(), ignore_errors=True)
