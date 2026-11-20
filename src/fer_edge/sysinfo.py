"""Thông tin máy và trạng thái nhiệt (dùng cho phần đo trên Raspberry Pi)."""
import json
import os
import platform
import re
import subprocess
import time
from pathlib import Path


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=3).stdout.strip()
    except Exception:
        return ""


def read_temp():
    """Nhiệt độ SoC (°C), NaN nếu không đọc được."""
    m = re.search(r"([\d.]+)", _run(["vcgencmd", "measure_temp"]))
    if m:
        return float(m.group(1))
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000
    except Exception:
        return float("nan")


def read_throttled():
    """Cờ giảm xung của firmware Pi: '0x0' = bình thường. Bit 2 = đang giảm xung, bit 18 = đã từng giảm xung."""
    m = re.search(r"0x[0-9a-fA-F]+", _run(["vcgencmd", "get_throttled"]))
    return m.group(0) if m else "n/a"


def cpu_freq_mhz():
    try:
        return int(Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq").read_text()) / 1000
    except Exception:
        return float("nan")


def governor():
    try:
        return Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
    except Exception:
        return "n/a"


def env_info():
    import numpy as np
    import onnxruntime as ort
    info = {"python": platform.python_version(), "os": platform.platform(), "machine": platform.machine(),
            "onnxruntime": ort.__version__, "numpy": np.__version__, "device": "CPU (ONNX Runtime)"}
    try:
        import cv2
        info["opencv"] = cv2.__version__
    except Exception:
        pass
    try:
        info["board"] = Path("/proc/device-tree/model").read_text().strip("\x00").strip()
    except Exception:
        pass
    return info


def system_info():
    info = {**env_info(), "kernel": platform.release(), "governor": governor()}
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                info["os_name"] = line.split("=", 1)[1].strip('"')
    except Exception:
        pass
    try:
        kb = int(re.search(r"MemTotal:\s+(\d+)", Path("/proc/meminfo").read_text()).group(1))
        info["ram_gb"] = round(kb / 1024 / 1024, 1)
    except Exception:
        pass
    info.update({k: os.environ[k] for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS") if k in os.environ})
    return info


def camera_info():
    """Những nguồn camera có thể dùng (không mở camera, chỉ liệt kê)."""
    out = {}
    try:
        from picamera2 import Picamera2
        cams = Picamera2.global_camera_info()
        out["picamera2"] = (f"{len(cams)} camera CSI: {[c.get('Model', '?') for c in cams]}" if cams
                            else "đã cài nhưng không thấy camera CSI nào")
    except ImportError:
        out["picamera2"] = "chưa cài (Pi Camera Module sẽ không dùng được; webcam USB vẫn chạy)"
    except Exception as e:
        out["picamera2"] = f"lỗi: {type(e).__name__}: {e}"
    if platform.system() == "Linux":
        vids = sorted(Path("/dev").glob("video*"))
        out["/dev/video*"] = ", ".join(p.name for p in vids) or "không có (chưa cắm webcam USB?)"
    return out


def is_pi():
    return "raspberry" in str(env_info().get("board", "")).lower()


def wait_cool(target_c=50.0, max_wait_s=300, quiet=False):
    """Chờ SoC nguội dưới target_c để mọi lần đo bắt đầu từ cùng trạng thái nhiệt."""
    t0 = time.time()
    if read_temp() != read_temp():        # NaN: máy không có cảm biến -> bỏ qua
        return
    while read_temp() > target_c and time.time() - t0 < max_wait_s:
        if not quiet:
            print(f"\r  chờ nguội: {read_temp():.1f}°C > {target_c}°C ...", end="", flush=True)
        time.sleep(5)
    if not quiet and time.time() - t0 > 1:
        print()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def append_csv(path, rows):
    import pandas as pd
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def pctl(a, q):
    import numpy as np
    return float(np.percentile(np.asarray(a, dtype=float), q))
