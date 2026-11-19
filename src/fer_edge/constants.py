"""Hằng số dùng chung."""

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# Mỗi cảm xúc một màu cố định (bảng màu phân loại phân biệt được cả với người mù màu)
EMOTION_COLORS = {
    "neutral": "#2a78d6", "happiness": "#eb6834", "surprise": "#1baf7a", "sadness": "#eda100",
    "anger": "#e87ba4", "disgust": "#008300", "fear": "#6250d6", "contempt": "#e34948",
}

# Tên thư mục thường gặp -> tên lớp chuẩn (dùng khi chấm điểm thư mục ảnh có nhãn)
LABEL_ALIASES = {
    "happy": "happiness", "sad": "sadness", "angry": "anger", "suprise": "surprise", "surprised": "surprise",
    "fearful": "fear", "disgusted": "disgust", "neural": "neutral", "contemptuous": "contempt",
}

VARIANTS = ["fp32", "int8_dyn", "int8_static"]
VARIANT_SUFFIX = {"fp32": "fp32", "int8_dyn": "int8d", "int8_static": "int8s"}
SUFFIX_VARIANT = {v: k for k, v in VARIANT_SUFFIX.items()}


def hex_to_bgr(h):
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)
