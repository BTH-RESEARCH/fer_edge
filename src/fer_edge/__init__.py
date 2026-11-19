"""fer-edge: nhận dạng biểu cảm khuôn mặt trên thiết bị biên (ONNX Runtime, không cần PyTorch).

    from fer_edge import FERModel, predict

    m = FERModel("resnet18")                 # tự tải mô hình lần đầu
    label, conf, probs = m.predict_image("khuon_mat.jpg")

    predict("anh/")        predict("video.mp4")        predict(0)   # webcam
"""
__version__ = "0.1.0"
__all__ = ["FERModel", "Pack", "open_pack", "predict", "build_pipeline", "__version__"]

_LAZY = {"FERModel": "model", "Pack": "pack", "open_pack": "pack",
         "predict": "runners", "build_pipeline": "runners"}


def __getattr__(name):                 # nạp chậm: `import fer_edge` không kéo theo cv2/pandas
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
