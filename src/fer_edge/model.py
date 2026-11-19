"""Suy luận bằng ONNX Runtime. Mọi quy tắc tiền xử lý đọc từ gói mô hình (fer.json), không viết cứng trong code."""
import numpy as np
from PIL import Image

from .pack import Pack, open_pack


def make_session(onnx_path, threads=0):
    import onnxruntime as ort
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])


def to_train_like(img, meta):
    """Đưa khuôn mặt về dạng giống ảnh huấn luyện.
    FERPlus: ảnh xám 48x48 (như FER2013) rồi nhân thành 3 kênh. Ảnh test sẵn 48x48 đi qua đây không đổi."""
    if meta.get("grayscale"):
        img = img.convert("L")
    r = meta.get("train_res")
    if r and min(img.size) > int(r):
        img = img.resize((int(r), int(r)), Image.BOX)
    return img.convert("RGB")


def preprocess_pil(img, meta):
    """PIL -> 1x3xSxS float32, giống hệt phép biến đổi lúc đánh giá khi huấn luyện."""
    s = int(meta["img_size"])
    img = to_train_like(img, meta).resize((s, s), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - np.asarray(meta["mean"], dtype=np.float32)) / np.asarray(meta["std"], dtype=np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


class FERModel:
    """Nạp một .ferpack và phân loại ảnh khuôn mặt.

        m = FERModel("resnet18-ferplus-fp32")      # tự tải nếu máy chưa có
        probs = m.predict_faces([anh_rgb_numpy])   # N x số lớp
    """

    def __init__(self, model="resnet18", threads=0, models_dir=None):
        self.pack = model if isinstance(model, Pack) else open_pack(model, models_dir)
        self.meta = self.pack.meta
        self.classes = self.pack.classes
        self.name = self.pack.name
        self.threads = threads
        self.session = make_session(self.pack.onnx, threads)
        self.input_name = self.session.get_inputs()[0].name
        self.warmup()

    # ---- tiền xử lý
    def preprocess(self, rgb_crops):
        """Danh sách ảnh RGB (numpy HxWx3 uint8) -> batch Nx3xSxS float32."""
        return np.concatenate([preprocess_pil(Image.fromarray(c), self.meta) for c in rgb_crops])

    # ---- suy luận
    def logits(self, x):
        if x.shape[0] == 1:
            return self.session.run(None, {self.input_name: x})[0]
        return np.concatenate([self.session.run(None, {self.input_name: x[i:i + 1]})[0]   # batch 1: như lúc chạy thật
                               for i in range(x.shape[0])])

    def predict(self, x):
        """Batch đã tiền xử lý -> xác suất (N x số lớp)."""
        return softmax(self.logits(x))

    def predict_faces(self, rgb_crops):
        return self.predict(self.preprocess(rgb_crops))

    def predict_image(self, path_or_pil):
        """Một ảnh đã là khuôn mặt -> (nhãn, độ tin cậy, xác suất các lớp)."""
        img = path_or_pil if hasattr(path_or_pil, "convert") else Image.open(path_or_pil)
        p = self.predict(preprocess_pil(img, self.meta))[0]
        k = int(np.argmax(p))
        return self.classes[k], float(p[k]), p

    def warmup(self, n=2):
        s = self.pack.img_size
        x = np.zeros((1, 3, s, s), np.float32)
        for _ in range(n):
            self.session.run(None, {self.input_name: x})

    def info(self):
        return {**self.pack.info(), "threads": self.threads or "mặc định của ONNX Runtime"}

    def __repr__(self):
        return f"<FERModel {self.name}>"
