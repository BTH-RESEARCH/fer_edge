# fer-edge

Nhận dạng biểu cảm khuôn mặt (FERPlus, 8 lớp) chạy bằng ONNX Runtime — trên laptop và Raspberry Pi 5.
Mô hình **tự tải về**, không cần chép tay, không cần PyTorch.

```bash
pip install "git+https://github.com/BTH-RESEARCH/fer-edge.git"
fer-edge predict --source 0
```

---

## 1. Cài đặt

Cần Python 3.9–3.12.

### Laptop (Windows)

```powershell
py -3 -m venv .venv
.venv\Scripts\activate
pip install "git+https://github.com/BTH-RESEARCH/fer-edge.git"
fer-edge doctor
```

### Raspberry Pi 5 / Linux / macOS

```bash
python3 -m venv --system-site-packages .venv     # --system-site-packages để thấy picamera2 của hệ thống
source .venv/bin/activate
pip install "git+https://github.com/BTH-RESEARCH/fer-edge.git"
fer-edge doctor
```

### Cài từ mã nguồn (để sửa code)

```bash
git clone https://github.com/BTH-RESEARCH/fer-edge.git
cd fer-edge
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e .
```

Mỗi lần mở terminal mới phải kích hoạt lại môi trường: `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`).

---

## 2. Tải mô hình

```bash
fer-edge models                      # xem danh sách
fer-edge download resnet18 apvit     # tải về máy
```

Mô hình được lưu ở `~/.cache/fer-edge/packs/` và dùng lại cho mọi lần chạy sau. Lệnh `predict` / `bench` cũng tự tải nếu chưa có.

| Tên | Mô hình | Ảnh vào | Ghi chú |
|---|---|---|---|
| `resnet18-ferplus-fp32` (tắt: `resnet18`) | ResNet-18 | 224 | nhanh |
| `resnet18-ferplus-int8d` / `-int8s` | ResNet-18 | 224 | lượng tử hóa INT8 (động / tĩnh) |
| `apvit-ferplus-fp32` (tắt: `apvit`) | APViT | 112 | chính xác hơn, chậm hơn |
| `apvit-ferplus-int8d` / `-int8s` | APViT | 112 | lượng tử hóa INT8 |

Có file `.ferpack` sẵn trên máy: đặt vào thư mục `models/` cạnh nơi chạy lệnh, hoặc chỉ đường dẫn thẳng:

```bash
fer-edge predict --source anh.jpg --model D:\tai_ve\resnet18-ferplus-fp32.ferpack
```

---

## 3. Chạy

Một lệnh duy nhất cho mọi loại đầu vào.

```bash
# webcam (0 là camera mặc định)
fer-edge predict --source 0

# webcam, tự dừng sau 60 giây, có ghi video
fer-edge predict --source 0 --seconds 60 --record

# một ảnh
fer-edge predict --source anh.jpg

# thư mục ảnh
fer-edge predict --source D:\anh\

# video
fer-edge predict --source video.mp4 --show

# dùng mô hình khác
fer-edge predict --source 0 --model apvit
```

**Ảnh đã là khuôn mặt cắt sẵn** (tập test FERPlus 48×48, RAF-DB) thì bỏ bước phát hiện mặt:

```bash
fer-edge predict --source D:\ferplus\test --detector none --name ferplus_test
```

Nếu thư mục con mang tên lớp (`neutral`, `happy`, `sad`, `angry`, `suprise`, `disgust`, `fear`, `contempt`) hoặc số `1..8`, chương trình tự lấy làm nhãn thật và tính Accuracy / MCA / F1.

**Phím khi chạy webcam:** `q` thoát · `s` chụp ảnh · `r` bật/tắt ghi video · `b` bật/tắt thanh xác suất.

### Tùy chọn hay dùng

| Tùy chọn | Mặc định | Ý nghĩa |
|---|---|---|
| `--model` | `resnet18` | tên mô hình hoặc đường dẫn `.ferpack` |
| `--detector` | `yunet` | `yunet` · `yolo` (cần `pip install "fer-edge[yolo]"`) · `none` |
| `--out-dir` | `outputs` | nơi lưu kết quả |
| `--name` | – | thêm tên vào thư mục kết quả |
| `--threads` | tự động | số luồng CPU |
| `--margin` | `0.15` | nới khung mặt mỗi phía |
| `--smoothing` | `ema` | làm mượt nhãn theo thời gian: `ema` · `majority` · `none` |
| `--seconds` | `0` | [webcam] tự dừng sau N giây |
| `--max-frames` | `0` | [video] chỉ xử lý N khung đầu |
| `--no-report` | – | không tạo báo cáo |

---

## 4. Kết quả được lưu ở đâu

Mỗi lần chạy tạo một thư mục riêng: `outputs/<kịch bản>_<tên>_<thời gian>/`

```
outputs/webcam_20261118-143000/
├── frames.csv           mỗi khung hình: số mặt, thời gian từng khâu, nhãn + xác suất 8 lớp (thô và đã làm mượt)
├── faces.csv            mỗi khuôn mặt: toạ độ khung, điểm phát hiện, nhãn, xác suất
├── run.json             mô hình, thiết bị, phiên bản thư viện, tham số đã dùng
├── recording.mp4        video đã ghi (nếu bấm r)   | annotated.mp4 (video) | annotated/ (ảnh)
├── snapshots/           ảnh đã chụp (nếu bấm s)
└── report/
    ├── report.md        bảng tóm tắt + hình, mở bằng VS Code hoặc đẩy lên GitHub là xem được
    ├── summary.json     các con số chính
    ├── fig_latency.png      thời gian từng khâu (trung vị, p95)
    ├── fig_emotions.png     phân bố cảm xúc
    ├── fig_confidence.png   phân bố độ tin cậy
    ├── fig_confusion.png    ma trận nhầm lẫn        (ảnh có nhãn)
    ├── fig_gallery.png      ảnh minh họa, có ảnh sai (ảnh)
    ├── fig_timeline.png     nhãn theo thời gian      (video / webcam)
    ├── fig_speed_time.png   tốc độ theo thời gian    (video / webcam)
    └── fig_frames.png       khung hình minh họa      (video / webcam)
```

Gộp nhiều lần chạy thành một bảng so sánh (`outputs/summary_<thời gian>.md` và `.csv`):

```bash
fer-edge report outputs/image_* outputs/video_* outputs/webcam_*
```

---

## 5. Đo hiệu năng trên Raspberry Pi 5

Trước khi đo: gắn quạt tản nhiệt, dùng nguồn 27 W, và cố định tần số CPU

```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

```bash
fer-edge download resnet18-ferplus-fp32 resnet18-ferplus-int8s apvit-ferplus-fp32 apvit-ferplus-int8s

fer-edge bench                       # độ trễ mọi mô hình đang có: trung vị, p95, FPS, RAM, nhiệt độ
fer-edge bench --threads 1 4         # thêm cấu hình 1 luồng
fer-edge bench resnet18-ferplus-int8s --soak 600    # chạy liên tục 10 phút, xem nhiệt làm giảm FPS bao nhiêu

fer-edge eval                        # chạy lại testset kèm trong gói, đối chiếu với kết quả lúc export

fer-edge predict --source video.mp4 --model resnet18-ferplus-int8s   # đo toàn bộ quy trình

fer-edge summarize --rt-fps 10       # bảng chính + đồ thị Pareto
```

Kết quả đo nằm trong `results/`:

```
results/
├── latency.csv             mỗi dòng một cấu hình: độ trễ, FPS, RAM, nhiệt độ, cờ giảm xung, phiên bản thư viện
├── raw/*.npy               độ trễ từng lần chạy
├── soak.csv, soak/*.csv    kết quả chạy liên tục
├── accuracy_device.csv     độ chính xác đo lại trên chính thiết bị
├── table_main.csv / .md    bảng chính cho bài báo
└── pareto_ferplus.png      đồ thị độ chính xác – độ trễ
```

Giao thức đo: batch 1, ảnh thật từ testset, ONNX Runtime CPU; mỗi cấu hình chạy trong một tiến trình riêng, bắt đầu khi SoC dưới 50 °C, làm nóng 50 lần rồi đo 500 lần, báo cáo trung vị và p95.

---

## 6. Dùng trong Python

```python
from fer_edge import FERModel, predict

m = FERModel("resnet18")                          # tự tải lần đầu
label, conf, probs = m.predict_image("khuon_mat.jpg")
print(label, conf)

run_dir = predict("video.mp4", model="apvit")     # trả về thư mục kết quả
```

---

## 7. Tạo mô hình mới (chỉ khi huấn luyện lại)

1. Chạy notebook trong `kaggle/export/` trên Kaggle (xem `kaggle/README.md`). Kết quả là các file `.ferpack` ở tab *Output*.
2. Tải các file đó lên **GitHub Releases** của repo này (*Releases → Draft a new release → kéo thả file*).
3. Cập nhật `src/fer_edge/models.json`: dán đoạn JSON mà notebook in ra, thay `<URL release>` bằng link thật.

Nếu đã có sẵn thư mục export kiểu cũ (`meta.json` + `model_*.onnx`), đóng gói bằng:

```bash
fer-edge pack duong/dan/export --out packs --update-registry
```

---

## Khắc phục sự cố

| Hiện tượng | Cách xử lý |
|---|---|
| `fer-edge: command not found` | Chưa kích hoạt môi trường ảo. Hoặc dùng `python -m fer_edge` thay cho `fer-edge`. |
| Tải mô hình lỗi | Tải tay file `.ferpack` theo link trong `fer-edge models`, đặt vào `~/.cache/fer-edge/packs/`. |
| `Không mở được camera 0` | Thử `--source 1`. Trên Windows kiểm tra quyền truy cập camera. |
| Webcam chạy chậm | Dùng `--model resnet18-ferplus-int8s`, hoặc giảm `--width 640 --height 480`. |
| Không có cửa sổ hiện lên (chạy qua SSH) | Thêm `--no-show`, hoặc chạy trên video đã quay sẵn. |
| Nhãn nhảy liên tục | Tăng độ mượt: `--smoothing ema --alpha 0.15`. |
| Nhãn hay sai với ảnh webcam | Chỉnh `--margin 0.25`, hoặc thêm `--align`. |
| Muốn xóa toàn bộ mô hình đã tải | Xóa thư mục `~/.cache/fer-edge/`. |
