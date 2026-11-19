# Export mô hình trên Kaggle

Chỉ cần làm lại khi huấn luyện mô hình mới.

| Notebook | Mô hình | Checkpoint cần |
|---|---|---|
| `export/fer_export_resnet18_kaggle.ipynb` | ResNet-18 (224 px) | `resnet18_sqrt/ferplus_s0_best.pt` |
| `export/fer_export_apvit_kaggle.ipynb` | APViT (112 px) | `apvit_sqrt/ferplus_s0_best.pt` |

`training/` là notebook huấn luyện gốc; các ô cấu hình, dữ liệu và định nghĩa mô hình của notebook export được copy nguyên từ đó. Sửa notebook huấn luyện thì tạo lại bằng:

```bash
python kaggle/build_export_notebooks.py
```

## Các bước

1. Kaggle → **Create → New Notebook → File → Import Notebook**, chọn notebook trong `export/`.
2. **Settings**: Accelerator **GPU T4 x1**, Internet **On**.
3. **Add Input**: dataset checkpoint (thư mục `resnet18_sqrt/`, `apvit_sqrt/`). Kaggle hay tự giải nén file `.pt` thành thư mục — notebook nhận ra và tự đóng gói lại. Không cần Add Input FER2013, notebook tự tải.
4. **Save Version → Save & Run All (Commit)**.
5. Tab **Output** → tải các file `.ferpack` trong `packs/`.
6. Tải các file đó lên **GitHub Releases**, rồi dán đoạn JSON mà notebook in ra vào `src/fer_edge/models.json` (thay `<URL release>` bằng link thật).

## Notebook làm gì

| Bước | Nội dung |
|---|---|
| Export | ONNX FP32, batch 1, opset 17 |
| Lượng tử hóa | `int8_dyn` (`quantize_dynamic`) và `int8_static` (QDQ, theo kênh, MinMax trên 200 ảnh train) |
| Kiểm tra tương đương | Trên toàn bộ tập test: PyTorch FP32 so với từng bản ONNX — tỉ lệ dự đoán trùng, max\|Δlogit\|, Acc/MCA/F1, recall từng lớp |
| Đóng gói | Mỗi biến thể thành một `.ferpack`: `fer.json` (lớp, kích thước ảnh, mean/std, quy tắc tiền xử lý, kết quả đo) + `model.onnx` + `testset/` (500 ảnh test + logits tham chiếu) |

Bản FP32 phải trùng PyTorch ≥ 99,5%; notebook in cảnh báo nếu không đạt.
Thời gian chạy trên T4: ResNet-18 khoảng 10–15 phút, APViT khoảng 20–30 phút.
