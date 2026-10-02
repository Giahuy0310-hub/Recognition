# YOLO Traffic Sign Detection

Project nhận diện biển báo giao thông bằng YOLO với **18 loại biển báo**. Project có giao diện Streamlit để nhận diện bằng **ảnh, video và webcam**, đồng thời có dashboard để theo dõi dữ liệu nhận diện.

## 1. Cấu trúc project

```text
TTTN/
├── app/
│   ├── recognition_app.py       # Giao diện nhận diện
│   └── analytics_dashboard.py   # Dashboard thống kê
├── core/
│   └── detection_engine.py      # Bộ máy xử lý nhận diện
├── demo/
│   └── demo_video.py            # Tạo video demo
├── research/
│   ├── hardcase_analysis.py     # Phân tích trường hợp khó
│   └── compare_models.py        # So sánh model
├── datasets/                    # Dataset
├── models/                      # Model YOLO
├── input/                       # Ảnh/video đầu vào
├── output/                      # Kết quả
├── runs/                        # Kết quả training
├── backup_before_reorganize/    # Bản sao lưu
├── requirements.txt
├── py
└── README.md
```

## 2. Các loại biển báo

Model nhận diện 18 lớp:

1. Cấm đỗ xe
2. Cấm dừng xe và đỗ xe
3. Cấm xe đi ngược chiều
4. Cấm xe 2 bánh và xe 3 bánh có động cơ
5. Giao nhau với đường không ưu tiên
6. Chú ý người đi bộ cắt ngang
7. Đi chậm
8. Cấm ô tô tải và xe khách
9. Tốc độ tối đa 50
10. Tốc độ tối đa 60
11. Tốc độ tối đa 80
12. Làn đường cho từng xe theo vạch kẻ đường
13. Cấm ô tô
14. Cấm rẽ phải
15. Đường một chiều
16. Cấm rẽ trái
17. Cấm ô tô rẽ phải
18. Cấm ô tô rẽ trái

## 3. Model

Model chính được fine-tune với các trường hợp khó từ video thực tế.

```text
runs/detect/traffic_sign_18class_hardcase/weights/best.pt
```

## 4. Chạy chương trình

Mở CMD tại thư mục `TTTN`:

```cmd
py -m streamlit run app\recognition_app.py
```

Giao diện cho phép:
- Chọn model
- Nhận diện ảnh
- Nhận diện video
- Sử dụng webcam
- Điều chỉnh Confidence
- Điều chỉnh kích thước ảnh
- Theo dõi kết quả nhận diện

## 5. Chạy video demo

```cmd
python demo\demo_video.py
```

Video đầu vào:

```text
input\compare_video.mp4
```

Video kết quả:

```text
output\hardcase_demo_result.mp4
```

## 6. Dashboard

Dashboard dùng dữ liệu đã được ghi lại từ quá trình nhận diện.

```cmd
py -m streamlit run app\analytics_dashboard.py
```

Dashboard hiển thị:
- Tổng số detection
- Các loại biển báo
- Confidence
- Nguồn dữ liệu
- Sự kiện cảnh báo
- Dữ liệu detection và event

Dashboard không chạy YOLO trực tiếp mà đọc dữ liệu log trong `output`.

## 7. Dataset

Dataset chính:

```text
datasets\traffic_sign_18class_hardcase\
```

Gồm:

```text
train\
valid\
test\
```

Dataset hardcase được bổ sung các hình ảnh đại diện cho những trường hợp model gặp khó khăn khi nhận diện trong video thực tế.

## 8. Research

### `hardcase_analysis.py`

Phân tích các frame khó nhận diện và hỗ trợ tạo dữ liệu hardcase.

### `compare_models.py`

So sánh kết quả nhận diện giữa các model trên cùng một video.

## 9. Cài đặt

```cmd
python -m pip install -r requirements.txt
```

Nên chạy các lệnh từ thư mục gốc `TTTN`.

## 10. Kết quả test

Kết quả test của model 18 lớp trước bước hardcase fine-tuning:

```text
Precision : 0.835
Recall    : 0.849
mAP50     : 0.881
mAP50-95  : 0.651
```

Các chỉ số trên được dùng làm mốc so sánh trước khi fine-tune hardcase.

## 11. Luồng hoạt động

```text
Ảnh / Video / Webcam
        │
        ▼
recognition_app.py
        │
        ▼
detection_engine.py
        │
        ▼
YOLO Model
        │
        ├── Detection
        ├── Tracking / Event
        └── Logging
        │
        ▼
output/
        │
        ▼
analytics_dashboard.py
```

## 12. Lưu ý

- Nên chạy lệnh từ thư mục gốc `TTTN`.
- Không xóa `runs` nếu vẫn cần model và kết quả training.
- Không xóa `datasets` nếu vẫn cần training hoặc kiểm tra model.
- `backup_before_reorganize` là bản sao lưu trước khi sắp xếp project.
