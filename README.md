# Mô Hình Hóa Huyết Áp Tâm Thu NHANES

> Môn học: MTH00051 — Toán Ứng Dụng và Thống Kê  
> Học kỳ: Học kỳ II, 2025–2026  
> Bộ dữ liệu: NHANES 2021–2023 (Khảo sát Kiểm tra Sức khỏe và Dinh dưỡng Quốc gia Hoa Kỳ)  
> Biến mục tiêu: Huyết áp tâm thu trung bình (`SYSTOLIC_TARGET`)

---

## Tổng Quan Dự Án

Dự án nghiên cứu, xây dựng và đánh giá các mô hình hồi quy tuyến tính nhằm dự đoán huyết áp tâm thu từ các chỉ số nhân khẩu học, nhân trắc học và sinh hóa máu thuộc bộ dữ liệu thực tế NHANES 2021–2023.

Tất cả thuật toán thống kê và mô hình học máy cốt lõi — bao gồm phương pháp bình phương bé nhất (OLS) đầy đủ và chọn lọc, hồi quy Ridge, hồi quy Ridge nhân (RBF Kernel), quy trình điền khuyết KNN, tính hệ số phóng đại phương sai (VIF), và kiểm chéo k-fold — đều được cài đặt thủ công từ đầu dựa trên nguyên lý toán học nền tảng, không phụ thuộc vào các mô hình dựng sẵn của scikit-learn hay statsmodels.

---

## Cấu Trúc Thư Mục

```
nhanes-blood-pressure-modeling/
├── part1/                          # Thuật toán thống kê cốt lõi (Cài đặt từ đầu)
│   ├── ols_implementation.py       # Nghiệm OLS, ma trận chiếu (hat matrix), minh họa Gauss-Markov
│   ├── ridge_lasso.py              # Hồi quy Ridge/Lasso, tính toán hệ số phóng đại phương sai VIF
│   ├── residual_analysis.py        # Các chỉ số (R², MAE, RMSE), suy diễn hệ số, vẽ đồ thị chẩn đoán
│   ├── cross_validation.py         # Kiểm chéo k-fold tổng quát
│   └── part1_notebook.ipynb        # Chứng minh lý thuyết và minh họa thực nghiệm
│
├── part2/                          # Pipeline dữ liệu và so sánh mô hình ứng dụng
│   ├── data/
│   │   ├── dataset/                # Chứa các file dữ liệu thô (.xpt) tải về từ CDC NHANES
│   │   ├── processed/              # Lưu trữ các file dữ liệu trung gian và tập dữ liệu sạch
│   │   ├── merge_nhanes.py         # Gộp các file .xpt thô dựa trên mã SEQN
│   │   └── prep_checkpoint.py      # Lọc outlier sinh học và tạo dữ liệu tiền xử lý
│   │
│   ├── data_pipeline.py            # Hệ thống DataPipeline (KNN imputer và EDA tự viết)
│   ├── residual_analysis.py        # Các hàm tính toán sai số và vẽ đồ thị chẩn đoán phần dư
│   ├── model_comparison.py         # Huấn luyện và đánh giá OLS đầy đủ, OLS chọn lọc và Ridge
│   ├── advanced_methods.py         # Hồi quy Ridge nhân (Kernel Ridge Regression)
│   └── part2_notebook.ipynb        # Phân tích hoàn chỉnh dữ liệu thực tế và tích hợp Unit Tests
│
├── requirements.txt                # Thư viện phụ thuộc (numpy, pandas, matplotlib, seaborn)
└── README.md                       # Hướng dẫn dự án
```

---

## Pipeline Tiền Xử Lý Dữ Liệu

Pipeline được thiết kế nhằm chuẩn bị dữ liệu và ngăn chặn hiện tượng rò rỉ dữ liệu giữa tập huấn luyện và tập kiểm thử.

1. **Gộp dữ liệu thô (`merge_nhanes.py`):** Kết hợp 10 phân hệ khảo sát của NHANES trên mã SEQN thành tập dữ liệu thô gồm 11.933 dòng.
2. **Làm sạch sinh học (`prep_checkpoint.py`):** Lọc bỏ các quan sát chứa giá trị phi sinh lý, tính trung bình các lần đo để tạo ra biến mục tiêu (`SYSTOLIC_TARGET`) và nhịp tim (`BPXOPLS`), thu được tập dữ liệu 7.515 dòng.
3. **DataPipeline (`data_pipeline.py`):** Thực hiện fit/transform tuần tự: điền khuyết bằng thuật toán KNN tự cài đặt kết hợp lọc outlier trên Donor Pool, mã hóa one-hot các biến phân loại, chuẩn hóa Z-score các đặc trưng liên tục, và thêm cột bias.

---

## Kết Quả So Sánh Mô Hình

Các mô hình được huấn luyện trên 80% dữ liệu (6.012 dòng) và kiểm thử độc lập trên 20% dữ liệu (1.503 dòng).

* **OLS Đầy đủ:** Hồi quy sử dụng toàn bộ 9 đặc trưng đầu vào.
* **OLS Chọn lọc:** Lọc các biến không có ý nghĩa thống kê bằng t-test ($p$-value $> 0.05$) và loại bỏ đa cộng tuyến bằng chỉ số VIF. Mô hình tối giản được rút gọn còn 4 đặc trưng: tuổi (`RIDAGEYR`), chỉ số khối cơ thể (`BMXBMI`), tổng cholesterol (`LBXTC`) và giới tính nữ (`RIAGENDR_2.0`).
* **Hồi Quy Ridge:** Giải quyết đa cộng tuyến bằng chính quy hóa L2 với tham số $\lambda \approx 37.28$ tối ưu qua kiểm chéo 5-fold.

### Kết quả trên tập kiểm thử:

| Mô hình | MAE (mmHg) | RMSE (mmHg) | Hệ số $R^2$ | Hệ số $R^2_{adj}$ | Số đặc trưng |
|---|---|---|---|---|---|
| OLS Đầy đủ | 11.1602 | 14.7232 | 0.2988 | 0.2946 | 9 |
| **OLS Chọn lọc**  | **11.1586** | **14.7127** | **0.2998** | **0.2980** | **4** |
| Ridge Regression | 11.1621 | 14.7196 | 0.2992 | 0.2950 | 9 |

Hệ thống OLS Chọn lọc đem lại khả năng tổng quát hóa tốt nhất trên dữ liệu thực tế, khẳng định tính hiệu quả của việc giảm đa cộng tuyến và tối giản số lượng biến.

---

## Hồi Quy Ridge Nhân Nâng Cao

Hồi quy Ridge nhân sử dụng nhân Gaussian (RBF Kernel) được tích hợp để mô hình hóa các mối quan hệ phi tuyến:

$$\hat{y}(x^*) = k(x^*)^\top (K + \lambda I)^{-1} y$$

Với các siêu tham số được tối ưu qua tìm kiếm lưới kết hợp kiểm chéo 5-fold ($\lambda^* = 0.1, \ell^* = 10$).

| Mô hình | MAE | RMSE | Hệ số $R^2$ |
|---|---|---|---|
| OLS Tuyến tính | 11.1602 | 14.7232 | 0.2988 |
| Kernel Ridge (RBF) | 11.0830 | 14.8792 | 0.2839 |

---

## Hướng Dẫn Tái Lập Thực Nghiệm

### 1. Cài đặt môi trường
Dự án yêu cầu Python từ phiên bản 3.10 trở lên. Cài đặt các thư viện phụ thuộc:
```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị dữ liệu
Tải các file dữ liệu định dạng `.xpt` tương ứng chu kỳ 2021–2023 từ trang chủ CDC NHANES và đặt vào thư mục `part2/data/dataset/`.

### 3. Thực thi
Chạy tuần tự các lệnh sau từ thư mục gốc của dự án:
```bash
# Bước 1: Gộp dữ liệu khảo sát thô
python part2/data/merge_nhanes.py

# Bước 2: Lọc sinh học và chuẩn bị tập dữ liệu tiền xử lý
python part2/data/prep_checkpoint.py

# Bước 3: Huấn luyện và so sánh các mô hình hồi quy chính
python part2/model_comparison.py

# Bước 4: Chạy mô hình hồi quy Ridge nhân nâng cao
python part2/advanced_methods.py
```

*Lưu ý: Các biểu đồ phân tích và chẩn đoán mô hình sẽ tự động lưu trong các thư mục `part1/` và `part2/` sau khi kết thúc thực thi.*

---

## Kiểm Định Độc Lập

Dự án tích hợp sẵn các ca kiểm định nhằm đảm bảo tính chính xác của các giải thuật tự thiết kế:
* **Phần 1 (`part1/part1_notebook.ipynb`):** Kiểm thử các giải thuật hồi quy OLS, ma trận chiếu, suy diễn hệ số, các chỉ số đánh giá và kiểm chéo k-fold.
* **Phần 2 (`part2/part2_notebook.ipynb`):** Kiểm thử 8 ca kiểm định liên quan đến các hàm thống kê thủ công trong `EDA` và tính toàn vẹn của `DataPipeline`.

Các bộ kiểm định được chạy trực tiếp bằng cách thực thi các ô code tương ứng ở cuối mỗi file notebook.
