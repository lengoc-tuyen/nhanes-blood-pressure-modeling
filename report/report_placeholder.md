# Report Placeholder

Đặt file `report.pdf` và `report.tex` (hoặc `report.md`) vào thư mục này trước khi nộp bài.

## Bố cục báo cáo (theo §3.1 đề bài)

```
report.pdf
├── Trang bìa  (tên, MSSV, nhóm, GV hướng dẫn, ngày)
├── Mục lục
│
├── PHẦN 1 — Lý Thuyết Data Fitting và OLS  (6.0đ)
│   1.1  Bài toán Data Fitting — phát biểu, mô hình tuyến tính
│   1.2  Giả thiết Gauss–Markov (GM1–GM5)
│   1.3  Nghiệm OLS: β̂ = (XᵀX)⁻¹Xᵀy — chứng minh, hat matrix H
│   1.4  Định lý Gauss–Markov (BLUE) + minh họa Monte Carlo
│   1.5  Đánh giá mô hình: R², R²_adj, F-test
│   1.6  Kiểm định hệ số: t-test, p-value, khoảng tin cậy 95%
│   1.7  Đa cộng tuyến: VIF — định nghĩa, VIF > 10
│   1.8  Regularization: Ridge (closed-form), Lasso (coord. descent), ridge trace
│   1.9  Phân tích phần dư: 4 biểu đồ chẩn đoán, nhận xét
│   1.10 Cross-validation: k-fold CV, CV score
│
├── PHẦN 2 — Ứng Dụng: Dự Đoán Huyết Áp Tâm Thu  (5.5đ)
│   2.1  Giới thiệu dữ liệu NHANES
│   2.2  EDA (thống kê mô tả, phân phối, tương quan, missing, outlier)
│   2.3  Tiền xử lý (outlier filter, BP averaging, DataPipeline, train/test split)
│   2.4  Xây dựng 3 mô hình (OLS full, OLS selective, Ridge)
│   2.5  Đánh giá và so sánh trên test set (bảng MAE/RMSE/R²)
│   2.6  Phân tích phần dư mô hình tốt nhất
│   2.7  Feature importance
│   2.8  Kernel Ridge Regression (bonus +0.5đ)
│   2.9  Kết luận Part 2
│
├── KẾT LUẬN TỔNG
├── TÀI LIỆU THAM KHẢO  (≥5)
└── PHỤ LỤC (phân công thành viên, bảng số liệu bổ sung)
```

## Kết quả chính cần trình bày

| Model | MAE | RMSE | R² | R²_adj | #Features |
|---|---|---|---|---|---|
| OLS Full | 11.1602 | 14.7232 | 0.2988 | 0.2946 | 9 |
| **OLS Selective** | **11.1586** | **14.7127** | **0.2998** | **0.2980** | **4** |
| Ridge (λ=37.28) | 11.1621 | 14.7196 | 0.2992 | 0.2950 | 9 |
| Kernel Ridge | 11.0830 | 14.8792 | 0.2839 | 0.2796 | — |

## Figures đã có sẵn

```
part1/
  fig_confidence_intervals.png
  fig_cv_lambda.png
  fig_gauss_markov.png
  fig_hat_matrix.png
  fig_residual_plots.png
  fig_ridge_trace.png
  fig_vif.png
part2/
  fig_feature_importance.png
  fig_model_comparison.png
  fig_residual_analysis.png
  fig_cv_lambda.png
  fig_kernel_cv_heatmap.png
  fig_kernel_predicted_vs_actual.png
```
