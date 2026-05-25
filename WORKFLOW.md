# WORKFLOW — Part 2: Data Pipeline & Model Training

> Tài liệu handoff cho thành viên nhóm.  
> Môn MTH00051 — Deadline: **30/5/2026 23:59**

---

## Tổng quan quy trình

```
[Raw NHANES .xpt files — 10 survey modules]
        │
        ▼  part2/data/merge_nhanes.py
[nhanes_stroke_analysis.csv]
  • 21 cột được chọn, có missing values
  • Dùng cho EDA trong part2_notebook.ipynb
        │
        ▼  part2/data/prep_checkpoint.py
[nhanes_pre_pipeline.csv]
  • 7,515 obs × 10 cột
  • Outlier sinh học đã lọc (BPXOSY ∈ [40,260], BMI ∈ [10,90]...)
  • 3 lần đo BP → 1 giá trị trung bình (SYSTOLIC_TARGET, BPXOPLS)
  • Missing values CÒN NGUYÊN — chưa impute
        │
        ▼  part2/data_pipeline.py — DataPipeline.load_and_split(seed=42)
   ┌────┴────┐
[X_train_df] [X_test_df]   +  y_train, y_test
  80% = 6,012 obs          20% = 1,503 obs
        │
        ▼  DataPipeline.fit(X_train_df)   ← CHỈ train, KHÔNG dùng test
  Học từ train:
  • impute_median_  : median của continuous cols (bỏ qua NaN)
  • impute_mode_    : mode của categorical cols
  • scaler_mean_    : mean sau impute
  • scaler_std_     : std  sau impute
  • category_maps_  : sorted unique categories
        │
        ▼  DataPipeline.transform(X)
  1. Drop SEQN
  2. Remap giá trị không hợp lệ → NaN  (SMQ020: 7.0, 9.0)
  3. Median impute continuous
  4. Mode impute categorical
  5. One-hot encode  (categories từ fit — drop first)
  6. Z-score: (x − mean) / std
  7. Thêm cột bias = 1 ở đầu
        │
[X_train_clean (6012×10)  /  X_test_clean (1503×10)]
  feature_names = [
    'bias', 'RIDAGEYR', 'BMXBMI', 'BPXOPLS', 'LBXTC', 'LBXSCR',
    'RIAGENDR_2.0', 'SMQ020_2.0', 'DIQ010_2.0', 'DIQ010_3.0'
  ]
        │
        ├─► Model 1 — OLS Full
        │   ols_fit(X_train, y_train)
        │   β̂ = (XᵀX)⁻¹Xᵀy    [từ part1/ols_implementation.py]
        │
        ├─► Model 2 — OLS Selective
        │   Step 1: p-value filter (α=0.05)
        │     coef_inference → t-test → giữ lại p ≤ 0.05
        │   Step 2: VIF filter (threshold=10)
        │     iterative removal: loại biến có VIF cao nhất > 10
        │   ols_fit(X_selected, y_train)
        │   Kết quả: 4 biến — RIDAGEYR, BMXBMI, LBXTC, RIAGENDR_2.0
        │
        ├─► Model 3 — Ridge Regression
        │   find_best_lambda: 5-fold CV, λ ∈ logspace(-3, 4, 50)
        │     kfold_cv(fit_func=ridge_fit)  [từ part1/cross_validation.py]
        │   best λ* = 37.276
        │   ridge_fit(X_train, y_train, lam=37.276)
        │
        ▼  evaluate_on_test_set(models, X_test, y_test)
```

### Kết quả trên Test Set

| Model | MAE | RMSE | R² | R²_adj | #Features |
|---|---|---|---|---|---|
| OLS Full | 11.1602 | 14.7232 | 0.2988 | 0.2946 | 9 |
| **OLS Selective** ★ | **11.1586** | **14.7127** | **0.2998** | **0.2980** | **4** |
| Ridge (λ=37.28) | 11.1621 | 14.7196 | 0.2992 | 0.2950 | 9 |

> OLS Selective tốt nhất: giảm nhiễu multicollinearity, R²_adj cao nhất với ít biến nhất.

---

## Bonus — Kernel Ridge Regression (`part2/advanced_methods.py`)

```
RBF kernel: K_ij = exp(−‖xᵢ−xⱼ‖² / 2ℓ²)
Fit:        α = (K + λI)⁻¹ y
Predict:    ŷ(x*) = k(x*)ᵀ α

Tune (λ*, ℓ*) via 5-fold CV grid search
  λ ∈ logspace(-1, 3, 8), ℓ ∈ logspace(-1, 1, 5)
  Subsample n=800 (kernel matrix O(n²), solve O(n³))
Best: λ*=0.1, ℓ*=10
```

| Model | MAE | RMSE | R² |
|---|---|---|---|
| OLS | 11.1602 | 14.7232 | 0.2988 |
| Kernel Ridge | 11.0830 | 14.8792 | 0.2839 |

---

## Cách chạy từng bước

```bash
# Bước 1 — Merge raw data (chỉ cần chạy 1 lần)
python part2/data/merge_nhanes.py

# Bước 2 — Tạo checkpoint pre-imputation
python part2/data/prep_checkpoint.py

# Bước 3 — Train models + đánh giá (lưu 4 figures)
python part2/model_comparison.py

# Bước 4 — Kernel Ridge (bonus, lưu 2 figures)
python part2/advanced_methods.py
```

---

## Cột dữ liệu sau DataPipeline

| Nhóm | Cột gốc | Xử lý |
|---|---|---|
| Continuous | RIDAGEYR, BMXBMI, BPXOPLS, LBXTC, LBXSCR | Median impute → Z-score |
| Categorical | RIAGENDR, SMQ020, DIQ010 | Mode impute → One-hot (drop first) |
| Target | SYSTOLIC_TARGET | Không scale, tách thành y |
| Drop | SEQN | ID — bỏ trước transform |

> Lưu ý: MCQ160E, MCQ160F, PAQ680, BMXWAIST, SLD012 không có trong `nhanes_pre_pipeline.csv`  
> (không được merge hoặc bị lọc ra). DataPipeline chỉ dùng các cột thực sự có mặt.

---

## Figures được tạo tự động

| File | Script | Nội dung |
|---|---|---|
| `part2/fig_feature_importance.png` | model_comparison.py | \|βⱼ\| bar chart cho 3 models |
| `part2/fig_model_comparison.png` | model_comparison.py | Grouped bar MAE/RMSE/R² |
| `part2/fig_residual_analysis.png` | model_comparison.py | 4 diagnostic plots (OLS selective) |
| `part2/fig_cv_lambda.png` | model_comparison.py | Ridge λ selection curve |
| `part2/fig_kernel_cv_heatmap.png` | advanced_methods.py | CV MSE heatmap (λ × ℓ) |
| `part2/fig_kernel_predicted_vs_actual.png` | advanced_methods.py | Predicted vs Actual (Kernel Ridge) |
| `part1/fig_*.png` (7 files) | part1 scripts | OLS theory illustrations |

---

## Trạng thái công việc

| Thành phần | Trạng thái | Ghi chú |
|---|---|---|
| `part1/` — tất cả scripts | ✅ Xong | OLS, Ridge, Lasso, CV, residuals |
| `part1/part1_notebook.ipynb` | ✅ Xong | |
| `part2/data/merge_nhanes.py` | ✅ Xong | |
| `part2/data/prep_checkpoint.py` | ✅ Xong | Output: nhanes_pre_pipeline.csv |
| `part2/data_pipeline.py` | ✅ Xong | DataPipeline class, no leakage |
| `part2/model_comparison.py` | ✅ Xong | 3 models + 4 figures |
| `part2/advanced_methods.py` | ✅ Xong | Kernel Ridge + 2 figures |
| `part2/part2_notebook.ipynb` | ⚠️ Cần hoàn thiện | EDA + trình bày kết quả |
| `report/report.pdf` | ❌ Chưa có | Xem `report/report_placeholder.md` |
| Nộp bài | ❌ Chưa | Deadline 30/5/2026 23:59 |

---

## Import dependencies giữa các file

```
part2/model_comparison.py
    └── imports from part1/ols_implementation.py   (ols_fit)
    └── imports from part1/ridge_lasso.py          (ridge_fit, vif)
    └── imports from part1/residual_analysis.py    (model_metrics, coef_inference, residual_plots)
    └── imports from part1/cross_validation.py     (kfold_cv)
    └── imports from part2/data_pipeline.py        (DataPipeline)

part2/advanced_methods.py
    └── imports from part1/ols_implementation.py   (ols_fit)
    └── imports from part1/residual_analysis.py    (model_metrics)
    └── imports from part2/data_pipeline.py        (DataPipeline)
```

> Tất cả import dùng `sys.path.insert(0, '../part1')` — không cần cài package.
