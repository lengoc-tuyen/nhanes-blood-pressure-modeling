"""
advanced_methods.py
===================
Thành viên D — Kỹ Thuật Nâng Cao (Bonus +0.5đ)
Môn: Toán Ứng Dụng và Thống Kê (MTH00051)

Triển khai Kernel Ridge Regression (closed-form) và so sánh với OLS thông thường.

Công thức:
    Kernel trick: ŷ(x) = k(x)ᵀ (K + λI)⁻¹ y
    RBF kernel  : k(x, x') = exp(-‖x − x'‖² / 2ℓ²)
    Gram matrix : K_ij = k(x_i, x_j)
"""

from __future__ import annotations

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — no GUI window needed
import matplotlib.pyplot as plt

# Import OLS từ Part 1 để so sánh
_PART1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'part1')
sys.path.insert(0, _PART1)

from ols_implementation import ols_fit
from residual_analysis import model_metrics


# ─────────────────────────────────────────────────────────────────────────────
# 1. RBF KERNEL
# ─────────────────────────────────────────────────────────────────────────────

def rbf_kernel(X1: np.ndarray, X2: np.ndarray, l: float = 1.0) -> np.ndarray:
    """
    Tính Gram matrix K với RBF (Gaussian) kernel.

    Công thức:
        k(x, x') = exp(−‖x − x'‖² / 2ℓ²)
        K_ij = k(X1_i, X2_j)

    Parameters
    ----------
    X1 : np.ndarray, shape (n1, d)
    X2 : np.ndarray, shape (n2, d)
    l  : length-scale ℓ > 0 (điều chỉnh độ rộng kernel)

    Returns
    -------
    K : np.ndarray, shape (n1, n2)
    """
    # ‖x_i − x_j‖² = ‖x_i‖² + ‖x_j‖² − 2 x_iᵀ x_j  (vectorized)
    sq_norm_1 = np.sum(X1 ** 2, axis=1, keepdims=True)   # (n1, 1)
    sq_norm_2 = np.sum(X2 ** 2, axis=1, keepdims=True).T  # (1, n2)
    sq_dist   = sq_norm_1 + sq_norm_2 - 2.0 * (X1 @ X2.T) # (n1, n2)
    sq_dist   = np.maximum(sq_dist, 0.0)                   # tránh số âm do float

    return np.exp(-sq_dist / (2.0 * l ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# 2. KERNEL RIDGE FIT
# ─────────────────────────────────────────────────────────────────────────────

def kernel_ridge_fit(
    X_train: np.ndarray,
    y_train: np.ndarray,
    lam: float = 1.0,
    l: float = 1.0,
) -> dict:
    """
    Huấn luyện Kernel Ridge Regression (closed-form).

    Công thức:
        α = (K + λI)⁻¹ y
        ŷ(x) = k(x)ᵀ α

    với K là Gram matrix của X_train.

    Parameters
    ----------
    X_train : np.ndarray, shape (n, d) — KHÔNG có cột bias
    y_train : np.ndarray, shape (n,)
    lam     : regularization λ ≥ 0
    l       : RBF length-scale ℓ

    Returns
    -------
    dict chứa: alpha (n,), K (n×n), X_train, lam, l, y_hat_train, rss
    """
    n = X_train.shape[0]
    K = rbf_kernel(X_train, X_train, l)              # (n, n)

    # α = (K + λI)⁻¹ y
    KrI   = K + lam * np.eye(n)
    alpha = np.linalg.solve(KrI, y_train)            # stable hơn inv()

    y_hat = K @ alpha                                 # fitted values trên train
    rss   = float(np.sum((y_train - y_hat) ** 2))

    return {
        'alpha'      : alpha,
        'K_train'    : K,
        'X_train'    : X_train,
        'lam'        : lam,
        'l'          : l,
        'y_hat'      : y_hat,
        'rss'        : rss,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. KERNEL RIDGE PREDICT
# ─────────────────────────────────────────────────────────────────────────────

def kernel_ridge_predict(
    X_train: np.ndarray,
    X_test: np.ndarray,
    alpha: np.ndarray,
    l: float = 1.0,
) -> np.ndarray:
    """
    Dự đoán với Kernel Ridge Regression.

    Công thức:
        ŷ(x*) = k(x*)ᵀ α  =  [k(x*, x_1), ..., k(x*, x_n)] · α

    Parameters
    ----------
    X_train : np.ndarray, shape (n, d) — dữ liệu huấn luyện (không bias)
    X_test  : np.ndarray, shape (m, d) — dữ liệu kiểm tra
    alpha   : np.ndarray, shape (n,)   — từ kernel_ridge_fit
    l       : RBF length-scale ℓ

    Returns
    -------
    y_pred : np.ndarray, shape (m,)
    """
    K_star = rbf_kernel(X_test, X_train, l)   # (m, n)
    return K_star @ alpha


# ─────────────────────────────────────────────────────────────────────────────
# 4. TUNE KERNEL RIDGE (CV Grid Search)
# ─────────────────────────────────────────────────────────────────────────────

def tune_kernel_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    lam_grid: np.ndarray | None = None,
    l_grid: np.ndarray | None = None,
    k: int = 5,
    seed: int = 42,
) -> tuple[float, float, dict]:
    """
    Tìm (λ*, ℓ*) tốt nhất cho Kernel Ridge bằng k-fold CV thủ công.

    Parameters
    ----------
    X_train  : (n, d)
    y_train  : (n,)
    lam_grid : mảng λ (mặc định logspace(-2, 3, 10))
    l_grid   : mảng ℓ (mặc định logspace(-1, 1, 5))
    k        : số fold
    seed     : random seed

    Returns
    -------
    best_lam   : float
    best_l     : float
    cv_results : dict {'lam_grid', 'l_grid', 'cv_matrix' (len_lam × len_l)}
    """
    if lam_grid is None:
        lam_grid = np.logspace(-2, 3, 10)
    if l_grid is None:
        l_grid = np.logspace(-1, 1, 5)

    np.random.seed(seed)
    n = len(y_train)
    idx = np.random.permutation(n)
    folds = np.array_split(idx, k)

    cv_matrix = np.zeros((len(lam_grid), len(l_grid)))

    for i, lam in enumerate(lam_grid):
        for j, l in enumerate(l_grid):
            fold_mses = []
            for f_idx in range(k):
                val_idx   = folds[f_idx]
                train_idx = np.concatenate([folds[m] for m in range(k) if m != f_idx])

                Xtr, ytr = X_train[train_idx], y_train[train_idx]
                Xval, yval = X_train[val_idx], y_train[val_idx]

                res   = kernel_ridge_fit(Xtr, ytr, lam=lam, l=l)
                y_hat = kernel_ridge_predict(Xtr, Xval, res['alpha'], l=l)
                fold_mses.append(float(np.mean((yval - y_hat) ** 2)))

            cv_matrix[i, j] = float(np.mean(fold_mses))

    best_flat = int(np.argmin(cv_matrix))
    best_i, best_j = divmod(best_flat, len(l_grid))
    best_lam = float(lam_grid[best_i])
    best_l   = float(l_grid[best_j])

    return best_lam, best_l, {
        'lam_grid' : lam_grid,
        'l_grid'   : l_grid,
        'cv_matrix': cv_matrix,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. SO SÁNH KERNEL RIDGE VS OLS
# ─────────────────────────────────────────────────────────────────────────────

def compare_kernel_vs_ols(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    lam: float = 1.0,
    l: float = 1.0,
) -> 'pd.DataFrame':
    """
    So sánh Kernel Ridge và OLS trên cùng dữ liệu.

    Parameters
    ----------
    X_train, y_train : dữ liệu train (không có bias)
    X_test, y_test   : dữ liệu test
    lam, l           : siêu tham số Kernel Ridge

    Returns
    -------
    dict chứa kết quả MAE, RMSE, R² của cả hai mô hình
    """
    import pandas as pd

    # OLS (cần thêm cột bias)
    X_tr_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_te_bias = np.hstack([np.ones((X_test.shape[0],  1)), X_test])

    ols_res  = ols_fit(X_tr_bias, y_train)
    y_ols    = X_te_bias @ ols_res['beta_hat']
    p        = X_train.shape[1]
    ols_m    = model_metrics(y_test, y_ols, p)

    # Kernel Ridge
    kr_res   = kernel_ridge_fit(X_train, y_train, lam=lam, l=l)
    y_kr     = kernel_ridge_predict(X_train, X_test, kr_res['alpha'], l=l)
    kr_m     = model_metrics(y_test, y_kr, p)

    rows = []
    for name, y_hat, metrics in [
        ('OLS',          y_ols, ols_m),
        ('Kernel Ridge', y_kr,  kr_m),
    ]:
        mae  = float(np.mean(np.abs(y_test - y_hat)))
        rmse = float(np.sqrt(np.mean((y_test - y_hat) ** 2)))
        rows.append({
            'Model' : name,
            'MAE'   : round(mae,  4),
            'RMSE'  : round(rmse, 4),
            'R2'    : round(metrics['r2'],     4),
            'R2_adj': round(metrics['r2_adj'], 4),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 6. VẼ HEATMAP CV
# ─────────────────────────────────────────────────────────────────────────────

def plot_cv_heatmap(cv_results: dict, best_lam: float, best_l: float) -> plt.Figure:
    """
    Heatmap CV MSE theo (λ, ℓ), đánh dấu tham số tốt nhất.

    Returns plt.Figure
    """
    import seaborn as sns

    lam_grid  = cv_results['lam_grid']
    l_grid    = cv_results['l_grid']
    cv_matrix = cv_results['cv_matrix']

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        cv_matrix,
        xticklabels=[f'{v:.2f}' for v in l_grid],
        yticklabels=[f'{v:.3f}' for v in lam_grid],
        annot=True, fmt='.1f', cmap='YlOrRd_r', ax=ax,
        cbar_kws={'label': 'CV MSE'},
    )
    ax.set_xlabel('Length-scale ℓ', fontsize=12)
    ax.set_ylabel('λ', fontsize=12)
    ax.set_title(f'Kernel Ridge CV — Best: λ={best_lam:.3f}, ℓ={best_l:.2f}',
                 fontsize=13, fontweight='bold')

    # Đánh dấu ô tốt nhất
    best_i = list(lam_grid).index(best_lam) if best_lam in lam_grid else \
             int(np.argmin(np.abs(lam_grid - best_lam)))
    best_j = list(l_grid).index(best_l) if best_l in l_grid else \
             int(np.argmin(np.abs(l_grid - best_l)))
    ax.add_patch(plt.Rectangle((best_j, best_i), 1, 1,
                                fill=False, edgecolor='blue', linewidth=3))
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import pandas as pd
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_pipeline import DataPipeline

    CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'data', 'nhanes_pre_pipeline.csv')

    CONTINUOUS  = ['RIDAGEYR', 'BMXBMI', 'BPXOPLS', 'LBXTC', 'LBXSCR']
    CATEGORICAL = ['RIAGENDR', 'SMQ020', 'DIQ010']
    INVALID     = {'SMQ020': [7.0, 9.0]}

    # Lấy dữ liệu từ DataPipeline
    pipe = DataPipeline(CONTINUOUS, CATEGORICAL,
                        target_col='SYSTOLIC_TARGET',
                        invalid_cat_values=INVALID)
    X_tr_df, X_te_df, y_train, y_test = pipe.load_and_split(CSV, seed=42)
    X_train_full, _ = pipe.fit_transform(X_tr_df)
    X_test_full,  _ = pipe.transform(X_te_df)

    # Bỏ cột bias (Kernel Ridge không dùng)
    X_train = X_train_full[:, 1:]
    X_test  = X_test_full[:, 1:]

    print("=== Kernel Ridge Regression ===\n")

    # Subsample để CV grid search nhanh hơn (Kernel Ridge O(n³))
    rng = np.random.default_rng(42)
    sub_n = min(800, len(y_train))
    sub_idx = rng.choice(len(y_train), sub_n, replace=False)
    X_cv, y_cv = X_train[sub_idx], y_train[sub_idx]

    # 1. Tune siêu tham số
    print(f"Đang chạy CV grid search trên {sub_n} mẫu (có thể mất ~30s)...")
    best_lam, best_l, cv_res = tune_kernel_ridge(
        X_cv, y_cv,
        lam_grid=np.logspace(-1, 3, 8),
        l_grid=np.logspace(-1, 1, 5),
        k=5, seed=42
    )
    print(f"Best λ = {best_lam:.4f}, Best ℓ = {best_l:.4f}")

    # 2. So sánh với OLS
    results = compare_kernel_vs_ols(X_train, y_train, X_test, y_test,
                                    lam=best_lam, l=best_l)
    print("\n─── So sánh trên Test Set ───")
    print(results.to_string(index=False))

    # 3. Vẽ biểu đồ
    fig1 = plot_cv_heatmap(cv_res, best_lam, best_l)

    # Scatter plot: predicted vs actual
    kr_res = kernel_ridge_fit(X_train, y_train, lam=best_lam, l=best_l)
    y_pred = kernel_ridge_predict(X_train, X_test, kr_res['alpha'], l=best_l)

    fig2, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.3, s=15, color='#4477AA', edgecolors='none')
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
    ax.set_xlabel('Actual SYSTOLIC_TARGET', fontsize=12)
    ax.set_ylabel('Predicted', fontsize=12)
    ax.set_title('Kernel Ridge: Predicted vs Actual', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig1.savefig(os.path.join(out_dir, 'fig_kernel_cv_heatmap.png'), dpi=150, bbox_inches='tight')
    fig2.savefig(os.path.join(out_dir, 'fig_kernel_predicted_vs_actual.png'), dpi=150, bbox_inches='tight')
    print("\nĐã lưu 2 biểu đồ Kernel Ridge.")
    plt.close('all')
