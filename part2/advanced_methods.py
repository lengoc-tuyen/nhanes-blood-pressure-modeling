"""
Kernel Ridge Regression (RBF kernel, closed-form) và so sánh với OLS.
"""

from __future__ import annotations

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_PART1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'part1')
sys.path.insert(0, _PART1)

# pyrefly: ignore [missing-import]
from ols_implementation import ols_fit
# pyrefly: ignore [missing-import]
from residual_analysis import model_metrics


def rbf_kernel(X1: np.ndarray, X2: np.ndarray, l: float = 1.0) -> np.ndarray:
    """Gram matrix với RBF kernel: K_ij = exp(−‖xi − xj‖² / 2ℓ²)."""
    sq_norm_1 = np.sum(X1 ** 2, axis=1, keepdims=True)
    sq_norm_2 = np.sum(X2 ** 2, axis=1, keepdims=True).T
    sq_dist   = np.maximum(sq_norm_1 + sq_norm_2 - 2.0 * (X1 @ X2.T), 0.0)
    return np.exp(-sq_dist / (2.0 * l ** 2))


def kernel_ridge_fit(
    X_train: np.ndarray,
    y_train: np.ndarray,
    lam: float = 1.0,
    l: float = 1.0,
) -> dict:
    """
    Huấn luyện Kernel Ridge Regression: α = (K + λI)⁻¹y.
    X_train không có cột bias.
    """
    n = X_train.shape[0]
    K     = rbf_kernel(X_train, X_train, l)
    alpha = np.linalg.solve(K + lam * np.eye(n), y_train)
    y_hat = K @ alpha

    return {
        'alpha'  : alpha,
        'K_train': K,
        'X_train': X_train,
        'lam'    : lam,
        'l'      : l,
        'y_hat'  : y_hat,
        'rss'    : float(np.sum((y_train - y_hat) ** 2)),
    }


def kernel_ridge_predict(
    X_train: np.ndarray,
    X_test: np.ndarray,
    alpha: np.ndarray,
    l: float = 1.0,
) -> np.ndarray:
    """Dự đoán: ŷ(x*) = k(x*, X_train)ᵀ α."""
    return rbf_kernel(X_test, X_train, l) @ alpha


def tune_kernel_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    lam_grid: np.ndarray | None = None,
    l_grid: np.ndarray | None = None,
    k: int = 5,
    seed: int = 42,
) -> tuple[float, float, dict]:
    """Grid search k-fold CV để tìm (λ*, ℓ*) tối ưu cho Kernel Ridge."""
    if lam_grid is None:
        lam_grid = np.logspace(-2, 3, 10)
    if l_grid is None:
        l_grid = np.logspace(-1, 1, 5)

    np.random.seed(seed)
    folds     = np.array_split(np.random.permutation(len(y_train)), k)
    cv_matrix = np.zeros((len(lam_grid), len(l_grid)))

    for i, lam in enumerate(lam_grid):
        for j, l in enumerate(l_grid):
            fold_mses = []
            for f_idx in range(k):
                val_idx   = folds[f_idx]
                train_idx = np.concatenate([folds[m] for m in range(k) if m != f_idx])
                res   = kernel_ridge_fit(X_train[train_idx], y_train[train_idx], lam=lam, l=l)
                y_hat = kernel_ridge_predict(X_train[train_idx], X_train[val_idx], res['alpha'], l=l)
                fold_mses.append(float(np.mean((y_train[val_idx] - y_hat) ** 2)))
            cv_matrix[i, j] = float(np.mean(fold_mses))

    best_i, best_j = divmod(int(np.argmin(cv_matrix)), len(l_grid))
    return float(lam_grid[best_i]), float(l_grid[best_j]), {
        'lam_grid' : lam_grid,
        'l_grid'   : l_grid,
        'cv_matrix': cv_matrix,
    }


def compare_kernel_vs_ols(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    lam: float = 1.0,
    l: float = 1.0,
) -> 'pd.DataFrame':
    """So sánh Kernel Ridge và OLS trên test set. X không có cột bias."""
    import pandas as pd

    X_tr_b = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_te_b = np.hstack([np.ones((X_test.shape[0],  1)), X_test])
    p      = X_train.shape[1]

    y_ols = X_te_b @ ols_fit(X_tr_b, y_train)['beta_hat']
    y_kr  = kernel_ridge_predict(X_train, X_test,
                                 kernel_ridge_fit(X_train, y_train, lam=lam, l=l)['alpha'], l=l)

    rows = []
    for name, y_hat in [('OLS', y_ols), ('Kernel Ridge', y_kr)]:
        m = model_metrics(y_test, y_hat, p)
        rows.append({
            'Model' : name,
            'MAE'   : round(float(np.mean(np.abs(y_test - y_hat))), 4),
            'RMSE'  : round(float(np.sqrt(np.mean((y_test - y_hat) ** 2))), 4),
            'R2'    : round(m['r2'],     4),
            'R2_adj': round(m['r2_adj'], 4),
        })
    return pd.DataFrame(rows)


def plot_cv_heatmap(cv_results: dict, best_lam: float, best_l: float) -> plt.Figure:
    """Heatmap CV MSE theo (λ, ℓ), đánh dấu tham số tốt nhất."""
    import seaborn as sns

    lam_grid  = cv_results['lam_grid']
    l_grid    = cv_results['l_grid']
    cv_matrix = cv_results['cv_matrix']

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(cv_matrix,
                xticklabels=[f'{v:.2f}' for v in l_grid],
                yticklabels=[f'{v:.3f}' for v in lam_grid],
                annot=True, fmt='.1f', cmap='YlOrRd_r', ax=ax,
                cbar_kws={'label': 'CV MSE'})
    ax.set_xlabel('Length-scale ℓ', fontsize=12)
    ax.set_ylabel('λ', fontsize=12)
    ax.set_title(f'Kernel Ridge CV — Best: λ={best_lam:.3f}, ℓ={best_l:.2f}',
                 fontsize=13, fontweight='bold')

    best_i = int(np.argmin(np.abs(lam_grid - best_lam)))
    best_j = int(np.argmin(np.abs(l_grid   - best_l)))
    ax.add_patch(plt.Rectangle((best_j, best_i), 1, 1,
                                fill=False, edgecolor='blue', linewidth=3))
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    import pandas as pd
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_pipeline import DataPipeline

    CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'data', 'processed', 'nhanes_pre_pipeline.csv')

    CONTINUOUS  = ['RIDAGEYR', 'BMXBMI', 'BPXOPLS', 'LBXTC', 'LBXSCR']
    CATEGORICAL = ['RIAGENDR', 'SMQ020', 'DIQ010']
    INVALID     = {'SMQ020': [7.0, 9.0]}

    pipe = DataPipeline(CONTINUOUS, CATEGORICAL,
                        target_col='SYSTOLIC_TARGET',
                        invalid_cat_values=INVALID)
    X_tr_df, X_te_df, y_train, y_test = pipe.load_and_split(CSV, seed=42)
    X_train = pipe.fit_transform(X_tr_df)[0][:, 1:]
    X_test  = pipe.transform(X_te_df)[0][:, 1:]

    print("=== Kernel Ridge Regression ===\n")

    rng     = np.random.default_rng(42)
    sub_idx = rng.choice(len(y_train), min(800, len(y_train)), replace=False)
    X_cv, y_cv = X_train[sub_idx], y_train[sub_idx]

    print(f"Đang chạy CV grid search trên {len(sub_idx)} mẫu...")
    best_lam, best_l, cv_res = tune_kernel_ridge(
        X_cv, y_cv,
        lam_grid=np.logspace(-1, 3, 8),
        l_grid=np.logspace(-1, 1, 5),
        k=5, seed=42,
    )
    print(f"Best λ = {best_lam:.4f}, Best ℓ = {best_l:.4f}")

    results = compare_kernel_vs_ols(X_train, y_train, X_test, y_test,
                                    lam=best_lam, l=best_l)
    print("\n─── So sánh trên Test Set ───")
    print(results.to_string(index=False))

    fig1 = plot_cv_heatmap(cv_res, best_lam, best_l)

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
