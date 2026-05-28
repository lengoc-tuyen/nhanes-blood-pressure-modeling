"""
Feature selection, huấn luyện và so sánh OLS / Ridge trên NHANES dataset.
Tất cả thuật toán dùng hàm from scratch từ part1.
"""

from __future__ import annotations

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

_PART1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'part1')
_PART2 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PART1)
sys.path.insert(0, _PART2)  # part2/residual_analysis.py (Cook's Distance) takes precedence over part1

# pyrefly: ignore [missing-import]
from ols_implementation import ols_fit
# pyrefly: ignore [missing-import]
from ridge_lasso import ridge_fit, vif
# pyrefly: ignore [missing-import]
from residual_analysis import model_metrics, coef_inference, residual_plots
# pyrefly: ignore [missing-import]
from cross_validation import kfold_cv

def select_features_by_pvalue(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    threshold: float = 0.05,
) -> tuple[np.ndarray, list[int], list[str]]:
    """Loại biến không có ý nghĩa thống kê (p-value > threshold). Intercept luôn giữ."""
    result = ols_fit(X_train, y_train)
    infer  = coef_inference(X_train, y_train,
                            result['beta_hat'], result['sigma2'])
    p_vals = infer['p_values']

    kept_indices = [0] + [j for j in range(1, X_train.shape[1])
                          if p_vals[j] <= threshold]
    kept_names   = [feature_names[j] for j in kept_indices]

    removed = [feature_names[j] for j in range(1, X_train.shape[1])
               if p_vals[j] > threshold]
    if removed:
        print(f"  p-value selection: loại {removed}")
    else:
        print("  p-value selection: giữ tất cả biến")

    return X_train[:, kept_indices], kept_indices, kept_names


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE SELECTION — VIF
# ─────────────────────────────────────────────────────────────────────────────

def select_features_by_vif(
    X_train: np.ndarray,
    feature_names: list[str],
    threshold: float = 10.0,
) -> tuple[np.ndarray, list[int], list[str]]:
    """Loại iteratively biến có VIF cao nhất (> threshold). Intercept luôn giữ."""
    current_idx   = list(range(X_train.shape[1]))
    current_names = list(feature_names)

    while True:
        feature_idx = [i for i, n in zip(current_idx, current_names) if n != 'bias']
        if len(feature_idx) < 2:
            break

        X_feat   = X_train[:, feature_idx]
        vif_vals = vif(X_feat)

        max_vif = float(vif_vals.max())
        if max_vif <= threshold:
            break

        worst_local      = int(vif_vals.argmax())
        worst_global_idx = feature_idx[worst_local]
        worst_name       = current_names[current_idx.index(worst_global_idx)]
        print(f"  VIF selection: loại '{worst_name}' (VIF={max_vif:.2f})")

        pos = current_idx.index(worst_global_idx)
        current_idx.pop(pos)
        current_names.pop(pos)

    return X_train[:, current_idx], current_idx, current_names


# ─────────────────────────────────────────────────────────────────────────────
# 3. TÌM LAMBDA TỐI ƯU CHO RIDGE
# ─────────────────────────────────────────────────────────────────────────────

def find_best_lambda(
    X_train: np.ndarray,
    y_train: np.ndarray,
    lambdas: np.ndarray | None = None,
    k: int = 5,
    seed: int = 42,
) -> tuple[float, dict]:
    """Chọn λ tối ưu cho Ridge bằng k-fold CV. Trả về (best_lambda, cv_results)."""
    if lambdas is None:
        lambdas = np.logspace(-3, 4, 50)

    np.random.seed(seed)
    cv_scores = []
    for lam in lambdas:
        res = kfold_cv(X_train, y_train, k=k,
                       fit_func=ridge_fit, fit_kwargs={'lam': float(lam)})
        cv_scores.append(res['cv_score'])

    best_idx    = int(np.argmin(cv_scores))
    best_lambda = float(lambdas[best_idx])

    return best_lambda, {'lambdas': lambdas, 'cv_scores': cv_scores, 'best_idx': best_idx}


# ─────────────────────────────────────────────────────────────────────────────
# 4. HUẤN LUYỆN NHIỀU MÔ HÌNH
# ─────────────────────────────────────────────────────────────────────────────

def train_multiple_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    pvalue_threshold: float = 0.05,
    vif_threshold: float = 10.0,
    cv_k: int = 5,
    seed: int = 42,
) -> dict:
    """Huấn luyện 3 mô hình: OLS full, OLS chọn biến (p-value + VIF), Ridge (CV λ)."""
    models = {}
    p = X_train.shape[1] - 1
    print("─── Model 1: OLS cơ bản ───")
    res1     = ols_fit(X_train, y_train)
    metrics1 = model_metrics(y_train, res1['y_hat'], p)
    print(f"  R²={metrics1['r2']:.4f}, R²_adj={metrics1['r2_adj']:.4f}, "
          f"F={metrics1['f_stat']:.2f} (p={metrics1['f_pvalue']:.4e})")

    models['ols_full'] = {
        **res1,
        'feature_names': list(feature_names),
        'kept_indices' : list(range(X_train.shape[1])),
        'metrics'      : metrics1,
    }

    # ── Model 2: OLS chọn biến ───────────────────────────────────────────────
    print("\n─── Model 2: OLS chọn biến ───")
    X_sel, kept_p, names_p = select_features_by_pvalue(
        X_train, y_train, feature_names, pvalue_threshold
    )
    X_sel, kept_v, names_v = select_features_by_vif(X_sel, names_p, vif_threshold)

    # Ánh xạ chỉ số về X_train gốc
    final_kept = [kept_p[i] for i in kept_v]

    res2     = ols_fit(X_sel, y_train)
    metrics2 = model_metrics(y_train, res2['y_hat'], X_sel.shape[1] - 1)
    print(f"  Giữ {len(names_v)} biến: {names_v}")
    print(f"  R²={metrics2['r2']:.4f}, R²_adj={metrics2['r2_adj']:.4f}")

    models['ols_selective'] = {
        **res2,
        'feature_names': names_v,
        'kept_indices' : final_kept,
        'metrics'      : metrics2,
    }

    print("\n─── Model 3: Ridge Regression ───")
    best_lam, cv_res = find_best_lambda(X_train, y_train, k=cv_k, seed=seed)
    print(f"  Best λ = {best_lam:.6f}  (CV MSE = {min(cv_res['cv_scores']):.4f})")

    res3     = ridge_fit(X_train, y_train, lam=best_lam)
    metrics3 = model_metrics(y_train, res3['y_hat'], p)
    print(f"  R²={metrics3['r2']:.4f}, R²_adj={metrics3['r2_adj']:.4f}")

    models['ridge'] = {
        **res3,
        'feature_names': list(feature_names),
        'kept_indices' : list(range(X_train.shape[1])),
        'lambda'       : best_lam,
        'cv_results'   : cv_res,
        'metrics'      : metrics3,
    }

    return models


# ─────────────────────────────────────────────────────────────────────────────
# 5. ĐÁNH GIÁ TRÊN TEST SET
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_on_test_set(
    models_dict: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Tính MAE, RMSE, R², R²_adj trên test set. Trả về DataFrame so sánh các mô hình."""
    rows = []

    for name, model in models_dict.items():
        beta       = model['beta_hat']
        kept_idx   = model.get('kept_indices', list(range(X_test.shape[1])))
        X_sub      = X_test[:, kept_idx]
        y_hat      = X_sub @ beta

        mae     = float(np.mean(np.abs(y_test - y_hat)))
        rmse    = float(np.sqrt(np.mean((y_test - y_hat) ** 2)))
        metrics = model_metrics(y_test, y_hat, len(kept_idx) - 1)

        rows.append({
            'Model'    : name,
            'MAE'      : round(mae,  4),
            'RMSE'     : round(rmse, 4),
            'R2'       : round(metrics['r2'],     4),
            'R2_adj'   : round(metrics['r2_adj'], 4),
            '#Features': len(kept_idx) - 1,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 6. VẼ FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(
    models_dict: dict,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """Horizontal bar chart |β_j| cho từng mô hình (bỏ intercept)."""
    n_models = len(models_dict)
    fig, axes = plt.subplots(1, n_models, figsize=figsize, squeeze=False)
    fig.suptitle("Feature Importance — |β_j| (sau Z-score chuẩn hóa)",
                 fontsize=13, fontweight='bold')
    colors = ['#4477AA', '#44AA77', '#EE6677']

    for ax, (name, model), color in zip(axes[0], models_dict.items(), colors):
        beta  = model['beta_hat'][1:]
        names = model['feature_names'][1:]
        order = np.argsort(np.abs(beta))
        ax.barh([names[i] for i in order], [abs(beta[i]) for i in order],
                color=color, edgecolor='white', alpha=0.85)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xlabel("|β_j|", fontsize=10)
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. VẼ SO SÁNH MÔ HÌNH
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(
    results_df: pd.DataFrame,
    figsize: tuple = (13, 4),
) -> plt.Figure:
    """Grouped bar chart so sánh MAE, RMSE, R² trên test set."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle("So Sánh Hiệu Năng Mô Hình (Test Set)", fontsize=13,
                 fontweight='bold')
    colors = ['#4477AA', '#44AA77', '#EE6677']

    for ax, metric, color in zip(axes, ['MAE', 'RMSE', 'R2'], colors):
        vals = results_df[metric].tolist()
        bars = ax.bar(results_df['Model'], vals, color=color,
                      edgecolor='white', alpha=0.85)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_ylabel(metric)
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + bar.get_height() * 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        ax.set_xticklabels(results_df['Model'], rotation=15, ha='right')

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 8. RESIDUAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def run_residual_analysis(
    best_model_key: str,
    models_dict: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> plt.Figure:
    """Vẽ 4 biểu đồ phân tích phần dư cho mô hình chỉ định."""
    model    = models_dict[best_model_key]
    beta     = model['beta_hat']
    kept_idx = model.get('kept_indices', list(range(X_test.shape[1])))

    y_hat     = X_test[:, kept_idx] @ beta
    residuals = y_test - y_hat

    return residual_plots(y_test, y_hat, residuals,
                          X=X_test[:, kept_idx],
                          title_prefix=f"Model: {best_model_key}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. VẼ CV LAMBDA CURVE
# ─────────────────────────────────────────────────────────────────────────────

def plot_cv_lambda(cv_results: dict, best_lambda: float) -> plt.Figure:
    """Vẽ đường CV MSE theo λ (log scale) và đánh dấu best λ."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogx(cv_results['lambdas'], cv_results['cv_scores'],
                color='#4477AA', linewidth=2)
    ax.axvline(best_lambda, color='red', linestyle='--',
               label=f'Best λ = {best_lambda:.4f}')
    ax.set_xlabel("λ (log scale)", fontsize=12)
    ax.set_ylabel("CV MSE", fontsize=12)
    ax.set_title("Cross-Validation: Chọn λ cho Ridge Regression",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Minh họa luồng DataPipeline → ModelComparison
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
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
    X_train, feat_names = pipe.fit_transform(X_tr_df)
    X_test,  _          = pipe.transform(X_te_df)

    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"Features: {feat_names}\n")

    models = train_multiple_models(X_train, y_train, feat_names, seed=42)

    print("\n─── KẾT QUẢ TRÊN TEST SET ───")
    results = evaluate_on_test_set(models, X_test, y_test)
    print(results.to_string(index=False))

    best_key = results.loc[results['R2'].idxmax(), 'Model']
    print(f"\nMô hình tốt nhất: {best_key}")

    fig1 = plot_feature_importance(models)
    fig2 = plot_model_comparison(results)
    fig3 = run_residual_analysis(best_key, models, X_test, y_test)
    fig4 = plot_cv_lambda(models['ridge']['cv_results'], models['ridge']['lambda'])
    plt.show()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig1.savefig(os.path.join(out_dir, 'fig_feature_importance.png'), dpi=150, bbox_inches='tight')
    fig2.savefig(os.path.join(out_dir, 'fig_model_comparison.png'),   dpi=150, bbox_inches='tight')
    fig3.savefig(os.path.join(out_dir, 'fig_residual_analysis.png'),  dpi=150, bbox_inches='tight')
    fig4.savefig(os.path.join(out_dir, 'fig_cv_lambda.png'),          dpi=150, bbox_inches='tight')
    print("Đã lưu 4 biểu đồ.")
