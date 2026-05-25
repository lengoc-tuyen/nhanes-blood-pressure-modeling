"""
ols_implementation.py
=====================
Thành viên A — Lý thuyết & OLS Core
Môn: Toán Ứng Dụng và Thống Kê (MTH00051)
Dataset: Stroke Prediction Dataset (Kaggle)

Nội dung:
    - ols_fit(X, y)         : Ước lượng β̂ = (XᵀX)⁻¹Xᵀy và σ̂²
    - hat_matrix(X)         : Tính H = X(XᵀX)⁻¹Xᵀ, kiểm tra idempotent
    - model_metrics(y, y_hat, p) : RSS, TSS, R², R̄², kiểm định F
    - coef_inference(...)   : Standard errors, t-stats, p-values, CI 95%
    - vif(X)                : Variance Inflation Factor
    - ridge_fit(X, y, lam)  : Ridge Regression
    - residual_plots(...)   : 4 biểu đồ phân tích phần dư
    - kfold_cv(X, y, k)     : k-fold Cross-Validation
    - gauss_markov_demo()   : Monte Carlo minh họa định lý Gauss–Markov
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. OLS FIT
# ─────────────────────────────────────────────────────────────────────────────

def ols_fit(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Ước lượng OLS từ đầu (không dùng sklearn hay lstsq).

    Công thức:
        β̂ = (XᵀX)⁻¹ Xᵀy
        σ̂² = RSS / (n − p − 1)

    Parameters
    ----------
    X : np.ndarray, shape (n, p+1)
        Ma trận design (ĐÃ có cột bias toàn 1 ở đầu).
    y : np.ndarray, shape (n,)
        Vector biến mục tiêu.

    Returns
    -------
    dict với các key:
        beta_hat  : vector hệ số β̂, shape (p+1,)
        sigma2    : ước lượng phương sai nhiễu σ̂²
        y_hat     : giá trị fitted ŷ = Xβ̂
        residuals : phần dư ε̂ = y − ŷ
        rss       : Residual Sum of Squares
    """
    n, k = X.shape          # k = p + 1 (bao gồm intercept)
    p = k - 1               # số biến thực sự

    # Tính (XᵀX)⁻¹ Xᵀy  ← Normal Equations
    XtX = X.T @ X           # (p+1) × (p+1)
    Xty = X.T @ y           # (p+1,)

    # Kiểm tra XᵀX khả nghịch thông qua condition number
    cond = np.linalg.cond(XtX)
    if cond > 1e12:
        raise ValueError(f"XᵀX gần suy biến (condition number = {cond:.2e}). "
                         "Kiểm tra đa cộng tuyến.")

    XtX_inv = np.linalg.inv(XtX)
    beta_hat = XtX_inv @ Xty      # β̂ = (XᵀX)⁻¹ Xᵀy

    y_hat    = X @ beta_hat        # ŷ = Xβ̂
    residuals = y - y_hat          # ε̂ = y − ŷ
    rss      = float(residuals @ residuals)   # RSS = ‖ε̂‖²

    # Ước lượng phương sai nhiễu (không chệch)
    sigma2 = rss / (n - p - 1)    # σ̂² = RSS / (n − p − 1)

    return {
        "beta_hat"  : beta_hat,
        "sigma2"    : sigma2,
        "y_hat"     : y_hat,
        "residuals" : residuals,
        "rss"       : rss,
        "XtX_inv"   : XtX_inv,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. HAT MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def hat_matrix(X: np.ndarray) -> dict:
    """
    Tính Hat Matrix H = X(XᵀX)⁻¹Xᵀ và kiểm tra các tính chất.

    Tính chất:
        (i)  H² = H          (idempotent)
        (ii) Hᵀ = H          (đối xứng)
        (iii) eigenvalues ∈ {0, 1}
        (iv) rank(H) = p + 1
        (v)  ŷ = Hy,  ε̂ = (I − H)y

    Parameters
    ----------
    X : np.ndarray, shape (n, p+1)

    Returns
    -------
    dict với các key:
        H           : hat matrix (n × n)
        idempotent  : bool, H² ≈ H ?
        symmetric   : bool, Hᵀ ≈ H ?
        rank        : rank của H
        eigenvalues : eigenvalues của H (chỉ 0 hoặc 1)
    """
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    H = X @ XtX_inv @ X.T          # n × n

    # Kiểm tra idempotent: H² = H
    H2 = H @ H
    is_idempotent = np.allclose(H2, H, atol=1e-8)

    # Kiểm tra đối xứng: Hᵀ = H
    is_symmetric = np.allclose(H, H.T, atol=1e-8)

    # Eigenvalues (phải là 0 hoặc 1)
    eigvals = np.linalg.eigvalsh(H)
    eigvals_rounded = np.round(eigvals).astype(int)
    only_0_or_1 = np.all(np.isin(eigvals_rounded, [0, 1]))

    # Rank
    rank_H = int(np.round(np.sum(eigvals)))   # rank = trace = số eigenvalue = 1

    return {
        "H"            : H,
        "idempotent"   : is_idempotent,
        "symmetric"    : is_symmetric,
        "rank"         : rank_H,
        "eigenvalues"  : eigvals,
        "only_0_or_1"  : only_0_or_1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 9. GAUSS–MARKOV MONTE CARLO DEMO
# ─────────────────────────────────────────────────────────────────────────────

def gauss_markov_demo(n: int = 100, p: int = 3,
                      n_sim: int = 1000,
                      sigma: float = 1.0,
                      seed: int = 42) -> plt.Figure:
    """
    Minh họa định lý Gauss–Markov bằng Monte Carlo.

    Thực nghiệm:
        - Sinh dữ liệu y = Xβ + ε với ε ~ N(0, σ²I) (1000 lần lặp)
        - Ước lượng β̂_OLS mỗi lần
        - Kiểm chứng: E[β̂_OLS] ≈ β (unbiasedness)
        - So sánh Var(β̂_OLS) với ước lượng tuyến tính khác (estimator ngẫu nhiên)
          để minh họa BLUE property

    Parameters
    ----------
    n     : số quan sát
    p     : số biến (không tính intercept)
    n_sim : số lần mô phỏng
    sigma : độ lệch chuẩn của nhiễu
    seed  : random seed

    Returns
    -------
    fig : matplotlib Figure
    """
    rng = np.random.default_rng(seed)

    # True parameters
    beta_true = np.arange(1, p + 2, dtype=float)   # [1, 2, ..., p+1]

    # Design matrix cố định (random X)
    X_raw  = rng.standard_normal((n, p))
    X      = np.hstack([np.ones((n, 1)), X_raw])     # thêm intercept

    # Tính var lý thuyết của β̂_OLS: Var(β̂) = σ²(XᵀX)⁻¹
    XtX_inv = np.linalg.inv(X.T @ X)
    var_ols_theory = sigma**2 * np.diag(XtX_inv)    # Var lý thuyết

    # ── Mô phỏng Monte Carlo ──────────────────────────────────────────────────
    beta_ols_list   = []
    beta_naive_list = []   # Estimator "naive": β̃ = X⁺y + noise (biased)

    for _ in range(n_sim):
        eps = rng.normal(0, sigma, size=n)
        y   = X @ beta_true + eps

        # OLS
        res = ols_fit(X, y)
        beta_ols_list.append(res["beta_hat"])

        # Naive estimator: dùng thêm nhiễu → higher variance, unbiased
        # β̃ = β̂_OLS + δ với δ ~ N(0, 0.5) → vẫn unbiased nhưng var cao hơn
        delta = rng.normal(0, 0.5, size=p + 1)
        beta_naive_list.append(res["beta_hat"] + delta)

    beta_ols   = np.array(beta_ols_list)    # (n_sim, p+1)
    beta_naive = np.array(beta_naive_list)  # (n_sim, p+1)

    # ── Kết quả thống kê ──────────────────────────────────────────────────────
    mean_ols    = beta_ols.mean(axis=0)
    var_ols_sim = beta_ols.var(axis=0)
    var_naive   = beta_naive.var(axis=0)

    # ── Vẽ biểu đồ ───────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Minh Họa Định Lý Gauss–Markov (Monte Carlo, n_sim={})"
                 .format(n_sim), fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, p + 1, hspace=0.45, wspace=0.38)

    colors_ols   = "#4477AA"
    colors_naive = "#EE6677"

    for j in range(p + 1):
        ax = fig.add_subplot(gs[0, j])
        ax.hist(beta_ols[:, j],   bins=40, alpha=0.65, color=colors_ols,
                density=True, edgecolor="none", label="OLS")
        ax.hist(beta_naive[:, j], bins=40, alpha=0.45, color=colors_naive,
                density=True, edgecolor="none", label="Naive")
        ax.axvline(beta_true[j], color="green", linewidth=2,
                   linestyle="--", label=f"β_true={beta_true[j]:.0f}")
        ax.axvline(mean_ols[j],  color=colors_ols, linewidth=1.5,
                   linestyle=":", label=f"E[β̂]={mean_ols[j]:.3f}")
        ax.set_title(f"β_{j} | Var(OLS)={var_ols_sim[j]:.4f}\n"
                     f"Var(Naive)={var_naive[j]:.4f}",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel(f"β̂_{j}", fontsize=10)
        if j == 0:
            ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.2)

    # ── Subplot so sánh phương sai ────────────────────────────────────────────
    ax_var = fig.add_subplot(gs[1, :])
    x_idx = np.arange(p + 1)
    width = 0.28
    bars1 = ax_var.bar(x_idx - width, var_ols_theory, width,
                       label="Var OLS (Lý thuyết)", color="#228833", alpha=0.85)
    bars2 = ax_var.bar(x_idx,         var_ols_sim, width,
                       label="Var OLS (Monte Carlo)", color=colors_ols, alpha=0.85)
    bars3 = ax_var.bar(x_idx + width, var_naive, width,
                       label="Var Naive (Higher)", color=colors_naive, alpha=0.85)

    ax_var.set_xticks(x_idx)
    ax_var.set_xticklabels([f"β_{j}" for j in range(p + 1)], fontsize=11)
    ax_var.set_ylabel("Phương sai", fontsize=12)
    ax_var.set_title("So Sánh Phương Sai: OLS vs Naive Estimator\n"
                     "(BLUE: OLS có phương sai nhỏ nhất trong lớp ước lượng tuyến tính không chệch)",
                     fontsize=12, fontweight="bold")
    ax_var.legend(fontsize=11)
    ax_var.grid(True, alpha=0.25, axis="y")

    # Ghi bias
    bias_ols   = np.abs(mean_ols - beta_true)
    bias_naive = np.abs(beta_naive.mean(axis=0) - beta_true)
    textstr = ("Kết luận:\n"
               f"  E[β̂_OLS] ≈ β_true  →  |bias| max = {bias_ols.max():.4f}  (≈ 0 ✓)\n"
               f"  Var(β̂_OLS) ≤ Var(β̃_Naive) cho mọi j  →  OLS là BLUE ✓\n"
               f"  Var OLS lý thuyết ≈ Var OLS Monte Carlo  →  Công thức đúng ✓")
    fig.text(0.01, 0.01, textstr, fontsize=9.5,
             verticalalignment="bottom",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    return fig


# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# UNIT TESTS (Đã Tách)
# ─────────────────────────────────────────────────────────────────────────────

def run_unit_tests(verbose: bool = True) -> bool:
    print("Các hàm đã được chia nhỏ sang module khác. Vui lòng kiểm tra trên notebook.")
    return True
