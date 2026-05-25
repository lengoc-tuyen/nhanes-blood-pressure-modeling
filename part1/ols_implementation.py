import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


def ols_fit(X: np.ndarray, y: np.ndarray) -> dict:
    """Ước lượng OLS: β̂ = (XᵀX)⁻¹Xᵀy."""
    n, k = X.shape
    p = k - 1

    XtX = X.T @ X
    Xty = X.T @ y

    cond = np.linalg.cond(XtX)
    if cond > 1e12:
        raise ValueError(f"XᵀX gần suy biến (condition number = {cond:.2e}). "
                         "Kiểm tra đa cộng tuyến.")

    XtX_inv  = np.linalg.inv(XtX)
    beta_hat = XtX_inv @ Xty
    y_hat    = X @ beta_hat
    residuals = y - y_hat
    rss      = float(residuals @ residuals)
    sigma2   = rss / (n - p - 1)

    return {
        "beta_hat"  : beta_hat,
        "sigma2"    : sigma2,
        "y_hat"     : y_hat,
        "residuals" : residuals,
        "rss"       : rss,
        "XtX_inv"   : XtX_inv,
    }


def hat_matrix(X: np.ndarray) -> dict:
    """Tính Hat Matrix H = X(XᵀX)⁻¹Xᵀ và kiểm tra các tính chất idempotent, đối xứng."""
    XtX_inv = np.linalg.inv(X.T @ X)
    H = X @ XtX_inv @ X.T

    H2 = H @ H
    eigvals = np.linalg.eigvalsh(H)

    return {
        "H"           : H,
        "idempotent"  : np.allclose(H2, H, atol=1e-8),
        "symmetric"   : np.allclose(H, H.T, atol=1e-8),
        "rank"        : int(np.round(np.sum(eigvals))),
        "eigenvalues" : eigvals,
        "only_0_or_1" : np.all(np.isin(np.round(eigvals).astype(int), [0, 1])),
    }


def gauss_markov_demo(n: int = 100, p: int = 3,
                      n_sim: int = 1000,
                      sigma: float = 1.0,
                      seed: int = 42) -> plt.Figure:
    """
    Minh họa định lý Gauss–Markov bằng Monte Carlo.
    So sánh phương sai OLS vs một estimator tuyến tính không chệch khác.
    """
    rng = np.random.default_rng(seed)
    beta_true = np.arange(1, p + 2, dtype=float)

    X_raw = rng.standard_normal((n, p))
    X     = np.hstack([np.ones((n, 1)), X_raw])

    XtX_inv      = np.linalg.inv(X.T @ X)
    var_ols_theory = sigma**2 * np.diag(XtX_inv)

    beta_ols_list   = []
    beta_naive_list = []

    for _ in range(n_sim):
        eps = rng.normal(0, sigma, size=n)
        y   = X @ beta_true + eps

        res = ols_fit(X, y)
        beta_ols_list.append(res["beta_hat"])

        delta = rng.normal(0, 0.5, size=p + 1)
        beta_naive_list.append(res["beta_hat"] + delta)

    beta_ols   = np.array(beta_ols_list)
    beta_naive = np.array(beta_naive_list)

    mean_ols    = beta_ols.mean(axis=0)
    var_ols_sim = beta_ols.var(axis=0)
    var_naive   = beta_naive.var(axis=0)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Minh Họa Định Lý Gauss–Markov (Monte Carlo, n_sim={n_sim})",
                 fontsize=14, fontweight="bold", y=0.98)
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
                     f"Var(Naive)={var_naive[j]:.4f}", fontsize=9, fontweight="bold")
        ax.set_xlabel(f"β̂_{j}", fontsize=10)
        if j == 0:
            ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.2)

    ax_var = fig.add_subplot(gs[1, :])
    x_idx  = np.arange(p + 1)
    width  = 0.28
    ax_var.bar(x_idx - width, var_ols_theory, width,
               label="Var OLS (Lý thuyết)", color="#228833", alpha=0.85)
    ax_var.bar(x_idx,         var_ols_sim,    width,
               label="Var OLS (Monte Carlo)", color=colors_ols, alpha=0.85)
    ax_var.bar(x_idx + width, var_naive,      width,
               label="Var Naive (Higher)", color=colors_naive, alpha=0.85)
    ax_var.set_xticks(x_idx)
    ax_var.set_xticklabels([f"β_{j}" for j in range(p + 1)], fontsize=11)
    ax_var.set_ylabel("Phương sai", fontsize=12)
    ax_var.set_title("So Sánh Phương Sai: OLS vs Naive Estimator",
                     fontsize=12, fontweight="bold")
    ax_var.legend(fontsize=11)
    ax_var.grid(True, alpha=0.25, axis="y")

    bias_ols = np.abs(mean_ols - beta_true)
    textstr = (f"E[β̂_OLS] ≈ β_true  →  |bias| max = {bias_ols.max():.4f}\n"
               f"Var(β̂_OLS) ≤ Var(β̃_Naive) cho mọi j  →  OLS là BLUE ✓")
    fig.text(0.01, 0.01, textstr, fontsize=9.5, verticalalignment="bottom",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    return fig
