import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats


def model_metrics(y: np.ndarray, y_hat: np.ndarray, p: int) -> dict:
    """Tính RSS, TSS, R², R̄², và kiểm định F."""
    n      = len(y)
    y_mean = np.mean(y)

    rss = float(np.sum((y - y_hat) ** 2))
    tss = float(np.sum((y - y_mean) ** 2))
    ess = tss - rss

    r2     = 1.0 - rss / tss
    r2_adj = 1.0 - (n - 1) / (n - p - 1) * (1.0 - r2)

    f_stat   = (ess / p) / (rss / (n - p - 1))
    f_pvalue = 1.0 - stats.f.cdf(f_stat, dfn=p, dfd=n - p - 1)

    return {
        "rss"     : rss,
        "tss"     : tss,
        "ess"     : ess,
        "r2"      : r2,
        "r2_adj"  : r2_adj,
        "f_stat"  : f_stat,
        "f_pvalue": f_pvalue,
        "n"       : n,
        "p"       : p,
    }


def coef_inference(X: np.ndarray, y: np.ndarray,
                   beta_hat: np.ndarray, sigma2: float,
                   alpha: float = 0.05) -> dict:
    """Tính standard errors, t-statistics, p-values và khoảng tin cậy (1−α)%."""
    n, k = X.shape
    p    = k - 1
    dof  = n - p - 1

    XtX_inv  = np.linalg.inv(X.T @ X)
    cov_beta = sigma2 * XtX_inv
    se       = np.sqrt(np.diag(cov_beta))
    t_stats  = beta_hat / se
    p_values = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stats), df=dof))
    t_crit   = stats.t.ppf(1.0 - alpha / 2, df=dof)

    return {
        "se"       : se,
        "t_stats"  : t_stats,
        "p_values" : p_values,
        "ci_lower" : beta_hat - t_crit * se,
        "ci_upper" : beta_hat + t_crit * se,
        "t_crit"   : t_crit,
        "dof"      : dof,
        "cov_beta" : cov_beta,
    }


def residual_plots(y: np.ndarray, y_hat: np.ndarray,
                   residuals: np.ndarray,
                   title_prefix: str = "OLS") -> plt.Figure:
    """Vẽ 4 biểu đồ phân tích phần dư: Residuals vs Fitted, Q-Q, Scale-Location, Histogram."""
    std_res      = residuals / (np.std(residuals) + 1e-12)
    sqrt_abs_res = np.sqrt(np.abs(std_res))

    fig = plt.figure(figsize=(13, 10))
    fig.suptitle(f"Phân Tích Phần Dư — {title_prefix}",
                 fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

    # Residuals vs Fitted
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_hat, residuals, alpha=0.45, s=18, color="#4477AA", edgecolors="none")
    ax1.axhline(0, color="red", linewidth=1.2, linestyle="--")
    idx = np.argsort(y_hat)
    try:
        p_ = np.poly1d(np.polyfit(y_hat[idx], residuals[idx], 3))
        ax1.plot(y_hat[idx], p_(y_hat[idx]), color="red", linewidth=1.5, alpha=0.7)
    except Exception:
        pass
    ax1.set_xlabel("Fitted values (ŷ)", fontsize=11)
    ax1.set_ylabel("Residuals (ε̂)", fontsize=11)
    ax1.set_title("Residuals vs Fitted", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.25)

    # Q-Q Plot
    ax2 = fig.add_subplot(gs[0, 1])
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    ax2.scatter(osm, osr, alpha=0.45, s=18, color="#44AA77", edgecolors="none")
    ax2.plot(osm, slope * np.array(osm) + intercept,
             color="red", linewidth=1.5, label=f"R={r:.4f}")
    ax2.set_xlabel("Theoretical Quantiles", fontsize=11)
    ax2.set_ylabel("Sample Quantiles", fontsize=11)
    ax2.set_title("Normal Q-Q Plot", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.25)

    # Scale-Location
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(y_hat, sqrt_abs_res, alpha=0.45, s=18, color="#AA4477", edgecolors="none")
    idx2 = np.argsort(y_hat)
    try:
        p2 = np.poly1d(np.polyfit(y_hat[idx2], sqrt_abs_res[idx2], 2))
        ax3.plot(y_hat[idx2], p2(y_hat[idx2]), color="red", linewidth=1.5, alpha=0.7)
    except Exception:
        pass
    ax3.set_xlabel("Fitted values (ŷ)", fontsize=11)
    ax3.set_ylabel("√|Standardized Residuals|", fontsize=11)
    ax3.set_title("Scale-Location\n(Kiểm tra homoscedasticity)", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.25)

    # Histogram of Standardized Residuals
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(std_res, bins=30, color="#AAAA44", edgecolor="white",
             alpha=0.75, density=True, label="Standardized Residuals")
    xr = np.linspace(std_res.min(), std_res.max(), 200)
    ax4.plot(xr, stats.norm.pdf(xr), color="red", linewidth=2, label="N(0,1)")
    ax4.set_xlabel("Standardized Residuals", fontsize=11)
    ax4.set_ylabel("Density", fontsize=11)
    ax4.set_title("Phân Phối Phần Dư vs N(0,1)", fontsize=12, fontweight="bold")
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.25)

    return fig
