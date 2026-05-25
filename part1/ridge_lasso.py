import numpy as np
import matplotlib.pyplot as plt
from ols_implementation import ols_fit


def vif(X: np.ndarray) -> np.ndarray:
    """
    Tính Variance Inflation Factor cho từng biến trong X.
    X không bao gồm cột intercept. VIF > 10 → đa cộng tuyến nghiêm trọng.
    """
    n, p = X.shape
    vif_values = np.zeros(p)

    for j in range(p):
        y_j          = X[:, j]
        X_other      = np.delete(X, j, axis=1)
        X_other_bias = np.hstack([np.ones((n, 1)), X_other])

        result  = ols_fit(X_other_bias, y_j)
        y_hat_j = result["y_hat"]

        tss_j = np.sum((y_j - np.mean(y_j)) ** 2)
        rss_j = np.sum((y_j - y_hat_j) ** 2)
        r2_j  = 1.0 - rss_j / tss_j if tss_j > 0 else 0.0

        vif_values[j] = 1.0 / (1.0 - r2_j) if r2_j < 1.0 else np.inf

    return vif_values


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> dict:
    """
    Ridge Regression: β̂_ridge = (XᵀX + λI)⁻¹Xᵀy.
    Intercept (cột 0) không bị penalize.
    """
    n, k = X.shape
    I_mod = np.eye(k)
    I_mod[0, 0] = 0.0

    beta_hat  = np.linalg.inv(X.T @ X + lam * I_mod) @ (X.T @ y)
    y_hat     = X @ beta_hat
    residuals = y - y_hat

    return {
        "beta_hat"  : beta_hat,
        "y_hat"     : y_hat,
        "residuals" : residuals,
        "rss"       : float(residuals @ residuals),
        "lambda"    : lam,
    }


def ridge_trace(X: np.ndarray, y: np.ndarray,
                lambdas: np.ndarray = None,
                feature_names: list = None,
                ax: plt.Axes = None) -> plt.Axes:
    """Vẽ Ridge Trace — đường hệ số theo λ (log scale)."""
    if lambdas is None:
        lambdas = np.logspace(-3, 5, 200)

    betas = np.array([ridge_fit(X, y, lam)["beta_hat"][1:] for lam in lambdas])

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    p      = betas.shape[1]
    labels = feature_names if feature_names else [f"β_{j+1}" for j in range(p)]

    for j in range(p):
        ax.plot(lambdas, betas[:, j], linewidth=1.8, label=labels[j])

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("λ (log scale)", fontsize=12)
    ax.set_ylabel("Hệ số hồi quy", fontsize=12)
    ax.set_title("Ridge Trace — Hệ số β theo λ", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    return ax


def lasso_fit(X: np.ndarray, y: np.ndarray, lam: float,
              max_iter: int = 1000, tol: float = 1e-4) -> dict:
    """Lasso Regression bằng Coordinate Descent."""
    n, k     = X.shape
    beta_hat = np.zeros(k)

    for _ in range(max_iter):
        beta_old = beta_hat.copy()

        r = y - X @ beta_hat + X[:, 0] * beta_hat[0]
        beta_hat[0] = np.mean(r)

        for j in range(1, k):
            r   = y - X @ beta_hat + X[:, j] * beta_hat[j]
            rho = X[:, j] @ r
            z   = np.sum(X[:, j] ** 2)
            if z == 0:
                continue
            beta_hat[j] = np.sign(rho) * max(np.abs(rho) - lam / 2, 0) / z

        if np.max(np.abs(beta_hat - beta_old)) < tol:
            break

    y_hat     = X @ beta_hat
    residuals = y - y_hat

    return {
        "beta_hat"  : beta_hat,
        "y_hat"     : y_hat,
        "residuals" : residuals,
        "rss"       : float(residuals @ residuals),
        "lambda"    : lam,
    }
