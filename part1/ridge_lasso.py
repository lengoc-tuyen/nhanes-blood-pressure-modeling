import numpy as np
import matplotlib.pyplot as plt
from ols_implementation import ols_fit

# 5. VIF
# ─────────────────────────────────────────────────────────────────────────────

def vif(X: np.ndarray) -> np.ndarray:
    """
    Tính Variance Inflation Factor cho từng biến trong X.

    Công thức:
        VIFⱼ = 1 / (1 − Rⱼ²)
    trong đó Rⱼ² là R² khi hồi quy biến Xⱼ theo các biến còn lại.

    VIF > 10 → đa cộng tuyến nghiêm trọng.

    Parameters
    ----------
    X : np.ndarray, shape (n, p)   — KHÔNG bao gồm cột intercept.

    Returns
    -------
    vif_values : np.ndarray, shape (p,)
    """
    n, p = X.shape
    vif_values = np.zeros(p)

    for j in range(p):
        # Biến j là y, các biến còn lại là X_others
        y_j     = X[:, j]
        X_other = np.delete(X, j, axis=1)

        # Thêm intercept
        X_other_bias = np.hstack([np.ones((n, 1)), X_other])

        # Fit OLS
        result = ols_fit(X_other_bias, y_j)
        y_hat_j = result["y_hat"]

        # Tính R²ⱼ
        tss_j = np.sum((y_j - np.mean(y_j)) ** 2)
        rss_j = np.sum((y_j - y_hat_j) ** 2)
        r2_j  = 1.0 - rss_j / tss_j if tss_j > 0 else 0.0

        # VIFⱼ = 1 / (1 − R²ⱼ)
        vif_values[j] = 1.0 / (1.0 - r2_j) if r2_j < 1.0 else np.inf

    return vif_values


# ─────────────────────────────────────────────────────────────────────────────
# 6. RIDGE REGRESSION
# ─────────────────────────────────────────────────────────────────────────────

def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> dict:
    """
    Ridge Regression (L2 regularization).

    Công thức:
        β̂_ridge = (XᵀX + λI)⁻¹ Xᵀy

    Note: X ở đây đã bao gồm cột bias. Thông thường regularization
    KHÔNG áp dụng cho intercept — ta sẽ xử lý điều đó bằng cách
    để phần tử đầu tiên của I là 0.

    Parameters
    ----------
    X   : np.ndarray, shape (n, p+1)  — đã có cột bias
    y   : np.ndarray, shape (n,)
    lam : float — hệ số regularization λ ≥ 0

    Returns
    -------
    dict chứa: beta_hat, y_hat, residuals, rss
    """
    n, k = X.shape

    # Ma trận I nhưng không penalize intercept (phần tử [0,0] = 0)
    I_mod = np.eye(k)
    I_mod[0, 0] = 0.0

    # β̂_ridge = (XᵀX + λI)⁻¹ Xᵀy
    XtX_reg  = X.T @ X + lam * I_mod
    beta_hat = np.linalg.inv(XtX_reg) @ (X.T @ y)

    y_hat     = X @ beta_hat
    residuals = y - y_hat
    rss       = float(residuals @ residuals)

    return {
        "beta_hat"  : beta_hat,
        "y_hat"     : y_hat,
        "residuals" : residuals,
        "rss"       : rss,
        "lambda"    : lam,
    }


def ridge_trace(X: np.ndarray, y: np.ndarray,
                lambdas: np.ndarray = None,
                feature_names: list = None,
                ax: plt.Axes = None) -> plt.Axes:
    """
    Vẽ Ridge Trace — đường hệ số theo λ.

    Parameters
    ----------
    X             : ma trận design (n, p+1), đã có bias
    y             : biến mục tiêu (n,)
    lambdas       : mảng các giá trị λ cần thử
    feature_names : tên đặc trưng (không bao gồm intercept)
    ax            : matplotlib Axes (tạo mới nếu None)
    """
    if lambdas is None:
        lambdas = np.logspace(-3, 5, 200)

    betas = []
    for lam in lambdas:
        res = ridge_fit(X, y, lam)
        betas.append(res["beta_hat"][1:])   # bỏ intercept
    betas = np.array(betas)    # (n_lambda, p)

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    p = betas.shape[1]
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


# ─────────────────────────────────────────────────────────────────────────────

def lasso_fit(X: np.ndarray, y: np.ndarray, lam: float, max_iter: int = 1000, tol: float = 1e-4) -> dict:
    """
    Lasso Regression (L1 regularization) bằng Coordinate Descent.
    """
    n, k = X.shape
    beta_hat = np.zeros(k)

    for _ in range(max_iter):
        beta_old = beta_hat.copy()
        
        # Intercept update (không bị phạt)
        y_pred = X @ beta_hat
        r = y - y_pred + X[:, 0] * beta_hat[0]
        beta_hat[0] = np.mean(r)
        
        # Coefficients update
        for j in range(1, k):
            y_pred = X @ beta_hat
            r = y - y_pred + X[:, j] * beta_hat[j]
            rho = X[:, j] @ r
            z = np.sum(X[:, j]**2)
            
            if z == 0:
                continue
                
            # Soft thresholding
            beta_hat[j] = np.sign(rho) * max(np.abs(rho) - lam / 2, 0) / z
            
        if np.max(np.abs(beta_hat - beta_old)) < tol:
            break
            
    y_hat = X @ beta_hat
    residuals = y - y_hat
    rss = float(residuals @ residuals)
    return {
        "beta_hat": beta_hat,
        "y_hat": y_hat,
        "residuals": residuals,
        "rss": rss,
        "lambda": lam
    }
