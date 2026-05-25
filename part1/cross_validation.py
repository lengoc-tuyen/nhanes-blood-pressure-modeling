import numpy as np
from ols_implementation import ols_fit

# 8. K-FOLD CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def kfold_cv(X: np.ndarray, y: np.ndarray, k: int = 5,
             fit_func=None, fit_kwargs: dict = None) -> dict:
    """
    k-Fold Cross-Validation.

    Công thức:
        CV(k) = (1/k) Σᵢ MSEᵢ

    Parameters
    ----------
    X          : ma trận design (n, p+1), đã có cột bias
    y          : biến mục tiêu (n,)
    k          : số fold (mặc định 5)
    fit_func   : hàm fit (mặc định: ols_fit). Phải trả về dict có 'beta_hat'.
    fit_kwargs : tham số thêm cho fit_func (ví dụ: lam=0.1 cho ridge)

    Returns
    -------
    dict chứa: cv_score (MSE trung bình), mse_folds, rmse_folds
    """
    if fit_func is None:
        fit_func = ols_fit
    if fit_kwargs is None:
        fit_kwargs = {}

    n = len(y)
    fold_size = n // k
    indices = np.arange(n)
    np.random.shuffle(indices)

    mse_folds  = []
    rmse_folds = []

    for i in range(k):
        # Lấy chỉ số của fold i
        val_start = i * fold_size
        val_end   = (i + 1) * fold_size if i < k - 1 else n
        val_idx   = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

        X_train, y_train = X[train_idx], y[train_idx]
        X_val,   y_val   = X[val_idx],   y[val_idx]

        # Train
        result   = fit_func(X_train, y_train, **fit_kwargs)
        beta_hat = result["beta_hat"]

        # Predict
        y_pred = X_val @ beta_hat
        mse_i  = float(np.mean((y_val - y_pred) ** 2))
        mse_folds.append(mse_i)
        rmse_folds.append(np.sqrt(mse_i))

    cv_score = float(np.mean(mse_folds))

    return {
        "cv_score"   : cv_score,
        "cv_rmse"    : float(np.sqrt(cv_score)),
        "mse_folds"  : mse_folds,
        "rmse_folds" : rmse_folds,
        "k"          : k,
    }


# ─────────────────────────────────────────────────────────────────────────────
