import numpy as np
from ols_implementation import ols_fit


def kfold_cv(X: np.ndarray, y: np.ndarray, k: int = 5,
             fit_func=None, fit_kwargs: dict = None) -> dict:
    """
    k-Fold Cross-Validation. Trả về CV MSE trung bình trên k folds.

    Parameters
    ----------
    fit_func   : hàm fit, mặc định ols_fit. Phải trả về dict có 'beta_hat'.
    fit_kwargs : tham số bổ sung cho fit_func (ví dụ: lam=0.1 cho ridge).
    """
    if fit_func is None:
        fit_func = ols_fit
    if fit_kwargs is None:
        fit_kwargs = {}

    n         = len(y)
    fold_size = n // k
    indices   = np.arange(n)
    np.random.shuffle(indices)

    mse_folds  = []
    rmse_folds = []

    for i in range(k):
        val_start = i * fold_size
        val_end   = (i + 1) * fold_size if i < k - 1 else n
        val_idx   = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

        result   = fit_func(X[train_idx], y[train_idx], **fit_kwargs)
        y_pred   = X[val_idx] @ result["beta_hat"]
        mse_i    = float(np.mean((y[val_idx] - y_pred) ** 2))

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
