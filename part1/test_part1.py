import unittest
import numpy as np
import sys
import os

# Thêm đường dẫn để import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ols_implementation import ols_fit, hat_matrix
from ridge_lasso import vif, ridge_fit, lasso_fit
from residual_analysis import model_metrics, coef_inference
from cross_validation import kfold_cv

class TestPart1(unittest.TestCase):

    def setUp(self):
        # Tạo dữ liệu mẫu lớn và ổn định hơn (30 dòng, 3 cột: bias + 2 đặc trưng)
        # Để đảm bảo bậc tự do luôn > 0 và không có đa cộng tuyến
        rng = np.random.default_rng(42)
        n = 30
        X_raw = rng.normal(0.0, 1.0, size=(n, 2))
        self.X_dummy = np.hstack([np.ones((n, 1)), X_raw])
        self.beta_true = np.array([2.0, 3.0, -1.5])
        self.y_dummy = self.X_dummy @ self.beta_true

    def test_ols_fit_perfect_fit(self):
        """Test 1 cho ols_fit: Trường hợp khớp hoàn hảo (no noise)"""
        res = ols_fit(self.X_dummy, self.y_dummy)
        np.testing.assert_array_almost_equal(res['beta_hat'], self.beta_true, decimal=5)
        self.assertAlmostEqual(res['rss'], 0.0, places=5)
        self.assertAlmostEqual(res['sigma2'], 0.0, places=5)

    def test_ols_fit_with_noise(self):
        """Test 2 cho ols_fit: Trường hợp có nhiễu"""
        rng = np.random.default_rng(42)
        y_noisy = self.y_dummy + rng.normal(0.0, 0.1, size=30)
        res = ols_fit(self.X_dummy, y_noisy)
        self.assertEqual(len(res['beta_hat']), 3)
        self.assertTrue(res['rss'] > 0.0)
        self.assertTrue(res['sigma2'] > 0.0)

    def test_hat_matrix_idempotent(self):
        """Test 1 cho hat_matrix: Kiểm tra tính chất idempotent H^2 = H"""
        res = hat_matrix(self.X_dummy)
        H = res['H']
        np.testing.assert_array_almost_equal(H @ H, H, decimal=6)
        self.assertTrue(res['idempotent'])

    def test_hat_matrix_properties(self):
        """Test 2 cho hat_matrix: Kiểm tra tính đối xứng và Rank"""
        res = hat_matrix(self.X_dummy)
        self.assertTrue(res['symmetric'])
        self.assertEqual(res['rank'], 3)
        self.assertTrue(res['only_0_or_1'])

    def test_model_metrics_perfect(self):
        """Test 1 cho model_metrics: Trường hợp hoàn hảo R2 = 1.0"""
        # Thêm sai số cực nhỏ 1e-10 để tránh chia cho 0 trong tính F-statistic
        res = model_metrics(self.y_dummy, self.y_dummy + 1e-10, p=2)
        self.assertAlmostEqual(res['r2'], 1.0, places=5)
        self.assertAlmostEqual(res['r2_adj'], 1.0, places=5)
        self.assertTrue(res['rss'] < 1e-18)

    def test_model_metrics_normal(self):
        """Test 2 cho model_metrics: Kiểm tra tính đúng đắn của RSS và TSS"""
        # Thêm sai số cố định
        err = np.zeros(30)
        err[0] = 0.1
        err[1] = -0.1
        y_hat = self.y_dummy + err
        res = model_metrics(self.y_dummy, y_hat, p=2)
        self.assertTrue(res['r2'] < 1.0)
        self.assertAlmostEqual(res['rss'], 0.02, places=5)
        self.assertEqual(res['n'], 30)

    def test_coef_inference_dof(self):
        """Test 1 cho coef_inference: Kiểm tra bậc tự do (dof)"""
        res = coef_inference(self.X_dummy, self.y_dummy, self.beta_true, sigma2=0.01)
        self.assertEqual(res['dof'], 30 - 2 - 1)

    def test_coef_inference_bounds(self):
        """Test 2 cho coef_inference: Kiểm tra CI chứa beta_true khi sigma2 nhỏ"""
        res = coef_inference(self.X_dummy, self.y_dummy, self.beta_true, sigma2=1e-6)
        for j in range(3):
            self.assertTrue(res['ci_lower'][j] <= self.beta_true[j] <= res['ci_upper'][j])

    def test_vif_no_multicollinearity(self):
        """Test 1 cho vif: Trường hợp các biến độc lập tuyến tính hoàn hảo"""
        # X không bao gồm bias trong hàm vif
        X_indep = self.X_dummy[:, 1:]
        vifs = vif(X_indep)
        self.assertEqual(len(vifs), 2)
        self.assertTrue(np.all(vifs < 5.0))

    def test_vif_collinear(self):
        """Test 2 cho vif: Trường hợp có đa cộng tuyến mạnh"""
        X_coll = np.zeros((30, 2))
        X_coll[:, 0] = self.X_dummy[:, 1]
        X_coll[:, 1] = self.X_dummy[:, 1] * 2.0 + 1e-4 * np.random.normal(0, 1, 30)
        vifs = vif(X_coll)
        self.assertTrue(np.all(vifs > 10.0))

    def test_ridge_fit_shrinkage(self):
        """Test 1 cho ridge_fit: Hệ số giảm dần khi lambda tăng"""
        res_ols = ridge_fit(self.X_dummy, self.y_dummy, lam=0.0)
        res_ridge = ridge_fit(self.X_dummy, self.y_dummy, lam=1000.0)
        # Hệ số OLS và Ridge co hẹp (Ridge nhỏ hơn OLS về độ lớn, trừ Intercept cột 0)
        self.assertTrue(np.abs(res_ridge['beta_hat'][1]) < np.abs(res_ols['beta_hat'][1]))
        self.assertTrue(np.abs(res_ridge['beta_hat'][2]) < np.abs(res_ols['beta_hat'][2]))

    def test_ridge_fit_zero_lambda(self):
        """Test 2 cho ridge_fit: lambda=0 trùng khớp hoàn toàn với OLS"""
        res_ols = ols_fit(self.X_dummy, self.y_dummy)
        res_ridge = ridge_fit(self.X_dummy, self.y_dummy, lam=0.0)
        np.testing.assert_array_almost_equal(res_ridge['beta_hat'], res_ols['beta_hat'], decimal=5)

    def test_lasso_fit_perfect(self):
        """Test 1 cho lasso_fit: Khớp tốt khi lambda rất nhỏ"""
        res = lasso_fit(self.X_dummy, self.y_dummy, lam=1e-5)
        np.testing.assert_array_almost_equal(res['beta_hat'], self.beta_true, decimal=2)

    def test_lasso_fit_shrinkage(self):
        """Test 2 cho lasso_fit: Hệ số co về 0 khi lambda lớn"""
        res = lasso_fit(self.X_dummy, self.y_dummy, lam=1000.0)
        # Hệ số không phải bias co về 0
        self.assertAlmostEqual(res['beta_hat'][1], 0.0, places=2)
        self.assertAlmostEqual(res['beta_hat'][2], 0.0, places=2)

    def test_kfold_cv_score_type(self):
        """Test 1 cho kfold_cv: Kiểm tra kiểu dữ liệu trả về"""
        res = kfold_cv(self.X_dummy, self.y_dummy, k=5)
        self.assertIsInstance(res['cv_score'], float)
        self.assertEqual(res['k'], 5)
        self.assertEqual(len(res['mse_folds']), 5)

    def test_kfold_cv_identical(self):
        """Test 2 cho kfold_cv: CV score cho mô hình khớp hoàn hảo có nhiễu rất nhỏ"""
        res = kfold_cv(self.X_dummy, self.y_dummy + 1e-6, k=3)
        self.assertTrue(res['cv_score'] < 1e-4)

if __name__ == '__main__':
    unittest.main()
