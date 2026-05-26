import unittest
import numpy as np
import pandas as pd
import sys
import os

# Thêm đường dẫn để import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_pipeline import EDA, DataPipeline

class TestPart2(unittest.TestCase):

    def setUp(self):
        # Dữ liệu kiểm định mẫu chứa các biến liên tục và phân loại có missing values
        self.df_dummy = pd.DataFrame({
            'RIDAGEYR': [20.0, 40.0, 60.0, 80.0, np.nan],
            'BMXBMI':   [18.0, 22.0, np.nan, 30.0, 26.0],
            'BPXOPLS':  [60.0, 70.0, 80.0, np.nan, 90.0],
            'LBXTC':    [150.0, np.nan, 200.0, 220.0, 180.0],
            'LBXSCR':   [0.6, 0.8, 1.0, 1.2, 0.9],  # Không có NaN
            'RIAGENDR': [1.0, 2.0, 1.0, 2.0, 1.0],
            'SMQ020':   [1.0, 2.0, 1.0, 2.0, 7.0],  # 7.0 là invalid
            'DIQ010':   [1.0, 2.0, 3.0, 1.0, 2.0],
            'SYSTOLIC_TARGET': [110.0, 120.0, 130.0, 140.0, 125.0]
        })

    def test_eda_mean_imple_no_nan(self):
        """Test 1 cho EDA.meanImple: Biến không có NaN"""
        eda = EDA(self.df_dummy[['LBXSCR']].copy())
        means = eda.meanImple()
        # Mean của [0.6, 0.8, 1.0, 1.2, 0.9] là 0.9
        self.assertAlmostEqual(means[0], 0.9, places=5)

    def test_eda_mean_imple_with_nan(self):
        """Test 2 cho EDA.meanImple: Biến có chứa NaN"""
        eda = EDA(self.df_dummy[['RIDAGEYR']].copy())
        means = eda.meanImple()
        # Mean của [20.0, 40.0, 60.0, 80.0] (bỏ qua NaN) là 50.0
        self.assertAlmostEqual(means[0], 50.0, places=5)

    def test_eda_median_quantiles_odd(self):
        """Test 1 cho EDA.medianAndQuantilesImple: Số lượng phần tử lẻ"""
        df_odd = pd.DataFrame({'val': [10.0, 20.0, 30.0, 40.0, 50.0]})
        eda = EDA(df_odd)
        quantiles = eda.medianAndQuantilesImple()
        q1, median_val, q3 = quantiles[0]
        self.assertAlmostEqual(median_val, 30.0, places=5)
        self.assertAlmostEqual(q1, 20.0, places=5)
        self.assertAlmostEqual(q3, 40.0, places=5)

    def test_eda_median_quantiles_even(self):
        """Test 2 cho EDA.medianAndQuantilesImple: Số lượng phần tử chẵn"""
        df_even = pd.DataFrame({'val': [10.0, 20.0, 30.0, 40.0]})
        eda = EDA(df_even)
        quantiles = eda.medianAndQuantilesImple()
        q1, median_val, q3 = quantiles[0]
        # Median của [10, 20, 30, 40] là 25.0
        self.assertAlmostEqual(median_val, 25.0, places=5)

    def test_eda_std_imple_no_nan(self):
        """Test 1 cho EDA.stdImple: Độ lệch chuẩn không có NaN"""
        df_std = pd.DataFrame({'val': [1.0, 2.0, 3.0]})
        eda = EDA(df_std)
        stds = eda.stdImple()
        # std của [1, 2, 3] với ddof=1 là 1.0
        self.assertAlmostEqual(stds[0], 1.0, places=5)

    def test_eda_std_imple_with_nan(self):
        """Test 2 cho EDA.stdImple: Có chứa NaN"""
        eda = EDA(self.df_dummy[['BMXBMI']].copy())
        stds = eda.stdImple()
        # Cột BMXBMI có [18.0, 22.0, 30.0, 26.0] (bỏ qua NaN)
        # Mean = 24.0, std (ddof=1) = sqrt(((18-24)^2 + (22-24)^2 + (30-24)^2 + (26-24)^2)/3) = sqrt(80/3)
        self.assertAlmostEqual(stds[0], np.sqrt(80.0/3.0), places=5)

    def test_pipeline_standardization(self):
        """Test 1 cho DataPipeline: Kiểm tra tính đúng đắn của chuẩn hóa Z-score"""
        CONTINUOUS  = ['RIDAGEYR', 'BMXBMI', 'BPXOPLS', 'LBXTC', 'LBXSCR']
        CATEGORICAL = ['RIAGENDR', 'SMQ020', 'DIQ010']
        INVALID     = {'SMQ020': [7.0, 9.0]}

        pipe = DataPipeline(CONTINUOUS, CATEGORICAL, target_col='SYSTOLIC_TARGET', invalid_cat_values=INVALID, k=2)
        df_large = pd.concat([self.df_dummy, self.df_dummy], ignore_index=True)

        pipe.fit(df_large)
        X_trans, feat_names = pipe.transform(df_large)

        # Trích xuất cột LBXSCR (index 5) - cột này không có NaN nên không bị ảnh hưởng bởi imputation
        lbxscr_trans = X_trans[:, 5]
        # Sau khi chuẩn hóa Z-score, mean trên tập Train phải bằng 0.0
        self.assertAlmostEqual(float(np.mean(lbxscr_trans)), 0.0, places=5)
        self.assertAlmostEqual(float(np.std(lbxscr_trans, ddof=1)), 1.0, places=5)

    def test_pipeline_no_nan_left(self):
        """Test 2 cho DataPipeline: Đảm bảo không còn bất kỳ giá trị NaN nào sau khi transform"""
        CONTINUOUS  = ['RIDAGEYR', 'BMXBMI', 'BPXOPLS', 'LBXTC', 'LBXSCR']
        CATEGORICAL = ['RIAGENDR', 'SMQ020', 'DIQ010']
        INVALID     = {'SMQ020': [7.0, 9.0]}

        pipe = DataPipeline(CONTINUOUS, CATEGORICAL, target_col='SYSTOLIC_TARGET', invalid_cat_values=INVALID, k=2)
        df_large = pd.concat([self.df_dummy, self.df_dummy], ignore_index=True)

        pipe.fit(df_large)
        X_trans, _ = pipe.transform(df_large)

        # Kiểm tra không có giá trị NaN nào trong mảng NumPy kết quả
        self.assertFalse(np.isnan(X_trans).any())

if __name__ == '__main__':
    unittest.main()
