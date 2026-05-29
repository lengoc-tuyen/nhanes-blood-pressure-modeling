"""
EDA  : thống kê mô tả, phát hiện outlier, imputation, visualisation.
DataPipeline : fit/transform theo chuẩn sklearn để ngăn data leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def load_data(path: str = 'data/processed/nhanes_pre_pipeline.csv') -> pd.DataFrame:
    return pd.read_csv(path)

# EDA CLASS  (dùng trong notebook)
class EDA:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def meanImple(self) -> list:
        ans = []
        for i in self.df:
            totalSum = 0
            numOfVal = 0
            for j in self.df[i]:
                if not pd.isna(j):
                    numOfVal += 1
                    totalSum += j
            if numOfVal > 0:
                totalSum /= numOfVal
            else:
                totalSum = 0
            ans.append(totalSum)
        return ans

    def medianAndQuantilesImple(self) -> list[list]:
        ans = []
        for i in self.df:
            tempList = sorted([x for x in self.df[i] if not pd.isna(x)])
            l = len(tempList)
            if l == 0:
                ans.append((None, None, None))
                continue
            q2 = (tempList[l//2] + tempList[l//2 - 1]) / 2 if l % 2 == 0 else tempList[l//2]
            idx1 = l // 4
            q1 = (tempList[idx1] + tempList[idx1 - 1]) / 2 if l % 4 == 0 else tempList[idx1]
            idx3 = (3 * l) // 4
            q3 = (tempList[idx3] + tempList[idx3 - 1]) / 2 if (3*l) % 4 == 0 else tempList[idx3]
            ans.append((q1, q2, q3))
        return ans

    def stdImple(self) -> list:
        ans = []
        for i in self.df:
            vals = [j for j in self.df[i] if not pd.isna(j)]
            if len(vals) <= 1:
                ans.append(0)
                continue
            mean_val = sum(vals) / len(vals)
            diff = sum((v - mean_val) ** 2 for v in vals) / (len(vals) - 1)
            ans.append(diff ** 0.5)
        return ans

    def minImple(self) -> list:
        return [min((x for x in self.df[c] if not pd.isna(x)), default=None) for c in self.df]

    def maxImple(self) -> list:
        return [max((x for x in self.df[c] if not pd.isna(x)), default=None) for c in self.df]

    def missingRate(self) -> list:
        return [self.df[c].isna().sum() / len(self.df[c]) for c in self.df]

    def plot_target_histogram(self) -> None:
        col = 'SYSTOLIC_TARGET'
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df[col].dropna(), kde=True, color='royalblue', bins=30)
        plt.title('Phân phối Huyết áp tâm thu trung bình', fontsize=15)
        plt.xlabel('Huyết áp (mmHg)', fontsize=12)
        plt.ylabel('Tần suất', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_target_boxplot(self) -> None:
        col = 'SYSTOLIC_TARGET'
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=self.df[col].dropna(), color='salmon', fliersize=5)
        plt.title('Boxplot Huyết áp tâm thu trung bình', fontsize=15)
        plt.xlabel('Huyết áp (mmHg)', fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_correlation_heatmap(self) -> None:
        num_cols = self.df.select_dtypes(include='number').columns.tolist()
        corr = self.df[num_cols].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Ma trận tương quan giữa các biến', fontsize=15)
        plt.tight_layout()
        plt.show()

    def check_duplicates(self) -> None:
        print(f"Dòng trùng hoàn toàn: {self.df.duplicated().sum()}")
        if 'SEQN' in self.df.columns:
            print(f"ID trùng: {self.df['SEQN'].duplicated().sum()}")

    def detectOutliers(self) -> None:
        limits = {
            'RIDAGEYR': (0.1, 100), 'BMXBMI': (10, 90),
            'BPXOSY1': (40, 260),   'BPXOSY2': (40, 260), 'BPXOSY3': (40, 260),
            'BPXOPLS1': (30, 220),  'BPXOPLS2': (30, 220), 'BPXOPLS3': (30, 220),
            'LBXTC': (50, 500),     'LBXSCR': (0.1, 10),
        }
        cat_valid = {'RIAGENDR': [1, 2], 'DIQ010': [1, 2, 3]}
        mask = pd.Series(True, index=self.df.index)
        for col, (lo, hi) in limits.items():
            if col in self.df.columns:
                mask &= (self.df[col].between(lo, hi) | self.df[col].isna())
        for col, vals in cat_valid.items():
            if col in self.df.columns:
                mask &= (self.df[col].isin(vals) | self.df[col].isna())
        removed = (~mask).sum()
        self.df = self.df[mask].copy()
        print(f"detectOutliers: loại {removed} hàng → còn {len(self.df)}")

    def detectOutliersIQR(self) -> None:
        iqr_cols = ['RIDAGEYR', 'BMXBMI', 'LBXTC', 'LBXSCR', 'BPXOPLS1', 'BPXOPLS2', 'BPXOPLS3']
        quantiles = self.medianAndQuantilesImple()
        all_cols = list(self.df.columns)
        mask = pd.Series(True, index=self.df.index)
        for col in iqr_cols:
            if col not in self.df.columns:
                continue
            idx = all_cols.index(col)
            q1, _, q3 = quantiles[idx]
            if q1 is None:
                continue
            iqr = q3 - q1
            mask &= (self.df[col].between(q1 - 2*iqr, q3 + 2*iqr) | self.df[col].isna())
        removed = (~mask).sum()
        self.df = self.df[mask].copy()
        print(f"detectOutliersIQR: loại {removed} hàng → còn {len(self.df)}")

    def createAverageColumns(self) -> None:
        self.df['SYSTOLIC_TARGET'] = self.df[['BPXOSY1','BPXOSY2','BPXOSY3']].mean(axis=1)
        self.df['BPXOPLS'] = self.df[['BPXOPLS1','BPXOPLS2','BPXOPLS3']].mean(axis=1)
        self.df.drop(columns=['BPXOSY1','BPXOSY2','BPXOSY3','BPXOPLS1','BPXOPLS2','BPXOPLS3'],
                     inplace=True, errors='ignore')

    def dropCol(self) -> None:
        self.df.drop(columns=['BMXHT'], inplace=True, errors='ignore')

    def impute_knn_all_types(self, k: int = 5) -> None:
        import math
        cont_cols = ['RIDAGEYR', 'BMXBMI', 'BPXOSY1', 'BPXOSY2', 'BPXOSY3',
                     'BPXOPLS1', 'BPXOPLS2', 'BPXOPLS3', 'LBXTC', 'LBXSCR']
        cat_cols  = ['RIAGENDR', 'DIQ010', 'SMQ020']
        all_target = cont_cols + cat_cols
        data_list = self.df.to_dict('records')
        min_max = {}
        for col in cont_cols:
            if col in self.df.columns:
                vals = [r[col] for r in data_list if r.get(col) is not None and not math.isnan(r[col])]
                if vals:
                    min_max[col] = (min(vals), max(vals))
        complete, missing_info = [], []
        for idx, row in enumerate(data_list):
            has_missing = any(row.get(c) is None or (isinstance(row.get(c), float) and math.isnan(row[c]))
                              for c in all_target if c in row)
            has_all_cont = all(row.get(c) is not None and not math.isnan(row[c])
                               for c in cont_cols if c in row)
            if has_all_cont:
                complete.append(row)
            if has_missing:
                missing_info.append((idx, row))
        print(f"k-NN (k={k}): {len(missing_info)} dòng cần xử lý...")
        for idx_m, row_m in missing_info:
            known = [c for c in cont_cols if c in row_m and row_m.get(c) is not None
                     and not math.isnan(row_m[c])]
            if not known:
                continue
            dists = []
            for rc in complete:
                s = sum(((row_m[c] - rc[c]) / (max(min_max[c][1] - min_max[c][0], 1e-8))) ** 2
                        for c in known)
                dists.append((s, rc))
            dists.sort(key=lambda x: x[0])
            nn = [r for _, r in dists[:k]]
            for c in cont_cols:
                if c in row_m and (row_m.get(c) is None or math.isnan(row_m[c])):
                    vals = [n[c] for n in nn if n.get(c) is not None and not math.isnan(n[c])]
                    if vals:
                        data_list[idx_m][c] = sum(vals) / len(vals)
            for c in cat_cols:
                if c in row_m and (row_m.get(c) is None or math.isnan(row_m[c])):
                    vals = [n[c] for n in nn if n.get(c) is not None and not math.isnan(n[c])]
                    if vals:
                        data_list[idx_m][c] = Counter(vals).most_common(1)[0][0]
        self.df = pd.DataFrame(data_list)
        print("k-NN imputation hoàn thành.")


# DATA PIPELINE CLASS  (fit/transform, không data leakage)
class DataPipeline:
    """
    Transformer fit/transform ngăn data leakage dùng KNN Imputer nâng cao.
    Quy trình: 
      1. Học scaler (mean/std) và category maps trên Train.
      2. Xây dựng Donor Pool từ các bản ghi đầy đủ của Train.
      3. Lọc Outlier trên Donor Pool bằng IQR * 2.5.
      4. Chuẩn hóa Z-score tĩnh cho cả Train và Test trước khi đo khoảng cách.
      5. KNN điền khuyết dùng khoảng cách Euclid chuẩn hóa và trọng số nghịch đảo khoảng cách.
      6. One-hot encoding & Thêm Intercept.
    """

    def __init__(self,
                 continuous_cols: list[str],
                 categorical_cols: list[str],
                 target_col: str = 'SYSTOLIC_TARGET',
                 invalid_cat_values: dict | None = None,
                 k: int = 5):
        self.continuous_cols    = list(continuous_cols)
        self.categorical_cols   = list(categorical_cols)
        self.target_col         = target_col
        self.invalid_cat_values = invalid_cat_values or {}
        self.k                  = k

        self.impute_median_: dict[str, float] = {}
        self.impute_mode_: dict[str, float]   = {}
        self.scaler_mean_: dict[str, float]   = {}
        self.scaler_std_: dict[str, float]    = {}
        self.category_maps_: dict[str, list]  = {}
        self.feature_names_: list[str]        = []
        self.cleaned_donor_pool_: pd.DataFrame = pd.DataFrame()
        self.is_fitted_: bool                 = False

    def _remap_invalid(self, df: pd.DataFrame) -> pd.DataFrame:
        """Thay các giá trị không hợp lệ của categorical cols bằng NaN."""
        for col, bad_vals in self.invalid_cat_values.items():
            if col in df.columns:
                df[col] = df[col].replace(bad_vals, float('nan'))
        return df

    @staticmethod
    def _compute_median(series: pd.Series) -> float:
        """Tính median thủ công từ series, bỏ qua NaN."""
        vals = sorted(series.dropna().tolist())
        n = len(vals)
        if n == 0:
            return 0.0
        if n % 2 == 1:
            return float(vals[n // 2])
        return float((vals[n // 2 - 1] + vals[n // 2]) / 2)

    @staticmethod
    def _compute_mode(series: pd.Series):
        """Tính mode (giá trị xuất hiện nhiều nhất), bỏ qua NaN."""
        counts = Counter(v for v in series.tolist() if not pd.isna(v))
        if not counts:
            return None
        return counts.most_common(1)[0][0]

    def _build_feature_names(self) -> list[str]:
        """Xây dựng danh sách tên cột theo thứ tự trong ma trận X."""
        names = ['bias']
        names += list(self.continuous_cols)
        for col in self.categorical_cols:
            cats = self.category_maps_.get(col, [])
            for cat in cats[1:]:
                names.append(f"{col}_{cat}")
        return names

    def fit(self, X_train: pd.DataFrame, y_train=None) -> 'DataPipeline':
        """Học các tham số và xây dựng Cleaned Donor Pool từ X_train."""
        df = X_train.copy()
        
        # Loại các cột định danh hoặc nhãn nếu có
        for col in ['SEQN', self.target_col]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
                
        df = self._remap_invalid(df)

        # 1. Tính toán giá trị Median/Mode làm fallback phụ
        for col in self.continuous_cols:
            if col in df.columns:
                self.impute_median_[col] = self._compute_median(df[col])
        for col in self.categorical_cols:
            if col in df.columns:
                self.impute_mode_[col] = self._compute_mode(df[col])

        # 2. Học Mean và Std cho biến liên tục
        for col in self.continuous_cols:
            if col in df.columns:
                self.scaler_mean_[col] = float(df[col].mean())
                self.scaler_std_[col]  = float(df[col].std(ddof=1)) if df[col].std(ddof=1) > 1e-10 else 1.0

        # 3. Học Category Maps cho biến phân loại
        for col in self.categorical_cols:
            if col in df.columns:
                self.category_maps_[col] = sorted(
                    v for v in df[col].dropna().unique()
                    if v not in self.invalid_cat_values.get(col, [])
                )

        # 4. Xây dựng Donor Pool từ các complete cases (bản ghi đầy đủ)
        all_features = self.continuous_cols + self.categorical_cols
        donor_pool = df.dropna(subset=all_features).copy()
        
        # 5. Lọc Outlier trên Donor Pool bằng IQR * 2.5
        if len(donor_pool) > 10:
            mask = pd.Series(True, index=donor_pool.index)
            for col in self.continuous_cols:
                vals = sorted(donor_pool[col].tolist())
                n = len(vals)
                q1 = vals[n // 4]
                q3 = vals[(3 * n) // 4]
                iqr = q3 - q1
                median_val = vals[n // 2]
                mask &= donor_pool[col].between(median_val - 2.5 * iqr, median_val + 2.5 * iqr)
            
            cleaned_donors = donor_pool[mask].copy()
            if len(cleaned_donors) > self.k:
                donor_pool = cleaned_donors
                
        self.cleaned_donor_pool_ = donor_pool.reset_index(drop=True)
        print(f"Donor pool sau lọc outlier sinh học/IQR: còn {len(self.cleaned_donor_pool_)} bản ghi sạch.")

        self.feature_names_ = self._build_feature_names()
        self.is_fitted_ = True
        return self

    def _impute_knn(self, df: pd.DataFrame) -> pd.DataFrame:
        """Điền khuyết bằng KNN nâng cao với trọng số khoảng cách."""
        df_imputed = df.copy()
        all_features = self.continuous_cols + self.categorical_cols
        
        # Chuyển đổi donor pool sang dạng ma trận chuẩn hóa phục vụ đo khoảng cách
        donors = self.cleaned_donor_pool_.copy()
        donors_norm = donors.copy()
        for col in self.continuous_cols:
            mean = self.scaler_mean_[col]
            std = self.scaler_std_[col]
            donors_norm[col] = (donors[col] - mean) / std

        # Impute từng dòng có chứa NaN
        records = df_imputed.to_dict('records')
        donor_records = donors.to_dict('records')
        donor_norm_records = donors_norm.to_dict('records')

        for idx, row in enumerate(records):
            missing_cols = [c for c in all_features if pd.isna(row.get(c))]
            if not missing_cols:
                continue

            known_cols = [c for c in all_features if not pd.isna(row.get(c))]
            if not known_cols:
                # Fallback nếu dòng khuyết toàn bộ đặc trưng
                for c in missing_cols:
                    if c in self.continuous_cols:
                        records[idx][c] = self.impute_median_.get(c, 0.0)
                    else:
                        records[idx][c] = self.impute_mode_.get(c)
                continue

            # Tách các đặc trưng liên tục/phân loại đã biết
            known_cont = [c for c in known_cols if c in self.continuous_cols]
            known_cat  = [c for c in known_cols if c in self.categorical_cols]

            # Chuẩn hóa giá trị đã biết của dòng cần điền khuyết
            row_norm = {}
            for c in known_cont:
                mean = self.scaler_mean_[c]
                std = self.scaler_std_[c]
                row_norm[c] = (row[c] - mean) / std
            for c in known_cat:
                row_norm[c] = row[c]

            # Tính khoảng cách đến tất cả các nhà tài trợ (NaN-Euclidean Distance)
            p_total = len(all_features)
            p_known = len(known_cols)
            scale_factor = p_total / p_known

            dists = []
            for d_idx, d_row in enumerate(donor_norm_records):
                # Khoảng cách của biến liên tục chuẩn hóa
                dist_sq = sum((row_norm[c] - d_row[c]) ** 2 for c in known_cont)
                # Khoảng cách của biến phân loại (1 nếu khác nhau, 0 nếu giống nhau)
                dist_sq += sum(0.0 if row_norm[c] == d_row[c] else 1.0 for c in known_cat)
                
                # Áp dụng Nan-Euclidean scaling factor
                d = np.sqrt(dist_sq * scale_factor)
                dists.append((d, donor_records[d_idx]))

            # Chọn k láng giềng gần nhất
            dists.sort(key=lambda x: x[0])
            neighbors = dists[:self.k]

            # Tính trọng số nghịch đảo khoảng cách
            eps = 1e-8
            weights = [1.0 / (d + eps) for d, _ in neighbors]
            sum_weights = sum(weights)

            # Điền khuyết các đặc trưng bị khuyết
            for c in missing_cols:
                if c in self.continuous_cols:
                    # Điền bằng trung bình có trọng số khoảng cách
                    weighted_sum = sum(w * n[c] for w, (_, n) in zip(weights, neighbors))
                    records[idx][c] = weighted_sum / sum_weights
                else:
                    # Điền bằng bầu chọn có trọng số (weighted voting) cho biến phân loại
                    votes = {}
                    for w, (_, n) in zip(weights, neighbors):
                        cat_val = n[c]
                        votes[cat_val] = votes.get(cat_val, 0.0) + w
                    records[idx][c] = max(votes, key=votes.get)

        return pd.DataFrame(records)

    def transform(self, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Áp dụng tham số và KNN Imputation từ fit() lên X. Trả về (X_array, feature_names)."""
        if not self.is_fitted_:
            raise RuntimeError("Gọi fit() trước khi transform().")

        df = X.copy()

        # Loại các cột định danh
        for col in ['SEQN', self.target_col]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        df = self._remap_invalid(df)

        # 1. KNN Imputation sạch sẽ không rò rỉ dữ liệu
        df = self._impute_knn(df)

        # 2. One-hot encoding với drop_first=True
        for col in self.categorical_cols:
            if col not in df.columns:
                continue
            cats = self.category_maps_.get(col, [])
            for cat in cats[1:]:
                df[f"{col}_{cat}"] = (df[col] == cat).astype(float)
            df.drop(columns=[col], inplace=True)

        # 3. Chuẩn hóa Z-score biến liên tục sau điền khuyết
        for col in self.continuous_cols:
            if col not in df.columns:
                continue
            mean = self.scaler_mean_.get(col, 0.0)
            std  = self.scaler_std_.get(col, 1.0)
            df[col] = (df[col] - mean) / std

        # 4. Thêm cột Intercept (bias) ở vị trí đầu tiên
        df.insert(0, 'bias', 1.0)

        # Đảm bảo thứ tự cột chính xác như đã định nghĩa
        for col in self.feature_names_:
            if col not in df.columns:
                df[col] = 0.0
        df = df[self.feature_names_]

        return df.to_numpy(dtype=float), list(self.feature_names_)

    def fit_transform(self, X_train: pd.DataFrame,
                      y_train=None) -> tuple[np.ndarray, list[str]]:
        """Fit rồi transform trên cùng X_train."""
        return self.fit(X_train, y_train).transform(X_train)

    def load_and_split(self,
                       csv_path: str,
                       test_size: float = 0.2,
                       seed: int = 42
                       ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """Load CSV, shuffle, split train/test. Trả về (X_train_df, X_test_df, y_train, y_test)."""
        df = pd.read_csv(csv_path)
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        n_test = int(len(df) * test_size)
        df_test  = df.iloc[:n_test].copy()
        df_train = df.iloc[n_test:].copy()

        y_train = df_train[self.target_col].to_numpy(dtype=float)
        y_test  = df_test[self.target_col].to_numpy(dtype=float)

        return df_train, df_test, y_train, y_test

    def summary(self) -> None:
        """In tóm tắt trạng thái của pipeline."""
        print("=== DataPipeline Summary (KNN Imputer Nâng Cao) ===")
        print(f"  Fitted: {self.is_fitted_}")
        if not self.is_fitted_:
            return
        print(f"  k-Neighbors: {self.k}")
        print(f"  Donor Pool Size: {len(self.cleaned_donor_pool_)} bản ghi sạch.")
        print(f"  Continuous ({len(self.continuous_cols)}): {self.continuous_cols}")
        print(f"  Categorical ({len(self.categorical_cols)}): {self.categorical_cols}")
        print("  Scaler (mean / std):")
        for k in self.continuous_cols:
            print(f"    {k}: mean={self.scaler_mean_.get(k,0):.4f}, std={self.scaler_std_.get(k,0):.4f}")
        print("  Category maps:")
        for k, v in self.category_maps_.items():
            print(f"    {k}: {v}")
        print(f"  Output features ({len(self.feature_names_)}): {self.feature_names_}")


# DEMO TEST PIPELINE
if __name__ == '__main__':
    import os

    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV = os.path.join(THIS_DIR, 'data', 'processed', 'nhanes_pre_pipeline.csv')

    CONTINUOUS  = ['RIDAGEYR', 'BMXBMI', 'BPXOPLS', 'LBXTC', 'LBXSCR']
    CATEGORICAL = ['RIAGENDR', 'SMQ020', 'DIQ010']
    INVALID     = {'SMQ020': [7.0, 9.0]}   # refused / don't know → NaN

    pipe = DataPipeline(CONTINUOUS, CATEGORICAL,
                        target_col='SYSTOLIC_TARGET',
                        invalid_cat_values=INVALID,
                        k=5)

    X_tr_df, X_te_df, y_train, y_test = pipe.load_and_split(CSV, seed=42)
    print(f"Train: {len(X_tr_df)} rows | Test: {len(X_te_df)} rows")

    X_train, feat_names = pipe.fit_transform(X_tr_df)
    X_test,  _          = pipe.transform(X_te_df)

    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_test  shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    pipe.summary()

    # Kiểm tra không leakage: mean_ chỉ từ train
    print("\n[Kiểm tra leakage] scaler_mean_ RIDAGEYR (train only):",
          pipe.scaler_mean_['RIDAGEYR'])
    print("[Kiểm tra leakage] mean RIDAGEYR train raw:",
          X_tr_df['RIDAGEYR'].mean())
