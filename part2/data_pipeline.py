import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def load_data():
    df = pd.read_csv('DATA/nhanes_stroke_analysis.csv')
    return df

class EDA:
    def __init__(self, df):
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
            if l % 2 == 0:
                q2 = (tempList[l//2] + tempList[l//2 - 1]) / 2
            else:
                q2 = tempList[l//2]
            idx1 = l // 4
            if l % 4 == 0:
                q1 = (tempList[idx1] + tempList[idx1 - 1]) / 2
            else:
                q1 = tempList[idx1]
            idx3 = (3 * l) // 4
            if (3 * l) % 4 == 0:
                q3 = (tempList[idx3] + tempList[idx3 - 1]) / 2
            else:
                q3 = tempList[idx3]

            ans.append((q1, q2, q3))
        return ans

    def stdImple(self) -> list:
        ans = []
        for i in self.df:
            sum_val = 0 
            numOfVal = 0
            for j in self.df[i]:
                if not pd.isna(j):
                    sum_val += j
                    numOfVal += 1
            
            if numOfVal <= 1:
                ans.append(0)
                continue
                
            mean_val = sum_val / numOfVal
            diff = 0
            for j in self.df[i]:
                if not pd.isna(j):
                    diff += (j - mean_val) ** 2
            
            diff /= (numOfVal - 1)
            ans.append(diff ** (1/2))
        return ans

    def minImple(self) -> list:
        ans = []
        for i in self.df:
            tempList = [x for x in self.df[i] if not pd.isna(x)]
            if not tempList:
                ans.append(None)
                continue
                
            minVal = tempList[0]
            for j in tempList:
                if j < minVal:
                    minVal = j
            ans.append(minVal)
        return ans
    
    def maxImple(self) -> list:
        ans = []
        for i in self.df:
            tempList = [x for x in self.df[i] if not pd.isna(x)]
            if not tempList:
                ans.append(None)
                continue
                
            maxVal = tempList[0]
            for j in tempList: 
                if j > maxVal:
                    maxVal = j
            ans.append(maxVal)
        return ans

    def missingRate(self) -> list:
        ans  = []
        for i in self.df:
            nan = [x for x in self.df[i] if pd.isna(x)]
            ans.append(len(nan)/len(self.df[i]))
        return ans
    

    def plot_target_histogram(self) -> None:
        self.df['SYSTOLIC_TARGET'] = self.df[['BPXOSY1', 'BPXOSY2', 'BPXOSY3']].mean(axis=1)
        
        plt.figure(figsize=(10, 6))
        
        sns.histplot(self.df['SYSTOLIC_TARGET'].dropna(), kde=True, color='royalblue', bins=30)
        
        plt.title('Phân phối Huyết áp tâm thu trung bình', fontsize=15)
        plt.xlabel('Huyết áp (mmHg)', fontsize=12)
        plt.ylabel('Tần suất (Số người)', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.show()

    def plot_target_boxplot(self) -> None:
        if 'SYSTOLIC_TARGET' not in self.df.columns:
            self.df['SYSTOLIC_TARGET'] = self.df[['BPXOSY1', 'BPXOSY2', 'BPXOSY3']].mean(axis=1)
        
        plt.figure(figsize=(10, 5))
        
        sns.boxplot(x=self.df['SYSTOLIC_TARGET'].dropna(), color='salmon', fliersize=5)
        
        plt.title('Biểu đồ Boxplot của Huyết áp tâm thu trung bình', fontsize=15)
        plt.xlabel('Huyết áp (mmHg)', fontsize=12)
        plt.show()


    def plot_correlation_heatmap(self) -> None:  
        self.df['SYSTOLIC_TARGET'] = self.df[['BPXOSY1', 'BPXOSY2', 'BPXOSY3']].mean(axis=1)
        self.df['nhiptim'] = self.df[['BPXOPLS1', 'BPXOPLS2', 'BPXOPLS3']].mean(axis=1)
        cols = [
            'SYSTOLIC_TARGET', 'SEQN','RIDAGEYR', 'RIAGENDR', 'BMXBMI', 
            'nhiptim', 'SMQ020', 'DIQ010', 'LBXTC', 'LBXSCR'
        ]        
        corr_matrix = self.df[cols].corr()
        
        plt.figure(figsize=(12, 10))
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        
        plt.title('Ma trận tương quan giữa các biến', fontsize=15)
        plt.show()
    
    def check_duplicates(self):    
        total_dup = self.df.duplicated().sum()
        print(f"Số dòng trùng lặp hoàn toàn: {total_dup}")
        id_dup = self.df['SEQN'].duplicated().sum()
        print(f"Số ID bị trùng: {id_dup}")
        if total_dup > 0:
            df = self.df.drop_duplicates()
            print("Đã xóa các dòng trùng lặp hoàn toàn.")

    def dropCol(self) -> None:
        colDrop = ['BMXHT']
        self.df.drop(columns=colDrop, inplace=True, errors='ignore')
        self.df.to_csv('DATA/nhanes_stroke_analysis.csv', index=False)
    
    def detectOutliers(self) -> None:
        initial_rows = len(self.df)
        
        is_valid = pd.Series(True, index=self.df.index)
        
        numeric_limits = {
            'RIDAGEYR': (0.1, 100.0),
            'BMXBMI': (10.0, 90.0),
            'BPXOSY1': (40.0, 260.0),
            'BPXOSY2': (40.0, 260.0),
            'BPXOSY3': (40.0, 260.0),
            'BPXOPLS1': (30.0, 220.0),
            'BPXOPLS2': (30.0, 220.0),
            'BPXOPLS3': (30.0, 220.0),
            'LBXTC': (50.0, 500.0),
            'LBXSCR': (0.1, 10.0)
        }
        for col, (min_val, max_val) in numeric_limits.items():
            if col in self.df.columns:
                condition = ((self.df[col] >= min_val) & (self.df[col] <= max_val)) | self.df[col].isna()
                is_valid &= condition
        
        categorical_limits = {
            'RIAGENDR': [1, 2],
            'DIQ010': [1, 2, 3]
        }
        for col, valid_codes in categorical_limits.items():
            if col in self.df.columns:
                condition = self.df[col].isin(valid_codes) | self.df[col].isna()
                is_valid &= condition

        df_invalid = self.df[~is_valid]
        
        if len(df_invalid) > 0:
            df_invalid.to_csv('nhanes_stroke_analysis_missing_bp.csv', index=False)
            print(f"--> Đã lưu {len(df_invalid)} dòng ngoại lai/lỗi vào file 'nhanes_stroke_analysis_missing_bp.csv'")
        else:
            print("--> Tuyệt vời! Không phát hiện dòng ngoại lai nào.")
            
        self.df = self.df[is_valid]

        file_goc_path = 'nhanes_stroke_analysis.csv' 
        self.df.to_csv(file_goc_path, index=False)
        
        dropped_rows = initial_rows - len(self.df)
        print(f"\n--- KẾT QUẢ LỌC BIOLOGICAL OUTLIERS ---")
        print(f"Số dòng ban đầu: {initial_rows}")
        print(f"Số dòng sau khi lọc sạch: {len(self.df)}")
        print(f"Tổng số dòng đã loại bỏ: {dropped_rows} dòng.")

    def detectOutliersIQR(self) -> None:
        initial_rows = len(self.df)
        is_valid = pd.Series(True, index=self.df.index)
        iqr_columns = ['RIDAGEYR', 'BMXBMI', 'LBXTC', 'LBXSCR', 'BPXOPLS1', 'BPXOPLS2', 'BPXOPLS3']
        
        arr = self.medianAndQuantilesImple()
        all_columns = list(self.df.columns)
        
        for col in iqr_columns:
            if col in self.df.columns:
                idx = all_columns.index(col)
                q1, _, q3 = arr[idx]
                
                if q1 is None or q3 is None:
                    continue
                    
                IQR = q3 - q1
                lower_bound = q1 - 2 * IQR
                upper_bound = q3 + 2 * IQR
                
                condition = ((self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)) | self.df[col].isna()
                is_valid &= condition
        self.df = self.df[is_valid]
        dropped_rows = initial_rows - len(self.df)
        
        print(f"Đã loại bỏ {dropped_rows} dòng vi phạm IQR.")
        file_goc_path = 'nhanes_stroke_analysis.csv' 
        self.df.to_csv(file_goc_path, index=False)

    def createAverageColumns(self) -> None:
        self.df['SYSTOLIC_TARGET'] = self.df[['BPXOSY1', 'BPXOSY2', 'BPXOSY3']].mean(axis=1)
        self.df['BPXOPLS'] = self.df[['BPXOPLS1', 'BPXOPLS2', 'BPXOPLS3']].mean(axis=1)
        
        cols_to_drop = ['BPXOSY1', 'BPXOSY2', 'BPXOSY3', 'BPXOPLS1', 'BPXOPLS2', 'BPXOPLS3']
        self.df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        file_path = 'nhanes_stroke_analysis.csv'
        self.df.to_csv(file_path, index=False)
        print(f"Đã gộp thành công")
    def impute_knn_all_types(self, k: int = 5) -> None:
        import math
        from collections import Counter

        continuous_cols = ['RIDAGEYR', 'BMXBMI', 'BPXOSY1', 'BPXOSY2', 'BPXOSY3', 
                           'BPXOPLS1', 'BPXOPLS2', 'BPXOPLS3', 'LBXTC', 'LBXSCR']
        categorical_cols = ['RIAGENDR', 'DIQ010', 'SMQ020']
        all_target_cols = continuous_cols + categorical_cols

        data_list = self.df.to_dict('records')

        min_max = {}
        for col in continuous_cols:
            if col in self.df.columns:
                vals = [row[col] for row in data_list if row[col] is not None and not math.isnan(row[col])]
                if vals:
                    min_max[col] = (min(vals), max(vals))

        complete_rows = []
        missing_rows_info = [] 

        for idx, row in enumerate(data_list):
            has_missing = any(row[col] is None or math.isnan(row[col]) for col in all_target_cols if col in row)
            
            has_all_cont = all(row[col] is not None and not math.isnan(row[col]) for col in continuous_cols if col in row)
            
            if has_all_cont:
                complete_rows.append(row)
            
            if has_missing:
                missing_rows_info.append((idx, row))

        print(f"--> Đang chạy k-NN PURE PYTHON (k={k}) cho {len(missing_rows_info)} dòng dữ liệu...")

        for idx_miss, row_miss in missing_rows_info:
            known_cont_cols = [c for c in continuous_cols if c in row_miss and row_miss[c] is not None and not math.isnan(row_miss[c])]
            
            if not known_cont_cols:
                continue

            distances = []
            for row_comp in complete_rows:
                sum_sq_diff = 0.0
                for col in known_cont_cols:
                    c_min, c_max = min_max[col]
                    c_range = c_max - c_min if c_max != c_min else 1.0
                    
                    val_miss_scaled = (row_miss[col] - c_min) / c_range
                    val_comp_scaled = (row_comp[col] - c_min) / c_range
                    
                    sum_sq_diff += (val_miss_scaled - val_comp_scaled) ** 2
                
                distances.append((sum_sq_diff, row_comp))

            distances.sort(key=lambda x: x[0])
            nearest_neighbors = [row for _, row in distances[:k]]

            for col in continuous_cols:
                if col in row_miss and (row_miss[col] is None or math.isnan(row_miss[col])):
                    neighbor_vals = [n[col] for n in nearest_neighbors if n[col] is not None and not math.isnan(n[col])]
                    if neighbor_vals:
                        data_list[idx_miss][col] = sum(neighbor_vals) / len(neighbor_vals)

            for col in categorical_cols:
                if col in row_miss and (row_miss[col] is None or math.isnan(row_miss[col])):
                    neighbor_classes = [n[col] for n in nearest_neighbors if n[col] is not None and not math.isnan(n[col])]
                    if neighbor_classes:
                        most_common_class = Counter(neighbor_classes).most_common(1)[0][0]
                        data_list[idx_miss][col] = most_common_class

        import pandas as pd
        self.df = pd.DataFrame(data_list)
        
        file_goc_path = 'nhanes_stroke_analysis.csv'
        self.df.to_csv(file_goc_path, index=False)
        print("--> Xử lý Missing Value bằng k-NN thuần Python thành công! Máy mát rượi.")

if __name__ == '__main__':
    df = load_data()
    eda = EDA(df)
    #eda.plot_target_histogram()
    #eda.plot_target_boxplot()
    #eda.plot_correlation_heatmap()
    #eda.check_duplicates()
    #eda.dropCol()
    #eda.detectOutliers()
    #eda.detectOutliersIQR()
    #eda.createAverageColumns()
    eda.impute_knn_all_types(20)
    print(eda.missingRate())
