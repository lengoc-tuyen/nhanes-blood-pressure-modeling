import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

        iqr_columns = ['RIDAGEYR', 'BMXBMI', 'LBXTC', 'LBXSCR']
        for col in iqr_columns:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 2 * IQR
                upper_bound = Q3 + 2 * IQR
                
                iqr_condition = ((self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)) | self.df[col].isna()
                is_valid &= iqr_condition

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
        print(f"\n--- KẾT QUẢ TÍCH HỢP BIOLOGICAL + IQR FILTER ---")
        print(f"Số dòng ban đầu: {initial_rows}")
        print(f"Số dòng sau khi lọc sạch hoàn toàn: {len(self.df)}")
        print(f"Tổng số dòng đã loại bỏ: {dropped_rows} dòng.")
if __name__ == '__main__':
    df = load_data()
    eda = EDA(df)
    #eda.plot_target_histogram()
    eda.plot_target_boxplot()
    #eda.plot_correlation_heatmap()
    #eda.check_duplicates()
    #eda.dropCol()
    #eda.detectOutliers()
