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
    

    def plot_target_histogram(self) -> none:
        self.df['SYSTOLIC_TARGET'] = self.df[['BPXOSY1', 'BPXOSY2', 'BPXOSY3']].mean(axis=1)
        
        plt.figure(figsize=(10, 6))
        
        sns.histplot(self.df['SYSTOLIC_TARGET'].dropna(), kde=True, color='royalblue', bins=30)
        
        plt.title('Phân phối Huyết áp tâm thu trung bình', fontsize=15)
        plt.xlabel('Huyết áp (mmHg)', fontsize=12)
        plt.ylabel('Tần suất (Số người)', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.show()

    def plot_target_boxplot(self) -> none:
        if 'SYSTOLIC_TARGET' not in self.df.columns:
            self.df['SYSTOLIC_TARGET'] = self.df[['BPXOSY1', 'BPXOSY2', 'BPXOSY3']].mean(axis=1)
        
        plt.figure(figsize=(10, 5))
        
        sns.boxplot(x=self.df['SYSTOLIC_TARGET'].dropna(), color='salmon', fliersize=5)
        
        plt.title('Biểu đồ Boxplot của Huyết áp tâm thu trung bình', fontsize=15)
        plt.xlabel('Huyết áp (mmHg)', fontsize=12)
        plt.show()


    def plot_correlation_heatmap(self) -> none:  
        self.df['SYSTOLIC_TARGET'] = self.df[['BPXOSY1', 'BPXOSY2', 'BPXOSY3']].mean(axis=1)
        cols = ['SYSTOLIC_TARGET', 'RIDAGEYR', 'BMXBMI', 'BMXWAIST', 
                'BMXHT', 'BMXWT', 'LBXTC', 'LBXSCR', 'SLD012', 'PAQ680']
        
        corr_matrix = self.df[cols].corr()
        
        plt.figure(figsize=(12, 10))
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        
        plt.title('Ma trận tương quan giữa các biến', fontsize=15)
        plt.show()


if __name__ == '__main__':
    df = load_data()
    eda = EDA(df)
    #eda.plot_target_histogram()
    #eda.plot_target_boxplot()
    eda.plot_correlation_heatmap()
