import pandas as pd
import os

def load_data():
    current_dir = os.path.dirname(__file__)
    
    csv_path = os.path.join(current_dir, '..', 'nhanes_stroke_analysis.csv')
    
    df = pd.read_csv(csv_path)
    return df

def process_and_save_data(df, csv_path):
    condition = df[['BPXOSY1', 'BPXOSY2', 'BPXOSY3']].isna().all(axis=1)
    df_missing = df[condition]
    
    df_cleaned = df[~condition]
    
    missing_csv_path = csv_path.replace('.csv', '_missing_bp.csv')
    df_missing.to_csv(missing_csv_path, index=False)
    print(f"Đã lưu {len(df_missing)} dòng (cả 3 cột đều NaN) vào file: {os.path.basename(missing_csv_path)}")
    
    df_cleaned.to_csv(csv_path, index=False)
    print(f"Đã ghi đè file gốc với {len(df_cleaned)} dòng dữ liệu còn lại.")
    
    return df_cleaned

if __name__ == '__main__':
    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, '..', 'nhanes_stroke_analysis.csv')
    
    df = pd.read_csv(csv_path)
    print("Kích thước dữ liệu BAN ĐẦU:", df.shape)
    
    df_cleaned = process_and_save_data(df, csv_path)
    print("Kích thước dữ liệu SAU KHI XOÁ:", df_cleaned.shape)
