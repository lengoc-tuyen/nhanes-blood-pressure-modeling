"""
prep_checkpoint.py
==================
Tạo file nhanes_pre_pipeline.csv: dữ liệu sau outlier removal + BP averaging
nhưng CHƯA impute — đây là input chuẩn cho DataPipeline.

Chạy: python part2/data/prep_checkpoint.py
Output: part2/data/nhanes_pre_pipeline.csv
"""

from pathlib import Path
import pandas as pd

# ─── Đường dẫn ───────────────────────────────────────────────────────────────
THIS_DIR = Path(__file__).parent
INPUT_CSV = THIS_DIR / "nhanes_stroke_analysis.csv"   # sau outlier removal, còn NaN
OUTPUT_CSV = THIS_DIR / "nhanes_pre_pipeline.csv"     # sau averaging, vẫn còn NaN

# Giới hạn sinh học để lọc outlier lần 2 (an toàn)
BIOLOGICAL_LIMITS = {
    'RIDAGEYR':  (0.1,  100.0),
    'BMXBMI':    (10.0,  90.0),
    'BPXOSY1':   (40.0, 260.0),
    'BPXOSY2':   (40.0, 260.0),
    'BPXOSY3':   (40.0, 260.0),
    'BPXOPLS1':  (30.0, 220.0),
    'BPXOPLS2':  (30.0, 220.0),
    'BPXOPLS3':  (30.0, 220.0),
    'LBXTC':     (50.0, 500.0),
    'LBXSCR':    (0.1,   10.0),
}

CATEGORICAL_VALID = {
    'RIAGENDR': [1, 2],
    'DIQ010':   [1, 2, 3],
}


def filter_biological_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Lọc các hàng vi phạm giới hạn sinh học."""
    mask = pd.Series(True, index=df.index)
    for col, (lo, hi) in BIOLOGICAL_LIMITS.items():
        if col in df.columns:
            mask &= (df[col].between(lo, hi) | df[col].isna())
    for col, valid in CATEGORICAL_VALID.items():
        if col in df.columns:
            mask &= (df[col].isin(valid) | df[col].isna())
    removed = (~mask).sum()
    if removed:
        print(f"  Biological outlier filter: loại {removed} hàng")
    return df[mask].copy()


def average_bp_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gộp 3 lần đo huyết áp thành 1 giá trị trung bình.
    BPXOSY1/2/3  → SYSTOLIC_TARGET
    BPXOPLS1/2/3 → BPXOPLS
    """
    systolic_cols = [c for c in ['BPXOSY1', 'BPXOSY2', 'BPXOSY3'] if c in df.columns]
    pulse_cols    = [c for c in ['BPXOPLS1', 'BPXOPLS2', 'BPXOPLS3'] if c in df.columns]

    if systolic_cols:
        df['SYSTOLIC_TARGET'] = df[systolic_cols].mean(axis=1)
        df.drop(columns=systolic_cols, inplace=True)

    if pulse_cols:
        df['BPXOPLS'] = df[pulse_cols].mean(axis=1)
        df.drop(columns=pulse_cols, inplace=True)

    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Loại bỏ các cột không dùng trong mô hình."""
    drop_cols = ['BMXHT', 'BMXWT', 'SEQN']  # SEQN giữ lại để split, bỏ sau
    drop_cols = [c for c in drop_cols if c in df.columns]
    # Chỉ bỏ BMXHT và BMXWT — giữ SEQN để dùng trong split
    to_drop = [c for c in ['BMXHT', 'BMXWT'] if c in df.columns]
    if to_drop:
        df.drop(columns=to_drop, inplace=True)
    return df


def main():
    print(f"Input:  {INPUT_CSV}")
    print(f"Output: {OUTPUT_CSV}\n")

    # 1. Load
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"Columns: {list(df.columns)}\n")

    # 2. Lọc outlier sinh học (an toàn — data đã qua bước này rồi, chạy lại để chắc)
    df = filter_biological_outliers(df)
    print(f"Sau outlier filter: {df.shape[0]} rows")

    # 3. Gộp BP
    df = average_bp_columns(df)
    print(f"Sau averaging BP: columns = {list(df.columns)}")

    # 4. Bỏ cột không dùng
    df = drop_unused_columns(df)

    # 5. Báo cáo missing values
    print("\nMissing values per column:")
    missing = df.isnull().sum()
    for col, cnt in missing[missing > 0].items():
        print(f"  {col}: {cnt} ({cnt/len(df)*100:.1f}%)")
    if missing.sum() == 0:
        print("  (Không có missing values)")

    # 6. Lưu
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {df.shape[0]} rows × {df.shape[1]} cols → {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
