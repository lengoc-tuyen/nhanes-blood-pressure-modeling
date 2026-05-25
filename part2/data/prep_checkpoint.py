"""
Pipeline đầy đủ: từ nhanes_raw.csv (output của merge_nhanes.py, ~12k dòng)
đến nhanes_pre_pipeline.csv (dữ liệu sạch, ~7500 dòng, 10 features).

Quy trình:
  1. Load raw merged CSV
  2. Chọn 14 cột mục tiêu
  3. Lọc: giữ dòng có ≥1 lần đo huyết áp (BPXOSY1/2/3)
  4. Lọc outlier sinh học
  5. Lưu checkpoint nhanes_stroke_analysis.csv (trước khi gộp BP)
  6. Gộp 3 lần đo BP thành trung bình (SYSTOLIC_TARGET, BPXOPLS)
  7. Báo cáo missing values
  8. Lưu nhanes_pre_pipeline.csv (10 cột, ~7500 dòng)
"""

from pathlib import Path
import pandas as pd


THIS_DIR   = Path(__file__).parent
INPUT_CSV  = THIS_DIR / "processed" / "nhanes_stroke_raw.csv"
ANALYSIS_CSV = THIS_DIR / "processed" / "nhanes_stroke_analysis.csv"
OUTPUT_CSV = THIS_DIR / "processed" /  "nhanes_pre_pipeline.csv"

# 14 cột được chọn từ raw merged data
SELECTED_COLS = [
    "SEQN",
    "RIDAGEYR",
    "RIAGENDR",
    "BMXBMI",
    "BPXOSY1",
    "BPXOSY2",
    "BPXOSY3",
    "BPXOPLS1",
    "BPXOPLS2",
    "BPXOPLS3",
    "SMQ020",
    "DIQ010",
    "LBXTC",
    "LBXSCR",
]

# Giới hạn sinh học — loại dòng có giá trị phi lý (NaN được phép)
BIOLOGICAL_LIMITS = {
    "RIDAGEYR":  (0.1,  100.0),
    "BMXBMI":    (10.0,  90.0),
    "BPXOSY1":   (40.0, 260.0),
    "BPXOSY2":   (40.0, 260.0),
    "BPXOSY3":   (40.0, 260.0),
    "BPXOPLS1":  (30.0, 220.0),
    "BPXOPLS2":  (30.0, 220.0),
    "BPXOPLS3":  (30.0, 220.0),
    "LBXTC":     (50.0, 500.0),
    "LBXSCR":    (0.1,   10.0),
}

CATEGORICAL_VALID = {
    "RIAGENDR": [1, 2],
    "DIQ010":   [1, 2, 3],
}


def filter_biological_outliers(df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col, (lo, hi) in BIOLOGICAL_LIMITS.items():
        if col in df.columns:
            mask &= df[col].between(lo, hi) | df[col].isna()
    for col, valid in CATEGORICAL_VALID.items():
        if col in df.columns:
            mask &= df[col].isin(valid) | df[col].isna()
    removed = (~mask).sum()
    if removed:
        print(f"  Biological outlier filter: loại {removed} dòng")
    return df[mask].copy()


def require_bp_reading(df: pd.DataFrame) -> pd.DataFrame:
    """Giữ dòng có ít nhất 1 lần đo huyết áp tâm thu không phải NaN."""
    bp_cols = [c for c in ["BPXOSY1", "BPXOSY2", "BPXOSY3"] if c in df.columns]
    mask = df[bp_cols].notna().any(axis=1)
    removed = (~mask).sum()
    print(f"  Lọc dòng không có BP reading: loại {removed} dòng")
    return df[mask].copy()


def average_bp_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gộp 3 lần đo huyết áp thành trung bình.
    BPXOSY1/2/3  → SYSTOLIC_TARGET
    BPXOPLS1/2/3 → BPXOPLS
    """
    sys_cols   = [c for c in ["BPXOSY1", "BPXOSY2", "BPXOSY3"]   if c in df.columns]
    pulse_cols = [c for c in ["BPXOPLS1", "BPXOPLS2", "BPXOPLS3"] if c in df.columns]

    if sys_cols:
        df["SYSTOLIC_TARGET"] = df[sys_cols].mean(axis=1)
        df.drop(columns=sys_cols, inplace=True)

    if pulse_cols:
        df["BPXOPLS"] = df[pulse_cols].mean(axis=1)
        df.drop(columns=pulse_cols, inplace=True)

    return df


def main():
    print(f"Input:  {INPUT_CSV}")
    print(f"Output: {OUTPUT_CSV}\n")

    # 1. Load raw merged data
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded: {df.shape[0]} dòng × {df.shape[1]} cột")

    # 2. Chọn 14 cột mục tiêu
    present = [c for c in SELECTED_COLS if c in df.columns]
    missing_cols = [c for c in SELECTED_COLS if c not in df.columns]
    if missing_cols:
        print(f"  CẢNH BÁO: thiếu cột {missing_cols}")
    df = df[present].copy()
    print(f"Sau chọn cột ({len(present)}): {df.shape[0]} dòng × {df.shape[1]} cột")

    # 3. Giữ dòng có ≥1 lần đo huyết áp
    df = require_bp_reading(df)
    print(f"Sau lọc BP: {df.shape[0]} dòng")

    # 4. Lọc outlier sinh học
    df = filter_biological_outliers(df)
    print(f"Sau outlier filter: {df.shape[0]} dòng")

    # 5. Lưu checkpoint (14 cột, trước khi gộp BP)
    df.to_csv(ANALYSIS_CSV, index=False)
    print(f"\nCheckpoint lưu: {df.shape[0]} dòng × {df.shape[1]} cột → {ANALYSIS_CSV}")

    # 6. Gộp BP thành trung bình
    df = average_bp_columns(df)
    print(f"Sau gộp BP: {list(df.columns)}")

    # 7. Báo cáo missing values
    print("\nMissing values per column:")
    missing = df.isnull().sum()
    for col, cnt in missing[missing > 0].items():
        print(f"  {col}: {cnt} ({cnt / len(df) * 100:.1f}%)")
    if missing.sum() == 0:
        print("  (Không có missing values)")

    # 8. Lưu output
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {df.shape[0]} dòng × {df.shape[1]} cột → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
