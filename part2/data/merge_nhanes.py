from pathlib import Path
from functools import reduce

import pandas as pd
import numpy as np


RAW_DIR = Path(__file__).parent / "dataset"  # Script's directory (DATA/dataset/)
OUT_DIR = Path(__file__).parent / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

files = {
    "demo": "DEMO_L.xpt",
    "bpx": "BPXO_L.xpt",
    "bmx": "BMX_L.xpt",
    "mcq": "MCQ_L.xpt",
    "smq": "SMQ_L.xpt",
    "alq": "ALQ_L.xpt",
    "paq": "PAQ_L.xpt",
    "diq": "DIQ_L.xpt",
    "tchol": "TCHOL_L.xpt",
    "biopro": "BIOPRO_L.xpt",
}

# Optional: Add SLD and LBX if available, but don't fail if missing
optional_files = {
    "slq": "SLQ_L.xpt",
    "trigly": "TRIGLY_L.xpt",
    "lux": "LUX_L.xpt",
}
for name, filename in optional_files.items():
    if (RAW_DIR / filename).exists():
        files[name] = filename

missing_files = [name for name in files.values() if not (RAW_DIR / name).exists()]
if missing_files:
    raise FileNotFoundError(f"Missing files in {RAW_DIR}: {missing_files}")

dfs = {
    name: pd.read_sas(RAW_DIR / filename, format="xport")
    for name, filename in files.items()
}

for name, df in dfs.items():
    if "SEQN" not in df.columns:
        raise ValueError(f"{files[name]} does not contain SEQN")

df = reduce(
    lambda left, right: left.merge(right, on="SEQN", how="left"),
    dfs.values()
)

# NHANES triglyceride in this cycle is named LBXTLG; expose LBXTR alias for consistency.
if "LBXTR" not in df.columns and "LBXTLG" in df.columns:
    df["LBXTR"] = df["LBXTLG"]

# Physical activity variable in this cycle is PAD680; expose PAQ680 alias.
if "PAQ680" not in df.columns and "PAD680" in df.columns:
    df["PAQ680"] = df["PAD680"]

# Don't fail on missing blood pressure mean - just compute if available
bp_cols = [col for col in ["BPXOSY1", "BPXOSY2", "BPXOSY3", "BPXOSY4", "BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"] if col in df.columns]
if bp_cols:
    df["SYSBP_MEAN"] = df[bp_cols].mean(axis=1)

selected_cols = [
    "SEQN",
    "RIDAGEYR",
    "RIAGENDR",
    "BMXHT",
    "BMXWT",
    "BMXBMI",
    "BMXWAIST",
    "BPXOSY1",
    "BPXOSY2",
    "BPXOSY3",
    "BPXOPLS1",
    "BPXOPLS2",
    "BPXOPLS3",
    "SMQ020",
    "DIQ010",
    "MCQ160E",
    "MCQ160F",
    "PAQ680",
    "SLD012",
    "LBXTC",
    "LBXSCR",
]

existing_cols = [col for col in selected_cols if col in df.columns]
missing_selected = [col for col in selected_cols if col not in df.columns]

data = df[existing_cols].copy()

raw_merged_path = OUT_DIR / "nhanes_stroke_raw.csv"
data.to_csv(raw_merged_path, index=False)

print("Merged shape:", df.shape)
print("Selected columns present:", len(existing_cols), "/", len(selected_cols))
if missing_selected:
    print("Missing columns:", missing_selected)
print("Output shape:", data.shape)
print("Saved to:", raw_merged_path)