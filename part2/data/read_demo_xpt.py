import argparse
import sys
import json
from pathlib import Path
import pandas as pd

try:
    import pyreadstat
except ImportError:
    pyreadstat = None
    print("Warning: pyreadstat not found. Metadata/Labels export will not work.")

def read_and_export_xpt(xpt_path: Path, out_name: str):
    path_str = str(xpt_path)
    df, meta = None, None

    # Ưu tiên dùng pyreadstat để lấy được Metadata (Nhãn)
    if pyreadstat is not None:
        try:
            df, meta = pyreadstat.read_xport(path_str)
            print("Successfully read with pyreadstat (Metadata included).")
        except Exception as e:
            print(f"Pyreadstat failed: {e}")

    # Fallback sang pandas nếu pyreadstat lỗi hoặc không có
    if df is None:
        try:
            df = pd.read_sas(path_str, format='xport', encoding='latin1')
            print("Read with pandas (No metadata labels).")
        except Exception as e:
            raise RuntimeError(f"Failed to read XPT file: {e}")

    # 1. Xuất file CSV
    csv_file = f"{out_name}.csv"
    df.to_csv(csv_file, index=False)
    print(f"✅ Saved Data to: {csv_file}")

    # 2. Xuất file JSON chứa nhãn (Labels)
    if meta is not None:
        # Tạo dictionary {Tên_viết_tắt: Nhãn_đầy_đủ}
        label_dict = dict(zip(meta.column_names, meta.column_labels))
        json_file = f"{out_name}_labels.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(label_dict, f, ensure_ascii=False, indent=4)
        print(f"✅ Saved Labels to: {json_file}")
    else:
        print("❌ No metadata found to create JSON labels.")

def main():
    p = argparse.ArgumentParser(description='Read XPT and export to CSV/JSON')
    p.add_argument('xpt', help='Path to .xpt file')
    # Set tên mặc định là datadone01
    p.add_argument('--out', default='datadone01', help='Output base name (default: datadone01)')
    
    args = p.parse_args()
    xpt_path = Path(args.xpt)

    if not xpt_path.exists():
        print(f"Error: File {xpt_path} not found.", file=sys.stderr)
        sys.exit(1)

    read_and_export_xpt(xpt_path, args.out)

if __name__ == '__main__':
    main()