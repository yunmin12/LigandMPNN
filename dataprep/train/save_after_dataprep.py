import os
import shutil
import pandas as pd
import argparse

def safe_copy(src: str, dst: str) -> bool:
    """
    if src exists, make a directory as dst ad copy.
    Success: True/Failure: False
    """
    if not os.path.isfile(src):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main():
    parser = argparse.ArgumentParser(description="Copy files based on CSV paths")
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to the CSV file containing file paths to copy'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default="./collected_files",
        help='Output directory to copy files into (default: collected_files)'
    )
    args = parser.parse_args()
    
    CSV_PATH = args.csv
    OUT_DIR = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    TARGET_COL = "target_blurred_path"
    OFFTARGET_COL = "offtarget_pose_path"

    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    copied = 0
    missing = 0
    bad_format = 0

    # 1) target_blurred_path -> OUT_DIR/target_blurred/<basename>
    for src in df.get(TARGET_COL, pd.Series(dtype=str)).dropna().astype(str):
        src = src.strip()
        if not src:
            continue

        dst = os.path.join(OUT_DIR, "target_blurred", os.path.basename(src))
        if safe_copy(src, dst):
            copied += 1
        else:
            print(f"[MISSING target] {src}")
            missing += 1

    # 2) offtarget_pose_path (sep=";") -> OUT_DIR/<COMPLEX_ID>/<basename>
    series = df.get(OFFTARGET_COL, pd.Series(dtype=str)).dropna().astype(str)

    for cell in series:
        for src in map(str.strip, cell.split(";")):
            if not src:
                continue

            # expected: .../runs/<COMPLEX_ID>/final/<file>
            marker = "/runs/"
            if marker not in src:
                print(f"[BAD FORMAT] (no /runs/) {src}")
                bad_format += 1
                continue

            after_runs = src.split(marker, 1)[1]  # "<COMPLEX_ID>/final/xxx.pdb"
            parts = after_runs.split("/", 2)
            if len(parts) < 2:
                print(f"[BAD FORMAT] (runs parse fail) {src}")
                bad_format += 1
                continue

            complex_id = parts[0]
            fname = os.path.basename(src)

            dst = os.path.join(OUT_DIR, "offtarget_poses", complex_id, fname)

            if safe_copy(src, dst):
                copied += 1
            else:
                print(f"[MISSING off-target] {src}")
                missing += 1

    print(f"Done. copied={copied}, missing={missing}, bad_format={bad_format}")
    print(f"Total targets: {len(df.get(TARGET_COL, []))}, Copied targets: {copied - bad_format - missing}")
    print(f"Total off-target files: {series.apply(lambda x: len(x.split(';')) if pd.notna(x) else 0).sum()}, Copied off-target files: {copied - bad_format - missing - len(df.get(TARGET_COL, []))}")
    print(f"Input CSV Path: {os.path.abspath(CSV_PATH)}")
    print(f"Output dir: {os.path.abspath(OUT_DIR)}")

if __name__ == "__main__":
    main()