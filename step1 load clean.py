"""
CJSJ Paper — Step 1: Load & clean PLFS person-level microdata (2022, 2025)
Renames year-specific column names to a common schema and filters to
respondents who were actually employed during the reference week (CWS).

Usage:
    python step1_load_clean.py

Outputs (written to /home/claude/cjsj/):
    plfs_2022_clean.pkl
    plfs_2025_clean.pkl
    plfs_pooled_clean.pkl
(pickle instead of parquet — no pyarrow available in this sandbox; swap
to_pickle/read_pickle for to_parquet/read_parquet if you later run this
somewhere with pyarrow installed)
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path("/mnt/user-data/uploads")
OUT_DIR = Path("/home/claude/cjsj")
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Column maps: {common_name: year_specific_column_name}
# NOTE: the 2022 map uses the names you supplied. It has NOT been validated
# against the actual cperv1.csv yet (that file isn't uploaded). Run
# validate_columns() once you upload it — it will tell you immediately if
# any of these names are wrong, rather than failing silently.
# ---------------------------------------------------------------------------

COLMAP_2022 = {
    "occ_code":   "b6q7_cperv1",     # NCO code, CWS
    "earn_reg":   "b6q9_cperv1",     # earnings, regular salaried/wage
    "earn_self":  "b6q10_cperv1",    # earnings, self-employed
    "sex":        "b4q5_cperv1",
    "age":        "b4q6_perv1",
    "edu_level":  "b4q8_cperv1",
    "sector":     "b1q3_cperv1",     # rural/urban
    "state":      "state_cperv1",
    "activity_status": "b6q5_cperv1",  # confirmed: matches acws code distribution 1:1
}

COLMAP_2025 = {
    "occ_code":   "ocu_cws",
    "earn_reg":   "ern_reg",
    "earn_self":  "ern_self",
    "sex":        "sex",
    "age":        "age",
    "edu_level":  "gedu_lvl",
    "sector":     "sec",
    "state":      "st",
    "activity_status": "acws",
}

# CWS activity-status codes classified as "actually worked this week"
# (first digit 1-5). Codes 61/62/71/72 (job but didn't work), 81/82
# (unemployed), and 9x (out of labor force) are dropped.
EMPLOYED_CODES = {11, 12, 21, 31, 32, 41, 42, 51, 52}


def validate_columns(path: Path, colmap: dict) -> dict:
    """Check which mapped columns actually exist in the raw file's header.
    Returns the colmap unchanged but prints a report — call this FIRST
    whenever you plug in a new/unverified colmap (i.e. for 2022)."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    print(f"\nValidating columns against {path.name} ({len(header)} cols found)")
    ok = True
    for common, raw in colmap.items():
        if raw is None:
            print(f"  [MISSING] {common}: no column name set")
            ok = False
        elif raw not in header:
            print(f"  [NOT FOUND] {common} -> '{raw}' is not in the file header")
            ok = False
        else:
            print(f"  [ok] {common} -> '{raw}'")
    if not ok:
        print("  ^ fix COLMAP before trusting this file's cleaned output.")
    return colmap


def load_and_clean(path: Path, colmap: dict, year: int) -> pd.DataFrame:
    raw_cols = {v for v in colmap.values() if v is not None}
    df = pd.read_csv(path, usecols=list(raw_cols))

    # rename to common schema
    rename_map = {v: k for k, v in colmap.items() if v is not None}
    df = df.rename(columns=rename_map)

    n_raw = len(df)

    # filter to employed (only possible if activity_status column exists)
    if "activity_status" in df.columns:
        df = df[df["activity_status"].isin(EMPLOYED_CODES)].copy()
    else:
        print(f"  WARNING ({year}): no activity_status column — "
              f"skipping employed-only filter, dropping rows with no occupation instead")
        df = df[df["occ_code"].notna()].copy()

    df["occ_code"] = pd.to_numeric(df["occ_code"], errors="coerce")
    df = df[df["occ_code"].notna()].copy()
    df["occ_code"] = df["occ_code"].astype(int)

    for c in ["earn_reg", "earn_self"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # combined earnings: whichever stream is nonzero (regular salaried vs self-employed)
    df["earnings"] = df[["earn_reg", "earn_self"]].max(axis=1)

    # flag unpaid family workers (activity code 21): employed, but earnings
    # is structurally 0 rather than missing. Keep them in the main sample
    # (real employment outcome, and AI exposure could plausibly affect
    # whether family businesses need to hire paid help) but flag so the
    # wage-growth regression can be run with/without them as a robustness check.
    df["is_unpaid_family_worker"] = (df["activity_status"] == 21) if "activity_status" in df.columns else False

    df["year"] = year

    print(f"  {year}: {n_raw:,} raw rows -> {len(df):,} employed rows kept "
          f"({len(df)/n_raw:.1%})")
    return df


if __name__ == "__main__":
    path_2025 = RAW_DIR / "cperv12025.csv"
    path_2022 = RAW_DIR / "cperv1.csv"

    validate_columns(path_2025, COLMAP_2025)
    df_2025 = load_and_clean(path_2025, COLMAP_2025, 2025)
    df_2025.to_pickle(OUT_DIR / "plfs_2025_clean.pkl")
    print(df_2025.head())

    if path_2022.exists():
        validate_columns(path_2022, COLMAP_2022)
        df_2022 = load_and_clean(path_2022, COLMAP_2022, 2022)
        df_2022.to_pickle(OUT_DIR / "plfs_2022_clean.pkl")

        pooled = pd.concat([df_2022, df_2025], ignore_index=True)
        pooled.to_pickle(OUT_DIR / "plfs_pooled_clean.pkl")
        print(f"\nPooled: {len(pooled):,} rows across both years")
    else:
        print(f"\n{path_2022.name} not found yet — upload it and re-run "
              f"to build the pooled dataset.")
