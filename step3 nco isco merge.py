"""
CJSJ Paper — Step 3: Bridge NCO-2015 (PLFS) occupation codes to ISCO-08
sub-major groups and merge in the India exposure scores.

Method: truncate NCO-2015's 3-digit occupation code to its first 2 digits.
This is NOT an approximation — per the DGET/NSDC official NCO-2015
documentation, NCO-2015's coding structure was explicitly built on ISCO-08's:
first digit = Major Group, first 2 digits = Sub-Major Group, using the SAME
numbering as ISCO-08 at those levels (divergence only starts at the minor-
group/occupation-family level, i.e. 3rd digit onward — which we don't rely on
here since we're merging at sub-major/2-digit resolution).

Empirical check: all 40 distinct 2-digit prefixes appearing in the actual
PLFS occupation codes are valid ISCO-08 sub-major codes. The only ISCO-08
sub-major codes absent are 01/02/03 (Armed Forces) -- expected, since PLFS
is a civilian labor force survey.
"""
import pandas as pd

OUT_DIR = "/home/claude/cjsj"

plfs = pd.read_pickle(f"{OUT_DIR}/plfs_pooled_clean.pkl")
exposure = pd.read_csv(f"{OUT_DIR}/india_isco2_exposure.csv")
exposure["isco_submajor"] = exposure["isco_submajor"].astype(str).str.zfill(2)

plfs["isco_submajor"] = plfs["occ_code"].astype(str).str[:2]

before = len(plfs)
merged = plfs.merge(exposure, on="isco_submajor", how="left")
matched = merged["mean_exposure"].notna().sum()
print(f"{matched}/{before} PLFS person-records matched to an exposure score ({matched/before:.1%})")

unmatched_codes = sorted(merged.loc[merged["mean_exposure"].isna(), "isco_submajor"].unique())
print(f"Unmatched ISCO sub-major codes in PLFS data: {unmatched_codes}")

merged.to_pickle(f"{OUT_DIR}/plfs_with_exposure.pkl")
print(f"\nSaved plfs_with_exposure.pkl: {merged.shape}")
print(merged[["year","occ_code","isco_submajor","earnings","mean_exposure"]].head())
