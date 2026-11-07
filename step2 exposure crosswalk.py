"""
CJSJ Paper — Step 2: Build India occupation-exposure data at ISCO-08 2-digit
(sub-major group) resolution.

Pipeline:
  1. Global Automation Atlas country_occupation_panel (O*NET-SOC level, India rows)
  2. BLS official ISCO-08 <-> 2010 SOC crosswalk
  3. Join (2) into (1) via SOC code, resolving ambiguous SOC->ISCO matches by
     modal assignment
  4. Aggregate to ISCO-08 2-digit group level

KNOWN LIMITATION (flag in Methods): the O*NET-SOC codes here are the current
(2019, aligned to 2018 SOC) taxonomy, but the BLS ISCO crosswalk we have maps
to the OLDER 2010 SOC. ~15% of India occupation rows (137/923) are titles
introduced in the 2018 SOC revision and have no 2010 SOC equivalent, so they
drop out unmatched. Coverage: 786/923 occupations (85%) across 38 ISCO-08
sub-major groups.
"""
import pandas as pd

RAW_DIR = "/mnt/user-data/uploads"
OUT_DIR = "/home/claude/cjsj"

# ---- 1. load & clean BLS crosswalk ----
cw = pd.read_csv(f"{RAW_DIR}/Untitled_spreadsheet_-_ISCO-08_to_2010_SOC.csv", skiprows=6)
cw.columns = [c.strip() for c in cw.columns]
cw["soc_code"] = cw["2010 SOC Code"].astype(str).str.strip()
cw["isco08_code"] = cw["ISCO-08 Code"].astype(str).str.strip().str.zfill(4)
cw["isco_submajor"] = cw["isco08_code"].str[:2]

# resolve SOC codes that map to >1 ISCO sub-major group -> take modal (most common) group
primary = (cw.groupby("soc_code")["isco_submajor"]
             .agg(lambda x: x.value_counts().idxmax())
             .reset_index())

# ---- 2. load India O*NET-SOC exposure data, strip O*NET suffix to get base SOC code ----
ind = pd.read_csv(f"{RAW_DIR}/country_occupation_panel_IND.csv")
ind["soc_code"] = ind["onet_soc_code"].str.split(".").str[0]

# ---- 3. merge ----
merged = ind.merge(primary, on="soc_code", how="left")

# MANUAL OVERRIDE: the entire SOC 2018 "15-12xx Computer Occupations" family
# (Systems Analysts, Programmers, Software Developers, Database Administrators,
# Network Architects, Web Developers, IT Project Managers, etc.) plus Data
# Scientists (15-2051, new in SOC 2018) is completely absent from the BLS
# 2010-vintage crosswalk -- SOC 2010's "15-11xx" numbering for this whole
# cluster was renumbered to "15-12xx" in the 2018 SOC revision. Left unmatched,
# this would silently drop ICT professionals -- ISCO-08 sub-major group 25,
# and arguably THE occupation group most central to an LLM-exposure study --
# entirely out of the analysis. These 16 base SOC codes unambiguously fit
# ISCO-08 group 25's definition (Information and Communications Technology
# Professionals: 251 Software/Applications Developers and Analysts, 252
# Database and Network Professionals), so they are hand-mapped here rather
# than silently dropped.
ICT_MANUAL_OVERRIDE = [
    "15-1211", "15-1212", "15-1221", "15-1231", "15-1232", "15-1241",
    "15-1242", "15-1243", "15-1244", "15-1251", "15-1252", "15-1253",
    "15-1254", "15-1255", "15-1299", "15-2051",
]
override_mask = merged["soc_code"].isin(ICT_MANUAL_OVERRIDE) & merged["isco_submajor"].isna()
n_override = override_mask.sum()
merged.loc[override_mask, "isco_submajor"] = "25"
print(f"Manually mapped {n_override} ICT professional rows to ISCO-08 group 25 "
      f"(SOC-2018 'Computer Occupations' family, absent from the 2010 crosswalk)")

n_matched = merged["isco_submajor"].notna().sum()
print(f"{n_matched}/{len(ind)} India occupations matched to an ISCO-08 sub-major group "
      f"({n_matched/len(ind):.1%})")

matched = merged.dropna(subset=["isco_submajor"]).copy()

# ---- 4. aggregate to ISCO-08 2-digit level ----
isco2_exposure = matched.groupby("isco_submajor").agg(
    n_occupations=("onet_soc_code", "count"),
    mean_exposure=("mean_exposure", "mean"),
    substitution_share=("substitution_share", "mean"),
    augmentation_share=("augmentation_share", "mean"),
    sub_only_share=("sub_only_share", "mean"),
    both_share=("both_share", "mean"),
).round(4).reset_index().sort_values("isco_submajor")

matched.to_pickle(f"{OUT_DIR}/india_occ_exposure_matched.pkl")
isco2_exposure.to_csv(f"{OUT_DIR}/india_isco2_exposure.csv", index=False)

print(f"\n{len(isco2_exposure)} ISCO-08 sub-major groups covered")
print(isco2_exposure.to_string(index=False))
