"""
CJSJ Paper — Step 4c: Finalize occupation-group wage-growth dataset for
regression. Median real earnings growth is the primary outcome (robust to
the independent-cross-section outlier issue found in the step 4b check);
mean-based growth is retained as a secondary/comparison column only.

Groups 71, 92, 93 (Building/Related Trades; Agricultural/Forestry/Fishery
Labourers; Labourers in Mining/Construction/Manufacturing/Transport) are
EXCLUDED from the regression sample: median cash earnings = Rs 0 in BOTH
2022 and 2025, so percentage growth is undefined. LIMITATION for the paper:
these groups have high rates of unpaid family labor / subsistence work, so
cash earnings don't meaningfully capture their economic activity -- this is
a real measurement gap in PLFS-based wage analysis for India, not a coding
artifact.
"""
import pandas as pd

OUT_DIR = "/home/claude/cjsj"
EXCLUDE_GROUPS = {"71", "92", "93"}

cpi_2022_months = [165.7,166.1,167.7,170.1,171.7,172.6,173.4,174.3,175.3,176.7,176.5,175.7]
cpi_2025_months = [193.4,192.5,192.0,192.6,193.0,194.2,196.1,197.0,197.0,197.3,197.9,198.0]
cpi_2022 = sum(cpi_2022_months) / 12
cpi_2025 = sum(cpi_2025_months) / 12
deflator = {2022: 1.0, 2025: cpi_2022 / cpi_2025}

df = pd.read_pickle(f"{OUT_DIR}/plfs_with_exposure.pkl")
df = df.dropna(subset=["mean_exposure"]).copy()
df["earnings_real"] = df.apply(lambda r: r["earnings"] * deflator[r["year"]], axis=1)

g = (df.groupby(["isco_submajor", "year"])
       .agg(n=("earnings_real", "count"),
            mean_real=("earnings_real", "mean"),
            median_real=("earnings_real", "median"))
       .reset_index())
wide = g.pivot(index="isco_submajor", columns="year", values=["n","mean_real","median_real"])
wide.columns = [f"{a}_{b}" for a, b in wide.columns]
wide = wide.reset_index()

wide["median_growth_pct"] = (wide["median_real_2025"] / wide["median_real_2022"] - 1) * 100
wide["mean_growth_pct"] = (wide["mean_real_2025"] / wide["mean_real_2022"] - 1) * 100

exposure = pd.read_csv(f"{OUT_DIR}/india_isco2_exposure.csv")
exposure["isco_submajor"] = exposure["isco_submajor"].astype(str).str.zfill(2)
controls = (df[df["year"] == 2022].groupby("isco_submajor").agg(
    pct_female=("sex", lambda x: (x == 2).mean() * 100),
    mean_age=("age", "mean"),
    pct_urban=("sector", lambda x: (x == 2).mean() * 100),
    mean_edu_level=("edu_level", "mean"),
).reset_index())

final = wide.merge(exposure, on="isco_submajor").merge(controls, on="isco_submajor")

n_before = len(final)
excluded = final[final["isco_submajor"].isin(EXCLUDE_GROUPS)]
final_regression = final[~final["isco_submajor"].isin(EXCLUDE_GROUPS)].copy()
print(f"Excluded {len(excluded)} groups (undefined median growth): "
      f"{sorted(excluded['isco_submajor'].tolist())}")
print(f"Regression-ready sample: {len(final_regression)} of {n_before} ISCO-08 groups\n")

final.to_csv(f"{OUT_DIR}/isco2_growth_all_groups.csv", index=False)
final_regression.to_csv(f"{OUT_DIR}/isco2_growth_regression_sample.csv", index=False)

display_cols = ["isco_submajor","n_occupations","mean_exposure","n_2022","n_2025",
                 "median_real_2022","median_real_2025","median_growth_pct","mean_growth_pct"]
pd.set_option("display.width", 160)
print(final_regression[display_cols].sort_values("mean_exposure", ascending=False).round(2).to_string(index=False))
