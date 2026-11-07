"""
CJSJ Paper — Step 5c: Person-level regression with clustered standard errors.

Specification (DiD-style, to directly test whether earnings GROWTH differs
by exposure, not just earnings LEVELS):
    log(earnings_real) ~ exposure * post2025 + exposure + post2025
                          + age + age^2 + female + edu_level + urban
Key coefficient: exposure x post2025 interaction.

Standard errors: cluster-robust (CR1, clustered by isco_submajor), REQUIRED
here since mean_exposure only takes ~38 distinct values across ~580K person
records -- a naive (non-clustered) SE would badly understate uncertainty
(Moulton problem).

Sample: unpaid family workers excluded (earnings = 0 by construction, not
meaningful for a log-earnings outcome). Unlike the group-level analysis,
this does NOT require dropping entire occupation groups (71/92/93) -- their
paid earners are retained; only the mechanically-zero unpaid-family
observations are dropped, at the individual level.
"""
import pandas as pd
import numpy as np
from scipy import stats

OUT_DIR = "/home/claude/cjsj"

cpi_2022_months = [165.7,166.1,167.7,170.1,171.7,172.6,173.4,174.3,175.3,176.7,176.5,175.7]
cpi_2025_months = [193.4,192.5,192.0,192.6,193.0,194.2,196.1,197.0,197.0,197.3,197.9,198.0]
cpi_2022 = sum(cpi_2022_months) / 12
cpi_2025 = sum(cpi_2025_months) / 12
deflator = {2022: 1.0, 2025: cpi_2022 / cpi_2025}

df = pd.read_pickle(f"{OUT_DIR}/plfs_with_exposure.pkl")
df = df.dropna(subset=["mean_exposure"]).copy()
df = df[~df["is_unpaid_family_worker"]].copy()
df = df[df["earnings"] > 0].copy()  # log requires positive earnings

df["earnings_real"] = df.apply(lambda r: r["earnings"] * deflator[r["year"]], axis=1)
df["log_earnings"] = np.log(df["earnings_real"])
df["post2025"] = (df["year"] == 2025).astype(float)
df["exposure_x_post"] = df["mean_exposure"] * df["post2025"]
df["age2"] = df["age"]**2
df["female"] = (df["sex"] == 2).astype(float)
df["urban"] = (df["sector"] == 2).astype(float)

print(f"n = {len(df):,} person-records after dropping unpaid family workers / zero earners")
print(f"n clusters (ISCO-08 sub-major groups) = {df['isco_submajor'].nunique()}")

feature_cols = ["mean_exposure", "post2025", "exposure_x_post", "age", "age2",
                 "female", "edu_level", "urban"]
y = df["log_earnings"].values
X = np.column_stack([np.ones(len(df))] + [df[c].values for c in feature_cols])
names = ["const"] + feature_cols
clusters = df["isco_submajor"].values
n, k = X.shape
n_clusters = len(np.unique(clusters))

XtX_inv = np.linalg.inv(X.T @ X)
beta = XtX_inv @ X.T @ y
resid = y - X @ beta

# cluster-robust sandwich (CR1)
meat = np.zeros((k, k))
for c in np.unique(clusters):
    Xc = X[clusters == c]
    uc = resid[clusters == c]
    score_c = Xc.T @ uc
    meat += np.outer(score_c, score_c)
cr_correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
cov_cr1 = cr_correction * (XtX_inv @ meat @ XtX_inv)
se = np.sqrt(np.diag(cov_cr1))
tvals = beta / se
pvals = 2 * (1 - stats.t.cdf(np.abs(tvals), df=n_clusters - 1))  # conservative: cluster-count df

r2 = 1 - np.sum(resid**2) / np.sum((y - y.mean())**2)

print(f"\n=== log(earnings_real) ~ exposure*post2025 + controls ===")
print(f"(cluster-robust SEs, {n_clusters} clusters, n={n:,})\n")
print(f"{'var':20s} {'coef':>10s} {'se':>10s} {'t':>8s} {'p':>8s}")
for nm, b, s, t, p in zip(names, beta, se, tvals, pvals):
    star = "*" if p < 0.05 else ("+" if p < 0.10 else "")
    print(f"{nm:20s} {b:10.4f} {s:10.4f} {t:8.2f} {p:8.3f} {star}")
print(f"\nR^2 = {r2:.4f}")

print("\n--- For comparison: naive (non-clustered, heteroskedasticity-robust HC1) SEs ---")
diag_resid2 = resid**2
meat_hc1 = X.T @ (X * diag_resid2[:, None])
cov_hc1 = (n / (n - k)) * (XtX_inv @ meat_hc1 @ XtX_inv)
se_hc1 = np.sqrt(np.diag(cov_hc1))
key_idx = names.index("exposure_x_post")
t_hc1 = beta[key_idx] / se_hc1[key_idx]
p_hc1 = 2 * (1 - stats.t.cdf(np.abs(t_hc1), df=n - k))
print(f"exposure_x_post: coef={beta[key_idx]:.4f}  clustered_se={se[key_idx]:.4f} (p={pvals[key_idx]:.3f})  "
      f"vs naive_HC1_se={se_hc1[key_idx]:.4f} (p={p_hc1:.3g})")
print("^ if the naive p-value looks far more 'significant' than the clustered one, that's the Moulton problem in action.")
