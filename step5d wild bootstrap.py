"""
CJSJ Paper — Step 5d: Wild cluster bootstrap (Cameron, Gelbach & Miller 2008)
robustness check on the exposure x post2025 coefficient, since 38 clusters
sits at the edge of where standard cluster-robust asymptotics are reliable.

Method: Wild Cluster Restricted (WCR) bootstrap.
  1. Fit the restricted model with exposure_x_post's coefficient constrained
     to 0 (i.e. omit it), imposing the null hypothesis.
  2. For each replication, draw one Rademacher weight (+1/-1, p=0.5) PER
     CLUSTER (not per observation) and apply it to that cluster's restricted
     residuals.
  3. Build a synthetic outcome from the restricted fitted values + reweighted
     residuals, refit the FULL model, recompute the cluster-robust t-stat for
     exposure_x_post.
  4. p-value = share of bootstrap |t*| >= |t_original| (two-sided).
"""
import pandas as pd
import numpy as np
import time

OUT_DIR = "/home/claude/cjsj"
np.random.seed(42)

cpi_2022_months = [165.7,166.1,167.7,170.1,171.7,172.6,173.4,174.3,175.3,176.7,176.5,175.7]
cpi_2025_months = [193.4,192.5,192.0,192.6,193.0,194.2,196.1,197.0,197.0,197.3,197.9,198.0]
cpi_2022 = sum(cpi_2022_months) / 12
cpi_2025 = sum(cpi_2025_months) / 12
deflator = {2022: 1.0, 2025: cpi_2022 / cpi_2025}

df = pd.read_pickle(f"{OUT_DIR}/plfs_with_exposure.pkl")
df = df.dropna(subset=["mean_exposure"]).copy()
df = df[~df["is_unpaid_family_worker"]].copy()
df = df[df["earnings"] > 0].copy()
df["earnings_real"] = df.apply(lambda r: r["earnings"] * deflator[r["year"]], axis=1)
df["log_earnings"] = np.log(df["earnings_real"])
df["post2025"] = (df["year"] == 2025).astype(float)
df["exposure_x_post"] = df["mean_exposure"] * df["post2025"]
df["age2"] = df["age"]**2
df["female"] = (df["sex"] == 2).astype(float)
df["urban"] = (df["sector"] == 2).astype(float)

feature_cols = ["mean_exposure", "post2025", "exposure_x_post", "age", "age2", "female", "edu_level", "urban"]
y = df["log_earnings"].values
X_full = np.column_stack([np.ones(len(df))] + [df[c].values for c in feature_cols])
names = ["const"] + feature_cols
key_idx = names.index("exposure_x_post")

# restricted design: drop exposure_x_post column
restr_idx = [i for i in range(X_full.shape[1]) if i != key_idx]
X_restr = X_full[:, restr_idx]

clusters = df["isco_submajor"].values
uniq_clusters, cluster_codes = np.unique(clusters, return_inverse=True)
n_clusters = len(uniq_clusters)
n, k_full = X_full.shape

XtX_inv_full = np.linalg.inv(X_full.T @ X_full)
XtX_inv_restr = np.linalg.inv(X_restr.T @ X_restr)

def cluster_robust_t(X, y_vec, XtX_inv, target_idx, cluster_codes, n_clusters):
    beta = XtX_inv @ (X.T @ y_vec)
    resid = y_vec - X @ beta
    k = X.shape[1]
    Xu = X * resid[:, None]
    meat = np.zeros((k, k))
    for c in range(n_clusters):
        s = Xu[cluster_codes == c].sum(axis=0)
        meat += np.outer(s, s)
    n_obs = X.shape[0]
    corr = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - k))
    cov = corr * (XtX_inv @ meat @ XtX_inv)
    se = np.sqrt(cov[target_idx, target_idx])
    return beta[target_idx] / se, beta[target_idx], se

# original (unrestricted) t-stat
t_orig, beta_orig, se_orig = cluster_robust_t(X_full, y, XtX_inv_full, key_idx, cluster_codes, n_clusters)
print(f"Original: coef={beta_orig:.4f}  clustered_se={se_orig:.4f}  t={t_orig:.3f}")

# restricted fit (null imposed: exposure_x_post excluded)
beta_restr = XtX_inv_restr @ (X_restr.T @ y)
fitted_restr = X_restr @ beta_restr
resid_restr = y - fitted_restr

B = 999
t_boot = np.empty(B)
t0 = time.time()
for b in range(B):
    rademacher = np.random.choice([-1.0, 1.0], size=n_clusters)
    w = rademacher[cluster_codes]
    y_boot = fitted_restr + w * resid_restr
    t_boot[b], _, _ = cluster_robust_t(X_full, y_boot, XtX_inv_full, key_idx, cluster_codes, n_clusters)
elapsed = time.time() - t0
print(f"{B} bootstrap replications in {elapsed:.1f}s")

p_boot = np.mean(np.abs(t_boot) >= np.abs(t_orig))
print(f"\nWild cluster bootstrap p-value (two-sided): {p_boot:.4f}")
print(f"(for comparison, asymptotic cluster-robust p-value was 0.025)")

ci_lo, ci_hi = np.percentile(t_boot, [2.5, 97.5])
print(f"Bootstrap t-stat distribution: 2.5th/97.5th pctile = [{ci_lo:.2f}, {ci_hi:.2f}], "
      f"vs original t = {t_orig:.2f}")
