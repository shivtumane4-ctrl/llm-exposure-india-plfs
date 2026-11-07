"""
Step 5e: Robustness of the exposure x post2025 coefficient to (i) education as
categorical dummies instead of one numeric code, and (ii) state fixed effects.
Clustered (CR1) SEs by ISCO-08 sub-major group, plus wild cluster restricted
(WCR) bootstrap, 999 Rademacher replications, seed 42.

Speed trick: only the t-stat on exposure_x_post is needed per replication.
Var(b_t) = corr * sum_c (a_t . s_c)^2, where a_t is row t of (X'X)^-1 and
s_c = X_c' u_c. With z = X a_t (fixed across replications), a_t . s_c =
sum_{i in c} z_i u_i, so each replication is O(n) via np.bincount.
Mathematically identical to the full CR1 sandwich element.
"""
import numpy as np, pandas as pd
from scipy import stats
import time

OUT = "/home/claude/cjsj"
cpi22 = np.mean([165.7,166.1,167.7,170.1,171.7,172.6,173.4,174.3,175.3,176.7,176.5,175.7])
cpi25 = np.mean([193.4,192.5,192.0,192.6,193.0,194.2,196.1,197.0,197.0,197.3,197.9,198.0])
defl = {2022: 1.0, 2025: cpi22 / cpi25}

df = pd.read_pickle(f"{OUT}/plfs_with_exposure.pkl").dropna(subset=["mean_exposure"])
df = df[~df["is_unpaid_family_worker"] & (df["earnings"] > 0)].copy()
df["log_earn"] = np.log(df["earnings"] * df["year"].map(defl))
df["post2025"] = (df["year"] == 2025).astype(float)
df["exposure_x_post"] = df["mean_exposure"] * df["post2025"]
df["age2"] = df["age"] ** 2
df["female"] = (df["sex"] == 2).astype(float)
df["urban"] = (df["sector"] == 2).astype(float)

# sort by cluster so bincount codes are contiguous (not required, just tidy)
df = df.sort_values("isco_submajor").reset_index(drop=True)
codes = pd.factorize(df["isco_submajor"])[0]
G = codes.max() + 1
y = df["log_earn"].values
n = len(y)

base = ["mean_exposure", "post2025", "exposure_x_post", "age", "age2", "female", "urban"]
edu_d = pd.get_dummies(df["edu_level"], prefix="edu", drop_first=True, dtype=float)
st_d = pd.get_dummies(df["state"], prefix="st", drop_first=True, dtype=float)

specs = {
    "A: baseline (edu numeric, no state FE)": (base + ["edu_level"], None, None),
    "B: edu dummies, no state FE":            (base, edu_d, None),
    "C: edu dummies + state FE":              (base, edu_d, st_d),
}

def build(cols, d1, d2):
    parts = [np.ones((n, 1)), df[cols].values]
    names = ["const"] + cols
    for d in (d1, d2):
        if d is not None:
            parts.append(d.values); names += list(d.columns)
    return np.hstack(parts), names

def cr1_t(X, yv, XtXi, ti, z):
    beta = XtXi @ (X.T @ yv)
    u = yv - X @ beta
    q = np.bincount(codes, weights=z * u, minlength=G)
    k = X.shape[1]
    corr = (G / (G - 1)) * ((n - 1) / (n - k))
    se = np.sqrt(corr * np.sum(q ** 2))
    return beta[ti] / se, beta[ti], se, beta, u

rows = []
for label, (cols, d1, d2) in specs.items():
    X, names = build(cols, d1, d2)
    ti = names.index("exposure_x_post")
    XtXi = np.linalg.inv(X.T @ X)
    z = X @ XtXi[ti]
    t0, b0, se0, beta, u = cr1_t(X, y, XtXi, ti, z)
    p_asym = 2 * (1 - stats.t.cdf(abs(t0), df=G - 1))
    r2 = 1 - np.sum(u ** 2) / np.sum((y - y.mean()) ** 2)

    # WCR bootstrap: restricted model drops exposure_x_post
    keep = [i for i in range(X.shape[1]) if i != ti]
    Xr = X[:, keep]
    br = np.linalg.solve(Xr.T @ Xr, Xr.T @ y)
    fit_r = Xr @ br
    res_r = y - fit_r
    np.random.seed(42)
    B = 999
    tb = np.empty(B)
    t_start = time.time()
    for b in range(B):
        w = np.random.choice([-1.0, 1.0], size=G)[codes]
        tb[b] = cr1_t(X, fit_r + w * res_r, XtXi, ti, z)[0]
    p_boot = np.mean(np.abs(tb) >= abs(t0))
    rows.append(dict(spec=label, k=X.shape[1], coef=b0, se=se0, t=t0,
                     p_asymptotic=p_asym, p_wild_boot=p_boot, r2=r2))
    print(f"{label}\n  k={X.shape[1]}  coef={b0:.4f}  CR1 SE={se0:.4f}  t={t0:.2f}  "
          f"p(asym)={p_asym:.3f}  p(WCR boot)={p_boot:.3f}  R2={r2:.4f}  "
          f"[{time.time()-t_start:.0f}s]")
    if label.startswith("C"):
        pd.DataFrame({"var": names, "coef": beta}).to_csv(f"{OUT}/specC_all_coefs.csv", index=False)
        main = [n_ for n_ in names if not n_.startswith(("edu_", "st_"))] 
        print("  Spec C core coefficients:")
        for nm in main:
            print(f"    {nm:18s} {beta[names.index(nm)]:9.4f}")

pd.DataFrame(rows).to_csv(f"{OUT}/robustness_fe_summary.csv", index=False)
