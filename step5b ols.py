"""
CJSJ Paper — Step 5b: OLS, no interaction term (per the RF exploratory
check in step 5a -- mean_exposure was the WEAKEST predictor of the five,
and the RF itself had no real out-of-sample predictive power at n=35, so
there's no empirical basis for adding an interaction term at this stage).

No statsmodels available in this sandbox -- implemented via numpy least
squares with HC1 heteroskedasticity-robust standard errors (White 1980).
"""
import pandas as pd
import numpy as np
from scipy import stats

OUT_DIR = "/home/claude/cjsj"
df = pd.read_csv(f"{OUT_DIR}/isco2_growth_regression_sample.csv")
y = df["median_growth_pct"].values
n = len(y)

def ols_robust(X_df, y, label):
    X = np.column_stack([np.ones(len(y)), X_df.values])
    names = ["const"] + list(X_df.columns)
    k = X.shape[1]
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    # HC1 robust covariance (White, d.f.-adjusted)
    meat = X.T @ np.diag(resid**2) @ X
    cov_hc1 = (n / (n - k)) * (XtX_inv @ meat @ XtX_inv)
    se = np.sqrt(np.diag(cov_hc1))
    tvals = beta / se
    pvals = 2 * (1 - stats.t.cdf(np.abs(tvals), df=n - k))
    r2 = 1 - np.sum(resid**2) / np.sum((y - y.mean())**2)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k)

    print(f"\n=== {label} (n={n}, robust HC1 SEs) ===")
    print(f"{'var':18s} {'coef':>10s} {'se':>10s} {'t':>8s} {'p':>8s}")
    for nm, b, s, t, p in zip(names, beta, se, tvals, pvals):
        star = "*" if p < 0.05 else ("+" if p < 0.10 else "")
        print(f"{nm:18s} {b:10.3f} {s:10.3f} {t:8.2f} {p:8.3f} {star}")
    print(f"R^2 = {r2:.3f}   Adj. R^2 = {adj_r2:.3f}")
    return dict(zip(names, zip(beta, se, pvals)))

# Model 1: bivariate, exposure only
ols_robust(df[["mean_exposure"]], y, "Model 1: median_growth ~ exposure")

# Model 2: full controls, no interaction
ols_robust(df[["mean_exposure","pct_female","mean_age","pct_urban","mean_edu_level"]], y,
           "Model 2: median_growth ~ exposure + controls")
