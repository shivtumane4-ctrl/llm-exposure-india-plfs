"""
CJSJ Paper — Step 5a: Random Forest feature importance (exploratory, no
interaction terms specified), run BEFORE any OLS specification is chosen.

Outcome: median real earnings growth (2022->2025), ISCO-08 group level.
Sample: 35 groups (71/92/93 excluded per step 4c -- undefined median growth).
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import LeaveOneOut, cross_val_predict

OUT_DIR = "/home/claude/cjsj"
df = pd.read_csv(f"{OUT_DIR}/isco2_growth_regression_sample.csv")

features = ["mean_exposure", "pct_female", "mean_age", "pct_urban", "mean_edu_level"]
X = df[features].values
y = df["median_growth_pct"].values

print(f"n = {len(df)} occupation groups, {len(features)} features")
print("Feature summary:")
print(df[features].describe().round(2))

# small-n RF: shallow trees, many estimators to stabilize, fixed seed for reproducibility
rf = RandomForestRegressor(n_estimators=1000, max_depth=3, min_samples_leaf=3,
                            random_state=42)
rf.fit(X, y)

print("\n--- Impurity-based feature importance ---")
imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print(imp.round(3))

print("\n--- Permutation importance (more reliable with n=35, correlated features) ---")
perm = permutation_importance(rf, X, y, n_repeats=200, random_state=42)
perm_imp = pd.Series(perm.importances_mean, index=features).sort_values(ascending=False)
perm_std = pd.Series(perm.importances_std, index=features)
for f in perm_imp.index:
    print(f"  {f:18s} mean={perm_imp[f]:+.3f}  std={perm_std[f]:.3f}")

# leave-one-out CV R^2, honest given n=35 (no holdout set otherwise viable)
loo = LeaveOneOut()
y_pred_loo = cross_val_predict(RandomForestRegressor(n_estimators=1000, max_depth=3,
                                                        min_samples_leaf=3, random_state=42),
                                 X, y, cv=loo)
ss_res = np.sum((y - y_pred_loo)**2)
ss_tot = np.sum((y - y.mean())**2)
loo_r2 = 1 - ss_res/ss_tot
print(f"\nLeave-one-out CV R^2: {loo_r2:.3f}  (in-sample R^2 would be optimistic with n=35 -- LOO is the honest number)")

# in-sample R2 for comparison
in_sample_r2 = rf.score(X, y)
print(f"In-sample R^2 (for reference/contrast only, NOT for reporting): {in_sample_r2:.3f}")
