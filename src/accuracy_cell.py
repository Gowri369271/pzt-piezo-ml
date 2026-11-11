# ===================== MODEL ACCURACY TEST (run after the training cells) =====================
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def score(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return dict(R2=r2_score(y, p),
                RMSE=float(np.sqrt(mean_squared_error(y, p))),
                MAE=mean_absolute_error(y, p),
                MAPE_pct=100 * np.mean(np.abs((y - p) / y)))

# ---- A. Held-out test set (the 14 rows the models never saw) ----
rows = []
for tgt in TARGETS:
    for name in GRIDS:
        model, _ = fitted[(tgt, name)]
        rows.append(dict(Target=tgt, Model=name, **score(Y_te[tgt], model.predict(X_te))))
test_tbl = pd.DataFrame(rows)
print("A. Held-out test set: R2")
display(test_tbl.pivot(index="Target", columns="Model", values="R2").round(4))
print("A. Held-out test set: MAPE (%)")
display(test_tbl.pivot(index="Target", columns="Model", values="MAPE_pct").round(2))

# ---- B. Leave-one-a/b-out validation (tougher, more honest) ----
# Neighbouring coverage rows at the same a/b are near-duplicates, so a random split is optimistic.
# Here every a/b ratio is held out in turn and predicted by a model trained on the other ratios.
X_all, Y_all, groups = df[FEATURES], df[TARGETS], df["a/b"].values
logo = LeaveOneGroupOut()
rows, oof = [], {}
for tgt in TARGETS:
    for name in GRIDS:
        model, _ = fitted[(tgt, name)]
        p = cross_val_predict(clone(model), X_all, Y_all[tgt], groups=groups, cv=logo, n_jobs=-1)
        oof[(tgt, name)] = p
        rows.append(dict(Target=tgt, Model=name, **score(Y_all[tgt], p)))
logo_tbl = pd.DataFrame(rows)
print("B. Leave-one-a/b-out: R2")
display(logo_tbl.pivot(index="Target", columns="Model", values="R2").round(4))
print("B. Leave-one-a/b-out: MAPE (%)")
display(logo_tbl.pivot(index="Target", columns="Model", values="MAPE_pct").round(2))

# ---- C. Best model per target (by leave-one-a/b-out R2) + where its errors are largest ----
# MAPE reflects typical-case error; R2 is dominated by the few extreme rows (e.g. a/b=4 at 10 % coverage).
# Tree models predict piecewise-constant steps, so they interpolate poorly between the few a/b values in this dataset.
best = logo_tbl.loc[logo_tbl.groupby("Target")["MAPE_pct"].idxmin()].set_index("Target")["Model"]
print("Best model per target (lowest leave-one-a/b-out MAPE):", best.to_dict())

fig, axes = plt.subplots(2, 2, figsize=(10, 9))
for ax, tgt in zip(axes.ravel(), TARGETS):
    y, p = Y_all[tgt].values, oof[(tgt, best[tgt])]
    ax.scatter(y, p, c=np.log(groups), cmap="viridis", s=25)
    lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_title(f"{tgt}: {best[tgt]} (leave-one-a/b-out)")
    ax.set_xlabel("True"); ax.set_ylabel("Predicted")
plt.tight_layout(); plt.show()

err = pd.DataFrame({"a/b": groups})
for tgt in TARGETS:
    err[tgt] = np.abs(Y_all[tgt].values - oof[(tgt, best[tgt])]) / Y_all[tgt].values * 100
print("C. Mean absolute % error by held-out a/b (edge ratios must extrapolate, so they are usually worst)")
display(err.groupby("a/b").mean().round(2))

# ---- D. Predict a new design ----
def predict_design(a_b, coverage_pct):
    if not (df["a/b"].min() <= a_b <= df["a/b"].max() and 10 <= coverage_pct <= 100):
        print("WARNING: outside the training range, so this is extrapolation and unreliable.")
    x = pd.DataFrame([[a_b, coverage_pct]], columns=FEATURES)
    return {tgt: float(fitted[(tgt, best[tgt])][0].predict(x)[0]) for tgt in TARGETS}

print("Prediction for a/b = 3.0, coverage = 15 %:")
print(predict_design(3.0, 15))
