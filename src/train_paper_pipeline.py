"""
Replicates the ML workflow of Islam et al. (2026), Energy Conversion and Management: X 31, 102173.

Pipeline (paper Fig. 9, Sec. 2.6, 3.2):
  1. Load dataset  -> features: a/b, Piezo_Area(%) ; targets: Voltage, Power, PowerDensity, EigenFrequency
  2. 80/20 train/test split, StandardScaler fitted on TRAIN only
  3. Separate model per target for RF, XGBoost, SVR, GBR; GridSearchCV with CV
  4. Metrics: R2, RMSE, MAE (train and test) + parity plots
  5. SHAP: mean |SHAP| bar plots and beeswarm plots per model/target
  6. Robustness on XGBoost: repeated 5-fold CV x30, learning curve, 500-bootstrap 95% prediction intervals
  7. Polynomial RSM baselines (quadratic, cubic) vs XGBoost

Usage: python train_paper_pipeline.py pzt_dataset_extracted.csv outputs
"""
import sys, os, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import (train_test_split, GridSearchCV, RepeatedKFold,
                                     cross_validate, learning_curve)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

CSV = sys.argv[1] if len(sys.argv) > 1 else "pzt_dataset_extracted.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "outputs"
os.makedirs(OUT, exist_ok=True)

SEED = 42          # not stated in the paper
CV_FOLDS = 5       # not stated in the paper for GridSearchCV
FEATURES = ["a/b", "Piezo_Area(%)"]
FEAT_LABELS = ["Apex-to-base ratio (a/b)", "Piezoelectric material coverage (%)"]
TARGETS = ["Voltage(V)", "Power(mW)", "PowerDensity(uW/mm3)", "EigenFrequency(Hz)"]

# Grids contain the best values reported in the paper's Table 6
GRIDS = {
    "RF": (RandomForestRegressor(random_state=SEED), {
        "n_estimators": [100, 300], "max_depth": [10, 20, 30],
        "min_samples_split": [2, 5], "min_samples_leaf": [1, 2], "bootstrap": [True]}),
    "XGBoost": (XGBRegressor(random_state=SEED, verbosity=0), {
        "learning_rate": [0.05, 0.1, 0.2], "max_depth": [3, 6, 9],
        "n_estimators": [200, 300], "subsample": [0.8, 1.0],
        "colsample_bytree": [1.0], "gamma": [0, 0.1]}),
    "SVR": (SVR(), {
        "C": [1, 10, 100], "epsilon": [0.01, 0.1],
        "gamma": ["scale", "auto", 1], "kernel": ["rbf"]}),
    "GBR": (GradientBoostingRegressor(random_state=SEED), {
        "learning_rate": [0.05, 0.1, 0.2], "max_depth": [3, 5],
        "n_estimators": [100, 300], "subsample": [0.8, 1.0],
        "max_features": ["sqrt"], "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 4]}),
}

def rmse(y, p): return float(np.sqrt(mean_squared_error(y, p)))

# ---------------------------------------------------------------- 1-2. data
df = pd.read_csv(CSV)
X, Y = df[FEATURES], df[TARGETS]
X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.20, random_state=SEED)
print(f"rows={len(df)}  train={len(X_tr)}  test={len(X_te)}")

# scaler fitted on training data only (used by SVR; trees are scale-invariant)
scaler = StandardScaler().fit(X_tr)

def make_estimator(name, params=None):
    base, _ = GRIDS[name]
    est = base.__class__(**{**base.get_params(), **(params or {})})
    return make_pipeline(StandardScaler(), est) if name == "SVR" else est

# ---------------------------------------------------------------- 3-4. train
rows, fitted = [], {}
for tgt in TARGETS:
    y_tr, y_te = Y_tr[tgt].values, Y_te[tgt].values
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, name in zip(axes.ravel(), GRIDS):
        base, grid = GRIDS[name]
        if name == "SVR":
            pipe = make_pipeline(StandardScaler(), base)
            grid = {f"svr__{k}": v for k, v in grid.items()}
        else:
            pipe = base
        gs = GridSearchCV(pipe, grid, cv=CV_FOLDS, scoring="r2", n_jobs=-1).fit(X_tr, y_tr)
        best = gs.best_estimator_
        fitted[(tgt, name)] = (best, gs.best_params_)
        p_tr, p_te = best.predict(X_tr), best.predict(X_te)
        rows.append(dict(Target=tgt, Model=name,
                         R2_train=r2_score(y_tr, p_tr), R2_test=r2_score(y_te, p_te),
                         RMSE_train=rmse(y_tr, p_tr), RMSE_test=rmse(y_te, p_te),
                         MAE_train=mean_absolute_error(y_tr, p_tr), MAE_test=mean_absolute_error(y_te, p_te),
                         best_params=str(gs.best_params_)))
        lo, hi = min(y_tr.min(), y_te.min()), max(y_tr.max(), y_te.max())
        ax.plot([lo, hi], [lo, hi], color="grey", label="Ideal line")
        ax.scatter(y_tr, p_tr, marker="^", c="yellowgreen", label="Training")
        ax.scatter(y_te, p_te, c="slateblue", label="Testing")
        ax.set_title(f"Model: {name}"); ax.set_xlabel(f"True {tgt}"); ax.set_ylabel(f"Predicted {tgt}")
        ax.text(0.45, 0.05, f"Train R2={rows[-1]['R2_train']:.3f}\nTest R2={rows[-1]['R2_test']:.3f}\n"
                            f"Test MAE={rows[-1]['MAE_test']:.3f}", transform=ax.transAxes, fontsize=8,
                bbox=dict(fc="white", alpha=.8))
        ax.legend(loc="upper left", fontsize=8)
        print(f"{tgt:22s} {name:8s} R2_test={rows[-1]['R2_test']:.4f}")
    fig.tight_layout(); fig.savefig(f"{OUT}/parity_{tgt.split('(')[0]}.png", dpi=150); plt.close(fig)

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/table6_model_performance.csv", index=False)

# ---------------------------------------------------------------- 5. SHAP
shap_summary = []
for tgt in TARGETS:
    for name in GRIDS:
        model, _ = fitted[(tgt, name)]
        if name == "SVR":
            expl = shap.Explainer(model.predict, X_tr)
            sv = expl(X_te).values
        else:
            expl = shap.TreeExplainer(model)
            sv = expl.shap_values(X_te)
        mean_abs = np.abs(sv).mean(axis=0)
        shap_summary.append(dict(Target=tgt, Model=name, **{f"meanAbsSHAP_{f}": m for f, m in zip(FEATURES, mean_abs)}))
        short = tgt.split("(")[0]
        plt.figure(figsize=(6, 3)); plt.barh(FEAT_LABELS[::-1], mean_abs[::-1], color="#0057a8")
        plt.xlabel("Mean Absolute SHAP Value"); plt.title(f"Model: {name} ({short})")
        plt.tight_layout(); plt.savefig(f"{OUT}/shap_bar_{short}_{name}.png", dpi=150); plt.close()
        shap.summary_plot(sv, X_te.rename(columns=dict(zip(FEATURES, FEAT_LABELS))), show=False, cmap=plt.get_cmap("coolwarm"))
        plt.title(f"Model: {name} ({short})"); plt.tight_layout()
        plt.savefig(f"{OUT}/shap_beeswarm_{short}_{name}.png", dpi=150); plt.close()
pd.DataFrame(shap_summary).to_csv(f"{OUT}/shap_importance.csv", index=False)

# ---------------------------------------------------------------- 6. XGBoost robustness
rob = []
rkf = RepeatedKFold(n_splits=5, n_repeats=30, random_state=SEED)
for tgt in TARGETS:
    y = Y[tgt].values
    xgb_best, xgb_params = fitted[(tgt, "XGBoost")]
    # repeated CV on the whole dataset
    cv = cross_validate(make_estimator("XGBoost", xgb_params), X, y, cv=rkf, n_jobs=-1,
                        scoring=("r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"))
    # learning curve (paper: 24 -> 120 samples = 20%..100% of the 120-sample training set)
    ts, _, val = learning_curve(make_estimator("XGBoost", xgb_params), X_tr, Y_tr[tgt].values,
                                train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0], cv=CV_FOLDS,
                                scoring="neg_root_mean_squared_error", n_jobs=-1)
    # bootstrap prediction intervals (500 models)
    rng = np.random.default_rng(SEED)
    preds = np.empty((500, len(X_te)))
    for b in range(500):
        idx = rng.integers(0, len(X_tr), len(X_tr))
        m = make_estimator("XGBoost", xgb_params).fit(X_tr.iloc[idx], Y_tr[tgt].values[idx])
        preds[b] = m.predict(X_te)
    lo, hi = np.percentile(preds, [2.5, 97.5], axis=0)
    y_te = Y_te[tgt].values
    rob.append(dict(Target=tgt,
        CV_R2=f"{cv['test_r2'].mean():.5f} +/- {cv['test_r2'].std():.5f}",
        CV_RMSE=f"{-cv['test_neg_root_mean_squared_error'].mean():.5f} +/- {cv['test_neg_root_mean_squared_error'].std():.5f}",
        CV_MAE=f"{-cv['test_neg_mean_absolute_error'].mean():.5f}",
        learning_curve_sizes=list(ts), learning_curve_val_RMSE=[round(-v.mean(), 4) for v in val],
        bootstrap_mean_R2=r2_score(y_te, preds.mean(axis=0)),
        PI95_coverage_pct=100 * np.mean((y_te >= lo) & (y_te <= hi)),
        mean_PI_width=float(np.mean(hi - lo))))
pd.DataFrame(rob).to_csv(f"{OUT}/table8_xgboost_robustness.csv", index=False)

# ---------------------------------------------------------------- 7. RSM baselines
rsm = []
for tgt in TARGETS:
    y_tr, y_te, y = Y_tr[tgt].values, Y_te[tgt].values, Y[tgt].values
    models = {"Quadratic RSM": make_pipeline(PolynomialFeatures(2), LinearRegression()),
              "Cubic RSM": make_pipeline(PolynomialFeatures(3), LinearRegression()),
              "XGBoost": make_estimator("XGBoost", fitted[(tgt, "XGBoost")][1])}
    for mname, m in models.items():
        m.fit(X_tr, y_tr); p = m.predict(X_te)
        cv = cross_validate(m, X, y, cv=rkf, n_jobs=-1, scoring=("r2", "neg_root_mean_squared_error"))
        rsm.append(dict(Target=tgt, Model=mname, Test_R2=r2_score(y_te, p), Test_RMSE=rmse(y_te, p),
                        Test_MAE=mean_absolute_error(y_te, p),
                        CV_R2=cv["test_r2"].mean(), CV_RMSE=-cv["test_neg_root_mean_squared_error"].mean()))
pd.DataFrame(rsm).to_csv(f"{OUT}/table7_rsm_vs_xgboost.csv", index=False)

print("\nDone. Outputs in", OUT)
print(res[["Target", "Model", "R2_test", "RMSE_test", "MAE_test"]].round(4).to_string(index=False))
