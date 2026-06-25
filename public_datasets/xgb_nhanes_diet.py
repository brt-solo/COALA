# ── Config ────────────────────────────────────────────────────────────────────
# Variant of xgb_nhanes.py where the only COALA-mutable features are dietary
# intake variables from the NHANES first 24-hour dietary recall (DR1TOT_J).
# All clinical/demographic features are kept for prediction but are constraints.
USE_SMOTE = False

import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

if USE_SMOTE:
    from imblearn.over_sampling import SMOTE

# ── Metrics helper ────────────────────────────────────────────────────────────
def print_metrics(y_true, y_prob, label=""):
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    print(f"\n── {label} ──")
    print(f"  ROC AUC:     {roc_auc_score(y_true, y_prob):.4f}")
    print(f"  AUPRC:       {average_precision_score(y_true, y_prob):.4f}")
    print(f"  Accuracy:    {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision:   {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  Recall:      {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  F1:          {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  Brier:       {brier_score_loss(y_true, y_prob):.4f}")

# ── Data loading ──────────────────────────────────────────────────────────────
data_dir = os.path.dirname(os.path.abspath(__file__))

demo  = pd.read_sas(os.path.join(data_dir, "DEMO_J.xpt"))
bmx   = pd.read_sas(os.path.join(data_dir, "BMX_J.xpt"))
bpx   = pd.read_sas(os.path.join(data_dir, "BPX_J.xpt"))
tchol = pd.read_sas(os.path.join(data_dir, "TCHOL_J.xpt"))
ghb   = pd.read_sas(os.path.join(data_dir, "GHB_J.xpt"))
smq   = pd.read_sas(os.path.join(data_dir, "SMQ_J.xpt"))
dr1   = pd.read_sas(os.path.join(data_dir, "DR1TOT_J.XPT"))

# Only keep reliable dietary recalls (status 1 = reliable, not missing)
dr1 = dr1[dr1["DR1DRSTZ"] == 1][
    ["SEQN", "DR1TKCAL", "DR1TCARB", "DR1TSUGR",
     "DR1TTFAT", "DR1TSFAT", "DR1TPROT", "DR1TFIBE", "DR1TSODI"]
]

df = (demo
      .merge(bmx,   on="SEQN", how="left")
      .merge(bpx,   on="SEQN", how="left")
      .merge(tchol, on="SEQN", how="left")
      .merge(ghb,   on="SEQN", how="left")
      .merge(smq,   on="SEQN", how="left")
      .merge(dr1,   on="SEQN", how="inner")    # inner: only keep those with valid diet recall
      .copy())

df = df[df["RIDAGEYR"] >= 20].copy()
df["y"] = (df["LBXGH"] >= 6.5).astype(int)

print(f"After diet recall merge: {len(df)} participants, "
      f"diabetes prevalence = {df['y'].mean():.3f}")

# ── Feature selection ─────────────────────────────────────────────────────────
# Constraints: demographics + clinical baseline (not mutable in COALA)
# Mutable:     dietary intake variables (the intervention modality)
feature_map = {
    # ── Constraints ──────────────────────────────────────────────────────────
    "RIDAGEYR": "Age (years)",
    "RIAGENDR": "Sex (male)",
    #"RIDRETH3": "Race/ethnicity",
    "INDFMPIR": "Poverty-income ratio",
    "BMXWAIST": "Waist circumference (cm)",
    #"BMXBMI":   "BMI",
    "BPXSY1":   "Systolic BP (mmHg)",
    "BPXDI1":   "Diastolic BP (mmHg)",
    "LBXTC":    "Total cholesterol (mg/dL)",
    "SMQ020":   "Smoked ≥100 cigarettes",
    # ── Mutable: dietary intake ───────────────────────────────────────────────
    "DR1TKCAL": "Energy (kcal)",
    "DR1TCARB": "Carbohydrates (g)",
    "DR1TSUGR": "Total sugars (g)",
    "DR1TTFAT": "Total fat (g)",
    "DR1TSFAT": "Saturated fat (g)",
    "DR1TPROT": "Protein (g)",
    "DR1TFIBE": "Dietary fiber (g)",
    "DR1TSODI": "Sodium (mg)",
}

X = df[list(feature_map.keys())].copy()
y = df["y"]
X = X.rename(columns=feature_map)

# ── Recode binary columns ─────────────────────────────────────────────────────
X["Sex (male)"]             = X["Sex (male)"].map({1.0: 1, 2.0: 0})
X["Smoked ≥100 cigarettes"] = X["Smoked ≥100 cigarettes"].map({1.0: 1, 2.0: 0})

# ── Sentinel codes → NaN ─────────────────────────────────────────────────────
X = X.replace({v: np.nan for v in [9996, 9998, 9999, 99999]})

# ── Hard physiological / domain bounds (fixed constants — safe before split) ──
# Dietary bounds reflect realistic 24-hr recall ranges in NHANES adults.
# Values beyond these are data-entry errors or physiologically implausible.
_phys_bounds = {
    "Diastolic BP (mmHg)":      (40,   130),
    "Systolic BP (mmHg)":       (70,   250),
    "Waist circumference (cm)": (60,   160),    # <60 implausible for adults 20+; >160 extreme obesity
    #"BMI":                      (15,    60),    # <15 not survivable; >60 extreme; 99th pct ~55
    "Total cholesterol (mg/dL)":(50,   600),
    "Age (years)":              (20,    85),
    "Poverty-income ratio":     (0,      5),
    "Energy (kcal)":            (500,  5000),   # <500 implausible; >5000 extreme
    "Carbohydrates (g)":        (0,    600),    # 95th pct ~450g in NHANES adults
    "Total sugars (g)":         (0,    300),    # 95th pct ~220g; 300 ≈ 99th pct
    "Total fat (g)":            (0,    200),    # 95th pct ~140g
    "Saturated fat (g)":        (0,     80),    # 95th pct ~50g
    "Protein (g)":              (0,    250),    # 95th pct ~175g
    "Dietary fiber (g)":        (0,     60),    # 95th pct ~32g
    "Sodium (mg)":              (200,  5500),   # 95th pct ~5,700 mg for men
}
for col, (lo, hi) in _phys_bounds.items():
    if col in X.columns:
        X[col] = X[col].where((X[col] >= lo) & (X[col] <= hi))

# Drop rows made NaN by sentinel codes or hard bounds — safe before split
# because these thresholds are fixed constants, not derived from data.
mask = X.notna().all(axis=1)
X = X[mask]
y = y[mask]

print("Columns:", X.columns.tolist())
print("X shape after hard bounds:", X.shape, "  diabetes prevalence:", y.mean().round(3))

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10, stratify=y
)
X_train = X_train.select_dtypes(include="number")
X_test  = X_test[X_train.columns]

# ── IQR outlier removal — fit on training data only to prevent leakage ────────
# Quantiles are derived from X_train alone; the same fences are then applied
# to X_test so test statistics never influence the thresholds.
_non_continuous = {"Sex (male)", "Race/ethnicity", "Smoked ≥100 cigarettes"}
_iqr_bounds = {}
for col in X_train.columns:
    if col in _non_continuous:
        continue
    q1, q3 = X_train[col].quantile(0.25), X_train[col].quantile(0.75)
    iqr = q3 - q1
    _iqr_bounds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

for col, (lo, hi) in _iqr_bounds.items():
    X_train[col] = X_train[col].where((X_train[col] >= lo) & (X_train[col] <= hi))
    X_test[col]  = X_test[col].where((X_test[col]  >= lo) & (X_test[col]  <= hi))

train_mask = X_train.notna().all(axis=1)
X_train, y_train = X_train[train_mask], y_train[train_mask]

test_mask = X_test.notna().all(axis=1)
X_test, y_test = X_test[test_mask], y_test[test_mask]

print(f"After IQR filter — train: {X_train.shape[0]}, test: {X_test.shape[0]}")

X_train_arr = X_train.to_numpy()
X_test_arr  = X_test.to_numpy()
best_features = list(X_train.columns)

# ── Class imbalance handling ──────────────────────────────────────────────────
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
scale_pos_weight = neg / pos
if USE_SMOTE:
    smote = SMOTE(random_state=42)
    X_train_arr, y_train = smote.fit_resample(X_train_arr, y_train)
else:
    print(f"scale_pos_weight = {scale_pos_weight:.2f}  (neg={neg}, pos={pos})")

# ── XGBoost hyperparameter search ────────────────────────────────────────────
cv = KFold(n_splits=5, shuffle=True, random_state=42)
param_distributions = {
    "n_estimators":     randint(100, 2000),
    "max_depth":        randint(2, 12),
    "learning_rate":    loguniform(0.005, 0.3),
    "subsample":        uniform(0.5, 0.5),
    "colsample_bytree": uniform(0.3, 0.7),
    "min_child_weight": randint(1, 20),
    "reg_alpha":        loguniform(1e-8, 10.0),
    "reg_lambda":       loguniform(1e-8, 10.0),
    "gamma":            uniform(0.0, 5.0),
}
search = RandomizedSearchCV(
    XGBClassifier(
        objective="binary:logistic",
        booster="gbtree",
        eval_metric="logloss",
        random_state=10,
        scale_pos_weight=(1.0 if USE_SMOTE else scale_pos_weight),
    ),
    param_distributions=param_distributions,
    n_iter=300,
    scoring="roc_auc",
    cv=cv,
    verbose=1,
    n_jobs=-1,
    random_state=42,
)
search.fit(X_train_arr, y_train)
print("\nXGBoost best CV ROC AUC:", search.best_score_)
xgb_final = search.best_estimator_

print_metrics(y_train, xgb_final.predict_proba(X_train_arr)[:, 1], "XGBoost — Train")
print_metrics(y_test,  xgb_final.predict_proba(X_test_arr)[:, 1],  "XGBoost — Test")

# ── Logistic Regression ───────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_arr)
X_test_sc  = scaler.transform(X_test_arr)
_cw = None if USE_SMOTE else "balanced"
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight=_cw)
lr.fit(X_train_sc, y_train)
print_metrics(y_train, lr.predict_proba(X_train_sc)[:, 1], "Logistic Regression — Train")
print_metrics(y_test,  lr.predict_proba(X_test_sc)[:, 1],  "Logistic Regression — Test")

# ── Random Forest ─────────────────────────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight=_cw)
rf.fit(X_train_arr, y_train)
print_metrics(y_train, rf.predict_proba(X_train_arr)[:, 1], "Random Forest — Train")
print_metrics(y_test,  rf.predict_proba(X_test_arr)[:, 1],  "Random Forest — Test")

# ── Save ─────────────────────────────────────────────────────────────────────
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
model_dir = os.path.join(repo_root, "models")
os.makedirs(model_dir, exist_ok=True)
with open(os.path.join(model_dir, "xgb_nhanes_diet.pkl"), "wb") as f:
    pickle.dump(xgb_final, f)
print("\nSaved model to", os.path.join(model_dir, "xgb_nhanes_diet.pkl"))

save_dir = os.path.join(repo_root, "models", "real")
os.makedirs(save_dir, exist_ok=True)
pd.DataFrame(X_train_arr, columns=best_features).to_csv(
    os.path.join(save_dir, "X_train_xgb_nhanes_diet.csv"), index=False)
pd.DataFrame(X_test_arr, columns=best_features).to_csv(
    os.path.join(save_dir, "X_test_xgb_nhanes_diet.csv"),  index=False)
print("Saved train/test CSVs to", save_dir)
