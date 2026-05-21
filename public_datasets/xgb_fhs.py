# ── Config ────────────────────────────────────────────────────────────────────
USE_SMOTE = False   # False → scale_pos_weight (XGB) + class_weight="balanced" (LR/RF)

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
url = (
    "https://ocw.mit.edu/courses/15-071-the-analytics-edge-spring-2017/"
    "5d689a024551e672313f7fd7eb1bee8d_framingham.csv"
)
df = pd.read_csv(url)

print(df.head())
print(df.columns)

# ── Preprocessing ─────────────────────────────────────────────────────────────
df["y"] = df["TenYearCHD"].astype(int)

keep_cols = df.columns[df.isna().mean() <= 0.20]
df = df[keep_cols].copy()
df = df.dropna()

X = df.drop(columns=["TenYearCHD", "y", "education"])
y = df["y"]
X = X.select_dtypes(include="number")

new_col_names = [
    "Sex (male)", "Age (years)", "Current smoker", "Cigarettes per day",
    "BP Medication", "Stroke history", "Hypertension", "Diabetes",
    "Total cholesterol", "Systolic BP", "Diastolic BP",
    "BMI", "Resting HR", "Blood glucose",
]
X.columns = new_col_names
X_cols = new_col_names

# ── Replace sentinel/outlier codes with NaN ───────────────────────────────────
# Large numeric sentinel codes (9999, 9996, 9998, etc.)
X = X.replace({v: np.nan for v in [9996, 9998, 9999, 99999]})

# Drop rows that still have any NaN after sentinel removal
X = X[X.notna().all(axis=1)]
y = y[X.index]

print("Columns after missingness filter:", X.columns.tolist())
print("X shape:", X.shape, "y mean:", y.mean())

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=8, stratify=y
)

X_train = X_train.select_dtypes(include="number")
X_test  = X_test.select_dtypes(include="number")
X_test  = X_test[X_train.columns]

X_train_arr = X_train.to_numpy()
X_test_arr  = X_test.to_numpy()
best_features = list(X_train.columns)

# ── Class imbalance handling ──────────────────────────────────────────────────
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
scale_pos_weight = neg / pos

if USE_SMOTE:
    smote = SMOTE(random_state=42)
    X_train_arr, y_train = smote.fit_resample(X_train_arr, y_train)
    print(f"After SMOTE — train size: {X_train_arr.shape[0]}, class balance: {y_train.mean():.3f}")
else:
    print(f"scale_pos_weight = {scale_pos_weight:.2f}  (neg={neg}, pos={pos})")

# ── XGBoost: hyperparameter search ───────────────────────────────────────────
cv = KFold(n_splits=5, shuffle=True, random_state=42)

param_distributions = {
    "n_estimators":     randint(20, 1000),
    "max_depth":        randint(2, 12),
    "learning_rate":    loguniform(0.005, 0.3),
    "subsample":        uniform(0.1, 0.5),
    "colsample_bytree": uniform(0.1, 0.7),
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
        random_state=42,
        scale_pos_weight=(1.0 if USE_SMOTE else scale_pos_weight),
    ),
    param_distributions=param_distributions,
    n_iter=500,
    scoring="roc_auc",
    cv=cv,
    verbose=1,
    n_jobs=-1,
    random_state=42,
)
search.fit(X_train_arr, y_train)

print("\nXGBoost best CV ROC AUC:", search.best_score_)
print("XGBoost best params:", search.best_params_)

xgb_final = search.best_estimator_

xgb_train_prob = xgb_final.predict_proba(X_train_arr)[:, 1]
xgb_test_prob  = xgb_final.predict_proba(X_test_arr)[:, 1]

print_metrics(y_train, xgb_train_prob, "XGBoost — Train")
print_metrics(y_test,  xgb_test_prob,  "XGBoost — Test")

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

# ── Save XGBoost model ────────────────────────────────────────────────────────
model_dir = os.path.expanduser("~/models")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "xgb_fhs.pkl")
with open(model_path, "wb") as f:
    pickle.dump(xgb_final, f)
print("\nSaved model to:", model_path)

# ── Save train / test arrays ──────────────────────────────────────────────────
save_dir = os.path.expanduser("~/COALA/models/real")
os.makedirs(save_dir, exist_ok=True)

X_train_df = pd.DataFrame(X_train_arr, columns=best_features)
X_test_df  = pd.DataFrame(X_test_arr,  columns=best_features)

X_train_df.to_csv(os.path.join(save_dir, "X_train_xgb_fhs.csv"), index=False)
X_test_df.to_csv( os.path.join(save_dir, "X_test_xgb_fhs.csv"),  index=False)
