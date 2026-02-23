import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.impute import KNNImputer

from tabpfn import TabPFNClassifier

'''
# -----------------------------
# Load NHANES tables
# -----------------------------
demo = pd.read_sas("DEMO_L.xpt")
bmx  = pd.read_sas("BMX_L.xpt")
lab  = pd.read_sas("GHB_L.xpt")
paq  = pd.read_sas("PAQ_L.xpt")
smq  = pd.read_sas("SMQ_L.xpt")
glu = pd.read_sas("GLU_L.xpt")
tc  = pd.read_sas("TCHOL_L.xpt")
hdl = pd.read_sas("HDL_L.xpt")

df = (
    demo.merge(bmx, on="SEQN", how="left")
        .merge(lab, on="SEQN", how="left")
        .merge(paq, on="SEQN", how="left")
        .merge(smq, on="SEQN", how="left")
        .merge(glu, on="SEQN", how="left")
        .merge(tc,  on="SEQN", how="left")
        .merge(hdl, on="SEQN", how="left")
)

# -----------------------------
# Target
# -----------------------------
df = df[df["RIDAGEYR"] >= 20].copy()
df["y"] = (df["LBXGH"] >= 6.5).astype(int)
final_features = [
    "RIDAGEYR",
    "RIAGENDR",
    "INDFMPIR",
    "DMDHHSIZ",
    "BMXBMI",
    "BMXWAIST",
    "PAD790Q",
    "SMQ020",
    "LBXTC",
    "LBDHDD",
    "y"
]

df = df[final_features].dropna().copy()
df = df[df['SMQ020'] != 7]  # remove "Refused.Don't know" in smoking
for col in final_features:
    print(col, np.unique(df[col], return_counts=True))

# -----------------------------
# Feature selection (your defensible set)
# -----------------------------
X = df.drop(columns=["y"])
#X = pd.get_dummies(X, columns=["RIAGENDR"], drop_first=True)  # yields RIAGENDR_2.0
y = df["y"]
'''

'''
import pandas as pd
from sklearn.datasets import fetch_openml

# Fetch dataset
data = fetch_openml(name="diabetes", version=1, as_frame=True)

df = data.frame

print(df.head())
print(df.columns)

df["y"] = (df["class"] == "tested_positive").astype(int)
X = df.drop(columns=["class", "y"])
y = df["y"]
'''

''''''
import pandas as pd

# MIT OCW copy of framingham.csv (has TenYearCHD as the label)
url = "https://ocw.mit.edu/courses/15-071-the-analytics-edge-spring-2017/5d689a024551e672313f7fd7eb1bee8d_framingham.csv"
df = pd.read_csv(url)

print(df.head())
print(df.columns)

# label
df["y"] = df["TenYearCHD"].astype(int)
# Drop columns with >20% missingness
keep_cols = df.columns[df.isna().mean() <= 0.20]
df = df[keep_cols].copy()
df = df.dropna()

# features (drop ID/label)
X = df.drop(columns=["TenYearCHD", "y", "education"])  # drop education for now due to multiple levels
y = df["y"]

# OPTIONAL: keep only numeric/ordinal/binary columns (Framingham is mostly numeric already)
X = X.select_dtypes(include="number")


new_col_names = ['Sex (male)', 'Age (years)', 'Current smoker', 'Cigarettes per day', 'BP Medication',
       'Stroke history', 'Hypertension', 'Diabetes', 'Total cholesterol', 'Systolic BP',
       'Diastolic BP', 'BMI', 'Resting HR', 'Blood glucose']
X.columns = new_col_names
print("Columns after missingness filter:", X.columns.tolist())
print("X shape:", X.shape, "y mean:", y.mean())


# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=8, stratify=y
)


''''''


#new_col_names = X_train.columns
# Keep only numeric (TabPFN expects numeric arrays)
X_train = X_train.select_dtypes(include="number")
X_test  = X_test.select_dtypes(include="number")

# Align columns (safety)
X_test = X_test[X_train.columns]

# -----------------------------
# Impute missing values
# -----------------------------
'''
imputer = KNNImputer(n_neighbors=10, weights="distance")
X_train_np = imputer.fit_transform(X_train)
X_test_np  = imputer.transform(X_test)
'''
X_train_np, X_test_np = X_train.to_numpy(), X_test.to_numpy()
# -----------------------------
# TabPFN model (pretrained)
# -----------------------------
# device="cuda" if you have GPU; otherwise "cpu"
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
model = TabPFNClassifier(device="cuda")

# Optional: quick CV on training set
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(model, X_train_np, y_train.to_numpy(), scoring="roc_auc", cv=cv, n_jobs=1)
print("TabPFN CV AUC mean:", cv_auc.mean(), "std:", cv_auc.std())

# Fit on full train
model.fit(X_train_np, y_train.to_numpy())

# Evaluate on holdout
train_probs = model.predict_proba(X_train_np)[:, 1]
print("Train AUC:", roc_auc_score(y_train, train_probs))
probs = model.predict_proba(X_test_np)[:, 1]
print("Holdout AUC:", roc_auc_score(y_test, probs))
print("Accuracy:", (model.predict(X_test_np) == y_test.to_numpy()).mean())
print("F1 Score:", 2 * (( (model.predict(X_test_np) == 1) & (y_test.to_numpy() == 1) ).sum() / ( (model.predict(X_test_np) == 1).sum() + (y_test.to_numpy() == 1).sum() )))
print("Precision:", ( ( (model.predict(X_test_np) == 1) & (y_test.to_numpy() == 1) ).sum() ) / ( (model.predict(X_test_np) == 1).sum() ) )
print("Recall:", ( ( (model.predict(X_test_np) == 1) & (y_test.to_numpy() == 1) ).sum() ) / ( (y_test.to_numpy() == 1).sum() ) )
print("Confusion Matrix:\n", pd.crosstab(y_test, model.predict(X_test_np), rownames=['Actual'], colnames=['Predicted'], margins=True))


import pickle, os

model_dir = os.path.expanduser("~/models")
os.makedirs(model_dir, exist_ok=True)


'''
with open(os.path.join(model_dir, "tabpfn_nhane.pkl"), "wb") as f:
    pickle.dump(model, f)

save_dir = os.path.expanduser("~/MAP-CF/nhane")  # or wherever you like
os.makedirs(save_dir, exist_ok=True)

print("Saved to:", os.path.join(save_dir, "tabpfn_nhane.pkl"))

X_train_df = pd.DataFrame(X_train_np, columns=X.columns)
X_test_df  = pd.DataFrame(X_test_np,  columns=X.columns)

X_train_df.to_csv(os.path.join(save_dir, "X_train_tabpfn_nhane.csv"), index=False)
X_test_df.to_csv(os.path.join(save_dir, "X_test_tabpfn_nhane.csv"),  index=False)
'''






# --- LOGISTIC REGRESSION BASELINE ---
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# 1) Create scaled versions (fit scaler on train only)
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_np)
X_test_scaled  = scaler.transform(X_test_np)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def eval_classifier(name, model, Xtr, Xte):
    cv_auc = cross_val_score(model, Xtr, y_train.to_numpy(), scoring="roc_auc", cv=cv, n_jobs=-1)
    model.fit(Xtr, y_train.to_numpy())
    probs = model.predict_proba(Xte)[:, 1]
    holdout_auc = roc_auc_score(y_test, probs)
    print(f"{name}: CV AUC mean={cv_auc.mean():.4f} std={cv_auc.std():.4f} | Holdout AUC={holdout_auc:.4f}")

# -----------------------------
# 2) Baseline models
# -----------------------------

# Logistic Regression (scaled)
logreg = LogisticRegression(max_iter=5000, solver="lbfgs")
eval_classifier("LogReg (scaled)", logreg, X_train_scaled, X_test_scaled)

# Linear SVM (needs calibration to get probabilities)
svm = LinearSVC()
svm_cal = CalibratedClassifierCV(svm, method="sigmoid", cv=3)
eval_classifier("Linear SVM (scaled + calibrated)", svm_cal, X_train_scaled, X_test_scaled)

# Random Forest (does NOT need scaling, but we can still use unscaled)
rf = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    class_weight="balanced_subsample"
)
eval_classifier("RandomForest (unscaled)", rf, X_train_np, X_test_np)

# -----------------------------
# 3) Optional: Linear Regression as a "linear probability model"
# (not recommended as a classifier, but included if you want)
# -----------------------------
linreg = LinearRegression()
cv_auc = cross_val_score(linreg, X_train_scaled, y_train.to_numpy(), scoring="roc_auc", cv=cv, n_jobs=-1)
linreg.fit(X_train_scaled, y_train.to_numpy())
pred = linreg.predict(X_test_scaled)  # not bounded to [0,1]
holdout_auc = roc_auc_score(y_test, pred)
print(f"LinearRegression (scaled, LPM): CV AUC mean={cv_auc.mean():.4f} std={cv_auc.std():.4f} | Holdout AUC={holdout_auc:.4f}")
