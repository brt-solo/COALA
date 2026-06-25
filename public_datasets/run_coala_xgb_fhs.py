import sys
import os
# Force stdout/stderr to flush immediately — needed for SLURM output capture
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

print("Starting run_coala_xgb_fhs.py", flush=True)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(REPO_ROOT)

import pickle
import json
import warnings
import pandas as pd
import numpy as np
print("Core imports done", flush=True)
from cf_search.map import mapcf_instance_classification
print("mapcf imported", flush=True)

warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH      = os.path.join(REPO_ROOT, "models", "xgb_fhs.pkl")
TRAIN_PATH      = os.path.join(REPO_ROOT, "models", "real", "X_train_xgb_fhs.csv")
REFERENCE_PATH  = os.path.join(REPO_ROOT, "models", "real", "X_test_xgb_fhs.csv")
FEATURE_JSON    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feature_categories_fhs.json")
OUTPUT_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xgb_fhs_output")
INIT_POP        = 1000
MAX_ITER        = 10000
METHOD          = "uniform"
MUTATION_RATE   = None
MAX_INDIVIDUALS = 500

# ── Load data ─────────────────────────────────────────────────────────────────
X_train_df = pd.read_csv(TRAIN_PATH)
if "Unnamed: 0" in X_train_df.columns:
    X_train_df = X_train_df.drop(columns="Unnamed: 0")

with open(FEATURE_JSON, "r") as f:
    feature_categories = json.load(f)

ref_df = pd.read_csv(REFERENCE_PATH)
if "Unnamed: 0" in ref_df.columns:
    ref_df = ref_df.drop(columns="Unnamed: 0")
ref_df = ref_df.iloc[:MAX_INDIVIDUALS, :]

if not all(col in ref_df.columns for col in X_train_df.columns):
    missing = [col for col in X_train_df.columns if col not in ref_df.columns]
    raise ValueError(f"Missing columns in reference data: {missing}")

# ── MAP-Elites config ─────────────────────────────────────────────────────────
params = {"random_init_batch": INIT_POP}

category_labels = list(feature_categories.keys())
num_categories = len(category_labels)

cell_feature_sets = {}
for i in range(num_categories):
    for j in range(num_categories):
        features = feature_categories[category_labels[i]].copy()
        if i != j:
            features += feature_categories[category_labels[j]]
        cell_feature_sets[(i, j)] = features

# ── Load model ────────────────────────────────────────────────────────────────
with open(MODEL_PATH, "rb") as f:
    wrapper = pickle.load(f)
print(f"Loaded model: {MODEL_PATH}")

# ── Run MAP-Elites ────────────────────────────────────────────────────────────
archive_dict = {}

for idx, row in ref_df[X_train_df.columns].iterrows():
    print(f"Processing individual {idx} / {len(ref_df)}...", flush=True)
    X_reference = row.to_numpy().astype(float)

    elite = mapcf_instance_classification(
        dim_map=num_categories,
        dim_x=X_train_df.shape[1],
        max_evals=MAX_ITER,
        params=params,
        cell_feature_sets=cell_feature_sets,
        X_train_df=X_train_df,
        feature_categories=feature_categories,
        X_reference=X_reference,
        wrapper=wrapper,
        method=METHOD,
        mutation_rate=MUTATION_RATE,
    )
    archive = elite.run()

    data_records = []
    for cell_index, data in archive.items():
        if data["best"]:
            best_vector, best_fitness = data["best"]
            record = dict(zip(X_train_df.columns, best_vector))
            record["fitness"] = best_fitness
            record["cell"] = str(cell_index)
            data_records.append(record)

    archive_dict[idx] = pd.DataFrame(data_records)

# ── Save archive ──────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, "counterfactuals.pkl")
with open(out_path, "wb") as f:
    pickle.dump(archive_dict, f)
print(f"Saved: {out_path}")
