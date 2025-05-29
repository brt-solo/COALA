import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import pickle
import json
import os
import torch
from cf_search.map import mapcf_instance
import warnings
import torch.nn as nn
warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)

# ─────────────────────────────────────────────────────────────
# 1. Setup: file paths and metadata
# ─────────────────────────────────────────────────────────────

# Define SimpleMLP (must match training definition)
class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)

model_info = {
    "linear_model": f"~/MAP-CF/synthetic/linear_model.pkl",
    "mlp_model": f"~/MAP-CF/synthetic/mlp_model.pth",
    #"hgb_model": f"~/MAP-CF/synthetic/hgboost_model.pkl"
    
    
}

train_df_file = f"~/MAP-CF/synthetic/synthetic_train.csv"
reference_file = f"~/MAP-CF/synthetic/synthetic_test.csv"
feature_json_file = "~/MAP-CF/synthetic/synthetic_feature_categories.json"

# Expand paths
train_df_file = os.path.expanduser(train_df_file)
reference_file = os.path.expanduser(reference_file)
feature_json_file = os.path.expanduser(feature_json_file)

# ─────────────────────────────────────────────────────────────
# 2. Load training data, features, and reference data
# ─────────────────────────────────────────────────────────────

X_train_df = pd.read_csv(train_df_file)
if "Unnamed: 0" in X_train_df.columns:
    X_train_df = X_train_df.drop(columns="Unnamed: 0")

with open(feature_json_file, "r") as f:
    feature_categories = json.load(f)

ref_df = pd.read_csv(reference_file)
if "Unnamed: 0" in ref_df.columns:
    ref_df = ref_df.drop(columns="Unnamed: 0")

if not all(col in ref_df.columns for col in X_train_df.columns):
    missing = [col for col in X_train_df.columns if col not in ref_df.columns]
    raise ValueError(f"Missing columns in reference data: {missing}")

# ─────────────────────────────────────────────────────────────
# 3. Shared MAP-Elites configuration
# ─────────────────────────────────────────────────────────────

params = {
    "min": np.array(X_train_df.min()),
    "max": np.array(X_train_df.max()),
    "random_init_batch": 1000
}

print(params)
print("min:", np.array(ref_df.min()), "max:", np.array(ref_df.max()))

category_labels = list(feature_categories.keys())
num_categories = len(category_labels)

cell_feature_sets = {}
for i in range(num_categories):
    for j in range(num_categories):
        features = feature_categories[category_labels[i]].copy()
        if i != j:
            features += feature_categories[category_labels[j]]
        cell_feature_sets[(i, j)] = features

# ─────────────────────────────────────────────────────────────
# 4. Loop through each model
# ─────────────────────────────────────────────────────────────

for model_name, model_path in model_info.items():
    model_path = os.path.expanduser(model_path)
    print(f"\n=== Running MAP-Elites for: {model_name} ===")

    # Load model
    if model_path.endswith(".pth"):
        if model_name == "mlp_model":
            input_dim = X_train_df.shape[1]
            wrapper = SimpleMLP(input_dim)
            wrapper.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
            wrapper.eval()
    else:
        with open(model_path, "rb") as f:
            wrapper = pickle.load(f)  # sklearn model

    # Run MAP-Elites for each individual
    archive_dict = {}
    total = len(ref_df)

    for idx, row in ref_df[X_train_df.columns].iterrows():
        print(f"Processing individual {idx} for model '{model_name}'...")
        X_reference = row.to_numpy().astype(float)

        elite = mapcf_instance(
            dim_map=num_categories,
            dim_x=X_train_df.shape[1],
            max_evals=10000,
            params=params,
            cell_feature_sets=cell_feature_sets,
            X_train_df=X_train_df,
            feature_categories=feature_categories,
            X_reference=X_reference,
            wrapper=wrapper
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

    # Save archive
    archive_filename = f"output/{model_name}_counterfactuals.pkl"
    with open(archive_filename, "wb") as f:
        pickle.dump(archive_dict, f)
    print(f"Saved: {archive_filename}")
