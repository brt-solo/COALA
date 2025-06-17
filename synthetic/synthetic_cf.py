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
import argparse
warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)

#parameter arguments


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Path to pth or pkl file")
parser.add_argument("--reference", type=str, required=True, help="Path to input features for samples of interest (csv file)")
parser.add_argument("--train", type=str, required=True, help="Path to input features of training set, used for setting min and max of counterfactuals (csv file)")
parser.add_argument("--feature_cat", type=str, required=True, help="Path to .json file describing which features are in which category")
parser.add_argument("--model_name", type=str, default="model")
parser.add_argument("--init_pop", type=int, default=1000, help="Initial random population (default=1000)")
parser.add_argument("--iter", type=int, default=5000, help="Total number of iterations (default=5000)")
parser.add_argument("--method", type=str, help="Crossover method, can choose from sbx, single_point, uniform")
parser.add_argument("--mutation_rate", type=float, default=None, help="mutation rate, default is None")
parser.add_argument("--output", type=str, default="model_cf_out", help="directory for storing outputs")


args = parser.parse_args()
#print(f"model: {args.model}")
#print(f"reference: {args.reference}")


# ─────────────────────────────────────────────────────────────
# 1. Setup: file paths and metadata
# ─────────────────────────────────────────────────────────────

# Define SimpleMLP (must match training definition)
class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            #nn.Dropout(0.1),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.model(x)

class PerfectWrapper:
    def __init__(self):
        # Define feature ordering if needed
        self.feature_names = ['G1', 'G2', 'G3', 'E1', 'E2', 'N1', 'N2', 'M1', 'M2']

    def predict(self, X):
        # If input is a DataFrame, convert to NumPy in correct order
        if hasattr(X, 'loc') or hasattr(X, 'iloc'):
            X = X[self.feature_names].values

        G1 = X[:, 0]
        G2 = X[:, 1]
        G3 = X[:, 2]
        E1 = X[:, 3]
        E2 = X[:, 4]
        N1 = X[:, 5]
        N2 = X[:, 6]
        M1 = X[:, 7]
        M2 = X[:, 8]

        y = (
            0.5 * G3 +
            0.7 * E2 +
            0.2 * N1 +
            0.3 * M1 +
            + 1.8 * G1 * E1 #interaction between 2 features
            + 1.8 * G2 * N2 * M2 # higher order interaction
        )
        return y

'''
model_info = {
    "perfect_model": f"~/MAP-CF/synthetic/perfect_model.pkl",
    "mlp_model": f"~/MAP-CF/synthetic/mlp_model.pth",
    "linear_model": f"~/MAP-CF/synthetic/linear_model.pkl",

    
    #"hgb_model": f"~/MAP-CF/synthetic/hgboost_model.pkl"
    
    
}
'''

model_path = args.model
model_name = args.model_name

#train_df_file = f"~/MAP-CF/synthetic/synthetic_train.csv"
#reference_file = f"~/MAP-CF/synthetic/synthetic_test.csv"
#feature_json_file = "~/MAP-CF/synthetic/synthetic_feature_categories_action.json"

train_df_file = args.train
reference_file = args.reference
feature_json_file = args.feature_cat

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
'''
params = {
    "min": np.array(X_train_df.min()),
    "max": np.array(X_train_df.max()),
    "random_init_batch": args.init_pop
}
'''
params = {
    "min": np.full(X_train_df.shape[1], -3.0),
    "max": np.full(X_train_df.shape[1], 3.0),
    "random_init_batch": args.init_pop
}


#print(params)
#print("min:", np.array(ref_df.min()), "max:", np.array(ref_df.max()))

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
    print(f"Processing individual {idx} for '{model_name}'...")
    X_reference = row.to_numpy().astype(float)

    elite = mapcf_instance(
        dim_map=num_categories,
        dim_x=X_train_df.shape[1],
        max_evals=args.iter,
        params=params,
        cell_feature_sets=cell_feature_sets,
        X_train_df=X_train_df,
        feature_categories=feature_categories,
        X_reference=X_reference,
        wrapper=wrapper,
        method=args.method,
        mutation_rate=args.mutation_rate
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
archive_filename = f"{args.output}/counterfactuals.pkl"
os.makedirs(os.path.dirname(archive_filename), exist_ok=True)
with open(archive_filename, "wb") as f:
    pickle.dump(archive_dict, f)
print(f"Saved: {archive_filename}")
