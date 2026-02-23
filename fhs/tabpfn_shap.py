import os, pickle
from pathlib import Path

import shap
import numpy as np
if not hasattr(np, "int"):
    np.int = int  # for shap compatibility
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # important for sbatch/headless
import matplotlib.pyplot as plt

import torch
print("CUDA available:", torch.cuda.is_available())

# ---- paths ----
outdir = Path(os.path.expanduser("~/MAP-CF/fhs/figures"))
outdir.mkdir(parents=True, exist_ok=True)

model_path = os.path.expanduser("~/models/tabpfn_fhs.pkl")

# ---- load data ----
X_val = pd.read_csv(os.path.expanduser("~/MAP-CF/fhs/X_test_tabpfn_fhs.csv"))

# ---- load model ----
with open(model_path, "rb") as f:
    wrapper = pickle.load(f)

# ---- SHAP prediction wrapper ----
# For binary classification, SHAP should explain a scalar output.
# Use probability of class 1 (or class 0 if that’s your “good” class).
def model_predict(X_numpy: np.ndarray) -> np.ndarray:
    # TabPFN expects numpy; returns (n, 2) for binary
    proba = wrapper.predict_proba(X_numpy)
    return proba[:, 1]   # <-- change to [:, 0] if you want class-0 probability

# ---- background data (IMPORTANT) ----
# KernelExplainer is very slow if you pass the whole dataset.
# Use a small background sample.
background = shap.sample(X_val.values, 100, random_state=0)

explainer = shap.KernelExplainer(model_predict, background)

# Explain a subset (also important for speed)
X_explain = X_val.iloc[:200, :].values

shap_values = explainer.shap_values(X_explain, nsamples=200)

# Plot needs the matching DataFrame rows
X_explain_df = X_val.iloc[:200].copy()

shap.summary_plot(shap_values, X_explain_df, plot_type="dot", show=False)
plt.tight_layout()
plt.savefig(outdir / "shap_summary_dot_unscaled.png", dpi=300, bbox_inches="tight")
plt.close()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# --- Assume you have your shap_matrix (n_samples, n_features) ---
shap_matrix = shap_values
# Step 1: Standardize SHAP values
scaler = StandardScaler()
shap_matrix_scaled = shap_matrix #scaler.fit_transform(shap_matrix)

# Step 2: Compute pairwise distances
pairwise_distances = pdist(shap_matrix_scaled, metric='euclidean')
distance_matrix = squareform(pairwise_distances)

def optimal_hclust_k(X, max_k=3, method='ward', metric='euclidean'):
    """
    Identify the optimal number of clusters for hierarchical clustering
    using silhouette score.

    Parameters:
    - X: array-like, shape (n_samples, n_features)
    - max_k: maximum number of clusters to consider
    - method: linkage method (default: 'ward')
    - metric: distance metric (default: 'euclidean')

    Returns:
    - best_k: int, number of clusters with highest silhouette score
    - best_labels: array of cluster labels for best_k
    - Z: linkage matrix (can be reused)
    """
    Z = linkage(X, method=method, metric=metric)
    best_score = -1
    best_k = None
    best_labels = None

    for k in range(2, max_k + 1):
        labels = fcluster(Z, t=k, criterion='maxclust')
        if len(set(labels)) <= 1:
            continue
        try:
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
        except Exception:
            continue

    return best_k, best_labels, Z
'''
# Step 3: Hierarchical clustering
Z = linkage(pairwise_distances, method='ward')

# Step 4: Cut tree into clusters
n_clusters = 8
height = 12
cluster_labels = fcluster(Z, n_clusters, criterion='maxclust')
#cluster_labels = fcluster(Z, t=height, criterion='distance')
n_clusters = len(cluster_labels)
# Step 5: Assign colors to cluster labels
palette = sns.color_palette("tab10", n_clusters)
cluster_color_mapping = dict(zip(np.unique(cluster_labels), palette))
row_colors = pd.Series(cluster_labels).map(cluster_color_mapping).to_numpy()

# Step 6: Create clustermap
sns.clustermap(distance_matrix,
                   row_linkage=Z,
                   col_linkage=Z,
                   row_colors=row_colors,
                   col_colors=row_colors,
                   cmap='coolwarm',  # <- makes sure the HEATMAP (distance matrix) is coolwarm!
                   center=np.median(distance_matrix),  # Centering at median distance
                   figsize=(12, 12),
                   cbar_kws={"label": "Euclidean Distance"})

plt.suptitle('Sample-Sample Similarity Heatmap (Colored by Clusters)', y=1.02)
plt.show()
'''
# Step 3: Hierarchical clustering using optimal number of clusters
max_k = 5  # Or any reasonable limit based on your data
best_k, cluster_labels, Z = optimal_hclust_k(shap_matrix_scaled, max_k=max_k, method='ward', metric='euclidean')

# Step 4: Assign colors to cluster labels
n_clusters = len(np.unique(cluster_labels))
palette = sns.color_palette("tab10", n_clusters)
cluster_color_mapping = dict(zip(np.unique(cluster_labels), palette))
row_colors = pd.Series(cluster_labels).map(cluster_color_mapping).to_numpy()

# Step 5: Compute pairwise distance again for the clustermap
pairwise_distances = pdist(shap_matrix_scaled, metric='euclidean')
distance_matrix = squareform(pairwise_distances)
Z = linkage(pairwise_distances, method='ward')  # Reuse same linkage as optimal_hclust_k




# Step 6: Create clustermap

plt.suptitle('Sample-Sample Similarity Heatmap (Colored by Clusters)', y=1.02)
plt.show()
g = sns.clustermap(
    distance_matrix,
    row_linkage=Z,
    col_linkage=Z,
    row_colors=row_colors,
    col_colors=row_colors,
    cmap="coolwarm",
    center=np.median(distance_matrix),
    figsize=(12, 12),
    cbar_kws={"label": "Euclidean Distance"},
)

g.fig.suptitle("Sample-Sample Similarity Heatmap (Colored by Clusters)", y=1.02)
g.fig.savefig(outdir / "shap_distance_clustermap.png", dpi=300, bbox_inches="tight")
plt.close(g.fig)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from scipy.stats import f


# --- Assume these are already defined ---
# - shap_values: SHAP matrix (n_samples, n_features)
# - cluster_labels: cluster assignments (n_samples,)
# - X_val: original unscaled DataFrame (n_samples, n_features)

shap_matrix = shap_values
feature_names = X_val.columns.tolist()
def eta2_threshold_from_alpha(alpha, n_samples, n_clusters):
    df_between = n_clusters - 1
    df_within = n_samples - n_clusters
    if df_between <= 0 or df_within <= 0:
        return 1.0  # fallback
    f_crit = f.ppf(1 - alpha, df_between, df_within)
    eta2_thresh = (f_crit * df_between) / (f_crit * df_between + df_within)
    return eta2_thresh

# Calculate η² threshold at p < 0.05
eta2_thresh = eta2_threshold_from_alpha(0.05, n_samples=shap_matrix.shape[0], n_clusters=len(np.unique(cluster_labels)))

# --- Optional: PCA for 2D projection ---
scaler = StandardScaler()
shap_matrix_scaled = scaler.fit_transform(shap_matrix)
pca = PCA(n_components=2)
shap_umap = pca.fit_transform(shap_matrix_scaled)

# --- Step 1: η² from SHAP values ---
F_shap, _ = f_classif(shap_matrix, cluster_labels)
eta2_shap = F_shap / (F_shap + shap_matrix.shape[0] - shap_matrix.shape[1] - 1)

# --- Step 2: η² from raw values ---
F_raw, _ = f_classif(X_explain_df, cluster_labels)
eta2_raw = F_raw / (F_raw + X_explain_df.shape[0] - X_explain_df.shape[1] - 1)

# --- Sort by SHAP η² ---
sorted_idx = np.argsort(-eta2_shap)
top_features = np.array(feature_names)[sorted_idx][:10]
top_eta2_shap = eta2_shap[sorted_idx][:10]
top_eta2_raw = eta2_raw[sorted_idx][:10]  # same feature order

# --- Step 3: Mean SHAP & Raw values per cluster ---
cluster_df = pd.DataFrame(shap_matrix, columns=feature_names)
cluster_df['cluster'] = cluster_labels
mean_shap_by_cluster = cluster_df.groupby('cluster').mean()

X_with_cluster = X_explain_df.copy()
X_with_cluster['cluster'] = cluster_labels
mean_raw_by_cluster = X_with_cluster.groupby('cluster').mean()



fig = plt.figure(figsize=(35, 12), dpi=500)  # slightly larger overall figure
gs = fig.add_gridspec(2, 3, width_ratios=[1.6, 2.2, 2.4])  # make Panel A thinner

# Panel A: PCA scatter
ax0 = fig.add_subplot(gs[:, 0])
scatter = ax0.scatter(shap_umap[:, 0], shap_umap[:, 1], c=cluster_labels, cmap='tab10', s=60)
ax0.set_xlabel('PC1', fontsize=20)
ax0.set_ylabel('PC2', fontsize=20)
ax0.tick_params(labelsize=16)
unique_clusters = np.unique(cluster_labels)
handles = [
    plt.Line2D([], [], marker='o', linestyle='', color=scatter.cmap(scatter.norm(i)), label=f'Cluster {i}')
    for i in unique_clusters
]
ax0.legend(handles=handles, title='Cluster', bbox_to_anchor=(0.7, 1), loc='upper left', fontsize=12, title_fontsize=13)
ax0.text(-0.1, 1.05, "A", transform=ax0.transAxes, fontsize=18, fontweight='bold', va='top')

# Panel B: SHAP η² barplot
ax1 = fig.add_subplot(gs[0, 1])
ax1.barh(top_features, top_eta2_shap, color='steelblue')
ax1.axvline(eta2_thresh, linestyle='dotted', color='red', linewidth=2)
ax1.set_xlabel('SHAP value effect size (η²)', fontsize=20)
ax1.tick_params(labelsize=16)
ax1.invert_yaxis()
ax1.text(-0.1, 1.05, "B", transform=ax1.transAxes, fontsize=18, fontweight='bold', va='top')

# Panel C: Raw η² barplot
ax2 = fig.add_subplot(gs[0, 2])
ax2.barh(top_features, top_eta2_raw, color='steelblue')
ax2.axvline(eta2_thresh, linestyle='dotted', color='red', linewidth=2)
ax2.set_xlabel('Original value effect size (η²)', fontsize=20)
ax2.tick_params(labelsize=16)
ax2.invert_yaxis()
ax2.text(-0.1, 1.05, "C", transform=ax2.transAxes, fontsize=18, fontweight='bold', va='top')

# Panel D: SHAP heatmap
ax3 = fig.add_subplot(gs[1, 1])
sns.heatmap(mean_shap_by_cluster[top_features], cmap='coolwarm', center=0,
            annot=True, fmt=".2f", ax=ax3, vmax=0.3, vmin=-0.1, cbar=True, annot_kws={"size": 12})
ax3.set_xlabel('Feature', fontsize=20)
ax3.set_ylabel('Cluster', fontsize=20)
ax3.tick_params(labelsize=16)
ax3.text(-0.1, 1.05, "D", transform=ax3.transAxes, fontsize=18, fontweight='bold', va='top')

# Panel E: Raw heatmap
ax4 = fig.add_subplot(gs[1, 2])
sns.heatmap(mean_raw_by_cluster[top_features], cmap='coolwarm',
            annot=True, fmt=".2f", ax=ax4, vmax=250, vmin=0, cbar=True, annot_kws={"size": 12})
ax4.set_xlabel('Feature', fontsize=20)
ax4.set_ylabel('Cluster', fontsize=20)
ax4.tick_params(labelsize=16)
ax4.text(-0.1, 1.05, "E", transform=ax4.transAxes, fontsize=18, fontweight='bold', va='top')

fig.subplots_adjust(wspace=0.6)
plt.show()
fig.savefig(outdir / "shap_cluster_panels.png", dpi=600, bbox_inches="tight")
plt.close(fig)


# --- CREATE SIMILARITY MATRIX/NETWORK ---
import snf
from scipy.spatial.distance import pdist, squareform
import networkx as nx

def get_shap_similarity(X, test_data, color_feature,
                   K=10, mu=0.5):
    data_views = []
    data_views.append(X)

    # --- similarity fusion ---
    fused_affinity = snf.make_affinity(data_views[0], metric='euclidean', K=K, mu=mu)

    n = fused_affinity.shape[0]

    # --- symmetric kNN mask ---
    K_eff = max(1, min(K, n - 1))
    top_idx = np.argsort(fused_affinity, axis=1)[:, ::-1]
    mask = np.zeros_like(fused_affinity, dtype=bool)
    for i in range(n):
        row = top_idx[i]
        row = row[row != i][:K_eff]
        mask[i, row] = True
    mask = np.logical_or(mask, mask.T)

    # --- build graph ---
    G = nx.Graph()
    vals = test_data[color_feature].to_numpy()
    for i in range(n):
        G.add_node(i, value=vals[i])

    ii, jj = np.where(mask)
    for i, j in zip(ii, jj):
        if i < j:
            w = fused_affinity[i, j]
            if w > 0:
                G.add_edge(i, j, weight=w)

    if G.number_of_edges() == 0:
        raise ValueError("Fused similarity graph has no edges. Try larger K or mu.")

    # --- PyTorch Geometric exports ---
    edges = list(G.edges())
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weights = np.array([G[u][v]['weight'] for u, v in edges])
    edge_attr = torch.tensor(edge_weights, dtype=torch.float32).view(-1, 1)

    return G, edge_index, edge_weights, edge_attr, fused_affinity


shap_matrix = shap_values


# Get outputs
G, edge_index, edge_weights, edge_attr, fused_affinity = get_shap_similarity(
    shap_matrix,
    X_explain_df,
    color_feature="Age (years)",
    K=15, mu=0.5
)

# Layout
pos = nx.spring_layout(G, seed=42, weight='weight', iterations=300)

# Node attributes
attrs_all = X_explain_df.to_dict(orient="index")  # {i: {col: val, ...}, ...}
nx.set_node_attributes(G, attrs_all)

# Edge styling
ew_norm = (edge_weights - edge_weights.min()) / (edge_weights.ptp() + 1e-12)
edge_colors = [plt.cm.inferno(x) for x in ew_norm]
edge_widths = 0.5 + 1.5 * ew_norm

# Node colors from attribute
vals = np.array([d["value"] for _, d in G.nodes(data=True)])
vmin, vmax = np.nanmin(vals), np.nanmax(vals)

# Plot
fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
ax.set_axis_off()
nx.draw(
    G, pos, ax=ax, with_labels=False,
    node_color=vals, cmap="viridis", vmin=vmin, vmax=vmax,
    node_size=50,
    edge_color=edge_colors,
    width=edge_widths
)

for u, v, d in G.edges(data=True):
    d["weight"] = float(d["weight"])     

out_path = Path("~/MAP-CF/fhs/shap_similarity_network.graphml").expanduser()
out_path.parent.mkdir(parents=True, exist_ok=True)

nx.write_graphml(G, out_path)
print("Saved to:", out_path)