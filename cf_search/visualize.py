from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import chi2_contingency
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
from umap import UMAP

# Optional: Wes Anderson palette fallback
try:
    from wesanderson import wes_palette
    use_wes_palette = True
except ImportError:
    use_wes_palette = False

'''

def plot_cellwise_grid(archive_dict, cell_feature_sets, feature_categories, plot_fn, plot_type_name="Grid", figsize_per_cell=(4, 4), **kwargs):
    categories = list(feature_categories.keys())
    n = len(categories)
    fig, axes = plt.subplots(n, n, figsize=(figsize_per_cell[0]*n, figsize_per_cell[1]*n), dpi=300)
    axes = np.atleast_2d(axes)

    for i in range(n):
        for j in range(n):
            ax = axes[n - 1 - j, i]
            cell_key = (i, j)
            if cell_key not in cell_feature_sets:
                ax.axis("off")
                continue

            df = extract_cell_data(archive_dict, cell_key)
            if df.empty:
                ax.axis("off")
                continue

            mutable_features = cell_feature_sets[cell_key]
            plot_fn(ax, df, cell_key, mutable_features=mutable_features, **kwargs)

    for i in range(n):
        # Stack the new title above the existing title (if any)
        current_title = axes[0, i].get_title()
        new_title = f"{categories[i]}\n{current_title}" if current_title else categories[i]
        axes[0, i].set_title(new_title, fontsize=10, pad=10)

        axes[i, 0].set_ylabel(categories[n - 1 - i], fontsize=12, rotation=0, ha='right', va='center')
    

    fig.subplots_adjust(hspace=0.2, wspace=0.6)
    fig.suptitle(plot_type_name, fontsize=16)
    return fig
'''

def plot_cellwise_grid(archive_dict, cell_feature_sets, feature_categories, plot_fn, plot_type_name="Grid", figsize_per_cell=(4, 4), colorbar_label="Predicted outcome", **kwargs):
    categories = list(feature_categories.keys())
    n = len(categories)
    fig, axes = plt.subplots(n, n, figsize=(figsize_per_cell[0]*n, figsize_per_cell[1]*n), dpi=300)
    axes = np.atleast_2d(axes)

    mappables = []

    for i in range(n):
        for j in range(n):
            ax = axes[n - 1 - j, i]
            cell_key = (i, j)
            if cell_key not in cell_feature_sets:
                ax.axis("off")
                continue

            df = extract_cell_data(archive_dict, cell_key)
            if df.empty:
                ax.axis("off")
                continue

            mutable_features = cell_feature_sets[cell_key]
            mappable = plot_fn(ax, df, cell_key, mutable_features=mutable_features, **kwargs)
            if mappable:
                mappables.append(mappable)

    # Set column labels (top row)
    for i in range(n):
        ax_top = axes[0, i]
        ax_top.set_title(categories[i], fontsize=12, pad=10)

    # Add separate row labels using fig.text instead of modifying subplot y-axis labels
    for j in range(n):
        fig.text(
            0.07,  # x-position (near left)
            0.8*(j) / n+0.2,  # y-position (center of the subplot row)
            categories[n - 1 - j],
            ha='right',
            va='center',
            fontsize=12,
            rotation=0,
            transform=fig.transFigure
        )

    if mappables:
        cbar = fig.colorbar(mappables[0], ax=axes, orientation='horizontal', fraction=0.02, pad=0.01)
        cbar.set_label(colorbar_label, fontsize=12)

    return fig


################ NETWORK ANALYSIS ###############################
def correlation_mutable_vs_constraint_cell(ax, df, cell_key, mutable_features, vmin=-1, vmax=1, cmap="viridis"):
    """
    Plot Pearson correlation heatmap between mutable and constraint features for a single cell.

    - Rows = constraint features
    - Columns = mutable features
    """
    if df.empty or len(mutable_features) < 2:
        ax.axis("off")
        return

    try:
        # Identify constraint features
        constraint_features = [
            col for col in df.columns
            if col not in mutable_features and col not in ["cell", "individual_id", "fitness"]
        ]
        if len(constraint_features) < 1:
            ax.axis("off")
            return

        # Subset and drop rows with any missing values
        df_sub = df[mutable_features + constraint_features].dropna()
        if df_sub.shape[0] < 3:
            ax.axis("off")
            return

        # Compute full correlation and subset to [constraint_rows, mutable_cols]
        corr = df_sub.corr(method="pearson")
        corr_matrix = corr.loc[constraint_features, mutable_features]

        sns.heatmap(
            corr_matrix,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            xticklabels=True,
            yticklabels=True
        )

        ax.set_title(f"{cell_key}", fontsize=6)
        ax.tick_params(labelsize=5)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    except Exception as e:
        print(f"[ERROR] Cell {cell_key} → {e}")
        ax.axis("off")

def correlation_heatmap_cell(ax, df, cell_key, mutable_features, vmin=-1, vmax=1, cmap="viridis"):
    """
    Plot a Pearson correlation heatmap for mutable features in a cell.
    Shows feature names on axes and inverts diagonal.
    """
    if df.empty or len(mutable_features) < 2:
        ax.axis("off")
        return

    try:
        df_mut = df[mutable_features].copy().dropna(axis=0)
        if df_mut.shape[0] < 3:
            ax.axis("off")
            return

        # Compute and flip correlation matrix
        corr_matrix = df_mut.corr(method="pearson")
        corr_matrix = corr_matrix.iloc[::-1, :]  # invert rows to flip diagonal

        sns.heatmap(
            corr_matrix,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            xticklabels=True,
            yticklabels=True,
            square=True
        )

        ax.set_title(f"{cell_key}", fontsize=6)

        # Improve label visibility
        ax.tick_params(labelsize=5)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    except Exception as e:
        print(f"[ERROR] Cell {cell_key} → {e}")
        ax.axis("off")


############# PCA and K MEAN CLUSTERING PIPELINE FUNCTIONS #########################
def plot_single_pca(ax, df, cell_key, mutable_features, color_feature, n_components=2, global_vmin=0, global_vmax=1):
    if len(mutable_features) < 2 or df.shape[0] < 2:
        ax.axis("off")
        return

    try:
        X = df[mutable_features].values
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)

        pca_df = pd.DataFrame(X_pca, columns=[f"PC{k+1}" for k in range(n_components)])
        pca_df[color_feature] = df[color_feature].values

        scatter = ax.scatter(
            pca_df["PC1"], pca_df["PC2"],
            c=pca_df[color_feature],
            cmap="viridis",
            vmin=global_vmin,
            vmax=global_vmax,
            alpha=0.7
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        #ax.set_xlim(-5, 5)
        #ax.set_ylim(-5, 5)

        # Ensure ticks and labels are shown
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        return scatter
    except Exception:
        ax.axis("off")


def eta2_bar_cell_pca_kmeans(ax, df, cell_key, mutable_features, top_n=5, max_clusters=5):
    if df.empty or len(mutable_features) < 2:
        ax.axis("off")
        return

    X = df[mutable_features].values
    if X.shape[0] < 3:
        ax.axis("off")
        return

    try:
        # PCA + KMeans
        pca = PCA(n_components=min(2, X.shape[1]))
        X_pca = pca.fit_transform(X)

        from cf_search.visualize import optimal_clusters
        k, _ = optimal_clusters(X_pca, max_clusters=max_clusters)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_pca)

        # Eta squared (η²)
        F_vals, _ = f_classif(X, labels)
        df_within = X.shape[0] - X.shape[1] - 1
        if df_within <= 0:
            ax.axis("off")
            return

        eta_sq = F_vals / (F_vals + df_within)
        valid_mask = np.isfinite(eta_sq)
        if not np.any(valid_mask):
            ax.axis("off")
            return

        eta_sq = eta_sq[valid_mask]
        filtered_feats = np.array(mutable_features)[valid_mask]

        sorted_idx = np.argsort(-eta_sq)
        top_feats = filtered_feats[sorted_idx][:top_n]
        top_eta = eta_sq[sorted_idx][:top_n]

        ax.barh(top_feats[::-1], top_eta[::-1])
        ax.set_xlim(0, 1)
    except Exception as e:
        print(f"[ERROR] Cell {cell_key} → {e}")
        ax.axis("off")

def eta2_bar_constraints_cell_pca_kmeans(ax, df, cell_key, mutable_features, top_n=5, max_clusters=5):
    if df.empty or len(mutable_features) < 2:
        ax.axis("off")
        return

    constraint_features = [col for col in df.columns if col not in mutable_features and col not in ['cell', 'individual_id', 'fitness']]
    if len(constraint_features) < 2:
        ax.axis("off")
        return

    X = df[mutable_features].values
    if X.shape[0] < 3:
        ax.axis("off")
        return

    try:
        # PCA + KMeans
        pca = PCA(n_components=min(2, X.shape[1]))
        X_pca = pca.fit_transform(X)

        from cf_search.visualize import optimal_clusters
        k, _ = optimal_clusters(X_pca, max_clusters=max_clusters)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_pca)

        X_constraints = df[constraint_features].values
        df_within = X_constraints.shape[0] - X_constraints.shape[1] - 1
        if df_within <= 0:
            ax.axis("off")
            return

        F_vals, _ = f_classif(X_constraints, labels)
        eta_sq = F_vals / (F_vals + df_within)
        eta_sq = np.clip(eta_sq, 0, 1)

        sorted_idx = np.argsort(-eta_sq)
        top_features = np.array(constraint_features)[sorted_idx[:top_n]]
        top_eta = eta_sq[sorted_idx[:top_n]]

        ax.barh(top_features[::-1], top_eta[::-1])
        ax.set_xlim(0, 1)
    except Exception as e:
        print(f"[ERROR] Cell {cell_key} → {e}")
        ax.axis("off")

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def tree_constraints_to_cluster_cell_pca_kmeans(ax, df, cell_key, mutable_features, top_n=5, max_clusters=5, max_depth=3):
    """
    For a single cell:
    - Cluster the counterfactuals by PCA of mutable features.
    - Train a decision tree to predict the cluster using constraint features.
    - Plot the decision tree on the provided matplotlib axis.

    Parameters:
    - ax: matplotlib axis
    - df: DataFrame for one cell
    - cell_key: (i,j) cell ID
    - mutable_features: list of mutable feature names
    - top_n: unused here, but consistent API with η² version
    - max_clusters: number of clusters for KMeans
    - max_depth: max tree depth
    """
    if df.empty or len(mutable_features) < 2:
        ax.axis("off")
        return

    # Constraint feature selection: same as eta2_bar_constraints_cell_pca_kmeans
    constraint_features = [col for col in df.columns 
                           if col not in mutable_features 
                           and col not in ['cell', 'individual_id', 'fitness']]
    if len(constraint_features) < 2:
        ax.axis("off")
        return

    X = df[mutable_features].values
    if X.shape[0] < 3:
        ax.axis("off")
        return
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    try:
        # Step 1: Cluster using PCA + KMeans
        pca = PCA(n_components=min(2, X.shape[1]))
        X_pca = pca.fit_transform(X_scaled)

        from cf_search.visualize import optimal_clusters
        k, _ = optimal_clusters(X_pca, max_clusters=max_clusters)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_pca)

        # Step 2: Train decision tree on constraint features
        X_constraints = df[constraint_features]
        valid_idx = X_constraints.dropna().index
        X_constraints = X_constraints.loc[valid_idx]
        y_labels = pd.Series(labels, index=df.index).loc[valid_idx]

        if X_constraints.shape[0] < 3:
            ax.axis("off")
            return

        X_scaled = StandardScaler().fit_transform(X_constraints)
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
        clf.fit(X_scaled, y_labels)

        # Compute cross-validated accuracy
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(clf, X_scaled, y_labels, cv=5)
        mean_accuracy = scores.mean()

        # Step 3: Plot the tree
        plot_tree(
            clf,
            feature_names=constraint_features,
            class_names=[str(c) for c in sorted(np.unique(labels))],
            filled=True,
            rounded=True,
            ax=ax,
            fontsize=2,
            precision=2
        )
        ax.set_title(f"Tree {cell_key} (acc={mean_accuracy:.2f})")

    except Exception as e:
        print(f"[ERROR] Cell {cell_key} → {e}")
        ax.axis("off")



############# HIERARCHAL CLUSTERING PIPELINE FUNCTIONS ######################
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier

def tree_constraints_to_hclust_cell(ax, df, cell_key, mutable_features, max_depth=3, precision=2, fontsize=3):
    """
    - Cluster counterfactuals by hierarchical clustering of mutable features
    - Train a decision tree and a random forest to predict cluster using constraint features
    - Plot the decision tree
    - Print CV accuracy for both models

    Parameters:
    - ax: matplotlib axis
    - df: DataFrame for the cell
    - cell_key: tuple (i, j)
    - mutable_features: list of mutable feature names
    - height: threshold for hierarchical clustering distance
    - max_depth: maximum depth of the decision tree
    - precision: decimals to show in box
    - fontsize: size of text in tree
    """
    if df.empty or len(mutable_features) < 2:
        ax.axis("off")
        return

    constraint_features = [col for col in df.columns
                           if col not in mutable_features and col not in ['cell', 'individual_id', 'fitness']]
    if len(constraint_features) < 2:
        ax.axis("off")
        return

    X = df[mutable_features].values
    if X.shape[0] < 3:
        ax.axis("off")
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    try:
        # Step 1: Hierarchical clustering
        from cf_search.visualize import optimal_hclust_k  # assume already defined
        k, labels, Z = optimal_hclust_k(X_scaled, max_k=5)

        # Step 2: Train decision tree and random forest on constraint features
        X_constraints = df[constraint_features]
        valid_idx = X_constraints.dropna().index
        X_constraints = X_constraints.loc[valid_idx]
        y_labels = pd.Series(labels, index=df.index).loc[valid_idx]

        if X_constraints.shape[0] < 3:
            ax.axis("off")
            return

        X_constraints_scaled = StandardScaler().fit_transform(X_constraints)

        # Train Decision Tree
        dt = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
        dt.fit(X_constraints_scaled, y_labels)
        dt_scores = cross_val_score(dt, X_constraints_scaled, y_labels, cv=5)
        dt_acc = dt_scores.mean()

        # Train Random Forest
        #rf = HistGradientBoostingClassifier(max_depth=max_depth, random_state=0)
        #rf_scores = cross_val_score(rf, X_constraints_scaled, y_labels, cv=5)
        #rf_acc = rf_scores.mean()

        # Print results
        #print(f"Cell {cell_key} – DT acc: {dt_acc:.2f}, RF acc: {rf_acc:.2f}")

        # Step 3: Plot decision tree
        plot_tree(
            dt,
            feature_names=constraint_features,
            class_names=[str(c) for c in sorted(np.unique(labels))],
            filled=True,
            rounded=True,
            ax=ax,
            precision=precision,
            fontsize=fontsize
        )
        ax.set_title(f"Tree {cell_key} (DT acc={dt_acc:.2f})")

    except Exception as e:
        print(f"[ERROR] Cell {cell_key} → {e}")
        ax.axis("off")


def plot_cf_clustermap_cell(ax, df, cell_key, mutable_features):
    if df.empty or len(mutable_features) < 2:
        ax.axis("off")
        return

    X = df[mutable_features].values
    if X.shape[0] < 3 or np.allclose(X.std(axis=0), 0):
        ax.set_title(f"{cell_key} – Low Var", fontsize=8)
        ax.axis("off")
        return

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Hierarchical clustering on samples
    row_dist = pdist(X_scaled, metric='euclidean')
    row_linkage = linkage(row_dist, method='ward')

    try:
        # Reorder based on clustering
        from scipy.cluster.hierarchy import leaves_list
        ordered_indices = leaves_list(row_linkage)
        X_reordered = X_scaled[ordered_indices, :]

        # Plot reordered distance matrix (sample x sample)
        dist_matrix = squareform(pdist(X_reordered, metric='euclidean'))
        sns.heatmap(
            dist_matrix,
            cmap='coolwarm',
            center=np.median(dist_matrix),
            xticklabels=False,
            yticklabels=False,
            cbar=False,
            ax=ax
        )

        ax.set_title(f"Cell {cell_key}", fontsize=8)
    except Exception:
        ax.axis("off")



def plot_cf_umap_cell_clusters(ax, df, cell_key, mutable_features):
    if df.empty or len(mutable_features) < 2:
        ax.axis("off")
        return

    X = df[mutable_features].values
    if X.shape[0] < 3:
        ax.axis("off")
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if np.unique(X_scaled, axis=0).shape[0] == 1:
        ax.scatter([0], [0], c='blue', s=10)
        ax.set_title(f"Cell {cell_key} (Identical)")
        ax.axis('off')
        return

    umap_model = UMAP(n_components=2, random_state=42, n_neighbors=2, min_dist=0.0, metric="euclidean")
    X_umap = umap_model.fit_transform(X_scaled)

    k, labels, Z = optimal_hclust_k(X_scaled, max_k=5)

    ax.scatter(X_umap[:, 0], X_umap[:, 1], c=labels, cmap='tab10', s=10, alpha=0.7)
    #ax.set_title(f"Cell {cell_key}")
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)


def plot_cf_umap_cell(ax, df, cell_key, mutable_features, color_feature=None, global_vmin=None, global_vmax=None):
    if df.empty or len(mutable_features) < 2:
        ax.axis("off")
        return

    X = df[mutable_features].values
    if X.shape[0] < 3:
        ax.axis("off")
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # If all rows are identical
    if np.unique(X_scaled, axis=0).shape[0] == 1:
        ax.scatter([0], [0], c='blue', s=10)
        ax.set_title(f"Cell {cell_key} (Identical)")
        ax.axis('off')
        return

    # UMAP projection
    umap_model = UMAP(n_components=2, random_state=42, n_neighbors=2, min_dist=0.0, metric="euclidean")
    X_umap = umap_model.fit_transform(X_scaled)

    # Hierarchical clustering
    k, labels, Z = optimal_hclust_k(X_scaled, max_k=5)

    if color_feature is not None and color_feature in df.columns:
        colors = df[color_feature].values
        scatter = ax.scatter(
            X_umap[:, 0], X_umap[:, 1],
            c=colors, cmap='viridis', s=10, alpha=0.7,
            vmin=global_vmin, vmax=global_vmax
        )
        return scatter  # for colorbar
    else:
        ax.scatter(X_umap[:, 0], X_umap[:, 1], c=labels, cmap='tab10', s=10, alpha=0.7)
        return None  # no colorbar needed

    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)


import warnings
from sklearn.exceptions import ConvergenceWarning

from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
import numpy as np

def eta_squared_1d(x, labels):
    groups = np.unique(labels)
    n = len(x)
    grand_mean = np.mean(x)
    
    ss_between = sum([len(x[labels == g]) * (np.mean(x[labels == g]) - grand_mean)**2 for g in groups])
    ss_total = sum((x - grand_mean)**2)
    
    if ss_total == 0:
        return 0.0  # Avoid divide by zero
    
    eta_sq = ss_between / ss_total
    return eta_sq


def plot_cf_eta2_bar_cell(ax, df, cell_key, mutable_features, top_n=5):
    if df.empty or len(mutable_features) < 2:
        ax.axis("off")
        return

    X = df[mutable_features].values
    if X.shape[0] < 3:
        ax.axis("off")
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    try:
        k, labels, _ = optimal_hclust_k(X_scaled, max_k=5)
        eta_dict = {}

        for feat in mutable_features:
            x = df[feat].values

            try:
                eta = eta_squared_1d(x, labels)
                eta_dict[feat] = eta
            except Exception as e:
                print(f"[WARN] {feat}: {e}")
                eta_dict[feat] = 0.0

        # Plot top
        sorted_items = sorted(eta_dict.items(), key=lambda x: -x[1])
        top_feats, top_etas = zip(*sorted_items[:top_n])
        ax.barh(top_feats[::-1], top_etas[::-1])
        ax.set_xlim(0, 1)

    except Exception as e:
        print(f"[ERROR] Cell {cell_key} → {e}")
        ax.axis("off")




def plot_cf_eta2_bar_cell_old(ax, df, cell_key, mutable_features, top_n=5):
    if df.empty or len(mutable_features) < 2:
        ax.axis("off")
        return

    X = df[mutable_features].values
    if X.shape[0] < 3:
        ax.axis("off")
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    try:
        # Cluster to get group labels
        k, labels, Z = optimal_hclust_k(X_scaled, max_k=5)

        eta_dict = {}
        n_samples = X.shape[0]

        for feat in mutable_features:
            x_feat = df[[feat]].values

            # Skip constant feature (zero variance)
            if np.var(x_feat) < 1e-8:
                eta_dict[feat] = 0.0
                continue

            try:
                F_val, _ = f_classif(x_feat, labels)
                df_within = n_samples - 1 - 1
                if df_within <= 0 or not np.isfinite(F_val[0]):
                    eta = 0.0
                else:
                    eta = F_val[0] / (F_val[0] + df_within)
                eta_dict[feat] = eta
            except Exception:
                eta_dict[feat] = 0.0

        # Sort and plot top N
        sorted_items = sorted(eta_dict.items(), key=lambda x: -x[1])
        top_feats, top_eta = zip(*sorted_items[:top_n])

        ax.barh(top_feats[::-1], top_eta[::-1])
        ax.set_xlim(0, 1)

    except Exception as e:
        print(f"[ERROR] Cell {cell_key} → {e}")
        ax.axis("off")





def plot_cf_eta2_bar_constraints_cell(ax, df, cell_key, mutable_features, top_n=5):
    if df.empty or len(mutable_features) < 2:
        ax.axis("off")
        return

    # Constraints = non-mutable columns (not in mutable_features, not "cell" or "individual_id")
    constraint_features = [col for col in df.columns if col not in mutable_features and col not in ['cell', 'individual_id', 'fitness']]
    if len(constraint_features) < 2:
        ax.axis("off")
        return

    X = df[mutable_features].values
    if X.shape[0] < 3:
        ax.axis("off")
        return
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    try:
        # Cluster based on mutable features
        k, labels, Z = optimal_hclust_k(X_scaled, max_k=5)

        # Compute eta squared for constraint features
        from sklearn.feature_selection import f_classif

        X_constraints_raw = df[constraint_features]
        # Filter out constant features
        variances = X_constraints_raw.var(axis=0)
        valid_mask = variances > 1e-8
        X_constraints = X_constraints_raw.loc[:, valid_mask]

        if X_constraints.shape[1] == 0:
            ax.axis("off")
            return

        # Safe f_classif call
        F_vals, _ = f_classif(X_constraints.values, labels)
        eta_sq = F_vals / (F_vals + X_constraints.shape[0] - X_constraints.shape[1] - 1)
        eta_sq = np.nan_to_num(eta_sq, nan=0.0)  # Replace NaNs with 0

        # Select top constraint features
        constraint_features_filtered = X_constraints.columns.tolist()
        sorted_idx = np.argsort(-eta_sq)
        top_features = np.array(constraint_features_filtered)[sorted_idx[:top_n]]
        top_eta = eta_sq[sorted_idx[:top_n]]



        # Plot barplot
        ax.barh(top_features[::-1], top_eta[::-1])
        ax.set_xlim(0, 1)
    except Exception:
        ax.axis("off")


def plot_cf_meanvalue_heatmap_cell(ax, df, cell_key, mutable_features):
    if df.empty or len(mutable_features) < 2:
        ax.axis("off")
        return

    X = df[mutable_features].values
    if X.shape[0] < 3:
        ax.axis("off")
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    k, labels, Z = optimal_hclust_k(X_scaled, max_k=5)

    df_tmp = df.copy()
    df_tmp['cluster'] = labels
    mean_by_cluster = df_tmp.groupby('cluster')[mutable_features].mean()

    sns.heatmap(mean_by_cluster, cmap='coolwarm', center=0, annot=True, fmt=".2f", ax=ax)
    #ax.set_title(f"Mean Values per Cluster\nCell {cell_key}")

def plot_cf_meanvalue_heatmap_constraints_cell(ax, df, cell_key, mutable_features):
    if df.empty or len(mutable_features) < 2:
        ax.axis("off")
        return

    # Constraints = non-mutable columns (not in mutable_features, not "cell" or "individual_id")
    constraint_features = [col for col in df.columns if col not in mutable_features and col not in ['cell', 'individual_id', 'fitness']]
    if len(constraint_features) < 2:
        ax.axis("off")
        return

    X = df[mutable_features].values
    if X.shape[0] < 3:
        ax.axis("off")
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    try:
        k, labels, Z = optimal_hclust_k(X_scaled, max_k=5)

        # Add cluster label to df and compute mean constraints per cluster
        df_tmp = df.copy()
        df_tmp['cluster'] = labels
        mean_by_cluster = df_tmp.groupby('cluster')[constraint_features].mean()

        sns.heatmap(mean_by_cluster, cmap='coolwarm', center=0, annot=True, fmt=".2f", ax=ax)
        #ax.set_title(f"Constraint Means\nCell {cell_key}", fontsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    except Exception:
        ax.axis("off")


def plot_cf_kde_cell(ax, df, cell_key, mutable_features, feature):
    if df.empty or feature not in df.columns:
        ax.axis("off")
        return

    # Cluster on mutable features if possible, else on all features
    features_for_clustering = mutable_features
    X = df[features_for_clustering].values
    if X.shape[0] < 3:
        ax.axis("off")
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    try:
        k, labels, Z = optimal_hclust_k(X_scaled, max_k=5)
        df = df.copy()
        df['cluster'] = labels

        for c in sorted(df['cluster'].unique()):
            if feature in df.columns:
                sns.kdeplot(df[df['cluster'] == c][feature], label=f'Cluster {c}', fill=True, ax=ax, alpha=0.3)

        ax.set_title(f"{feature} by Cluster\nCell {cell_key}")
        ax.legend()
    except Exception:
        ax.axis("off")




def optimal_clusters(X, max_clusters=5):
    """Determines the optimal number of clusters using silhouette score."""
    if X.shape[0] < 3:
        return 1, -1  # Not enough samples

    best_k = 2
    best_score = -1

    for k in range(2, min(max_clusters, X.shape[0]) + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels) if k > 1 else -1
        if score > best_score:
            best_score = score
            best_k = k

    return best_k, best_score


def optimal_hclust_k(X, max_k=10, method='ward', metric='euclidean'):
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


def run_pairwise_tests(df, cluster_labels, features):
    df['Cluster'] = cluster_labels
    sig_pairs = {}
    non_sig_pairs = {}
    for feature in features:
        sig_pairs[feature] = []
        non_sig_pairs[feature] = []
        if df[feature].nunique() <= 2 and set(df[feature].dropna().unique()).issubset({0, 1}):
            table = pd.crosstab(df['Cluster'], df[feature])
            for i in table.index:
                for j in table.index:
                    if i < j:
                        sub = table.loc[[i, j]]
                        if sub.shape[1] == 2:
                            try:
                                stat, p, _, _ = chi2_contingency(sub)
                                if p < 0.05:
                                    sig_pairs[feature].append((i, j, p))
                                else:
                                    non_sig_pairs[feature].append((i, j, p))
                            except ValueError:
                                continue
        else:
            tukey = pairwise_tukeyhsd(endog=df[feature], groups=df['Cluster'], alpha=0.05)
            for res in tukey.summary().data[1:]:
                group1 = res[0]
                group2 = res[1]
                pval = res[3]
                if pval < 0.05:
                    sig_pairs[feature].append((group1, group2, pval))
                else:
                    non_sig_pairs[feature].append((group1, group2, pval))

    return sig_pairs, non_sig_pairs

def plot_cluster_heatmap(ax, data, cluster_labels, features, sig_pairs=None):
    """
    Plot a heatmap of feature values averaged per cluster, with significance markers.

    Parameters:
    - ax: matplotlib axis to plot into
    - data: numpy array or DataFrame with feature data
    - cluster_labels: array of cluster assignments
    - features: list of feature names
    - sig_pairs: dict mapping feature names to significant cluster pairs
    """
    df_plot = pd.DataFrame(data, columns=features)
    df_plot['Cluster'] = cluster_labels
    cluster_means = df_plot.groupby('Cluster').mean().round(2)

    annotations = pd.DataFrame("", index=cluster_means.index, columns=cluster_means.columns)

    if sig_pairs:
        symbol_map = ['¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹', '¹⁰']
        for f in features:
            for pair_idx, pair in enumerate(sig_pairs.get(f, [])):
                a, b, _ = pair
                sym = symbol_map[pair_idx % len(symbol_map)]
                for idx in (a, b):
                    try:
                        current = annotations.loc[idx, f]
                        mean_str = str(cluster_means.loc[idx, f])
                        if mean_str not in current:
                            annotations.loc[idx, f] = mean_str + sym
                        else:
                            if sym not in current:
                                annotations.loc[idx, f] += sym
                    except:
                        continue

    heatmap = sns.heatmap(
        cluster_means.T,
        cmap="viridis",
        annot=annotations.T.values,
        fmt="",
        xticklabels=True,
        yticklabels=features,
        ax=ax
    )
    ax.set_yticklabels(features, rotation=0, fontsize=16)
    # Make colorbar labels bigger
    if heatmap.collections:
        cbar = heatmap.collections[0].colorbar
        cbar.ax.tick_params(labelsize=16)
        ax.set_xlabel("Cluster", fontsize = 16)
        #ax.set_ylabel("Feature")


def plot_all_clusters_by_heatmap(archive_dict, cell_feature_sets, feature_categories, max_clusters=5, n_components=2, show=True):
    categories = list(feature_categories.keys())
    n = len(categories)

    fig, axes = plt.subplots(n, n, figsize=(7 * n, 6 * n), dpi=500)
    axes = np.atleast_2d(axes)

    for i in range(n):
        for j in range(n):
            ax = axes[n - 1 - j, i]  # Flip Y-axis to match grid layout
            cell_key = (i, j)

            if cell_key not in cell_feature_sets:
                ax.axis("off")
                continue

            df = extract_cell_data(archive_dict, cell_key)
            if df.empty:
                ax.axis("off")
                continue

            mutable_features = cell_feature_sets[cell_key]
            non_mutable_features = [feat for feat in df.drop(columns=['cell', 'individual_id']).columns if feat not in mutable_features]
            df = df.apply(pd.to_numeric, errors='coerce')

            if df.empty or len(mutable_features) < 2 or len(non_mutable_features) < 2:
                ax.axis("off")
                continue

            try:
                X_mutable = df[mutable_features].values
                X_non_mutable = df[non_mutable_features].values

                if X_mutable.shape[0] < 2 or X_mutable.shape[1] < 2:
                    raise ValueError(f"Not enough samples/features in Cell ({i},{j})")

                pca = PCA(n_components=min(n_components, X_mutable.shape[1]))
                X_pca = pca.fit_transform(X_mutable)

                optimal_k, _ = optimal_clusters(X_pca, max_clusters=max_clusters)
                kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_pca)

                sig_pairs, _ = run_pairwise_tests(df.copy(), cluster_labels, non_mutable_features)

                plot_cluster_heatmap(ax, X_non_mutable, cluster_labels, non_mutable_features, sig_pairs)

            except ValueError as e:
                print(f"Skipping Cell ({i},{j}): {e}")
                ax.axis("off")

    # Add category labels
    for i in range(n):
        axes[0, i].set_title(categories[i], fontsize=24, pad=20)
    for i in range(n):
        axes[i, 0].set_ylabel(categories[n - 1 - i], fontsize=24, rotation=0, ha='right', va='center')
    
    fig.subplots_adjust(hspace=0.2, wspace=0.8)
    plt.suptitle("Heatmap of Non-Mutable Feature Means by Category Pair", fontsize=36)
    

    if show:
        plt.show()
    else:
        return fig

def plot_all_mutable_clusters_by_heatmap(archive_dict, cell_feature_sets, feature_categories, max_clusters=5, n_components=2, show=True):
    categories = list(feature_categories.keys())
    n = len(categories)

    fig, axes = plt.subplots(n, n, figsize=(7 * n, 6 * n), dpi=500)
    axes = np.atleast_2d(axes)

    for i in range(n):
        for j in range(n):
            ax = axes[n - 1 - j, i]  # Flip Y-axis to match grid layout
            cell_key = (i, j)

            if cell_key not in cell_feature_sets:
                ax.axis("off")
                continue

            df = extract_cell_data(archive_dict, cell_key)
            if df.empty:
                ax.axis("off")
                continue

            mutable_features = cell_feature_sets[cell_key]
            df = df.apply(pd.to_numeric, errors='coerce')

            if df.empty or len(mutable_features) < 2:
                ax.axis("off")
                continue

            try:
                X_mutable = df[mutable_features].values

                if X_mutable.shape[0] < 2 or X_mutable.shape[1] < 2:
                    raise ValueError(f"Not enough samples/features in Cell ({i},{j})")

                pca = PCA(n_components=min(n_components, X_mutable.shape[1]))
                X_pca = pca.fit_transform(X_mutable)

                optimal_k, _ = optimal_clusters(X_pca, max_clusters=max_clusters)
                kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_pca)

                sig_pairs, _ = run_pairwise_tests(df.copy(), cluster_labels, mutable_features)

                plot_cluster_heatmap(ax, X_mutable, cluster_labels, mutable_features, sig_pairs)

            except ValueError as e:
                print(f"Skipping Cell ({i},{j}): {e}")
                ax.axis("off")

    # Add category labels
    for i in range(n):
        axes[0, i].set_title(categories[i], fontsize=24, pad=20)
    for i in range(n):
        axes[i, 0].set_ylabel(categories[n - 1 - i], fontsize=24, rotation=0, ha='right', va='center')

    fig.subplots_adjust(hspace=0.2, wspace=0.8)
    plt.suptitle("Heatmap of Mutable Feature Means by Category Pair", fontsize=36)

    if show:
        plt.show()
    else:
        return fig
    
    fig.subplots_adjust(hspace=0.3, wspace=0.7)
    plt.suptitle("Mutable Feature Mean Heatmap by Cell with Pairwise Posthoc Tests", fontsize=14)
    plt.show()


from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler

def plot_all_pcas_grid_by_cluster(archive_dict, cell_feature_sets, max_clusters=5, n_components=2, show=True):
    """Plots PCA for all cells, using hierarchical clustering on original values."""

    cell_indices = list(cell_feature_sets.keys())
    max_x = max(cell[0] for cell in cell_indices) + 1
    max_y = max(cell[1] for cell in cell_indices) + 1

    fig, axes = plt.subplots(max_y, max_x, figsize=(4 * max_x, 4 * max_y))
    axes = np.atleast_2d(axes)

    for i in range(max_x):
        for j in range(max_y):
            ax = axes[max_y - 1 - j, i]
            cell_key = (i, j) if (i, j) in cell_indices else (j, i) if (j, i) in cell_indices else None

            if cell_key:
                df = extract_cell_data(archive_dict, cell_key)

                if df.empty:
                    ax.set_title(f"Cell ({i},{j})\nNo Data", fontsize=8)
                    ax.axis("off")
                    continue

                changed_features = [feat for feat in cell_feature_sets[cell_key] if feat in df.columns]
                if len(changed_features) < 2:
                    ax.set_title(f"Cell ({i},{j})\nNo Features", fontsize=8)
                    ax.axis("off")
                    continue

                X = df[changed_features].values
                try:
                    if X.shape[0] < 2 or X.shape[1] < 2:
                        raise ValueError("Too few samples/features")

                    # Hierarchical clustering on original scaled values
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    Z = linkage(X_scaled, method="ward")
                    
                    # Determine optimal number of clusters (e.g., silhouette or fixed max_k)
                    from sklearn.metrics import silhouette_score
                    best_k, best_score = 2, -1
                    for k_try in range(2, min(max_clusters, len(X_scaled)) + 1):
                        labels_try = fcluster(Z, k_try, criterion='maxclust')
                        score = silhouette_score(X_scaled, labels_try) if len(set(labels_try)) > 1 else -1
                        if score > best_score:
                            best_k = k_try
                            best_score = score
                    labels = fcluster(Z, best_k, criterion='maxclust')

                    # PCA for 2D visualization
                    pca = PCA(n_components=n_components)
                    X_pca = pca.fit_transform(X)

                    # Plot
                    colors = sns.color_palette("colorblind", best_k)
                    ax.scatter(
                        X_pca[:, 0], X_pca[:, 1],
                        c=[colors[l - 1] for l in labels],
                        alpha=0.7, s=30
                    )

                    ax.set_title(f"Cell ({i},{j}) - {best_k} Clusters", fontsize=8)
                    ax.set_xticks([])
                    ax.set_yticks([])

                except Exception as e:
                    print(f"Skipping Cell ({i},{j}): {e}")
                    ax.set_title(f"Cell ({i},{j})\nSkipped", fontsize=8)
                    ax.axis("off")
            else:
                ax.set_title(f"Cell ({i},{j})\nEmpty", fontsize=8)
                ax.axis("off")

    fig.subplots_adjust(hspace=0.1, wspace=0.)
    plt.suptitle("Hierarchical Clustering + PCA of Counterfactuals by Cell", fontsize=14)
    if show:
        plt.show()
    else:
        return fig, axes

def plot_all_pcas_grid_by_cluster_old(archive_dict, cell_feature_sets, max_clusters=5, n_components=2, show=True):
    """Plots PCA with clustering for all cells."""


    cell_indices = list(cell_feature_sets.keys())
    max_x = max(cell[0] for cell in cell_indices) + 1
    max_y = max(cell[1] for cell in cell_indices) + 1

    fig, axes = plt.subplots(max_y, max_x, figsize=(4 * max_x, 4 * max_y))
    axes = np.atleast_2d(axes)

    for i in range(max_x):
        for j in range(max_y):
            ax = axes[max_y - 1 - j, i]
            cell_key = (i, j) if (i, j) in cell_indices else (j, i) if (j, i) in cell_indices else None

            if cell_key:
                df = extract_cell_data(archive_dict, cell_key)

                if df.empty:
                    ax.set_title(f"Cell ({i},{j})\nNo Data", fontsize=8)
                    ax.axis("off")
                    continue

                changed_features = [feat for feat in cell_feature_sets[cell_key] if feat in df.columns]
                if len(changed_features) < 2:
                    ax.set_title(f"Cell ({i},{j})\nNo Features", fontsize=8)
                    ax.axis("off")
                    continue

                X = df[changed_features].values
                try:
                    if X.shape[0] < 2 or X.shape[1] < 2:
                        raise ValueError("Too few samples/features")

                    pca = PCA(n_components=n_components)
                    X_pca = pca.fit_transform(X)

                    # Cluster
                    k, score = optimal_clusters(X_pca, max_clusters)
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(X_pca)

                    pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)])
                    pca_df["Cluster"] = labels

                    # Palette
                    if use_wes_palette:
                        colors = wes_palette("GrandBudapest2", k, type="discrete")
                    else:
                        colors = sns.color_palette("colorblind", k)

                    ax.scatter(
                        pca_df["PC1"], pca_df["PC2"],
                        c=[colors[label] for label in pca_df["Cluster"]],
                        alpha=0.7
                    )

                    ax.set_title(f"Cell ({i},{j}) - {k} Clusters", fontsize=8)
                    ax.set_xticks([])
                    ax.set_yticks([])

                except Exception as e:
                    print(f"Skipping Cell ({i},{j}): {e}")
                    ax.set_title(f"Cell ({i},{j})\nSkipped", fontsize=8)
                    ax.axis("off")
            else:
                ax.set_title(f"Cell ({i},{j})\nEmpty", fontsize=8)
                ax.axis("off")

    fig.subplots_adjust(hspace=0.1, wspace=0.)
    plt.suptitle("Clustered PCA of Counterfactuals by Cell", fontsize=14)
    if show:
        plt.show()
    else:
        return fig, axes


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

def plot_all_pcas_grid(archive_dict, cell_feature_sets, feature_categories, color_feature, n_components=2, show=True):
    """
    Plots PCA results for all valid cells in a heatmap-style grid with category-labeled axes.

    Parameters:
    - archive_dict: Dict of DataFrames from MAP-Elites.
    - cell_feature_sets: Dict of (i,j) tuples to list of feature names.
    - feature_categories: Dict of category name → list of features (to label axes).
    - color_feature: Feature used for point coloring.
    - n_components: Number of PCA components to plot (default=2).
    - show: Whether to call plt.show() (set False for Streamlit).

    Returns:
    - fig, axes if show=False
    """

    # Extract category names from feature_categories
    categories = list(feature_categories.keys())
    n = len(categories)

    # Compute global min/max for color scaling
    all_fitness = []
    for cell_key in cell_feature_sets:
        df = extract_cell_data(archive_dict, cell_key)
        if color_feature in df.columns:
            all_fitness.extend(df[color_feature].dropna().tolist())
    global_vmin = min(all_fitness) if all_fitness else 0
    global_vmax = max(all_fitness) if all_fitness else 1

    # Create figure and axes grid
    fig, axes = plt.subplots(n, n, figsize=(4 * n, 4 * n))
    axes = np.atleast_2d(axes)
    scatter = None

    for i in range(n):
        for j in range(n):
            ax = axes[n - 1 - j, i]  # Flip Y-axis so (0,0) is bottom-left
            cell_key = (i, j)

            if cell_key not in cell_feature_sets:
                ax.axis("off")
                continue

            df = extract_cell_data(archive_dict, cell_key)
            if df.empty:
                ax.axis("off")
                continue

            changed_features = [f for f in cell_feature_sets[cell_key] if f in df.columns]
            if len(changed_features) < 2:
                ax.axis("off")
                continue

            try:
                X = df[changed_features].values
                if X.shape[0] < 2:
                    raise ValueError("Too few samples")

                pca = PCA(n_components=n_components)
                X_pca = pca.fit_transform(X)

                pca_df = pd.DataFrame(X_pca, columns=[f"PC{k+1}" for k in range(n_components)])
                pca_df[color_feature] = df[color_feature].values

                scatter = ax.scatter(
                    pca_df["PC1"], pca_df["PC2"],
                    c=pca_df[color_feature],
                    cmap="viridis",
                    vmin=global_vmin,
                    vmax=global_vmax,
                    alpha=0.7
                )

                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlim(-5, 5)
                ax.set_ylim(-5, 5)


            except Exception:
                ax.axis("off")

    # Add x-axis labels at the top
    for i in range(n):
        axes[0, i].set_title(categories[i], fontsize=24, pad=20)

    # Add y-axis labels on the left side
    for i in range(n):
        axes[i, 0].set_ylabel(categories[n - 1 - i], fontsize=24, rotation=0, ha='right', va='center')

    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    if scatter:
        cbar = fig.colorbar(scatter, ax=axes, orientation="horizontal", fraction=0.02, pad=0.05)
        cbar.set_label(color_feature, fontsize=24)
        cbar.ax.tick_params(labelsize=24)

    plt.suptitle("PCA of Counterfactuals by Category Pair", fontsize=16)

    if show:
        plt.show()
    else:
        return fig, axes




def extract_cell_data(archive_dict, cell_index):
    """
    Extracts counterfactuals from all individuals for a specific cell.

    Parameters:
    - archive_dict: Dictionary of DataFrames per individual.
    - cell_index: Tuple (i, j) representing the cell.

    Returns:
    - Combined DataFrame of counterfactuals from all individuals in that cell.
    """
    cell_data = []

    for individual_id, df in archive_dict.items():
        selected_row = df[df["cell"] == str(cell_index)].copy()
        if not selected_row.empty:
            selected_row.loc[:, "individual_id"] = individual_id
            cell_data.append(selected_row)

    if cell_data:
        return pd.concat(cell_data, ignore_index=True)
    else:
        return pd.DataFrame()


def plot_fitness_heatmap(archive_dict, individual_id, figsize=(6, 5), cmap="viridis", show=True, save_path=None):
    """
    Plots a heatmap of fitness values for a given individual's counterfactuals.

    Parameters:
    - archive_dict: Dictionary of individual DataFrames with 'cell' and 'fitness'.
    - individual_id: ID of the individual to plot.
    - figsize: Size of the figure (default 6x5).
    - cmap: Colormap for the heatmap.
    - show: Whether to call plt.show() (default True).
    - save_path: If provided, saves the plot to this file path.
    """
    if individual_id not in archive_dict:
        print(f"❌ Individual {individual_id} not found in archive.")
        return

    individual_df = archive_dict[individual_id]
    unique_cells = [eval(cell) if isinstance(cell, str) else cell for cell in individual_df["cell"].unique()]
    max_x = max(cell[0] for cell in unique_cells) + 1
    max_y = max(cell[1] for cell in unique_cells) + 1

    fitness_matrix = np.full((max_x, max_y), np.nan)

    for _, row in individual_df.iterrows():
        cell_tuple = eval(row["cell"]) if isinstance(row["cell"], str) else row["cell"]
        i, j = cell_tuple
        fitness_matrix[i, j] = row["fitness"]
        fitness_matrix[j, i] = row["fitness"]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(fitness_matrix, annot=True, cmap=cmap, fmt=".2f", square=True, linewidths=0.5, ax=ax)

    ax.invert_yaxis()
    ax.set_xlabel("Feature Category 2")
    ax.set_ylabel("Feature Category 1")
    ax.set_title(f"Fitness Heatmap for Individual {individual_id}")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        return fig, ax


