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
import os
import random
os.environ["NUMBA_DISABLE_JIT"] = "1"  # Ensures deterministic UMAP
random.seed(42)
np.random.seed(42)
from umap import UMAP
import torch
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)


# Optional: Wes Anderson palette fallback
try:
    from wesanderson import wes_palette
    use_wes_palette = True
except ImportError:
    use_wes_palette = False

def fused_similarity_network(
    archive_dict,
    cell_feature_sets,
    feature_categories,
    test_data,
    color_feature
):
    data_views = []
    categories = list(feature_categories.keys())
    n = len(categories)
    #go through each cell and make a distance matrix
    for i in range(n):
        for j in range(n):
            cell_key = (i, j)

            df = extract_cell_data(archive_dict, cell_key)

            mutable_features = cell_feature_sets[cell_key]
            #print(f"{cell_key} columns: {df.columns.tolist()}")
            #print(f"{cell_key}: {mutable_features}")
            missing = [f for f in mutable_features if f not in df.columns]
            if missing:
                print(f"[{cell_key}] Missing features: {missing}")
                print(f"Available in df: {df.columns.tolist()}")
                continue  # skip this cell
            X = df[mutable_features].values
            data_views.append(X)

    # Now fuse the similarity networks
    affinities = snf.make_affinity(data_views, metric='euclidean', K=5, mu=0.5)
    fused_affinity = snf.snf(affinities)

    G = nx.Graph()
    n = fused_affinity.shape[0]

    for i in range(n):
        G.add_node(i, color=test_data[color_feature].iloc[i])

    K = 5  # number of neighbors
    for i in range(n):
        # Get indices of top K neighbors (excluding self)
        top_k = np.argsort(fused_affinity[i])[::-1][1:K+1]
        for j in top_k:
            weight = fused_affinity[i, j]
            if weight > 0:
                G.add_edge(i, j, weight=weight)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=500)
    pos = nx.spring_layout(G, seed=42)
    edge_weights = np.array([G[u][v]['weight'] for u, v in G.edges])

    norm = plt.Normalize(vmin=-1, vmax=1)
    node_colors = [plt.cm.viridis(norm(G.nodes[i]['color'])) for i in G.nodes]

    # Normalize edge weights
    edge_norm = plt.Normalize(vmin=edge_weights.min(), vmax=edge_weights.max())
    edge_colors = [plt.cm.Blues(edge_norm(w)) for w in edge_weights]

    nx.draw(
        G, pos, ax=ax, with_labels=False,
        node_color=node_colors, node_size=50,
        edge_color=edge_colors, edge_cmap=plt.cm.Blues,
        width=0.5  # Use manually normalized colors
    )
    return fig


def plot_cellwise_grid(
    archive_dict,
    cell_feature_sets,
    feature_categories,
    plot_fn,
    plot_type_name="Grid",
    figsize_per_cell=(4, 4),
    colorbar_label="Predicted outcome",
    legend=False,
    wspace=0.5,
    hspace=0.5,
    bottom=0.15,
    top=0.9,
    left=0.1,
    right=0.95,
    title_fontsize=24,
    rowlabel_fontsize=24,
    rowlabel_offset=0.05,
    rowlabel_ybias=0.2,
    **kwargs
):
    categories = list(feature_categories.keys())
    n = len(categories)
    fig, axes = plt.subplots(n, n, figsize=(figsize_per_cell[0]*n, figsize_per_cell[1]*n), dpi=300)
    axes = np.atleast_2d(axes)

    mappables = []
    handles_labels = []

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

                # For legend mode, collect handles + labels (if returned)
                if legend and hasattr(mappable, 'legend_elements'):
                    handles, labels = mappable.legend_elements()
                    handles_labels.append((handles, labels))
    '''
    # Set column labels (top row)
    for i in range(n):
        ax_top = axes[0, i]
        ax_top.set_title(categories[i], fontsize=title_fontsize, pad=10)


    # Add separate row labels using fig.text instead of modifying subplot y-axis labels
    row_height = 1.0 / (n+1)  # height of each subplot row
    for j in range(n):
        y_pos = bottom + row_height * (j) + row_height*0.4  # center of each row
        fig.text(
            rowlabel_offset,
            y_pos,
            #categories[n - 1 - j],
            categories[j],
            ha='right',
            va='center',
            fontsize=rowlabel_fontsize,
            rotation=0,
            transform=fig.transFigure
        )

    '''
    '''
    if legend and handles_labels:
        # Flatten handles/labels and deduplicate
        all_handles = sum((hl[0] for hl in handles_labels), [])
        all_labels = sum((hl[1] for hl in handles_labels), [])
        unique = {lbl: h for h, lbl in zip(all_handles, all_labels)}  # deduplicate
        fig.legend(
            handles=list(unique.values()),
            labels=list(unique.keys()),
            loc='lower center',
            bbox_to_anchor=(0.5, -0.01), 
            ncol=5,
            frameon=True,
            title="Cluster",
            fontsize=16,
            handletextpad=0.4
        )
    '''
    if not legend and mappables:
        cbar_ax = fig.add_axes([0.2, -0.1, 0.6, 0.02])  # Lower Y position
        #cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7])  # Right side, vertical
        cbar = fig.colorbar(mappables[0], cax=cbar_ax, orientation='horizontal')
        cbar.set_label(colorbar_label, fontsize=20)

    fig.subplots_adjust(wspace=wspace, hspace=hspace, bottom=bottom, top=top, left=left, right=right)
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

############# Similarity network of cells ##########################################
from snf import compute
import snf
from scipy.spatial.distance import pdist, squareform
import networkx as nx

def plot_similarity(ax, df, cell_key, mutable_features, color_feature, n_components=2, global_vmin=0, global_vmax=1):
    if len(mutable_features) < 2 or df.shape[0] < 2:
        ax.axis("off")
        return

    try:
        X = df[mutable_features].copy()
        dist = squareform(pdist(X, metric='euclidean'))

        # Use SNF to compute the similarity (affinity) matrix
        similarity = snf.make_affinity(X, metric='euclidean', K=20, mu=0.5)  # K: neighbors, mu: scaling factor

        G = nx.Graph()
        n = similarity.shape[0]

        for i in range(n):
            G.add_node(i, color=df[color_feature].iloc[i])

        K = 5  # number of neighbors
        for i in range(n):
            # Get indices of top K neighbors (excluding self)
            top_k = np.argsort(similarity[i])[::-1][1:K+1]
            for j in top_k:
                weight = similarity[i, j]
                if weight > 0:
                    G.add_edge(i, j, weight=weight)


        pos = nx.spring_layout(G, seed=42)
        edge_weights = [G[u][v]['weight'] for u, v in G.edges]

        norm = plt.Normalize(vmin=global_vmin, vmax=global_vmax)
        node_colors = [plt.cm.viridis(norm(G.nodes[i]['color'])) for i in G.nodes]

        # Normalize edge weights
        edge_norm = plt.Normalize(vmin=np.min(edge_weights), vmax=np.max(edge_weights))
        edge_colors = [plt.cm.inferno(edge_norm(w)) for w in edge_weights]

        nx.draw(
            G, pos, ax=ax, with_labels=False,
            node_color=node_colors, node_size=50,
            edge_color=edge_colors, edge_cmap=plt.cm.inferno,
            width=0.5  # Use manually normalized colors
        )


    except Exception as e:
        print(f"Error in plot_similarity (SNF): {e}")
        ax.axis("off")


def plot_similarity_manual(ax, df, cell_key, mutable_features, color_feature, n_components=2, global_vmin=0, global_vmax=1):
    if len(mutable_features) < 2 or df.shape[0] < 2:
        ax.axis("off")
        return

    try:
        # Avoid modifying original df
        X = df[mutable_features].copy()

        # Calculate Euclidean distance
        dist = squareform(pdist(X, metric='euclidean'))

        # Estimate sigma — average pairwise distance is a common heuristic
        sigma = np.mean(dist)

        # Gaussian similarity kernel
        similarity = np.exp(-dist**2 / (2 * sigma**2))

        G = nx.Graph()
        n = similarity.shape[0]

        # Add nodes (color stored as attribute)
        for i in range(n):
            G.add_node(i, color=df[color_feature].iloc[i])

        # Add edges for top-k similar neighbors
        k = min(5, n - 1)  # avoid index error when n < 5
        for i in range(n):
            top_k = np.argsort(similarity[i])[::-1][1:k+1]
            for j in top_k:
                G.add_edge(i, j, weight=similarity[i, j])

        # Layout and draw
        pos = nx.spring_layout(G, seed=42)
        edge_weights = [G[u][v]['weight'] for u, v in G.edges]

        # Normalize node colors
        norm = plt.Normalize(vmin=global_vmin, vmax=global_vmax)
        node_colors = [plt.cm.viridis(norm(G.nodes[i]['color'])) for i in G.nodes]
        ax.clear()
        nx.draw(
            G, pos, ax=ax, with_labels=False,
            node_color=node_colors, node_size=50,
            edge_color=edge_weights, edge_cmap=plt.cm.Blues
        )

    except Exception as e:
        print(f"Error in plot_similarity: {e}")
        ax.axis("off")

from sklearn.cluster import SpectralClustering
def plot_similarity_by_cluster(ax, df, cell_key, mutable_features, cluster_method, reducer, max_k=5):
    if df.empty or len(mutable_features) < 2 or df.shape[0] < 3:
        ax.axis("off")
        return

    try:
        # Extract and scale mutable features
        X = df[mutable_features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)



        # For plotting into 2D
        if reducer == "PCA":
            reducer_model = PCA(n_components=2)
            X_embedded = reducer_model.fit_transform(X)
            X_plotting = X_embedded
        elif reducer == "UMAP":
            reducer_model = UMAP(n_components=2, random_state=42, n_neighbors=5, min_dist=0.3)
            X_embedded = reducer_model.fit_transform(X)
            X_plotting = X_embedded
        else:
            X_embedded = X  # No dimensionality reduction
            reducer_model = PCA(n_components=2)
            X_plotting = reducer_model.fit_transform(X) #for plotting still reduce dimensionality

        # Clustering
        if cluster_method == "hierarchal":
            k, labels, Z = optimal_hclust_k(X, max_k=max_k)
        else:  # e.g., kmeans
            k, labels = optimal_clusters(X_embedded, max_k=max_k)
            labels = labels + 1


        X = df[mutable_features]
        # Calculate Euclidean distance
        dist = squareform(pdist(X, metric='euclidean'))

        # Estimate sigma — average pairwise distance is a common heuristic
        sigma = np.mean(dist)

        # Gaussian similarity kernel
        similarity = np.exp(-dist**2 / (2 * sigma**2))

        # Edge case: all rows are identical after scaling
        if np.unique(X_scaled, axis=0).shape[0] == 1:
            ax.scatter([0], [0], c='blue', s=10)
            ax.set_title(f"Cell {cell_key} (Identical)")
            ax.axis('off')
            return
        

        n_clusters = 4  # or however many clusters you expect

        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',  # because we're passing a similarity matrix
            assign_labels='kmeans',  # or 'discretize'
            random_state=42
        )

        labels = spectral.fit_predict(similarity)


        G = nx.Graph()
        n = similarity.shape[0]

        # Add nodes (color stored as attribute)
        for i in range(n):
            G.add_node(i, color=labels)

        # Add edges for top-k similar neighbors
        k = min(5, n - 1)  # avoid index error when n < 5
        for i in range(n):
            top_k = np.argsort(similarity[i])[::-1][1:k+1]
            for j in top_k:
                G.add_edge(i, j, weight=similarity[i, j])

        # Layout and draw
        pos = nx.spring_layout(G, seed=42)
        edge_weights = [G[u][v]['weight'] for u, v in G.edges]

        # Normalize node colors
        #norm = plt.Normalize(vmin=global_vmin, vmax=global_vmax)
        #node_colors = [plt.cm.viridis(norm(G.nodes[i]['color'])) for i in G.nodes]

        nx.draw(
            G, pos, ax=ax, with_labels=False,
            node_color=labels, node_size=50,
            edge_color=edge_weights, edge_cmap=plt.cm.Blues
        )
    except Exception as e:
        ax.axis("off")
        print(f"Error in cell {cell_key}: {e}")

############# PCA and K MEAN CLUSTERING PIPELINE FUNCTIONS #########################
def plot_counterfactuals(ax, df, cell_key, mutable_features, color_feature, n_components=2, global_vmin=0, global_vmax=1):
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
        #ax.set_xlim(-5, 5)
        #ax.set_ylim(-5, 5)

        # Ensure ticks and labels are shown
        ax.set_xlabel("PC1", fontsize=20)
        ax.set_ylabel("PC2", fontsize=20)
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.tick_params(axis='x', labelsize=16)  # Increase x-axis tick label size
        ax.tick_params(axis='y', labelsize=16)  # Increase y-axis tick label size
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        return scatter
    except Exception:
        ax.axis("off")

def plot_by_cluster(ax, df, cell_key, mutable_features, cluster_method, reducer, max_k=5):
    if df.empty or len(mutable_features) < 2 or df.shape[0] < 3:
        ax.axis("off")
        return

    try:
        # Extract and scale mutable features
        X = df[mutable_features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Edge case: all rows are identical after scaling
        if np.unique(X_scaled, axis=0).shape[0] == 1:
            ax.scatter([0], [0], c='blue', s=10)
            ax.set_title(f"Cell {cell_key} (Identical)")
            ax.axis('off')
            return

        # For plotting into 2D
        if reducer == "PCA":
            reducer_model = PCA(n_components=2)
            X_embedded = reducer_model.fit_transform(X)
            X_plotting = X_embedded
        elif reducer == "UMAP":
            reducer_model = UMAP(n_components=2, random_state=42, n_neighbors=5, min_dist=0.3)
            X_embedded = reducer_model.fit_transform(X)
            X_plotting = X_embedded
        else:
            X_embedded = X  # No dimensionality reduction
            reducer_model = PCA(n_components=2)
            X_plotting = reducer_model.fit_transform(X) #for plotting still reduce dimensionality

        # Clustering
        if cluster_method == "hierarchal":
            k, labels, Z = optimal_hclust_k(X, max_k=max_k)
        else:  # e.g., kmeans
            k, labels = optimal_clusters(X_embedded, max_k=max_k)
            labels = labels + 1


        # Scatter plot with cluster-based coloring
        scatter = ax.scatter(
            X_plotting[:, 0], X_plotting[:, 1],
            c=labels,
            cmap='tab10',
            alpha=0.7,
            s=60
        )
        # Add legend
        unique_clusters = np.unique(labels)
        handles = [
            plt.Line2D([], [], marker='o', linestyle='', 
                       color=scatter.cmap(scatter.norm(i)), 
                       label=f'Cluster {i}')
            for i in unique_clusters
        ]
        ''''''
        ax.legend(handles=handles, title='Cluster', 
                  bbox_to_anchor=(0.6, 1), loc='upper left', 
                  fontsize=12, title_fontsize=12)
        
        ax.set_xlabel("PC1", fontsize=20)
        ax.set_ylabel("PC2", fontsize=20)
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.tick_params(axis='x', labelsize=16)  # Increase x-axis tick label size
        ax.tick_params(axis='y', labelsize=16)  # Increase y-axis tick label size
        return scatter
    except Exception as e:
        ax.axis("off")
        print(f"Error in cell {cell_key}: {e}")



from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def tree_constraints_to_cluster_kmeans(ax, df, cell_key, mutable_features, max_k=5, max_depth=3):
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
        '''
        # Step 1: Cluster using PCA + KMeans
        pca = PCA(n_components=min(2, X.shape[1]))
        X_pca = pca.fit_transform(X_scaled)
        '''
        from cf_search.visualize import optimal_clusters
        k, labels = optimal_clusters(X, max_k=max_k) #use unscaled counterfactuals to preserve differences
        labels = labels+1
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
            fontsize=10,
            precision=2,
            label="root"  # This shows only the split condition and Gini index
        )

        #ax.set_title(f"Tree {cell_key} (acc={mean_accuracy:.2f})")

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
    umap_model = UMAP(n_components=2, random_state=42, n_neighbors=2, min_dist=0.2, metric="euclidean", n_jobs=1)
    X_umap = umap_model.fit_transform(X_scaled)

    # Hierarchical clustering
    #k, labels, Z = optimal_hclust_k(X_scaled, max_k=5)
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

from scipy.stats import f

def eta2_threshold_from_alpha(alpha, n_samples, n_clusters):
    df_between = n_clusters - 1
    df_within = n_samples - n_clusters
    if df_between <= 0 or df_within <= 0:
        return 1.0  # skip: can't compute
    f_crit = f.ppf(1 - alpha, df_between, df_within)
    eta2_thresh = (f_crit * df_between) / (f_crit * df_between + df_within)
    return eta2_thresh

def plot_cf_eta2_bar_cell(ax, df, cell_key, mutable_features, cluster_method, reducer, top_n=6, max_k=5, verbose=False):
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
        # For plotting into 2D
        if reducer == "PCA":
            reducer_model = PCA(n_components=2)
            X_embedded = reducer_model.fit_transform(X)
        elif reducer == "UMAP":
            reducer_model = UMAP(n_components=2, random_state=42, n_neighbors=5, min_dist=0.3)
            X_embedded = reducer_model.fit_transform(X)
        else:
            X_embedded = X  # No dimensionality reduction

        # Clustering
        if cluster_method == "hierarchal":
            k, labels, Z = optimal_hclust_k(X, max_k=max_k)
        else:  # e.g., kmeans
            k, labels = optimal_clusters(X_embedded, max_k=max_k)
            labels = labels + 1

        eta_dict = {}

        for feat in mutable_features:
            x = df[feat].values

            if np.var(x) < 1e-8:
                eta_dict[feat] = 0.0
                continue

            try:
                eta = eta_squared_1d(x, labels)
                if np.isnan(eta) or not np.isfinite(eta):
                    eta = 0.0
            except Exception as e:
                if verbose:
                    print(f"[WARN] {feat}: {e}")
                eta = 0.0

            eta_dict[feat] = eta
        n_samples = X.shape[0]
        n_clusters = len(np.unique(labels))
        eta_thresh = eta2_threshold_from_alpha(0.05, n_samples, n_clusters)

        # Draw the line
        ax.axvline(eta_thresh, color='red', linestyle='--', linewidth=1.5)

        sorted_items = sorted(eta_dict.items(), key=lambda x: -x[1])
        top_feats, top_etas = zip(*sorted_items[:top_n]) if any(eta_dict.values()) else ([], [])
        ax.set_xlabel('$\eta^2$', fontsize=20)
        ax.tick_params(axis='x', labelsize=20)  # Increase x-axis tick label size
        ax.tick_params(axis='y', labelsize=20)  # Increase y-axis tick label size
        if top_feats:
            ax.barh(top_feats[::-1], top_etas[::-1])
            ax.set_xlim(0, 1)
        else:
            ax.axis("off")

    except Exception as e:
        print(f"[ERROR] Cell {cell_key} → {e}")
        ax.axis("off")






def plot_cf_eta2_bar_constraints_cell(ax, df, cell_key, mutable_features, cluster_method, reducer, top_n=6, max_k=5, verbose=False):
    if df.empty or len(mutable_features) < 2:
        ax.axis("off")
        return

    X = df[mutable_features].values
    if X.shape[0] < 3:
        ax.axis("off")
        return
    
    # Constraints = non-mutable columns (not in mutable_features, not metadata columns)
    constraint_features = [
        col for col in df.columns 
        if col not in mutable_features and col not in ['cell', 'individual_id', 'fitness']
    ]
    if len(constraint_features) < 2:
        ax.axis("off")
        return
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    try:
        # For plotting into 2D
        if reducer == "PCA":
            reducer_model = PCA(n_components=2)
            X_embedded = reducer_model.fit_transform(X)
        elif reducer == "UMAP":
            reducer_model = UMAP(n_components=2, random_state=42, n_neighbors=5, min_dist=0.3)
            X_embedded = reducer_model.fit_transform(X)
        else:
            X_embedded = X  # No dimensionality reduction

        # Clustering
        if cluster_method == "hierarchal":
            k, labels, Z = optimal_hclust_k(X, max_k=max_k)
        else:  # e.g., kmeans
            k, labels = optimal_clusters(X_embedded, max_k=max_k)
            labels = labels + 1


        eta_dict = {}

        for feat in constraint_features:
            x = df[feat].values

            if np.var(x) < 1e-8:
                eta_dict[feat] = 0.0
                continue

            try:
                eta = eta_squared_1d(x, labels)
                if np.isnan(eta) or not np.isfinite(eta):
                    eta = 0.0
            except Exception as e:
                if verbose:
                    print(f"[WARN] {feat}: {e}")
                eta = 0.0

            eta_dict[feat] = eta
        
        n_samples = X.shape[0]
        n_clusters = len(np.unique(labels))
        eta_thresh = eta2_threshold_from_alpha(0.05, n_samples, n_clusters)

        # Draw the line
        ax.axvline(eta_thresh, color='red', linestyle='--', linewidth=1.5)

        sorted_items = sorted(eta_dict.items(), key=lambda x: -x[1])
        top_feats, top_etas = zip(*sorted_items[:top_n]) if any(eta_dict.values()) else ([], [])
        ax.tick_params(axis='x', labelsize=20)  # Increase x-axis tick label size
        ax.tick_params(axis='y', labelsize=20)  # Increase y-axis tick label size
        if top_feats:
            ax.barh(top_feats[::-1], top_etas[::-1])
            ax.set_xlim(0, 1)
        else:
            ax.axis("off")
        ax.set_xlabel('$\eta^2$', fontsize=20)
    except Exception as e:
        print(f"[ERROR] Cell {cell_key} → {e}")
        ax.axis("off")


def plot_cf_meanvalue_heatmap_cell(ax, df, cell_key, cluster_method, mutable_features, reducer, max_k=5):
    if df.empty or len(mutable_features) < 2:
        ax.axis("off")
        return

    X = df[mutable_features].values
    if X.shape[0] < 3:
        ax.axis("off")
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # For plotting into 2D
    if reducer == "PCA":
        reducer_model = PCA(n_components=2)
        X_embedded = reducer_model.fit_transform(X)
    elif reducer == "UMAP":
        reducer_model = UMAP(n_components=2, random_state=42, n_neighbors=5, min_dist=0.3)
        X_embedded = reducer_model.fit_transform(X)
    else:
        X_embedded = X  # No dimensionality reduction

    # Clustering
    if cluster_method == "hierarchal":
        k, labels, Z = optimal_hclust_k(X, max_k=max_k)
    else:  # e.g., kmeans
        k, labels = optimal_clusters(X_embedded, max_k=max_k)
        labels = labels + 1


    df_tmp = df.copy()
    df_tmp['cluster'] = labels
    mean_by_cluster = df_tmp.groupby('cluster')[mutable_features].mean()

    sns.heatmap(mean_by_cluster, cmap='coolwarm', center=0, annot=True, fmt=".2f", ax=ax)
    ax.tick_params(axis='x', labelsize=18)  # Increase x-axis tick label size
    ax.tick_params(axis='y', labelsize=18)  # Increase y-axis tick label size
    #ax.set_xlabel("Mutable features", fontsize=16)
    ax.set_ylabel("Cluster", fontsize=16)

    #ax.set_title(f"Mean Values per Cluster\nCell {cell_key}")

def plot_cf_meanvalue_heatmap_constraints_cell(ax, df, cell_key, cluster_method, mutable_features, reducer, max_k=5):
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
        # For plotting into 2D
        if reducer == "PCA":
            reducer_model = PCA(n_components=2)
            X_embedded = reducer_model.fit_transform(X)
        elif reducer == "UMAP":
            reducer_model = UMAP(n_components=2, random_state=42, n_neighbors=5, min_dist=0.3)
            X_embedded = reducer_model.fit_transform(X)
        else:
            X_embedded = X  # No dimensionality reduction

        # Clustering
        if cluster_method == "hierarchal":
            k, labels, Z = optimal_hclust_k(X, max_k=max_k)
        else:  # e.g., kmeans
            k, labels = optimal_clusters(X_embedded, max_k=max_k)
            labels = labels + 1

        # Add cluster label to df and compute mean constraints per cluster
        df_tmp = df.copy()
        df_tmp['cluster'] = labels
        mean_by_cluster = df_tmp.groupby('cluster')[constraint_features].mean()

        sns.heatmap(mean_by_cluster, cmap='coolwarm', center=0, annot=True, fmt=".2f", ax=ax)
        #ax.set_title(f"Constraint Means\nCell {cell_key}", fontsize=8)
        #ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.tick_params(axis='x', labelsize=20)  # Increase x-axis tick label size
        ax.tick_params(axis='y', labelsize=20)  # Increase y-axis tick label size
        #ax.set_xlabel("Constraint features", fontsize=16)
        ax.set_ylabel("Cluster", fontsize=16)

    except Exception:
        ax.axis("off")

def plot_cf_kde_cell(ax, df, cell_key, mutable_features, feature, max_k=5, min_variance=1e-4):
    if df.empty or feature not in df.columns:
        ax.axis("off")
        return

    if len(mutable_features) < 2 or df.shape[0] < 3:
        ax.axis("off")
        return

    # Cluster on mutable features
    X = df[mutable_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    try:
        k, labels, Z = optimal_clusters(X, max_k=max_k)
        df = df.copy()
        df['cluster'] = labels

        plotted = False

        for c in sorted(df['cluster'].unique()):
            cluster_values = df[df['cluster'] == c][feature]
            if cluster_values.var() <= min_variance:
                print(f"[Cell {cell_key}] Skipping Cluster {c} due to low variance: {cluster_values.var():.2e}")
                continue

            sns.kdeplot(cluster_values, label=f'Cluster {c}', fill=True, ax=ax, alpha=0.3)
            plotted = True

        if plotted:
            #ax.set_ylim(0, 20)
            ax.set_title(f"{feature} by Cluster\nCell {cell_key}")
            ax.legend(fontsize=6)
            print("Y max after plotting:", ax.get_ylim()[1])
        else:
            ax.axis("off")

    except Exception as e:
        print(f"[Cell {cell_key}] Error: {e}")
        ax.axis("off")





def optimal_clusters(X, max_k=5):
    """Determines the optimal number of clusters using silhouette score."""
    if X.shape[0] < 3:
        return 1, -1  # Not enough samples

    best_k = 2
    best_score = -1

    for k in range(2, min(max_k, X.shape[0]) + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels) if k > 1 else -1
        if score > best_score:
            best_score = score
            best_k = k
            best_labels=labels

    return best_k, best_labels


def optimal_hclust_k(X, max_k=5, method='ward', metric='euclidean'):
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


