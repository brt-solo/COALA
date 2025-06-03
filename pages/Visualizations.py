import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import io
import matplotlib.pyplot as plt
from cf_search.visualize import (
    extract_cell_data,
    plot_single_pca,
    plot_fitness_heatmap,
    plot_all_pcas_grid,
    plot_all_pcas_grid_by_cluster,
    plot_all_clusters_by_heatmap,
    plot_all_mutable_clusters_by_heatmap,
    plot_cf_clustermap_cell,
    plot_cf_umap_cell,
    plot_cf_umap_cell_clusters,
    plot_cf_eta2_bar_cell,
    plot_cf_meanvalue_heatmap_cell,
    plot_cf_kde_cell,
    plot_cellwise_grid,
    plot_cf_meanvalue_heatmap_constraints_cell,
    plot_cf_eta2_bar_constraints_cell,
    eta2_bar_constraints_cell_pca_kmeans,
    eta2_bar_cell_pca_kmeans,
    tree_constraints_to_cluster_cell_pca_kmeans,
    tree_constraints_to_hclust_cell

)

st.markdown("---")
st.header("Phase 2: Visualize Counterfactuals")

st.subheader("Load Archive for Visualization")

use_generated = st.checkbox("Use archive just generated", value=True)

if use_generated and 'archive_dict' not in locals():
    st.warning("No archive has been generated yet in this session.")
    use_generated = False

archive_data = None

if use_generated:
    archive_data = archive_data
else:
    uploaded_pkl = st.file_uploader("Upload your own .pkl archive", type="pkl", key="archive_upload")
    if uploaded_pkl:
        archive_data = pickle.load(uploaded_pkl)

if 'feature_categories' not in locals() or not use_generated:
    st.subheader("🧬pload Feature Categories (.json)")
    uploaded_categories = st.file_uploader("Upload feature_categories.json", type="json", key="feature_cat")
    
    if uploaded_categories:
        feature_categories = json.load(uploaded_categories)
        category_labels = list(feature_categories.keys())
        num_categories = len(category_labels)

        cell_feature_sets = {}
        for i in range(num_categories):
            for j in range(num_categories):
                features = feature_categories[category_labels[i]].copy()
                if i != j:
                    features += feature_categories[category_labels[j]]
                cell_feature_sets[(i, j)] = features

if 'feature_categories' not in locals():
    st.warning("Feature categories are required for visualization.")
    st.stop()

category_labels = list(feature_categories.keys())
num_categories = len(category_labels)


if archive_data:
    st.subheader("Select Cell to Visualize")
    
    i = st.number_input("Cell row index (i)", min_value=0, max_value=num_categories - 1, value=0)
    j = st.number_input("Cell column index (j)", min_value=0, max_value=num_categories - 1, value=1)
    selected_cell = (i, j)

    if st.button("Extract Data for Cell"):
        cell_df = extract_cell_data(archive_data, cell_index=selected_cell)

        if cell_df.empty:
            st.warning("⚠️ No counterfactuals found in the selected cell.")
        else:
            st.success(f"Found {len(cell_df)} counterfactuals.")
            st.dataframe(cell_df)

            csv = cell_df.to_csv(index=False).encode()
            st.download_button("⬇ Download Cell Data as CSV", csv, file_name=f"cell_{i}_{j}_data.csv")



st.subheader("PCA Pipeline")

if archive_data and 'feature_categories' in locals():
    # === 1. Upload optional reference for PCA coloring ===
    def clean_csv(uploaded_file):
        df = pd.read_csv(uploaded_file)
        return df.drop(columns="Unnamed: 0") if "Unnamed: 0" in df.columns else df

    ref_for_pca = None
    uploaded_ref = st.file_uploader("(Optional) Upload reference individuals file for coloring", type="csv", key="pca_ref")

    if uploaded_ref:
        ref_for_pca = clean_csv(uploaded_ref)
        st.success("Uploaded reference file will be used for coloring.")
    elif 'ref_df' in locals():
        ref_for_pca = ref_df
        st.info("ℹ️ Using previously uploaded reference individuals for coloring.")
    else:
        st.warning("⚠️ No reference individuals uploaded. Cannot color PCA plots.")

    # === 2. Upload predicted fitness ===
    yval_series = None
    yval_file = st.file_uploader("(Optional) Upload predicted fitness (y_val.csv)", type="csv")

    if yval_file:
        yval_df = clean_csv(yval_file)
        yval_series = yval_df.iloc[:, 0]
        st.success("Loaded predicted fitness values.")

        # Add predicted fitness to archive
        for idx, df in archive_data.items():
            if len(df) > 0 and idx < len(yval_series):
                archive_data[idx]["predicted_fitness"] = yval_series.iloc[idx]

    # === 3. Configure color feature ===
    if ref_for_pca is not None:
        color_options = ["fitness"]
        if yval_series is not None:
            color_options.append("predicted_fitness")
        color_options += list(ref_for_pca.columns)

        color_feature = st.selectbox("Feature to color by", options=color_options)

        # === 4. Plot selection ===
        plot_type = st.selectbox("Choose plot type", [
            "PCA",
            "eta2 (mutable)",
            "eta2 (constraints)",
            "Tree (constraints → cluster)"
        ])
        plot_now = st.button("Generate Grid Plot", key=f"generate_{plot_type}")

        if plot_now:
            with st.spinner("Generating plot..."):
                from cf_search.visualize import (
                    plot_cellwise_grid,
                    plot_single_pca,
                    eta2_bar_cell_pca_kmeans,
                    eta2_bar_constraints_cell_pca_kmeans,
                    tree_constraints_to_cluster_cell_pca_kmeans
                )

                fig = None  # Ensure fig is defined
                if plot_type == "PCA":
                    # Compute global min/max for coloring
                    all_vals = []
                    for cell_key in cell_feature_sets:
                        df = extract_cell_data(archive_data, cell_key)
                        if color_feature in df.columns:
                            all_vals.extend(df[color_feature].dropna().tolist())

                    if not all_vals:
                        st.warning(f"No values found for `{color_feature}`.")
                        st.stop()

                    global_vmin, global_vmax = min(all_vals), max(all_vals)

                    fig = plot_cellwise_grid(
                        archive_data,
                        cell_feature_sets,
                        feature_categories,
                        plot_single_pca,
                        plot_type_name="PCA of Counterfactuals",
                        color_feature=color_feature,
                        global_vmin=global_vmin,
                        global_vmax=global_vmax,
                    )

                elif plot_type == "eta2 (mutable)":
                    fig = plot_cellwise_grid(
                        archive_data,
                        cell_feature_sets,
                        feature_categories,
                        eta2_bar_cell_pca_kmeans,
                        plot_type_name="η² for Mutable Features"
                    )

                elif plot_type == "eta2 (constraints)":
                    fig = plot_cellwise_grid(
                        archive_data,
                        cell_feature_sets,
                        feature_categories,
                        eta2_bar_constraints_cell_pca_kmeans,
                        plot_type_name="η² for Constraint Features"
                    )
                elif plot_type == "Tree (constraints → cluster)":
                    # Extract mutable features from feature_categories
                    fig = plot_cellwise_grid(
                        archive_data,
                        cell_feature_sets,
                        feature_categories,
                        tree_constraints_to_cluster_cell_pca_kmeans,
                        plot_type_name="Decision Trees (Constraints → Cluster)"
                    )
                st.session_state.pca_grid_fig = fig
                # Option to download as high-res PNG
                png_buffer = io.BytesIO()
                st.session_state.pca_grid_fig.savefig(
                    png_buffer,
                    format="png",
                    dpi=1000,  # High resolution
                    bbox_inches='tight'
                )
                png_buffer.seek(0)
                st.download_button(
                    label="⬇ Download PCA Grid as PNG",
                    data=png_buffer,
                    file_name="pca_grid_fig.png",
                    mime="image/png"
                )

                if fig:
                    st.pyplot(fig)
                else:
                    st.warning("⚠️ No figure was generated.")


def comment():
        
    '''

    if archive_data and 'feature_categories' in locals():
        # Upload optional reference for coloring
        ref_for_pca = None

        uploaded_ref = st.file_uploader("(Optional) Upload reference individuals file for coloring", type="csv", key="pca_ref")
        if uploaded_ref:
            ref_for_pca = pd.read_csv(uploaded_ref)
            if "Unnamed: 0" in ref_for_pca.columns:
                ref_for_pca = ref_for_pca.drop(columns="Unnamed: 0")
            st.success("Uploaded reference file will be used for coloring.")
        elif 'ref_df' in locals():
            ref_for_pca = ref_df
            st.info("ℹ️ Using previously uploaded reference individuals for coloring.")
        else:
            st.warning("⚠️ No reference individuals uploaded. Cannot color PCA plots.")
            ref_for_pca = None

        yval_file = st.file_uploader("(Optional) Upload predicted fitness (y_val.csv)", type="csv")
        if yval_file:
            yval_df = pd.read_csv(yval_file)
            if "Unnamed: 0" in yval_df.columns:
                yval_df = yval_df.drop(columns="Unnamed: 0")
            yval_series = yval_df.iloc[:, 0]  # first column
            st.success("Loaded predicted fitness values.")
        else:
            yval_series = None
        if yval_series is not None:
            for idx, df in archive_data.items():
                if len(df) > 0 and idx < len(yval_series):
                    archive_data[idx]["predicted_fitness"] = yval_series.iloc[idx]

        if ref_for_pca is not None:
            available_color_features = ["fitness"]
            if yval_series is not None:
                available_color_features.append("predicted_fitness")
            available_color_features += list(ref_for_pca.columns)

            color_feature = st.selectbox("Feature to color by", options=available_color_features)

            if "pca_grid_fig" not in st.session_state:
                st.session_state.pca_grid_fig = None

            if st.button("Generate PCA Grid"):
                with st.spinner("Running PCA for each cell..."):
                    fig, _ = plot_all_pcas_grid(
                        archive_dict=archive_data,
                        cell_feature_sets=cell_feature_sets,
                        feature_categories=feature_categories,
                        color_feature=color_feature,
                        n_components=2,
                        show=False
                    )
                    st.session_state.pca_grid_fig = fig
                    # Option to download as high-res PNG
                    png_buffer = io.BytesIO()
                    st.session_state.pca_grid_fig.savefig(
                        png_buffer,
                        format="png",
                        dpi=1000,  # High resolution
                        bbox_inches='tight'
                    )
                    png_buffer.seek(0)
                    st.download_button(
                        label="⬇ Download PCA Grid as PNG",
                        data=png_buffer,
                        file_name="pca_grid_fig.png",
                        mime="image/png"
                    )

            if st.session_state.pca_grid_fig:
                st.pyplot(st.session_state.pca_grid_fig)
    '''
    pass




st.subheader("PCA Grid with Clustering")
max_k = st.slider("Max number of clusters", min_value=2, max_value=10, value=5)

if archive_data and 'feature_categories' in locals():
    if "cluster_grid_fig" not in st.session_state:
        st.session_state.cluster_grid_fig = None

    if st.button("Generate PCA Grid with Clustering"):
        with st.spinner("Running PCA + K-Means for each cell..."):
            fig, _ = plot_all_pcas_grid_by_cluster(
                archive_dict=archive_data,
                cell_feature_sets=cell_feature_sets,
                max_clusters=max_k,
                n_components=2,
                show=False
            )
            st.session_state.cluster_grid_fig = fig
    if st.session_state.cluster_grid_fig:
        st.pyplot(st.session_state.cluster_grid_fig)

st.subheader("Posthoc Heatmap Grid ")

if archive_data and 'feature_categories' in locals():
    
    if "heatmap_constraints_fig" not in st.session_state:
        st.session_state.heatmap_constraints_fig = None
    if st.button("Run Heatmap Grid with Posthoc Tests (Constraints)"):
        with st.spinner("Running clustering + posthoc testing..."):

            fig = plot_all_clusters_by_heatmap(
                archive_dict=archive_data,
                cell_feature_sets=cell_feature_sets,
                feature_categories=feature_categories,
                max_clusters=max_k,
                show=False
            )
            st.session_state.heatmap_constraints_fig = fig

            # Option to download as high-res PNG
            png_buffer = io.BytesIO()
            st.session_state.heatmap_constraints_fig.savefig(
                png_buffer,
                format="png",
                dpi=600,  # High resolution
                bbox_inches='tight'
            )
            png_buffer.seek(0)
            st.download_button(
                label="⬇ Download Heatmap (Constraints) as PNG",
                data=png_buffer,
                file_name="heatmap_constraints.png",
                mime="image/png"
            )

    if st.session_state.heatmap_constraints_fig:
        st.pyplot(st.session_state.heatmap_constraints_fig)




    if "heatmap_mutable_fig" not in st.session_state:
        st.session_state.heatmap_mutable_fig = None
    if st.button("Run Heatmap Grid with Posthoc Tests (Mutable)"):
        with st.spinner("Running clustering + posthoc testing..."):
            fig = plot_all_mutable_clusters_by_heatmap(
                archive_dict=archive_data,
                cell_feature_sets=cell_feature_sets,
                max_clusters=max_k,
                feature_categories=feature_categories,
                n_components=2,
                show=False
            )
            st.session_state.heatmap_mutable_fig = fig

            # Option to download as high-res PNG
            png_buffer = io.BytesIO()
            st.session_state.heatmap_mutable_fig.savefig(
                png_buffer,
                format="png",
                dpi=600,  # High resolution
                bbox_inches='tight'
            )
            png_buffer.seek(0)
            st.download_button(
                label="⬇ Download Heatmap (Mtuable) as PNG",
                data=png_buffer,
                file_name="heatmap_mutable.png",
                mime="image/png"
            )
    if st.session_state.heatmap_mutable_fig:
        st.pyplot(st.session_state.heatmap_mutable_fig)


st.subheader("📊 Per-Cell Hierarchical Clustering Visualizations")

plot_type = st.selectbox("Choose plot type", [
    "Clustermap",
    "UMAP",
    "UMAP (clustered)",
    "η² barplot",
    "η² barplot (constraints)",
    "Mean value heatmap (mutable)",
    "Mean value heatmap (constraints)",
    "KDE by feature",
    "Tree (constraints → HClust)"
])

if plot_type == "KDE by feature":
    selected_feature = st.selectbox("Choose feature for KDE", options=[f for group in feature_categories.values() for f in group])

plot_now = st.button("Generate Grid Plot")



if plot_now:
    with st.spinner("Generating plot..."):
        from cf_search.visualize import (
            plot_cellwise_grid,
            plot_cf_clustermap_cell,
            plot_cf_umap_cell,
            plot_cf_umap_cell_clusters,
            plot_cf_eta2_bar_cell,
            plot_cf_eta2_bar_constraints_cell,
            plot_cf_meanvalue_heatmap_cell, 
            plot_cf_meanvalue_heatmap_constraints_cell,
            plot_cf_kde_cell,
            tree_constraints_to_hclust_cell
        )

        if plot_type == "Clustermap":
            fig = plot_cellwise_grid(archive_data, cell_feature_sets, feature_categories, plot_cf_clustermap_cell, "Clustermap Grid")
        elif plot_type == "UMAP":
            fig = plot_cellwise_grid(archive_data, cell_feature_sets, feature_categories, plot_cf_umap_cell, "UMAP", color_feature=color_feature)
        elif plot_type == "UMAP (clustered)":
            fig = plot_cellwise_grid(archive_data, cell_feature_sets, feature_categories, plot_cf_umap_cell_clusters, "UMAP by Cluster")
        elif plot_type == "η² barplot":
            fig = plot_cellwise_grid(archive_data, cell_feature_sets, feature_categories, plot_cf_eta2_bar_cell, "Top η² Features", max_k=max_k)
        elif plot_type == "η² barplot (constraints)":
            fig = plot_cellwise_grid(archive_data, cell_feature_sets, feature_categories, plot_cf_eta2_bar_constraints_cell, "Top η² Features", max_k=max_k)
        elif plot_type == "Mean value heatmap (mutable)":
            fig = plot_cellwise_grid(archive_data, cell_feature_sets, feature_categories, plot_cf_meanvalue_heatmap_cell, "Mean Values per Cluster", max_k=max_k)
        elif plot_type == "Mean value heatmap (constraints)":
            fig = plot_cellwise_grid(archive_data, cell_feature_sets,  feature_categories, plot_cf_meanvalue_heatmap_constraints_cell, "Constraint Mean Heatmaps", max_k=max_k)
        elif plot_type == "KDE by feature":
            fig = plot_cellwise_grid(archive_data, cell_feature_sets, feature_categories, plot_cf_kde_cell, f"KDE of {selected_feature}", feature=selected_feature)
        elif plot_type == "Tree (constraints → HClust)":
            fig = plot_cellwise_grid(
                archive_data,
                cell_feature_sets,
                feature_categories, 
                tree_constraints_to_hclust_cell,
                plot_type_name="Decision Trees (Constraints → Hierarchical Cluster)"
            )
        st.session_state.hierarchal_clustering = fig
        # Option to download as high-res PNG
        png_buffer = io.BytesIO()
        st.session_state.hierarchal_clustering.savefig(
            png_buffer,
            format="png",
            dpi=1000,  # High resolution
            bbox_inches='tight'
        )
        png_buffer.seek(0)
        st.download_button(
            label="⬇ Download Cluster Grid as PNG",
            data=png_buffer,
            file_name="hierarchal_clustering.png",
            mime="image/png"
        )
        st.pyplot(fig)


# =====================
# 📊 Correlation Heatmaps Section
# =====================
st.markdown("## 🔗 Feature Correlation Heatmaps (per MAP-Elites cell)")

plot_corr = st.button("Generate Correlation Heatmaps")

if plot_corr:
    with st.spinner("Computing cellwise Pearson correlations..."):
        from cf_search.visualize import correlation_heatmap_cell, plot_cellwise_grid

        fig = plot_cellwise_grid(
            archive_data,
            cell_feature_sets,
            feature_categories,
            correlation_heatmap_cell,
            plot_type_name="Feature Correlation Heatmaps",
            figsize_per_cell=(2, 2)  # adjust for clarity
        )
        st.pyplot(fig)

st.markdown("## 🔁 Mutable–Constraint Correlation Heatmaps")
if st.button("Generate Cross-Correlation Heatmaps"):
    with st.spinner("Generating..."):
        from cf_search.visualize import correlation_mutable_vs_constraint_cell
        fig = plot_cellwise_grid(
            archive_data,
            cell_feature_sets,
            feature_categories,
            correlation_mutable_vs_constraint_cell,
            plot_type_name="Mutable vs Constraint Correlation",
            figsize_per_cell=(3, 2.5)
        )
        st.pyplot(fig)
