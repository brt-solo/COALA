import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import io
from cf_search.map import mapcf_instance  # Your MAP-Elites class


st.set_page_config(layout="wide")
'''
st.markdown(
    """
    <style>
    .stApp > header div[data-testid="stStatusWidget"] {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)
'''

st.title("MAP-Elites Counterfactual Explorer")

if "figures" not in st.session_state:
    st.session_state.figures = []

# ─────────────────────────────────────────────────────────────────────
# 1. Upload all required files
# ─────────────────────────────────────────────────────────────────────
model_file = st.file_uploader("Upload trained model (.pkl)", type="pkl")
train_df_file = st.file_uploader("Upload training DataFrame (.csv)", type="csv")
feature_json_file = st.file_uploader("Upload feature categories (.json)", type="json")
reference_file = st.file_uploader("Upload reference individual(s) (.csv)", type="csv")

if model_file and train_df_file and feature_json_file and reference_file:
    # Load model and data
    model = pickle.load(model_file)
    X_train_df = pd.read_csv(train_df_file)
    if "Unnamed: 0" in X_train_df.columns:
        X_train_df = X_train_df.drop(columns="Unnamed: 0")

    feature_categories = json.load(feature_json_file)

    ref_df = pd.read_csv(reference_file)
    if "Unnamed: 0" in ref_df.columns:
        ref_df = ref_df.drop(columns="Unnamed: 0")

    st.subheader("Uploaded Reference Data")
    st.dataframe(ref_df)
    st.write(f"✅ Expected number of columns: {len(X_train_df.columns)}")
    st.write(f"✅ Column names: {list(X_train_df.columns)}")

    if not all(col in ref_df.columns for col in X_train_df.columns):
        missing = [col for col in X_train_df.columns if col not in ref_df.columns]
        st.error(f"Reference file is missing required columns: {missing}")
        st.stop()

    # ─────────────────────────────────────────────────────────────
    # 2. Set up MAP-Elites config
    # ─────────────────────────────────────────────────────────────
    params = {
        "min": np.array(X_train_df.min()),
        "max": np.array(X_train_df.max()),
        "random_init_batch": 5000
    }
    category_labels = list(feature_categories.keys())
    num_categories = len(category_labels)

    # Create symmetric cell_feature_sets
    cell_feature_sets = {}
    for i in range(num_categories):
        for j in range(num_categories):
            features = feature_categories[category_labels[i]].copy()
            if i != j:
                features += feature_categories[category_labels[j]]
            cell_feature_sets[(i, j)] = features

    # ─────────────────────────────────────────────────────────────
    # 3. Run counterfactual generation for all rows
    # ─────────────────────────────────────────────────────────────
    if st.button("Generate Counterfactuals for All Individuals"):
        archive_dict = {}
        progress_bar = st.progress(0)
        total = len(ref_df)

        for idx, row in ref_df[X_train_df.columns].iterrows():
            st.write(f"🔄 Processing individual index: {idx}")
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
                wrapper=model
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
            progress_bar.progress((idx + 1) / total)

        # Save archive to an in-memory pickle file
        buffer = io.BytesIO()
        pickle.dump(archive_dict, buffer)
        buffer.seek(0)

        st.success("Counterfactuals generated for all individuals!")
        st.download_button("⬇ Download .pkl Archive", buffer, file_name="counterfactuals_archive.pkl")

else:
    st.info("📂 Please upload all required files to begin.")


