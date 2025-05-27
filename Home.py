import streamlit as st




st.set_page_config(layout="centered")

st.title("Welcome to the MAP-Elites Counterfactual Explorer")

st.markdown("""
Welcome! This app allows you to:
- Generate counterfactuals using MAP-Elites
- Visualize those counterfactuals across feature categories
- Explore PCA plots, clustering, and posthoc significance tests

### Use the sidebar to get started:
- Go to **Counterfactuals** to generate new results
- Then head to **Visualizations** to explore the output

You can upload your own model, data, and reference individuals on each page.

---

Built using Streamlit · Developed by Bryant Han
""")
