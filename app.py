# app.py
"""MediScan Streamlit application.

Provides:
- Home page with branding and navigation
- Symptom Checker (model inference)
- Prediction History (session state)
- Model Insights (metrics visualisation)
- About page

All UI follows the branding and disclaimer rules defined in the project brief.
"""

import json
import os
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Paths to artifacts
# ---------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.joblib")
LABEL_ENC_PATH = os.path.join(BASE_DIR, "models", "label_encoder.joblib")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_names.json")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")

# Load artifacts (cached)
@st.cache_resource

def load_artifacts():
    model = joblib.load(MODEL_PATH)
    le = joblib.load(LABEL_ENC_PATH)
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        symptoms = json.load(f)
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return model, le, symptoms, metrics

model, label_enc, SYMPTOMS, METRICS = load_artifacts()

# ---------------------------
# Session state helpers
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------
# Page layout helpers
# ---------------------------
def set_page_config(page_title: str, layout: str = "centered"):
    st.set_page_config(page_title=page_title, layout=layout, initial_sidebar_state="expanded")

# Branding constants
BRAND_NAME = "MediScan"
TAGLINE = "Smarter Health Insights, Powered by Machine Learning."
DISCLAIMER = (
    "**MediScan is an educational Machine Learning demonstration and is not a medical diagnostic tool.**\n"
    "Its predictions may be inaccurate and must not replace professional medical advice. "
    "Do not use this application to self‑diagnose, delay treatment, or make medication decisions."
)

# ---------------------------
# Navigation sidebar
# ---------------------------
set_page_config("MediScan")
st.sidebar.title(BRAND_NAME)
st.sidebar.markdown(f"*{TAGLINE}*")
page = st.sidebar.radio("Navigation", ["Home", "Symptom Checker", "Prediction History", "Model Insights", "About MediScan"])  # noqa: E501

# ---------------------------
# Home Page
# ---------------------------
if page == "Home":
    st.title(BRAND_NAME)
    st.header(TAGLINE)
    st.image("https://raw.githubusercontent.com/google/logo/main/Google_2015_logo.svg", width=200)  # placeholder logo
    st.markdown("---")
    st.subheader("Welcome to MediScan")
    st.write(
        "MediScan provides an interactive demonstration of how Machine Learning can be used to "
        "analyze common health symptoms and suggest possible conditions. This tool is intended for "
        "educational purposes only and should never be used as a substitute for professional medical advice."
    )
    st.markdown("**How it works**")
    st.markdown(
        "1️⃣ Select your symptoms.\n"
        "2️⃣ The ML model analyses the selected features.\n"
        "3️⃣ View preliminary prediction insights."
    )
    st.markdown("---")
    cols = st.columns(4)
    features = ["Intelligent Symptom Analysis", "Machine Learning Predictions", "Model Performance Insights", "Educational Health Information"]
    icons = ["🧠", "🤖", "📊", "📚"]
    for c, f, ic in zip(cols, features, icons):
        c.markdown(f"{ic} **{f}**")
    st.markdown("---")
    st.markdown(DISCLAIMER)
    if st.button("Start Symptom Check"):
        st.experimental_rerun()
        # Switching to Symptom Checker page on next render
        st.session_state['__page'] = "Symptom Checker"

# ---------------------------
# Symptom Checker Page
# ---------------------------
elif page == "Symptom Checker":
    st.title("Symptom Checker")
    st.markdown(DISCLAIMER)
    selected = st.multiselect("Select your symptoms", options=SYMPTOMS)
    if st.button("Clear All"):
        selected = []
        st.experimental_rerun()
    if st.button("Predict"):
        if not selected:
            st.error("Please select at least one symptom before predicting.")
        else:
            # Build feature vector
            input_vec = np.array([1 if s in selected else 0 for s in SYMPTOMS]).reshape(1, -1)
            pred_idx = model.predict(input_vec)[0]
            pred_label = label_enc.inverse_transform([pred_idx])[0]
            # Probability / score (if model supports predict_proba)
            score = None
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_vec)[0][pred_idx]
                score = float(prob)
            # Save to history
            st.session_state.history.append({
                "timestamp": datetime.datetime.now().isoformat(timespec='seconds'),
                "symptoms": selected,
                "condition": pred_label,
                "score": score,
            })
            # Display result
            st.subheader("Prediction Result")
            st.success(f"**Possible condition:** {pred_label}")
            if score is not None:
                st.info(f"Model confidence (estimated probability): {score:.2%}")
            st.markdown("---")
            st.markdown("**Selected symptoms:**")
            st.write(", ".join(selected))
            st.markdown("---")
            st.markdown(
                "*This is a preliminary insight generated by a machine‑learning model trained on a synthetic dataset. "
                "It does not constitute a medical diagnosis. If you are experiencing severe or worsening symptoms, "
                "please contact a qualified healthcare professional or local emergency services.*"
            )
            if st.button("Another Check"):
                st.experimental_rerun()

# ---------------------------
# Prediction History Page
# ---------------------------
elif page == "Prediction History":
    st.title("Prediction History (Session)")
    st.markdown(DISCLAIMER)
    if not st.session_state.history:
        st.info("No predictions made in this session yet.")
    else:
        for i, entry in enumerate(reversed(st.session_state.history), 1):
            st.markdown(f"**#{i} – {entry['timestamp']}**")
            st.write(f"*Symptoms:* {', '.join(entry['symptoms'])}")
            st.write(f"*Predicted condition:* {entry['condition']}")
            if entry['score'] is not None:
                st.write(f"*Model confidence:* {entry['score']:.2%}")
            st.markdown("---")
    if st.button("Clear History"):
        st.session_state.history = []
        st.experimental_rerun()

# ---------------------------
# Model Insights Page
# ---------------------------
elif page == "Model Insights":
    st.title("Model Insights")
    st.markdown(DISCLAIMER)
    st.subheader("Dataset Overview")
    st.write(f"Number of samples: {METRICS['dataset_samples']}")
    st.write(f"Number of symptom features: {METRICS['symptom_features']}")
    st.write(f"Number of condition classes: {METRICS['condition_classes']}")
    st.write(f"Synthetic dataset: {METRICS.get('synthetic_dataset', False)}")
    st.subheader("Model Comparison")
    comp_df = pd.DataFrame(METRICS['metrics_per_model']).T
    comp_df = comp_df.reset_index().rename(columns={"index": "Model"})
    st.table(comp_df)
    st.subheader("Selected Model: " + METRICS['selected_model'])
    # Confusion matrix heatmap
    cm = np.array(METRICS['confusion_matrix'])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_enc.classes_, yticklabels=label_enc.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
    # Feature importance (if available)
    if METRICS.get('feature_importance'):
        imp = METRICS['feature_importance']
        imp_df = pd.DataFrame({"Symptom": SYMPTOMS, "Importance": imp})
        imp_df = imp_df.sort_values(by="Importance", ascending=False)
        st.subheader("Feature Importance (Top 10)")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.barplot(x="Importance", y="Symptom", data=imp_df.head(10), ax=ax2, palette="viridis")
        ax2.set_xlabel("Importance")
        ax2.set_ylabel("")
        st.pyplot(fig2)
    else:
        st.info("Feature importance is not applicable for the selected model.")

# ---------------------------
# About MediScan Page
# ---------------------------
elif page == "About MediScan":
    st.title("About MediScan")
    st.markdown(DISCLAIMER)
    st.subheader("Project Overview")
    st.write(
        "**Project Name:** MediScan\n"
        "**Organization:** MediScan Diagnostics\n"
        "**Purpose:** Demonstrate how Machine Learning can be employed to provide early health insights "
        "based on common symptoms. This educational prototype showcases data preparation, model training, "
        "evaluation, and a user‑friendly interface."
    )
    st.subheader("Technology Stack")
    st.write(
        "- Python 3\n"
        "- Pandas, NumPy\n"
        "- Scikit‑learn (Decision Tree, Random Forest, Multinomial NB)\n"
        "- Streamlit for the web UI\n"
        "- Matplotlib & Seaborn for visualisation\n"
        "- Joblib for model persistence"
    )
    st.subheader("Data Disclaimer")
    st.write(
        "The dataset used to train the models is synthetic and created solely for educational purposes. "
        "It does **not** reflect real patient data, and the model performance metrics are illustrative only."
    )
    st.subheader("Contact & License")
    st.write(
        "This project is open‑source and may be shared under the MIT License. For any questions, "
        "please reach out to the MediScan Diagnostics team."
    )

