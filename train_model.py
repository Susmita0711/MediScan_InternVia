# train_model.py
"""Synthetic dataset generation, model training, evaluation, and serialization for MediScan.

This script creates a reproducible synthetic symptom‑disease dataset, trains three classifiers,
selects the best based on weighted F1‑score, and saves the chosen model along with supporting
metadata required by the Streamlit app.
"""

import os
import json
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import joblib

# ---------------------------
# Configuration
# ---------------------------
RANDOM_STATE = 42
N_SAMPLES = 1500
SYMPTOMS = [
    "Fever",
    "Cough",
    "Headache",
    "Sore Throat",
    "Runny Nose",
    "Sneezing",
    "Fatigue",
    "Body Ache",
    "Chills",
    "Nausea",
    "Vomiting",
    "Diarrhea",
    "Loss of Appetite",
    "Nasal Congestion",
    "Muscle Pain",
    "Weakness",
    "Dizziness",
    "Shortness of Breath",
]
# Simple condition list for the synthetic data
CONDITIONS = [
    "Common Cold",
    "Influenza",
    "COVID-19",
    "Strep Throat",
    "Allergic Rhinitis",
]

OUTPUT_DIR = "models"
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.joblib")
FEATURES_PATH = os.path.join(OUTPUT_DIR, "feature_names.json")
LABEL_ENC_PATH = os.path.join(OUTPUT_DIR, "label_encoder.joblib")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# Synthetic data generation
# ---------------------------
np.random.seed(RANDOM_STATE)

def generate_sample():
    """Generate a single synthetic record.
    The symptom pattern is loosely based on typical disease presentations.
    Returns a tuple (features list, condition string).
    """
    condition = np.random.choice(CONDITIONS)
    # Symptom likelihood matrix (condition x symptom)
    likelihood = {
        "Common Cold":        [0.6, 0.7, 0.4, 0.3, 0.8, 0.6, 0.5, 0.3, 0.2, 0.1, 0.05, 0.05, 0.2, 0.7, 0.2, 0.3, 0.1, 0.05],
        "Influenza":          [0.9, 0.6, 0.7, 0.4, 0.4, 0.2, 0.8, 0.6, 0.7, 0.3, 0.2, 0.15, 0.5, 0.3, 0.6, 0.5, 0.2, 0.2],
        "COVID-19":           [0.85, 0.65, 0.5, 0.2, 0.3, 0.15, 0.75, 0.5, 0.7, 0.25, 0.1, 0.2, 0.4, 0.25, 0.55, 0.4, 0.25, 0.5],
        "Strep Throat":       [0.7, 0.4, 0.6, 0.8, 0.2, 0.1, 0.5, 0.3, 0.3, 0.15, 0.05, 0.05, 0.3, 0.15, 0.25, 0.2, 0.1, 0.05],
        "Allergic Rhinitis":  [0.2, 0.3, 0.2, 0.1, 0.9, 0.85, 0.3, 0.2, 0.1, 0.05, 0.02, 0.03, 0.15, 0.95, 0.2, 0.1, 0.05, 0.02],
    }
    probs = likelihood[condition]
    symptoms = [1 if np.random.rand() < p else 0 for p in probs]
    return symptoms, condition

# Generate data
features_list = []
labels = []
for _ in range(N_SAMPLES):
    feats, lab = generate_sample()
    features_list.append(feats)
    labels.append(lab)

X = np.array(features_list, dtype=int)
Y = np.array(labels)

# Encode target labels
le = LabelEncoder()
Y_enc = le.fit_transform(Y)

# Train / test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y_enc,
    test_size=0.2,
    stratify=Y_enc,
    random_state=RANDOM_STATE,
)

# ---------------------------
# Model training & evaluation
# ---------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "Multinomial NB": MultinomialNB(),
}

results = {}
for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    results[name] = {
        "model": clf,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "y_pred": y_pred,
    }

# Choose best model by weighted F1‑score
best_name = max(results, key=lambda k: results[k]["f1"])
best_model = results[best_name]["model"]

# Save artifacts
joblib.dump(best_model, MODEL_PATH)
joblib.dump(le, LABEL_ENC_PATH)
with open(FEATURES_PATH, "w", encoding="utf-8") as f:
    json.dump(SYMPTOMS, f, ensure_ascii=False, indent=2)

# Compile metrics for the dashboard
metrics = {
    "dataset_samples": int(len(X)),
    "symptom_features": len(SYMPTOMS),
    "condition_classes": len(CONDITIONS),
    "selected_model": best_name,
    "metrics_per_model": {
        name: {
            "accuracy": round(info["accuracy"], 3),
            "precision": round(info["precision"], 3),
            "recall": round(info["recall"], 3),
            "f1": round(info["f1"], 3),
        }
        for name, info in results.items()
    },
    "confusion_matrix": confusion_matrix(y_test, results[best_name]["y_pred"]).tolist(),
    "classification_report": classification_report(
        y_test,
        results[best_name]["y_pred"],
        target_names=le.classes_,
        output_dict=True,
    ),
    "feature_importance": (
        best_model.feature_importances_.tolist()
        if hasattr(best_model, "feature_importances_")
        else None
    ),
    "synthetic_dataset": True,
    "random_state": RANDOM_STATE,
}

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print("Training complete. Model saved to", MODEL_PATH)
print("Metrics saved to", METRICS_PATH)
