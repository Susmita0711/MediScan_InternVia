# MediScan

**Project Name:** MediScan
**Organization:** MediScan Diagnostics
**Tagline:** Smarter Health Insights, Powered by Machine Learning.

---

## 🎯 Project Overview
MediScan is an **educational prototype** that showcases how Machine Learning can be applied to provide early health insights based on a user‑selected set of symptoms. It is **not** a clinically validated diagnostic tool.

- **Backend:** Python 3, Pandas, NumPy, Scikit‑learn
- **ML Models:** Decision Tree, Random Forest, Multinomial Naive Bayes (trained on a synthetic dataset)
- **Frontend:** Streamlit
- **Visualization:** Matplotlib, Seaborn
- **Model Persistence:** Joblib

---

## 📦 Getting Started
### Prerequisites
- Python 3.13 (or newer) 
- `pip` (comes with Python)

### Installation
```bash
# Clone the repo (if you haven't already)
git clone https://github.com/Susmita0711/MediScan_InternVia.git
cd MediScan_InternVia

# (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt   # you can generate this with `pip freeze > requirements.txt`
```

### Train the Model (first run only)
```bash
python train_model.py
```
This will generate the `models/` folder containing:
- `best_model.joblib`
- `feature_names.json`
- `label_encoder.joblib`
- `metrics.json`

### Run the App
```bash
streamlit run app.py
```
Open the URL shown in the console (usually `http://localhost:8501`).

---

## 🛠️ Usage
1. **Home** – Introduction, branding, and navigation.
2. **Symptom Checker** – Select any combination of symptoms, click **Predict**, and view the model’s suggested condition plus a confidence score.
3. **Prediction History** – Session‑based log of past predictions (no personal data stored).
4. **Model Insights** – Dataset size, model comparison, confusion matrix, feature importance, and evaluation metrics.
5. **About MediScan** – Project details, tech stack, synthetic‑data disclaimer, and licensing.

---

## ⚠️ Medical Disclaimer
**MediScan is an educational Machine Learning demonstration and is not a medical diagnostic tool.** Its predictions may be inaccurate and must not replace professional medical advice. Do not use this application to self‑diagnose, delay treatment, or make medication decisions. Always consult a qualified healthcare professional for medical concerns.

---

## 📄 License
This project is released under the **MIT License**. Feel free to use, modify, and distribute it for educational purposes.

---

## 🙏 Acknowledgements
- Streamlit team for the easy‑to‑use UI framework.
- Scikit‑learn community for the powerful ML library.
- All contributors who helped shape this prototype.
