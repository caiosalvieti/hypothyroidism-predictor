
# 🩺 Hypothyroidism-Predictor

A Streamlit web-app + training pipeline that **analyses thyroid-hormone data, visualises PCA patterns and predicts hypothyroidism** with a 99 %-accurate Random-Forest model.

<p align="center">
  <img src="docs/screenshot_ui.png" width="640" alt="UI demo">
</p>

| Stage | File | What happens |
|-------|------|--------------|
| **1. Training** | `train.py` | cleans raw `hypothyroid.data`, runs GridSearchCV (KNN • RF • LogReg), saves<br>`data/cleaned_dataset.csv` + `models/best_model.pkl` |
| **2. Predict API** | `predicty.py` | loads the model, returns **(prediction, probability, contextual analysis)** |
| **3. Web UI** | `app.py` | Streamlit form → calls `predicty` → shows result, SHAP-style stats & histograms |

---

## 📊 Quick PCA Story

| PC pair | Most important variable(s) | Visual insight |
|---------|---------------------------|----------------|
| **PC1 vs PC2** | `T3`, `TT4`, `T4U` | high T3/TT4 ↔ low TSH; hypothyroid patients cluster where `T4U` ↑ |
| **PC1 vs PC3** | `Age` | older patients trend to higher T3/TT4 |
| **PC1 vs PC4** | `TSH` | hypothyroid cluster at **high TSH & low T3/TT4** |
| <sub>(see `/notebooks/pca_plots.ipynb` for all six scatter-plots)</sub> |

---

## 🤖 Best model (after GridSearch)

| Metric | Value |
|--------|-------|
| Accuracy | **0.993** |
| Precision | 0.99 |
| Recall | 0.96 |
| F1-score | 0.97 |
| Matthews CC | 0.88 |

Confusion matrix  

[[TP 379 FN 0]
[FP 6 TN 18]]


The model was saved as `models/best_model.pkl` (joblib-pickle).  
Opening a `.pkl` in a text editor shows binary noise – load it in Python:

```python
import joblib
model = joblib.load("models/best_model.pkl")
print(model)
🚀 Run locally

git clone https://github.com/<your-user>/hypothyroidism-predictor.git
cd hypothyroidism-predictor

python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1) train (optional – model already committed)
python3 train.py

# 2) launch web-app
streamlit run app.py
Open http://localhost:8501 in your browser.

🌐 One-click deploy

Push the repo to GitHub.
Go to https://share.streamlit.io › New app
Fill in:
Repository  : <your-user>/hypothyroidism-predictor
Branch      : main
Main file   : app.py
Press Deploy → public URL appears in seconds.
📁 Project structure

├── app.py                 ← Streamlit frontend
├── predicty.py            ← model loader + context analysis
├── train.py               ← end-to-end training pipeline
├── data/
│   └── cleaned_dataset.csv
├── models/
│   └── best_model.pkl
├── requirements.txt
└── raw/
    └── hypothyroid.data   ← original UCI file
➕ Roadmap

 Add SHAP force-plots for individual explanations
 Batch CSV upload for clinicians
 Export ONNX model for language-agnostic inference
 Dockerfile + GitHub Actions CI/CD
Citation
Dataset: Thyroid Disease Data Set, UCI Machine-Learning Repository
Model & code © 2025 Caio Salvieti — MIT License


---

### How to use

1. Copy every line above into `README.md` (replace `<your-user>` / repo name).  
2. Add a screenshot at `docs/screenshot_ui.png` (or change the path).  
3. Commit & push -- your GitHub repo now looks polished and portfolio-ready.

