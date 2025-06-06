# train.py – end‑to‑end training & export script
"""
Run this script once to:
1. Load and clean the raw hypothyroid dataset
2. Train several ML models with GridSearchCV (KNN, Random Forest, Logistic Regression)
3. Pick the best model by cross‑validated accuracy
4. Persist:
   • cleaned dataset  → data/cleaned_dataset.csv
   • trained model    → models/best_model.pkl

Usage (inside project root):
    python3 train.py  # uses default file locations

Prereqs: Have requirements.txt installed & folders `data/` and `models/` created (script will create if absent).
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

RAW_PATH = "raw/hypothyroid.data"   # adjust if needed
CLEAN_PATH = "data/cleaned_dataset.csv"
MODEL_PATH = "models/best_model.pkl"

# Ensure folders exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ------------------------------------------------------------------
# 1. LOAD & CLEAN DATA
# ------------------------------------------------------------------
print("📥 Loading raw data …")
raw = pd.read_csv(RAW_PATH, header=None)

# Rename columns (mapping taken from original script)
raw.rename(columns={
    0: "diseases",  1: "age", 2: "sex", 3: "on_thyroxine", 4: "query_on_thyroxine",
    5: "on_antithyroid_medication", 6: "thyroid_surgery", 7: "query_hypothyroid",
    8: "query_hyperthyroid", 9: "pregnant", 10: "sick", 11: "tumor", 12: "lithium",
    13: "goitre", 14: "TSH_measured", 15: "TSH", 16: "T3_measured", 17: "T3",
    18: "TT4_measured", 19: "TT4", 20: "T4U_measured", 21: "T4U", 22: "FTI_measured",
    23: "FTI", 24: "TBG_measured", 25: "TBG"
}, inplace=True)

# Convert ? to NaN and categorical replacement
df = raw.replace("?", np.nan)
df.replace({"hypothyroid": 1, "negative": 0, "F": 1, "M": 0, "f": 0, "t": 1, "y": 1, "n": 0}, inplace=True)

# Drop unused columns
cols_to_drop = ["TBG_measured", "query_hypothyroid", "query_hyperthyroid", "pregnant", "sick", "tumor",
                "sex", "lithium", "goitre", "on_thyroxine", "query_on_thyroxine",
                "on_antithyroid_medication", "thyroid_surgery", "TSH_measured", "TBG",
                "T3_measured", "TT4_measured", "T4U_measured", "FTI_measured"]
df.drop(columns=cols_to_drop, inplace=True)

# Remove rows with missing values
df.dropna(inplace=True)

# Save cleaned dataset for later use by Streamlit app
print(f"💾 Saving cleaned dataset → {CLEAN_PATH}")
df.to_csv(CLEAN_PATH, index=False)

# ------------------------------------------------------------------
# 2. FEATURE / TARGET SPLIT
# ------------------------------------------------------------------
y = df["diseases"].astype(int)
X = df.drop(columns=["diseases"]).astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ------------------------------------------------------------------
# 3. DEFINE MODELS & GRIDS
# ------------------------------------------------------------------
models_and_grids = [
    ("KNN", Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())]),
     {"knn__n_neighbors": [3, 5, 7], "knn__weights": ["uniform", "distance"]}),

    ("RandomForest", Pipeline([("scaler", StandardScaler()), ("rf", RandomForestClassifier(random_state=42))]),
     {"rf__n_estimators": [100, 150], "rf__max_depth": [None, 10]}),

    ("LogReg", Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))]),
     {"lr__C": [0.1, 1, 10]})
]

best_model = None
best_acc = 0
best_name = ""

# ------------------------------------------------------------------
# 4. TRAIN & SELECT BEST
# ------------------------------------------------------------------
for name, pipe, grid in models_and_grids:
    print(f"\n🔄 Training {name} …")
    search = GridSearchCV(pipe, param_grid=grid, cv=5, scoring="accuracy", n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"   • best params: {search.best_params_}")
    pred = search.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"   • test accuracy: {acc:.3f}")

    if acc > best_acc:
        best_acc = acc
        best_model = search.best_estimator_
        best_name = name

print(f"\n🏆 Best model: {best_name} (accuracy {best_acc:.3f})")
print(classification_report(y_test, best_model.predict(X_test)))

# ------------------------------------------------------------------
# 5. SAVE BEST MODEL
# ------------------------------------------------------------------
print(f"💾 Saving best model → {MODEL_PATH}")
joblib.dump(best_model, MODEL_PATH)
print("✅ Training pipeline complete.")
