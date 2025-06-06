import joblib
import pandas as pd

# Carregar modelo e dataset
model = joblib.load("models/best_model.pkl")
data  = pd.read_csv("data/cleaned_dataset.csv")

def predicty(input_data: dict):
    df = pd.DataFrame([input_data])

    # Predição
    prediction  = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    # Análise contextual simples
    analysis = {}
    for col in df.columns:
        val  = df[col][0]
        mean = data[col].mean()
        std  = data[col].std()
        analysis[col] = {
            "value": val,
            "mean": mean,
            "std": std,
            "percentile": (data[col] < val).mean() * 100,
            "is_outlier": abs(val - mean) > 2 * std
        }

    return prediction, probability, analysis
