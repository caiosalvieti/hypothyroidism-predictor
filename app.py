# app.py  – Streamlit frontend
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from predicty import predicty   # nossa função de previsão

st.set_page_config(page_title="Hypothyroidism Predictor", layout="centered")
st.title("🩺 Hypothyroidism Diagnostic App")

st.markdown("""
Este app prevê a probabilidade de hipotireoidismo com um modelo treinado.
Preencha os exames laboratoriais abaixo:
""")

# --- Formulário de entrada ---
with st.form("prediction_form"):
    age  = st.slider("Age", 0, 100, 35)
    T3   = st.number_input("T3",  min_value=0.0, value=1.0)
    TT4  = st.number_input("TT4", min_value=0.0, value=100.0)
    T4U  = st.number_input("T4U", min_value=0.0, value=1.0)
    submit = st.form_submit_button("Predict")

if submit:
    input_data = {"age": age, "T3": T3, "TT4": TT4, "T4U": T4U}

    try:
        prediction, probability, analysis = predicty(input_data)
    except Exception as e:
        st.error(f"Erro na previsão: {e}")
        st.stop()

    # --- Resultado ---
    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.error(f"Hypothyroidism (Prob: {probability:.2f})")
    else:
        st.success(f"Normal (Prob: {probability:.2f})")

    # --- Análise contextual ---
    st.subheader("🧪 Contextual Analysis")
    for key, val in analysis.items():
        st.write(f"**{key}** → valor {val['value']:.2f} | média {val['mean']:.2f} | desvio {val['std']:.2f} | percentil {val['percentile']:.1f}%")
        if val['is_outlier']:
            st.warning(f"⚠️ {key} é outlier (±2 σ)")

    # --- Histograma comparativo ---
    st.subheader("📈 Distribution")
    ref = pd.read_csv("data/cleaned_dataset.csv")

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col, val in zip(axs, ["T3", "TT4", "T4U"], [T3, TT4, T4U]):
        ax.hist(ref[col], bins=20, alpha=0.6)
        ax.axvline(val, color="red", linestyle="--")
        ax.set_title(col)
    st.pyplot(fig)

st.markdown("---")
st.caption("Developed by Caio Salvieti · Powered by Streamlit · 2025")
