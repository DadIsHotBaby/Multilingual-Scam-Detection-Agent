import streamlit as st
from transformers import pipeline

st.title("🌐 Multilingual Scam Detector")

model_path = "./model"
clf = pipeline("text-classification", model=model_path, tokenizer=model_path)

text = st.text_area("Enter message in any language:")

if text:
    pred = clf(text)[0]
    label = "🚨 Scam" if pred["label"] == "LABEL_1" else "✅ Not Scam"
    st.markdown(f"**Prediction**: {label}")
    st.markdown(f"**Confidence**: {round(pred['score'] * 100, 2)}%")
