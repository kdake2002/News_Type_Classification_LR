import streamlit as st
import pickle
import os

# Load model and vectorizer
@st.cache_resource
def load_model():
    with open("news_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("vectorizer.pkl", "rb") as vec_file:
        vectorizer = pickle.load(vec_file)
    return model, vectorizer

model, vectorizer = load_model()

# Streamlit UI
st.set_page_config(page_title="News Category Classifier", layout="centered")

st.title("📰 News Category Prediction App")
st.markdown("Enter a news headline and description to predict its category.")

# Input from user
headline = st.text_input("Headline")
description = st.text_area("Short Description")

if st.button("Predict Category"):
    if not headline or not description:
        st.warning("Please enter both headline and short description.")
    else:
        full_text = headline + " " + description
        text_vector = vectorizer.transform([full_text])
        prediction = model.predict(text_vector)[0]
        st.success(f"🧠 Predicted Category: **{prediction}**")
