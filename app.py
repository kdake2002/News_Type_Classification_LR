import streamlit as st
import pickle

# Set Streamlit page config
st.set_page_config(page_title="News Category Classifier", layout="centered")

# Inject CSS for background image
st.markdown("""
    <style>
    .stApp {
        background-image: url('https://cdn.pixabay.com/photo/2016/02/01/00/56/news-1172463_1280.jpg');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }

    .main {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 10px;
    }

    h1, h3, .stTextInput > div > div, .stTextArea > div > div {
        color: #1c1c1c;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='main'>", unsafe_allow_html=True)
st.title("📰 News Category Prediction App")
st.markdown("Enter a news headline and short description, and I’ll predict the category for you.")

# Load model and vectorizer
@st.cache_resource
def load_model():
    with open("news_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("vectorizer.pkl", "rb") as vec_file:
        vectorizer = pickle.load(vec_file)
    return model, vectorizer

model, vectorizer = load_model()

# Input
headline = st.text_input("✏️ Headline")
description = st.text_area("📝 Short Description")

if st.button("🔍 Predict Category"):
    if not headline or not description:
        st.warning("Please enter both headline and description.")
    else:
        full_text = headline + " " + description
        text_vector = vectorizer.transform([full_text])
        prediction = model.predict(text_vector)[0]
        st.success(f"🧠 Predicted Category: **{prediction}**")

st.markdown("</div>", unsafe_allow_html=True)
