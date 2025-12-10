import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

# -----------------------------
# Load Model & Tokenizer
# -----------------------------
@st.cache_resource
def load_sentiment_model():
    return load_model("model.h5")

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.json", "r") as f:
        tok_json = json.load(f)
    return tokenizer_from_json(tok_json)

model = load_sentiment_model()
tokenizer = load_tokenizer()

MAX_LEN = 200  # same padding length you trained with

# -----------------------------
# Prediction Function
# -----------------------------
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="pre")
    pred = model.predict(padded)[0][0]

    sentiment = "Positive 😊" if pred > 0.5 else "Negative 😡"
    confidence = float(pred) if pred > 0.5 else float(1 - pred)

    return sentiment, round(confidence, 3)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🧠 Simple Sentence Sentiment Analyzer")
st.write("Type any sentence below and the model will predict whether it is **Positive** or **Negative**.")

# User Input
user_input = st.text_area("Enter your text here:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        sentiment, confidence = predict_sentiment(user_input)
        st.success(f"**Sentiment:** {sentiment}")
        st.info(f"**Confidence:** {confidence}")
