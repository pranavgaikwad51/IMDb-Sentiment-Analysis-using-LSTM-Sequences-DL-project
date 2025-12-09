import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

# ---------------------
# Load Model & Tokenizer
# ---------------------
@st.cache_resource
def load_sentiment_model():
    model = load_model("model.h5")
    return model

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.json", "r") as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)
    return tokenizer

model = load_sentiment_model()
tokenizer = load_tokenizer()

MAX_LEN = 200  # use same max length used during training

# ---------------------
# Prediction Function
# ---------------------
def predict_sentiment(text_list):
    seq = tokenizer.texts_to_sequences(text_list)
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='pre')
    preds = model.predict(padded)
    sentiments = ["Positive" if p > 0.5 else "Negative" for p in preds]
    return sentiments, preds.flatten()

# ---------------------
# Streamlit UI
# ---------------------
st.title("🧠 Comment Filter - Sentiment Analyzer")
st.write("Upload a CSV file and the app will automatically classify comments into **Positive** and **Negative** groups.")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Detect column containing text
    possible_cols = ["review", "text", "comment", "comments", "feedback"]
    col = None
    for c in possible_cols:
        if c in df.columns:
            col = c
            break
            
    if col is None:
        st.error("❌ No text column found. Please make sure your CSV has a column like 'review' or 'comment'.")
    else:
        st.success(f"📌 Text column detected: `{col}`")

        st.write("⬇ Processing comments... please wait.")

        sentiments, confidence = predict_sentiment(df[col].tolist())

        df["Sentiment"] = sentiments
        df["Confidence Score"] = confidence.round(3)

        st.subheader("📊 Sentiment Distribution")
        counts = df["Sentiment"].value_counts()

        st.bar_chart(counts)

        st.subheader("🔍 Filtered Results")

        pos_df = df[df["Sentiment"] == "Positive"]
        neg_df = df[df["Sentiment"] == "Negative"]

        st.write(f"### ✅ Positive Comments ({len(pos_df)})")
        st.dataframe(pos_df.head(10))

        st.write(f"### ❌ Negative Comments ({len(neg_df)})")
        st.dataframe(neg_df.head(10))

        # Download buttons
        st.subheader("📁 Download Filtered Results")

        st.download_button(
            label="⬇ Download Positive Comments",
            data=pos_df.to_csv(index=False),
            file_name="Positive_Comments.csv",
            mime="text/csv"
        )

        st.download_button(
            label="⬇ Download Negative Comments",
            data=neg_df.to_csv(index=False),
            file_name="Negative_Comments.csv",
            mime="text/csv"
        )

else:
    st.info("📤 Please upload a CSV file to begin.")
