import os
import json
import re
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime

# ---------- CONFIG ----------
st.set_page_config(page_title="YouTube Trend Predictor", page_icon="📈", layout="wide")
sns.set_theme(style="whitegrid")

# ---------- HELPERS ----------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def safe_read_csv(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except Exception:
        return pd.read_csv(path, encoding="latin-1")

@st.cache_data
def load_artifacts():
    model = load_pickle("../models/trend_model.pkl")
    tfidf = load_pickle("../models/tfidf_vectorizer.pkl")
    scaler = load_pickle("../models/scaler.pkl")
    return model, tfidf, scaler


STOPWORDS = set("""
a an the and or but if while of to for in on at from by with as is are was were be have has had not no you your we they it this that these those our
""".split())

def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join(w for w in text.split() if w not in STOPWORDS)
    return text

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    needed = ["video_id","title","tags","description","views","likes","dislikes","comment_count","publish_time"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Please include these in your CSV.")
    return df

def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")
    df = df.dropna(subset=["publish_time"])
    df["publish_hour"] = df["publish_time"].dt.hour
    df["views"] = pd.to_numeric(df["views"], errors="coerce").fillna(0)
    df["likes"] = pd.to_numeric(df["likes"], errors="coerce").fillna(0)
    df["dislikes"] = pd.to_numeric(df["dislikes"], errors="coerce").fillna(0)
    df["comment_count"] = pd.to_numeric(df["comment_count"], errors="coerce").fillna(0)
    denom = (df["views"] + 1)
    df["like_ratio"] = df["likes"] / denom
    df["dislike_ratio"] = df["dislikes"] / denom
    df["comment_ratio"] = df["comment_count"] / denom
    df["engagement"] = (df["likes"] + df["comment_count"]) / denom
    df["title_clean"] = df["title"].astype(str).apply(clean_text)
    df["tags_clean"] = df["tags"].astype(str).apply(clean_text)
    df["desc_clean"] = df["description"].astype(str).apply(clean_text)
    df["combined_text"] = df["title_clean"] + " " + df["tags_clean"] + " " + df["desc_clean"]
    return df

def featurize(df: pd.DataFrame, tfidf, scaler):
    X_text = tfidf.transform(df["combined_text"]).toarray()
    X_num = df[["like_ratio","dislike_ratio","comment_ratio","engagement","publish_hour"]].values
    X_num_scaled = scaler.transform(X_num)
    X = np.hstack([X_text, X_num_scaled])
    return X

# ---------- LOAD ARTIFACTS ----------
model, tfidf, scaler = load_artifacts()

# ---------- SIDEBAR ----------
st.sidebar.title("📦 Upload Your YouTube Data")
uploaded = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
prob_slider = st.sidebar.slider("Trending Probability Sensitivity", 0.10, 0.90, 0.50, 0.01)
st.sidebar.caption("👉 Lower = More videos predicted as trending, Higher = Only top ones considered trending.")
st.sidebar.markdown("---")
st.sidebar.caption("CSV must include: video_id, title, tags, description, views, likes, dislikes, comment_count, publish_time (+ optional country).")

# ---------- LOAD DATA ----------
if uploaded is not None:
    df_raw = pd.read_csv(uploaded)
    source_label = "Your Uploaded CSV"
else:
    sample_path = "data/dashboard_sample.csv"
    if os.path.exists(sample_path):
        df_raw = safe_read_csv(sample_path)
        source_label = "Sample Demo Dataset"
    else:
        st.warning("Please upload a CSV file to start.")
        st.stop()

# ---------- APP HEADER ----------
st.title("📈 YouTube Trend Prediction System")

# ✨ Info Box
st.info(
    """
    ℹ️ **What does this app do?**

    This app helps you find which YouTube videos are most likely to trend based on their **views, likes, comments, publish time, and keywords**.

    Just upload your YouTube data CSV — the app will calculate a **Trending Probability (%)** for each video and tell you whether it’s likely to trend 🔥 or not yet ⚪.
    """,
    icon="💡"
)
st.caption(f"Data Source: **{source_label}**")

# ---------- MAIN TAB STRUCTURE ----------
tab1, tab2 = st.tabs(["🔮 Predict Trending Videos", "📊 Analytics Overview"])

# =======================
# TAB 1: TREND PREDICTION
# =======================
with tab1:
    st.subheader("🔮 Predict Which Videos May Trend")
    try:
        df_in = ensure_columns(df_raw)
        df_prep = enrich_features(df_in)

        # Make predictions
        X = featurize(df_prep, tfidf, scaler)
        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= prob_slider).astype(int)

        # Create user-friendly output
        out = pd.DataFrame({
            "Video Title": df_prep["title"].astype(str).str.slice(0, 100),
            "Views": df_prep["views"].astype(int),
            "Likes": df_prep["likes"].astype(int),
            "Comments": df_prep["comment_count"].astype(int),
            "Published On": df_prep["publish_time"].astype(str),
            "Engagement Rate (%)": np.round(df_prep["engagement"] * 100, 2),
            "Trending Probability (%)": np.round(probs * 100, 2),
            "Prediction": np.where(preds == 1, "🔥 Likely to Trend", "⚪ Not Trending Yet")
        })

        st.dataframe(out.head(30), use_container_width=True)

        # Metrics section
        st.markdown("### Summary:")
        colA, colB = st.columns(2)
        with colA:
            st.metric("Predicted Trending Videos", int((preds == 1).sum()))
        with colB:
            st.metric("Average Trending Probability", f"{probs.mean() * 100:.1f}%")

        st.download_button(
            "⬇️ Download Predictions (CSV)",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="YouTube_Trend_Predictions.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Something went wrong: {e}")

# =======================
# TAB 2: SIMPLE ANALYTICS
# =======================
with tab2:
    st.subheader("📊 Quick Insights")
    try:
        base = df_prep.copy()
    except NameError:
        base = enrich_features(ensure_columns(df_raw))

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5,3))
        sns.histplot(base["engagement"], bins=40, ax=ax)
        ax.set_title("Engagement Distribution")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(5,3))
        sns.scatterplot(x=base["views"], y=base["likes"], s=10, ax=ax)
        ax.set_title("Views vs Likes")
        ax.set_xscale("log"); ax.set_yscale("log")
        st.pyplot(fig)

    st.markdown("### Common Words in Video Titles")
    from collections import Counter
    words = " ".join(base["title_clean"]).split()
    top_words = Counter(words).most_common(20)
    word_df = pd.DataFrame(top_words, columns=["Word", "Count"])
    st.bar_chart(word_df.set_index("Word"))

    st.markdown("### How the Model Thinks 🤖")
    st.write("""
    The model considers several factors such as **views, likes, comments, publish time,**
    and **keywords** in the title, tags, and description to estimate the likelihood of a video trending.
    """)
