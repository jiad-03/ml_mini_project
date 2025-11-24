# app.py
import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from datetime import datetime

# Optional: transformers pipeline (if you installed transformers & torch)
try:
    from transformers import pipeline
    hf_available = True
except Exception:
    hf_available = False

st.set_page_config(page_title="Mental-Health Sentiment Demo", layout="centered")

st.title("Mental-Health Aware Sentiment & Emotion Demo")
st.markdown("Paste a short message (social post, chat message). App returns sentiment + simple emotion hints.")

# sidebar
model_choice = st.sidebar.selectbox("Model choice", ("VADER (fast)", "Transformer (higher quality)" if hf_available else "VADER (fast)"))
max_history = st.sidebar.slider("History size", 5, 200, 50)

# state for history
if "history" not in st.session_state:
    st.session_state.history = []

st.markdown("### Try these examples:")
examples = [
    "I feel overwhelmed lately and can't focus on anything.",
    "Today was great! I feel really proud of myself.",
    "I'm anxious about the exam tomorrow.",
    "I don't want to talk to anyone right now.",
]

chosen_example = st.selectbox("Pick an example:", [""] + examples)
if chosen_example:
    txt = chosen_example
else:
    txt=""
txt = st.text_area("Enter text (1-500 chars):", height=140, max_chars=1000)

if st.button("Analyze"):
    if not txt.strip():
        st.warning("Please enter some text.")
    else:
        # analyze with VADER
        analyzer = SentimentIntensityAnalyzer()
        vs = analyzer.polarity_scores(txt)
        compound = vs["compound"]
        # map to simple label
        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"

        # optional transformer emotions/sentiment
        transformer_out = None
        if hf_available and model_choice.startswith("Transformer"):
            try:
                sentiment_pipe = pipeline("sentiment-analysis", truncation=True)
                transformer_out = sentiment_pipe(txt)[0]  # dict with label, score
            except Exception as e:
                transformer_out = {"error": str(e)}

        # simple emotion heuristics (keywords)
        emotions = {"sadness":0, "joy":0, "anger":0, "fear":0}
        low_txt = txt.lower()
        for k in emotions.keys():
            if k in low_txt:
                emotions[k] += 1
        # add based on sentiment
        if label == "Positive":
            emotions["joy"] += 1
        if label == "Negative":
            emotions["sadness"] += 1

        result = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "text": txt,
            "vader_compound": compound,
            "vader_label": label,
            "transformer": transformer_out,
            "emotions": emotions
        }

        # save to history
        st.session_state.history.insert(0, result)
        # limit size
        st.session_state.history = st.session_state.history[:max_history]

# show last analysis
if st.session_state.history:
    latest = st.session_state.history[0]
    st.subheader("Latest analysis")
    st.write("**Text:**", latest["text"])
    st.write("**VADER compound:**", round(latest["vader_compound"], 3), "— **", latest["vader_label"], "**")
    if latest["transformer"]:
        st.write("**Transformer output:**", latest["transformer"])
    st.write("**Emotion hints (simple):**", latest["emotions"])

    # plot history of compound scores
    df = pd.DataFrame([{"time":h["time"], "compound":h["vader_compound"]} for h in st.session_state.history])
    df["time"] = pd.to_datetime(df["time"])
    st.subheader("History of scores")
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(df["time"], df["compound"], marker="o")
    ax.set_ylim(-1.05,1.05)
    ax.set_ylabel("VADER compound score")
    ax.set_xlabel("Time")
    st.pyplot(fig)

    st.subheader("Saved history (latest first)")
    st.dataframe(df.sort_values("time", ascending=False).reset_index(drop=True))
    import json
    import csv
    import io

    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["time", "text", "compound", "label"])

    for h in st.session_state.history:
        writer.writerow([h["time"], h["text"], h["vader_compound"], h["vader_label"]])

    st.download_button(
        label="Download History as CSV",
        data=csv_buffer.getvalue(),
        file_name="sentiment_history.csv",
        mime="text/csv"
    )
else:
    st.info("No analyses yet — enter text and press Analyze.")
