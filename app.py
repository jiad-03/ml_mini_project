import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import base64
import json
from datetime import datetime
from transformers import pipeline

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Emotion Analyzer",
    layout="centered",
)

# ---------------------------------------------------------
# LOAD REAL EMOTION MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )

emotion_model = load_emotion_model()

# ---------------------------------------------------------
# CUSTOM CSS FOR BEAUTIFUL UI
# ---------------------------------------------------------
st.markdown("""
<style>

    .main { background-color: #F9FAFB; }

    .title {
        font-size: 40px;
        font-weight: 800;
        text-align: center;
        color: #333;
    }

    .subtitle {
        font-size: 18px;
        text-align: center;
        color: #555;
        margin-bottom: 25px;
    }

    .card {
        background: white;
        border-radius: 18px;
        padding: 25px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
        margin-top: 20px;
    }

    .emoji {
        font-size: 60px;
        text-align: center;
        animation: bounce 1.5s infinite;
    }

    @keyframes bounce {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-6px); }
        100% { transform: translateY(0px); }
    }

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------
st.markdown("<div class='title'>Emotion Analyzer 🎭</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Analyze emotions, understand tone, view stats & download a report.</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# EMOTION MAPPING (HuggingFace → Your Labels)
# ---------------------------------------------------------
emotion_map = {
    "sadness": ("Sadness 😔", "This message carries feelings of sadness, heaviness or emotional struggle."),
    "joy": ("Happiness 😊", "This text expresses joy, positivity and an uplifting tone."),
    "anger": ("Anger 😡", "There is frustration, irritation or strong negative feelings in this message."),
    "fear": ("Fear 😨", "This message shows anxiety, worry or emotionally tense feelings."),
    "surprise": ("Surprise 😲", "The message expresses amazement, shock or something unexpected."),
    "neutral": ("Neutral 🙂", "The tone is balanced and does not show strong emotions.")
}

all_labels = list(emotion_map.keys())

# ---------------------------------------------------------
# THEME TOGGLE
# ---------------------------------------------------------
theme = st.toggle("🌗 Dark Theme")
if theme:
    st.write("<style>body { background-color: #111; color: #EEE; }</style>", unsafe_allow_html=True)

# ---------------------------------------------------------
# INPUT BOX
# ---------------------------------------------------------
st.write("### Enter your message:")
user_text = st.text_area("", height=150)

analyze = st.button("Analyze Emotion 🔍")

# ---------------------------------------------------------
# EMOTION HISTORY STORAGE
# ---------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------
# PREDICTION FUNCTION
# ---------------------------------------------------------
def predict_emotion(text):
    output = emotion_model(text)
    scores = sorted(output[0], key=lambda x: x["score"], reverse=True)
    top = scores[0]
    return top["label"], scores

# ---------------------------------------------------------
# MAIN ANALYSIS
# ---------------------------------------------------------
if analyze:
    if user_text.strip() == "":
        st.warning("Please type something to analyze.")
    else:
        label, all_scores = predict_emotion(user_text)

        emotion, interpretation = emotion_map[label]
        score_dict = {s["label"]: s["score"] for s in all_scores}
        sentiment_score = round(score_dict[label] * 100, 2)

        # Save history
        st.session_state.history.append({
            "text": user_text,
            "emotion": emotion,
            "score": sentiment_score,
            "time": datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        })

        # Result Card
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        st.markdown(f"<div class='emoji'>{emotion.split()[-1]}</div>", unsafe_allow_html=True)
        st.write(f"## 🎯 Emotion Detected: **{emotion}**")

        st.write("### 📝 Interpretation:")
        st.write(interpretation)

        # Sentiment Strength Meter
        st.write("### 🔥 Emotion Strength:")
        st.progress(sentiment_score / 100)
        st.write(f"**Strength:** {sentiment_score}%")

        # Human-readable summary
        st.write("### 💡 Summary Review:")
        st.success(
            f"The message strongly reflects **{emotion}**. "
            f"Overall tone can be described as: **{interpretation}**"
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # Optional statistics
        with st.expander("📊 View Detailed Statistics"):
            fig, ax = plt.subplots()
            ax.bar(
                [emotion_map[l][0] for l in all_labels],
                [score_dict.get(l, 0) for l in all_labels]
            )
            ax.set_title("Emotion Probability Distribution")
            ax.set_ylabel("Probability")
            st.pyplot(fig)

        # Downloadable Report
        report_data = {
            "text": user_text,
            "emotion": emotion,
            "strength": sentiment_score,
            "interpretation": interpretation,
        }
        report_str = json.dumps(report_data, indent=4)

        st.download_button(
            label="📄 Download Emotion Report",
            data=report_str,
            file_name="emotion_report.txt",
            mime="text/plain",
        )

# ---------------------------------------------------------
# EMOTION HISTORY
# ---------------------------------------------------------
if len(st.session_state.history) > 0:
    st.write("## 🕒 Analysis History")
    for entry in reversed(st.session_state.history[-5:]):
        with st.expander(f"{entry['time']} – {entry['emotion']} ({entry['score']}%)"):
            st.write(f"**Message:** {entry['text']}")
            st.write(f"**Emotion:** {entry['emotion']}")
            st.write(f"**Strength:** {entry['score']}%")
