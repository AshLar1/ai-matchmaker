import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt

# ------------------------------
# 🖼️ Page Config
# ------------------------------
st.set_page_config(page_title="AI Matchmaker", page_icon="💘", layout="centered")

# ------------------------------
# 📂 Load AI Matching Files
# ------------------------------
@st.cache_data
def load_data():
    user_profiles = pd.read_csv("user_profiles_lda15.csv")
    topic_labels = pd.read_json("topic_labels_15.json", typ='series')
    cosine_sim_df = pd.read_csv("cosine_similarity_matrix.csv", header=None)
    cosine_sim_df = cosine_sim_df.select_dtypes(include=[np.number])
    cosine_sim = cosine_sim_df.astype(float).values
    return user_profiles, cosine_sim, topic_labels

user_profiles, similarity_matrix, topic_labels = load_data()

# Optional model loading
try:
    rf_model = joblib.load("rf_model.pkl")
    shap_explainer = joblib.load("shap_explainer.pkl")
    has_model = True
except:
    has_model = False

# ------------------------------
# 💡 Recommend Matches Function
# ------------------------------
def interpret_compatibility(user_score, match_score):
    diff = abs(user_score - match_score)
    if diff < 0.1:
        return "Your tone is similar — you're both upbeat and friendly 🎉"
    elif user_score > match_score:
        return "This person might balance your high energy with calmness 🧘‍♂️"
    else:
        return "You may energize their mellow vibe 🔄"

def recommend_matches(user_id, top_n=5):
    if user_id not in user_profiles["user_id"].values:
        return pd.DataFrame(columns=["user_id", "topic_label", "sentiment_score", "tone_match"])

    idx = user_profiles[user_profiles["user_id"] == user_id].index[0]
    sim_scores = similarity_matrix[idx]
    top_indices = sim_scores.argsort()[::-1][1:top_n+1]
    user_sentiment = user_profiles.loc[idx, "sentiment_score"]

    matches_df = user_profiles.iloc[top_indices][["user_id", "lda_topic_15", "sentiment_score"]].copy()
    matches_df["topic_label"] = matches_df["lda_topic_15"].map(topic_labels)
    matches_df["tone_match"] = matches_df["sentiment_score"].apply(
        lambda s: interpret_compatibility(user_sentiment, s)
    )

    return matches_df[["user_id", "topic_label", "sentiment_score", "tone_match"]]

# ------------------------------
# 🖥️ Web UI: Match Finder
# ------------------------------
st.title("💘 AI Matchmaker")

user_id = st.selectbox("Choose your user ID", user_profiles["user_id"])
top_n = st.slider("How many matches to show?", 1, 10, 3)

if st.button("Find Matches"):
    matches = recommend_matches(user_id, top_n)
    st.write("### 🔍 Recommended Matches")
    st.dataframe(matches, use_container_width=True)

    # Optional: Show SHAP explainability for first match
    if has_model:
        st.markdown("### 🧠 Why was this match suggested?")
        try:
            first_match_vector = matches.iloc[0].drop(labels=["user_id", "topic_label", "tone_match"]).values.reshape(1, -1)
            shap_values = shap_explainer.shap_values(first_match_vector)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            shap.summary_plot(shap_values, features=first_match_vector, feature_names=["sentiment_score"], plot_type="bar")
            st.pyplot()
        except:
            st.warning("⚠️ Could not generate SHAP plot for this match.")

# ------------------------------
# 💬 Simulated Chat Analyzer
# ------------------------------
st.markdown("---")
st.subheader("🗨️ Simulated Chat (Test Your Style)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

chat_input = st.text_area("💬 Write a message to someone you're matched with")

if st.button("Analyze Message"):
    if chat_input.strip():
        st.session_state.chat_history.append(chat_input)

        # Simulated scoring
        sentiment_score = round(len(chat_input) % 10 / 10, 2)
        topic = "Travel" if "beach" in chat_input.lower() else "General"

        st.success("✅ Message received and analyzed.")
        st.write("🧠 Sentiment Score (simulated):", sentiment_score)
        st.write("🧪 Detected Topic:", topic)

        avg_length = sum(len(m) for m in st.session_state.chat_history) / len(st.session_state.chat_history)
        st.write("📏 Avg message length:", round(avg_length, 1))

        st.markdown("### 📜 Chat History")
        for msg in st.session_state.chat_history:
            st.markdown(f"- {msg}")
    else:
        st.warning("Please type something to analyze.")

# ------------------------------
# 💾 Save Profile (Simulated)
# ------------------------------
if st.button("Save My Style (Simulated)"):
    st.success("Your messaging style has been saved. We'll use it to recommend better matches!")
