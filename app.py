
import streamlit as st
import pandas as pd

# Mock user data
mock_user_profiles = pd.DataFrame({
    "user_id": [f"user_00{i}" for i in range(1, 6)],
    "lda_topic": ["Travel", "Music", "Food", "Books", "Fitness"],
    "sentiment_score": [0.3, 0.6, 0.5, 0.7, 0.4],
})

def mock_recommend_matches(user_id, top_n=3):
    others = mock_user_profiles[mock_user_profiles["user_id"] != user_id]
    return others.sample(n=top_n)

# Streamlit UI
st.title("💘 AI Matchmaker - Demo")

user_id = st.selectbox("Choose your user ID", mock_user_profiles["user_id"])
top_n = st.slider("How many matches to show?", 1, 4, 3)

if st.button("Find Matches"):
    matches = mock_recommend_matches(user_id, top_n)
    st.write("### Recommended Matches")
    st.dataframe(matches)
