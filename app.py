import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("fake_social_profiles.csv")
X = df.drop("is_fake", axis=1)
y = df["is_fake"]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Streamlit UI
st.set_page_config(page_title="Fake Profile Detector", page_icon="🕵️", layout="centered")

st.markdown(
    """
    <div style='text-align: center; padding: 10px'>
        <h1 style='color:#800080;'>🕵️‍♀️ Fake Profile Detector</h1>
        <p style='font-size:18px;'>Enter social media profile details to detect if it's fake or real</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3064/3064197.png", width=150)
st.sidebar.header("👤 Profile Inputs")

# Sidebar inputs
username_length = st.sidebar.slider("Username Length", 3, 20, 8)
bio_length = st.sidebar.slider("Bio Length", 0, 150, 50)
num_followers = st.sidebar.number_input("Number of Followers", min_value=0, value=500)
num_following = st.sidebar.number_input("Number of Following", min_value=0, value=1000)
posts_count = st.sidebar.number_input("Number of Posts", min_value=0, value=5)
engagement_rate = st.sidebar.slider("Engagement Rate", 0.0, 1.0, 0.05)
has_profile_pic = st.sidebar.radio("Has Profile Picture?", ["Yes", "No"])
has_profile_pic = 1 if has_profile_pic == "Yes" else 0

# Predict Button
if st.button("🚀 Predict Now"):
    user_input = np.array([[username_length, bio_length, num_followers, num_following,
                            posts_count, engagement_rate, has_profile_pic]])
    prediction = model.predict(user_input)

    st.markdown("---")
    if prediction[0] == 1:
        st.error("⚠️ This profile is likely **FAKE**!")
        st.markdown("💡 Fake profiles often have zero posts, weird bio/usernames, or thousands of followings.")
    else:
        st.success("✅ This profile appears to be **REAL**!")
        st.markdown("🎉 Great! This user seems authentic based on the inputs.")
    st.markdown("---")

# Expand for info
with st.expander("ℹ️ How does it work?"):
    st.markdown("""
    - This app uses a machine learning model trained on profile characteristics.
    - Features like number of followers, posts, profile picture, and engagement rate help determine authenticity.
    - Built by Sreelekha using ❤️ and AI.
    """)
