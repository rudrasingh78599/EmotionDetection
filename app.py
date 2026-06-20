import streamlit as st
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="EmotionSense AI",
    page_icon="😊",
    layout="centered"
)

# =========================
# LOAD FILES
# =========================
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# =========================
# EMOTION MAPPING
# =========================
emotion_map = {
    0: "sadness",
    1: "anger",
    2: "love",
    3: "surprise",
    4: "fear",
    5: "joy"
}

emotion_emojis = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲"
}

# =========================
# TITLE
# =========================
st.title("😊 EmotionSense AI")
st.markdown(
    "Analyze text and predict the underlying emotion using an XGBoost Machine Learning model."
)

# =========================
# TEXT INPUT
# =========================
user_text = st.text_area(
    "Enter your text",
    height=150,
    placeholder="Example: I am feeling very happy today!"
)

# =========================
# PREDICTION
# =========================
if st.button("Analyze Emotion"):

    if not user_text.strip():
        st.warning("Please enter some text.")

    else:
        try:

            # Transform text
            X = vectorizer.transform([user_text])

            # Predict
            prediction = model.predict(X)

            # Convert prediction number to emotion
            emotion = emotion_map[int(prediction[0])]

            # Get emoji
            emoji = emotion_emojis.get(emotion, "🤖")

            # Display result
            st.success(f"{emoji} Predicted Emotion: {emotion.title()}")

            # Debug Info
            with st.expander("Debug Info"):
                st.write("Raw Prediction:", prediction)
                st.write("Detected Emotion:", emotion)

            st.balloons()

        except Exception as e:
            st.error(f"Error: {e}")

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("📌 About")

    st.write("""
    **EmotionSense AI**

    Detect emotions from text using Machine Learning.

    **Model:** XGBoost Classifier

    **Vectorization:** Bag of Words (CountVectorizer)

    **Framework:** Streamlit

    **Supported Emotions**
    - 😊 Joy
    - 😢 Sadness
    - 😠 Anger
    - 😨 Fear
    - ❤️ Love
    - 😲 Surprise
    """)