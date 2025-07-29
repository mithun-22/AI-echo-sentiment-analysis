import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load model and vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Preprocessing function
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# Streamlit UI
st.set_page_config(page_title="AI Echo - Sentiment Analyzer", layout="centered")
st.title("\U0001F4AC AI Echo - ChatGPT Review Sentiment Analyzer")
st.markdown("Classify ChatGPT reviews into Positive, Neutral, or Negative")

user_input = st.text_area("Enter a ChatGPT Review:", height=150)

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        cleaned = preprocess(user_input)
        vect_text = vectorizer.transform([cleaned])
        prediction = model.predict(vect_text)[0]

        st.subheader("\U0001F4C8 Sentiment Result")
        if prediction == "Positive":
            st.success("✅ This review is Positive")
        elif prediction == "Neutral":
            st.info("ℹ️ This review is Neutral")
        else:
            st.error("❌ This review is Negative")

        st.markdown("---")
        st.markdown("\U0001F916 **Model Info:** Logistic Regression | TF-IDF Features")
        st.markdown("\U0001F4E6 **Vector Size:** {} features".format(vect_text.shape[1]))
