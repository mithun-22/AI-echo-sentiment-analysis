import streamlit as st
import joblib

model = joblib.load('D:\ai echo\sentiment_model.pkl')
vectorizer = joblib.load('D:\ai echo\tfidf_vectorizer.pkl')

st.title("AI Echo – Sentiment Classifier")

user_input = st.text_area("Enter a ChatGPT Review:")
if st.button("Predict Sentiment"):
    processed = preprocess(user_input)
    vect = vectorizer.transform([processed])
    prediction = model.predict(vect)[0]
    st.success(f"Predicted Sentiment: **{prediction}**")
