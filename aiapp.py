import streamlit as st
import pandas as pd
import joblib
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import LabelEncoder

# NLTK downloads
nltk.download('stopwords')
nltk.download('wordnet')

# -------------------- Load model & vectorizer --------------------
MODEL_PATH = r"D:\ai echo\sentiment_model1.pkl"  # Replace with your new .pkl
VECTORIZER_PATH = r"D:\ai echo\tfidf_vectorizer1.pkl"  # Replace with your new .pkl
DATA_PATH = r"D:\ai echo\chatgpt_style_reviews_dataset.xlsx"  # Dataset path

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="AI Echo - Sentiment Analyzer", layout="centered")
st.title("💬 AI Echo - ChatGPT Review Sentiment Analyzer")
st.markdown("Classify ChatGPT reviews into **Positive**, **Neutral**, or **Negative**")

@st.cache_data
def load_data():
    return pd.read_excel(DATA_PATH)

df = load_data()
st.write("📄 **Columns in your dataset:**", df.columns.tolist())

# -------------------- Predictions for dataset --------------------
df['cleaned'] = df['review'].astype(str).apply(preprocess)
vect_transformed = vectorizer.transform(df['cleaned'])
df['sentiment'] = model.predict(vect_transformed)

# If model outputs numeric labels, decode them
if df['sentiment'].dtype != 'object':
    label_decoder = LabelEncoder()
    label_decoder.classes_ = ['Negative', 'Neutral', 'Positive']  # Match your training order
    df['sentiment'] = label_decoder.inverse_transform(df['sentiment'])

# -------------------- User input --------------------
manual_sentiment = {
    "good": "Positive",
    "great": "Positive",
    "excellent": "Positive",
    "amazing": "Positive",
    "love": "Positive",
    "bad": "Negative",
    "terrible": "Negative",
    "awful": "Negative",
    "worst": "Negative",
    "hate": "Negative",
    "okay": "Neutral",
    "average": "Neutral"
}

user_input = st.text_area("✏️ Enter a ChatGPT Review:", height=150)
cleaned = ""
vect_text = None

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter a review to analyze.")
    else:
        cleaned = preprocess(user_input)
        vect_text = vectorizer.transform([cleaned])
        ...

        if cleaned in manual_sentiment:
            prediction = manual_sentiment[cleaned]
        else:
            prediction_code = model.predict(vect_text)[0]
            if isinstance(prediction_code, (int, float)):
                prediction = label_decoder.inverse_transform([prediction_code])[0]
            else:
                prediction = prediction_code

        st.subheader("📊 Sentiment Result")
        if prediction == "Positive":
            st.success("✅ This review is **Positive**")
        elif prediction == "Neutral":
            st.info("ℹ️ This review is **Neutral**")
        else:
            st.error("❌ This review is **Negative**")

        st.markdown("---")
        st.markdown("🤖 **Model:** Latest Logistic Regression | TF-IDF with overrides")
        st.markdown(f"📦 **Vector Size:** {vect_text.shape[1]} features")


vect_text = vectorizer.transform([cleaned])  # Always compute TF-IDF vector
if cleaned in manual_sentiment:
    prediction = manual_sentiment[cleaned]
else:
    prediction_code = model.predict(vect_text)[0]
    if isinstance(prediction_code, (int, float)):
        prediction = label_decoder.inverse_transform([prediction_code])[0]
    else:
        prediction = prediction_code

# Now vect_text exists for display
st.markdown(f"📦 **Vector Size:** {vect_text.shape[1]} features")


# -------------------- Analysis Sections --------------------

# 1. Overall Sentiment Distribution
st.subheader("1️⃣ Overall Sentiment Distribution")
sentiment_counts = df['sentiment'].value_counts(normalize=True) * 100
st.bar_chart(sentiment_counts)

# 2. Sentiment vs Rating
if 'rating' in df.columns:
    st.subheader("2️⃣ Sentiment vs Rating")
    cross_tab = pd.crosstab(df['rating'], df['sentiment'], normalize='index') * 100
    st.dataframe(cross_tab.style.format("{:.1f}"))
    fig, ax = plt.subplots()
    cross_tab.plot(kind='bar', stacked=True, ax=ax)
    st.pyplot(fig)

# 3. Word Clouds per Sentiment
st.subheader("3️⃣ Keyword Clouds per Sentiment")
for sentiment in ["Positive", "Neutral", "Negative"]:
    st.markdown(f"**{sentiment} Reviews**")
    text = " ".join(df[df['sentiment'] == sentiment]['cleaned'])
    if text.strip():
        wc = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

# 4. Sentiment Over Time
if 'date' in df.columns:
    st.subheader("4️⃣ Sentiment Over Time")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    df['month'] = df['date'].dt.to_period('M').astype(str)
    time_sentiment = df.groupby(['month', 'sentiment']).size().unstack().fillna(0)
    st.line_chart(time_sentiment)

# 5. Verified vs Non-Verified
if 'verified_purchase' in df.columns:
    st.subheader("5️⃣ Verified vs Non-Verified Reviews")
    vp_sent = pd.crosstab(df['verified_purchase'], df['sentiment'], normalize='index') * 100
    st.dataframe(vp_sent.style.format("{:.1f}"))
    fig, ax = plt.subplots()
    vp_sent.plot(kind='bar', stacked=True, ax=ax)
    st.pyplot(fig)

# 6. Review Length vs Sentiment
st.subheader("6️⃣ Review Length vs Sentiment")
df['review_length'] = df['review'].apply(lambda x: len(str(x).split()))
fig, ax = plt.subplots()
sns.boxplot(x='sentiment', y='review_length', data=df, ax=ax)
st.pyplot(fig)

# 7. Sentiment by Location
if 'location' in df.columns:
    st.subheader("7️⃣ Sentiment by Location (Top 10)")
    top_locs = df['location'].value_counts().head(10).index
    loc_sent = df[df['location'].isin(top_locs)]
    loc_chart = pd.crosstab(loc_sent['location'], loc_sent['sentiment'], normalize='index') * 100
    st.bar_chart(loc_chart)

# 8. Platform Sentiment
if 'platform' in df.columns:
    st.subheader("8️⃣ Platform Sentiment")
    platform_sent = pd.crosstab(df['platform'], df['sentiment'], normalize='index') * 100
    st.dataframe(platform_sent.style.format("{:.1f}"))
    fig, ax = plt.subplots()
    platform_sent.plot(kind='bar', stacked=True, ax=ax)
    st.pyplot(fig)

# 9. Sentiment by ChatGPT Version
if 'version' in df.columns:
    st.subheader("9️⃣ Sentiment by ChatGPT Version")
    version_sent = pd.crosstab(df['version'], df['sentiment'], normalize='index') * 100
    st.dataframe(version_sent.style.format("{:.1f}"))

# 10. Topic Modeling on Negative Reviews
st.subheader("🔟 Common Negative Feedback Themes")
negative_reviews = df[df['sentiment'] == "Negative"]
if not negative_reviews.empty:
    neg_vectorized = vectorizer.transform(negative_reviews['cleaned'])
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(neg_vectorized)
    terms = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_terms = [terms[i] for i in topic.argsort()[-10:]]
        st.markdown(f"- **Topic #{topic_idx+1}:** {', '.join(top_terms)}")
else:
    st.info("No negative reviews available for topic modeling.")

