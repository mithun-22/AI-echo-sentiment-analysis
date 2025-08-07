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

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
# Load model and vectorizer
model = joblib.load(r"D:\ai echo\sentiment_model.pkl")
vectorizer = joblib.load(r"D:\ai echo\tfidf_vectorizer.pkl")

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing function
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

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv(r"D:\\ai echo\\chatgpt_style_reviews_dataset.xlsx - Sheet1.csv")

df = load_data()
st.write("Columns in your dataset:", df.columns.tolist())

# Predict sentiment
df['cleaned'] = df['review'].astype(str).apply(preprocess)
vect_transformed = vectorizer.transform(df['cleaned'])
df['sentiment'] = model.predict(vect_transformed)

# Decode predicted labels if model output is numeric
if df['sentiment'].dtype != 'object':
    label_decoder = LabelEncoder()
    label_decoder.classes_ = ['Negative', 'Neutral', 'Positive']  # Adjust based on training
    df['sentiment'] = label_decoder.inverse_transform(df['sentiment'])
else:
    df['sentiment'] = df['sentiment']

# User input section
user_input = st.text_area("Enter a ChatGPT Review:", height=150)
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        cleaned = preprocess(user_input)
        vect_text = vectorizer.transform([cleaned])
        prediction_code = model.predict(vect_text)[0]

        # Decode if prediction is numeric
        if isinstance(prediction_code, (int, float)):
            prediction = label_decoder.inverse_transform([prediction_code])[0]
        else:
            prediction = prediction_code

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

# 1. Overall Sentiment Distribution
st.subheader("1. Overall Sentiment Distribution")
sentiment_counts = df['sentiment'].value_counts(normalize=True) * 100
st.bar_chart(sentiment_counts)

# 2. Sentiment vs Rating
if 'rating' in df.columns:
    st.subheader("2. Sentiment vs Rating")
    cross_tab = pd.crosstab(df['rating'], df['sentiment'], normalize='index') * 100
    st.dataframe(cross_tab.style.format("{:.1f}"))
    fig, ax = plt.subplots()
    cross_tab.plot(kind='bar', stacked=True, ax=ax)
    st.pyplot(fig)

# 3. Keywords per Sentiment
st.subheader("3. Keyword Clouds per Sentiment")
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
    st.subheader("4. Sentiment Over Time")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    df['month'] = df['date'].dt.to_period('M').astype(str)
    time_sentiment = df.groupby(['month', 'sentiment']).size().unstack().fillna(0)
    st.line_chart(time_sentiment)

# 5. Sentiment by Verified Purchase
if 'verified_purchase' in df.columns:
    st.subheader("5. Verified vs Non-Verified Reviews")
    vp_sent = pd.crosstab(df['verified_purchase'], df['sentiment'], normalize='index') * 100
    st.dataframe(vp_sent.style.format("{:.1f}"))
    fig, ax = plt.subplots()
    vp_sent.plot(kind='bar', stacked=True, ax=ax)
    st.pyplot(fig)

# 6. Review Length vs Sentiment
st.subheader("6. Review Length vs Sentiment")
df['review_length'] = df['review'].apply(lambda x: len(str(x).split()))
fig, ax = plt.subplots()
sns.boxplot(x='sentiment', y='review_length', data=df, ax=ax)
st.pyplot(fig)

# 7. Sentiment by Location
if 'location' in df.columns:
    st.subheader("7. Sentiment by Location (Top 10)")
    top_locs = df['location'].value_counts().head(10).index
    loc_sent = df[df['location'].isin(top_locs)]
    loc_chart = pd.crosstab(loc_sent['location'], loc_sent['sentiment'], normalize='index') * 100
    st.bar_chart(loc_chart)

# 8. Platform Sentiment
if 'platform' in df.columns:
    st.subheader("8. Platform Sentiment")
    platform_sent = pd.crosstab(df['platform'], df['sentiment'], normalize='index') * 100
    st.dataframe(platform_sent.style.format("{:.1f}"))
    fig, ax = plt.subplots()
    platform_sent.plot(kind='bar', stacked=True, ax=ax)
    st.pyplot(fig)

# 9. Sentiment by ChatGPT Version
if 'version' in df.columns:
    st.subheader("9. Sentiment by ChatGPT Version")
    version_sent = pd.crosstab(df['version'], df['sentiment'], normalize='index') * 100
    st.dataframe(version_sent.style.format("{:.1f}"))

# 10. Common Negative Feedback Themes (Topic Modeling)
st.subheader("10. Common Negative Feedback Themes")
negative_reviews = df[df['sentiment'] == "Negative"]
if not negative_reviews.empty:
    neg_vectorized = vectorizer.transform(negative_reviews['cleaned'])
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(neg_vectorized)
    terms = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_terms = [terms[i] for i in topic.argsort()[-10:]]
        topics.append("Topic #{}: {}".format(topic_idx+1, ", ".join(top_terms)))
    for topic in topics:
        st.markdown(f"- {topic}")
else:
    st.info("No negative reviews available for topic modeling.")

        st.markdown("\U0001F916 **Model Info:** Logistic Regression | TF-IDF Features")
        st.markdown("\U0001F4E6 **Vector Size:** {} features".format(vect_text.shape[1]))

