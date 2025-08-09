# AI-echo-sentiment-analysis
📊 AI Echo is a Natural Language Processing (NLP) application designed to analyze ChatGPT user reviews and classify them into Positive, Neutral, or Negative sentiments.
It helps in:

Understanding user satisfaction trends

Identifying common complaints

Tracking sentiment changes over time

Comparing platform-specific experiences

The project includes:

Machine Learning Model (TF-IDF + Logistic Regression, optionally BERT embeddings)

Streamlit Dashboard for real-time predictions and analysis

Data visualizations such as word clouds, sentiment timelines, and topic modeling

🚀 Features
✅ Model Features
Text preprocessing (stopword removal, lemmatization)

TF-IDF vectorization with n-grams

Manual sentiment overrides for short keywords like "good", "bad"

Option to use BERT embeddings for better accuracy

📊 Dashboard Features
Real-time sentiment prediction for custom text

Sentiment distribution charts

Sentiment vs rating analysis

Word clouds for each sentiment category

Sentiment trends over time

Verified vs Non-verified user sentiment

Review length vs sentiment

Sentiment by location, platform, version

Topic modeling for negative reviews

📊 Dataset
The dataset contains ChatGPT review data with the following fields:

date – Review submission date

title – Short review title

review – Full review text

rating – Numerical rating (1–5)

username – Reviewer name

helpful_votes – Number of helpful votes

platform – Web/Mobile

language – Language code

location – Country

version – ChatGPT version

verified_purchase – Yes/No

review_length – Number of words in review

📌 File: chatgpt_style_reviews_dataset.xlsx

📊 Dashboard Preview
The Streamlit dashboard includes:

Sentiment Prediction for any input review

Overall Sentiment Distribution chart

Sentiment vs Rating analysis

Word Clouds for Positive, Neutral, Negative reviews

Sentiment Over Time trends

Verified vs Non-verified Sentiment

Review Length vs Sentiment

Sentiment by Location/Platform/Version

Topic Modeling for negative feedback

🛠️ Technologies Used
Python

Streamlit – Dashboard UI

Pandas, NumPy – Data handling

scikit-learn – Machine learning

NLTK – Text preprocessing

WordCloud – Keyword visualization

Matplotlib, Seaborn – Data visualization

Transformers (Hugging Face) – BERT embeddings

🔮 Future Improvements
Deploy as a web app using Streamlit Cloud / AWS / Azure

Add multi-language sentiment analysis

Improve neutral sentiment detection

Implement real-time data fetching from app store reviews
