import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def preprocess_text(text, remove_stopwords=True, lemmatize=True):
    # Remove URLs, mentions, and hashtags
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+|\#", "", text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    if remove_stopwords:
        tokens = [word for word in tokens if word.lower() not in stop_words]
    
    # Lemmatize
    if lemmatize:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return " ".join(tokens)

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Analyze sentiment using VADER
def analyze_sentiment_vader(text):
    sentiment = analyzer.polarity_scores(text)
    if sentiment['compound'] >= 0.05:
        return "Positive"
    elif sentiment['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Analyze sentiment using TextBlob
def analyze_sentiment_textblob(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Train a machine learning model
def train_model(X_train, y_train):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    return vectorizer, model

# Streamlit App
st.title("Sentiment Analysis App")
st.write("Upload a CSV file with a 'text' column to perform sentiment analysis.")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(df.head())

    # Add checkboxes for preprocessing options
    remove_stopwords = st.checkbox("Remove Stopwords", value=True)
    lemmatize = st.checkbox("Lemmatize Text", value=True)

    # Preprocess data
    if 'text' in df.columns:
        df['cleaned_text'] = df['text'].apply(lambda x: preprocess_text(x, remove_stopwords, lemmatize))
        st.write("Data Preprocessing Completed.")

        # Add a dropdown for model selection
        model_option = st.selectbox(
            "Choose a Sentiment Analysis Model:",
            ("VADER", "TextBlob", "Machine Learning (Naive Bayes)")
        )

        # Perform sentiment analysis
        st.write("Performing Sentiment Analysis...")
        if model_option == "VADER":
            df['sentiment'] = df['cleaned_text'].apply(analyze_sentiment_vader)
        elif model_option == "TextBlob":
            df['sentiment'] = df['cleaned_text'].apply(analyze_sentiment_textblob)
        elif model_option == "Machine Learning (Naive Bayes)":
            df['target'] = df['sentiment'].map({'Positive': 1, 'Negative': 0, 'Neutral': 2})
            X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['target'], test_size=0.2, random_state=42)
            vectorizer, model = train_model(X_train, y_train)
            df['sentiment'] = df['cleaned_text'].apply(lambda x: "Positive" if model.predict(vectorizer.transform([x]))[0] == 1 else "Negative")

        # Display sentiment distribution
        st.write("Sentiment Distribution:")
        sentiment_counts = df['sentiment'].value_counts()
        st.bar_chart(sentiment_counts)

        # Word Cloud for Positive and Negative Words
        st.write("Word Cloud for Positive Words:")
        positive_words = " ".join(df[df['sentiment'] == 'Positive']['cleaned_text'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_words)
        st.image(wordcloud.to_array(), use_container_width=True)

        st.write("Word Cloud for Negative Words:")
        negative_words = " ".join(df[df['sentiment'] == 'Negative']['cleaned_text'])
        wordcloud = WordCloud(width=800, height=400, background_color='black').generate(negative_words)
        st.image(wordcloud.to_array(), use_container_width=True)

        # Add polarity scores to the DataFrame
        df['vader_polarity'] = df['cleaned_text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
        df['textblob_polarity'] = df['cleaned_text'].apply(lambda x: TextBlob(x).sentiment.polarity)

        # Display polarity scores
        st.write("Polarity Scores:")
        st.write(df[['cleaned_text', 'vader_polarity', 'textblob_polarity']].head())

        # Generate a summary report
        st.write("Summary Report:")
        st.write("Overall Sentiment Distribution:")
        st.write(df['sentiment'].value_counts())

        st.write("Most Common Positive Words:")
        st.write(pd.Series(positive_words.split()).value_counts().head(5))

        st.write("Most Common Negative Words:")
        st.write(pd.Series(negative_words.split()).value_counts().head(5))

        # Download results
        st.write("Download Results:")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="sentiment_analysis_results.csv",
            mime="text/csv",
        )
    else:
        st.error("The dataset must contain a 'text' column.")