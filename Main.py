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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import sqlite3
import hashlib

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Initialize SQLite database for user authentication
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Hash password for security
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Add a new user to the database
def add_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed_password = hash_password(password)
    try:
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
        conn.commit()
        st.success("User created successfully!")
    except sqlite3.IntegrityError:
        st.error("Username already exists.")
    finally:
        conn.close()

# Authenticate a user
def authenticate_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed_password = hash_password(password)
    c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, hashed_password))
    user = c.fetchone()
    conn.close()
    return user is not None

# Initialize the database
init_db()

# Text preprocessing function
def preprocess_text(text, remove_stopwords=True, lemmatize=True):
    # Remove URLs, mentions, and hashtags
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+|\#", "", text)
    
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    if remove_stopwords:
        tokens = [word for word in tokens if word.lower() not in stop_words]
    
    # Lemmatize
    if lemmatize:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Filter out short words (e.g., 1-2 characters)
    tokens = [word for word in tokens if len(word) > 2]
    
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

# Generate PDF report
def generate_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Add title
    pdf.cell(200, 10, txt="Sentiment Analysis Report", ln=True, align="C")
    
    # Add sentiment distribution
    pdf.cell(200, 10, txt="Sentiment Distribution:", ln=True)
    sentiment_counts = df['sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        pdf.cell(200, 10, txt=f"{sentiment}: {count}", ln=True)
    
    # Add most common words
    pdf.cell(200, 10, txt="Most Common Positive Words:", ln=True)
    positive_words = " ".join(df[df['sentiment'] == 'Positive']['cleaned_text'])
    common_positive = pd.Series(positive_words.split()).value_counts().head(5)
    for word, count in common_positive.items():
        pdf.cell(200, 10, txt=f"{word}: {count}", ln=True)
    
    pdf.cell(200, 10, txt="Most Common Negative Words:", ln=True)
    negative_words = " ".join(df[df['sentiment'] == 'Negative']['cleaned_text'])
    common_negative = pd.Series(negative_words.split()).value_counts().head(5)
    for word, count in common_negative.items():
        pdf.cell(200, 10, txt=f"{word}: {count}", ln=True)
    
    # Save PDF
    pdf_output = pdf.output(dest="S").encode("latin1")
    return pdf_output

# Streamlit App
st.title("Sentiment Analysis App")

# Check if the user is authenticated
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

# Sidebar for user authentication or user info
with st.sidebar:
    if not st.session_state['authenticated']:
        st.header("User Authentication")
        auth_option = st.radio("Choose an option:", ("Login", "Create Account"))

        if auth_option == "Login":
            st.subheader("Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if authenticate_user(username, password):
                    st.session_state['authenticated'] = True
                    st.session_state['username'] = username
                    st.success("Login successful!")
                else:
                    st.error("Invalid username or password.")
        elif auth_option == "Create Account":
            st.subheader("Create Account")
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            if st.button("Create Account"):
                if new_password == confirm_password:
                    add_user(new_username, new_password)
                else:
                    st.error("Passwords do not match.")
    else:
        st.header(f"Welcome, {st.session_state['username']}!")
        if st.button("Logout"):
            st.session_state['authenticated'] = False
            st.session_state['username'] = None
            st.experimental_rerun()

# Only show the sentiment analysis app if the user is authenticated
if st.session_state['authenticated']:
    st.write("Upload a CSV file with a 'text' column to perform sentiment analysis.")

    # Add a sidebar for user inputs
    with st.sidebar:
        st.header("Settings")
        model_option = st.selectbox(
            "Choose a Sentiment Analysis Model:",
            ("VADER", "TextBlob", "Machine Learning (Naive Bayes)")
        )
        remove_stopwords = st.checkbox("Remove Stopwords", value=True)
        lemmatize = st.checkbox("Lemmatize Text", value=True)

    # Upload dataset
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        # Try reading the file with different encodings
        encodings_to_try = ['utf-8', 'latin-1', 'ISO-8859-1', 'Windows-1252']
        df = None

        for encoding in encodings_to_try:
            try:
                uploaded_file.seek(0)  # Reset file pointer
                df = pd.read_csv(uploaded_file, encoding=encoding)
                st.write(f"File successfully read with {encoding} encoding.")
                break
            except UnicodeDecodeError:
                continue
        else:
            st.error("Failed to read the file with all attempted encodings.")
            st.stop()

        # Display dataset preview
        st.header("Dataset Preview")
        st.write(df.head())

        # Preprocess data
        if 'text' in df.columns:
            df['cleaned_text'] = df['text'].apply(lambda x: preprocess_text(x, remove_stopwords, lemmatize))
            st.write("Data Preprocessing Completed.")

            # Perform sentiment analysis
            st.header("Sentiment Analysis Results")
            if model_option == "VADER":
                df['sentiment'] = df['cleaned_text'].apply(analyze_sentiment_vader)
            elif model_option == "TextBlob":
                df['sentiment'] = df['cleaned_text'].apply(analyze_sentiment_textblob)
            elif model_option == "Machine Learning (Naive Bayes)":
                # First, perform sentiment analysis using VADER to create the 'sentiment' column
                df['sentiment'] = df['cleaned_text'].apply(analyze_sentiment_vader)
                
                # Map the 'sentiment' column to create the 'target' column
                df['target'] = df['sentiment'].map({'Positive': 1, 'Negative': 0, 'Neutral': 2})
                
                # Split the data and train the model
                X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['target'], test_size=0.2, random_state=42)
                vectorizer, model = train_model(X_train, y_train)
                
                # Predict sentiment using the trained model
                df['sentiment'] = df['cleaned_text'].apply(lambda x: "Positive" if model.predict(vectorizer.transform([x]))[0] == 1 else "Negative")

            # Add polarity scores to the DataFrame
            df['vader_polarity'] = df['cleaned_text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
            df['textblob_polarity'] = df['cleaned_text'].apply(lambda x: TextBlob(x).sentiment.polarity)

            # Generate positive and negative words for word cloud and most common words
            positive_words = " ".join(df[df['sentiment'] == 'Positive']['cleaned_text'])
            negative_words = " ".join(df[df['sentiment'] == 'Negative']['cleaned_text'])

            # Create tabs for different dashboards
            tab1, tab2, tab3, tab4 = st.tabs(["Graphs", "Polarity Scores", "Summary Report", "Machine Learning"])

            # Graphs Dashboard
            with tab1:
                st.header("Graphs")

                # Display sentiment distribution
                st.subheader("Sentiment Distribution")
                sentiment_counts = df['sentiment'].value_counts()
                st.bar_chart(sentiment_counts)

                # Word Cloud for Positive and Negative Words
                st.subheader("Word Cloud Visualizations")

                st.write("Word Cloud for Positive Words:")
                wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_words)
                st.image(wordcloud_positive.to_array(), use_container_width=True)

                st.write("Word Cloud for Negative Words:")
                wordcloud_negative = WordCloud(width=800, height=400, background_color='black').generate(negative_words)
                st.image(wordcloud_negative.to_array(), use_container_width=True)

            # Polarity Scores Dashboard
            with tab2:
                st.header("Polarity Scores")
                st.write("Polarity Scores:")
                st.write(df[['cleaned_text', 'vader_polarity', 'textblob_polarity']].head())

            # Summary Report Dashboard
            with tab3:
                st.header("Summary Report")
                st.write("Overall Sentiment Distribution:")
                st.write(df['sentiment'].value_counts())

                st.write("Most Common Positive Words:")
                st.write(pd.Series(positive_words.split()).value_counts().head(5))

                st.write("Most Common Negative Words:")
                st.write(pd.Series(negative_words.split()).value_counts().head(5))

            # Machine Learning Dashboard
            with tab4:
                if model_option == "Machine Learning (Naive Bayes)":
                    st.header("Machine Learning")
                    st.write("Model Performance:")
                    y_pred = model.predict(vectorizer.transform(X_test))
                    st.write("Accuracy:", accuracy_score(y_test, y_pred))
                    st.write("Precision:", precision_score(y_test, y_pred, average='weighted'))
                    st.write("Recall:", recall_score(y_test, y_pred, average='weighted'))
                    st.write("F1-Score:", f1_score(y_test, y_pred, average='weighted'))
                    st.write("Classification Report:")
                    st.write(classification_report(y_test, y_pred))

                    # Confusion Matrix
                    st.write("Confusion Matrix:")
                    cm = confusion_matrix(y_test, y_pred)
                    st.write(cm)
                    
                    # Plot confusion matrix
                    st.write("Confusion Matrix Visualization:")
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    st.pyplot(fig)
                else:
                    st.write("Machine Learning Dashboard is only available when using the Naive Bayes model.")

            # Download results
            st.header("Download Results")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="sentiment_analysis_results.csv",
                mime="text/csv",
            )

            # Generate and download PDF report
            st.write("Generate PDF Report:")
            pdf_output = generate_pdf(df)
            st.download_button(
                label="Download PDF",
                data=pdf_output,
                file_name="sentiment_analysis_report.pdf",
                mime="application/pdf",
            )
        else:
            st.error("The dataset must contain a 'text' column.")

    # Add a reset button
    if st.button("Reset App"):
        st.experimental_rerun()

    # Add a help section
    with st.expander("How to Use This App"):
        st.write("""
        1. **Upload a CSV file** containing a 'text' column.
        2. Choose a **sentiment analysis model** from the sidebar.
        3. Customize **preprocessing options** (e.g., remove stopwords, lemmatize).
        4. View the **sentiment distribution**, **word clouds**, and **polarity scores**.
        5. Download the results as a **CSV** or **PDF** report.
        """)

    # Add an about section
    with st.expander("About This App"):
        st.write("""
        This app performs **sentiment analysis** on text data using multiple models:
        - **VADER**: A rule-based model for sentiment analysis.
        - **TextBlob**: A simple and easy-to-use model for sentiment analysis.
        - **Machine Learning (Naive Bayes)**: A machine learning model trained on your data.
        
        The app also provides **visualizations**, **model evaluation metrics**, and the ability to download results as a **CSV** or **PDF** report.
        """)

    # Add a contact section
    with st.expander("Contact"):
        st.write("""
        For feedback or questions, please contact:
        - **Email**: your.email@example.com
        - **GitHub**: [your-github-profile](https://github.com/your-github-profile)
        """)
else:
    st.warning("Please log in or create an account to access the sentiment analysis app.")