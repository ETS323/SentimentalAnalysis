# Comprehensive Sentiment Analysis Application with Custom Word Labeling

# Core Libraries
import streamlit as st
import pandas as pd
import re
import nltk
import sqlite3
import hashlib
from tqdm import tqdm

# NLP Processing
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Sentiment Analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go

# Reporting
from fpdf import FPDF

# Initialize application configuration
st.set_page_config(
    page_title="Sentiment Analyzer Pro",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {background-color: #4a90e2; color: white;}
    .stDownloadButton>button {background-color: #34a853;}
    .stAlert {border-left: 5px solid #4a90e2;}
    h1 {color: #2c3e50;}
    h2 {color: #34495e;}
    .css-1aumxhk {background-color: #ffffff;}
    </style>
    """, unsafe_allow_html=True)

# Initialize NLP resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
tqdm.pandas()

# Initialize NLP components
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
analyzer = SentimentIntensityAnalyzer()

# Database Configuration
def init_db():
    """Initialize SQLite database for user management and custom labels"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS custom_labels (
            username TEXT,
            word TEXT,
            sentiment TEXT,
            PRIMARY KEY (username, word),
            FOREIGN KEY (username) REFERENCES users(username)
        )
    ''')
    conn.commit()
    conn.close()

def hash_password(password):
    """Secure password hashing using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def add_user(username, password):
    """Add new user to database with validation"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed_password = hash_password(password)
    try:
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', 
                (username, hashed_password))
        conn.commit()
        st.success("Account created successfully! Please login.")
    except sqlite3.IntegrityError:
        st.error("Username already exists.")
    finally:
        conn.close()

def authenticate_user(username, password):
    """Authenticate user credentials"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed_password = hash_password(password)
    c.execute('SELECT * FROM users WHERE username = ? AND password = ?', 
            (username, hashed_password))
    user = c.fetchone()
    conn.close()
    return user is not None

init_db()

# Custom Label Functions
def get_custom_sentiment(text, username, remove_stopwords=True, lemmatize=True):
    """Check text for custom labeled words and return dominant sentiment"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT word, sentiment FROM custom_labels WHERE username = ?', (username,))
    custom_labels = c.fetchall()
    conn.close()
    
    # Preprocess text and custom words using current settings
    cleaned_text = preprocess_text(text, remove_stopwords, lemmatize).split()
    preprocessed_labels = {
        preprocess_word(word, remove_stopwords, lemmatize): sentiment 
        for word, sentiment in custom_labels
    }
    
    found_sentiments = []
    for token in cleaned_text:
        if token in preprocessed_labels:
            found_sentiments.append(preprocessed_labels[token])
    
    if not found_sentiments:
        return None
    
    # Return most common sentiment
    from collections import Counter
    return Counter(found_sentiments).most_common(1)[0][0]

def preprocess_word(word, remove_stopwords, lemmatize):
    """Preprocess individual words for label matching"""
    word = re.sub(r"[^\w\s]", "", word.lower())
    tokens = word_tokenize(word)
    if remove_stopwords:
        tokens = [w for w in tokens if w not in stop_words]
    if lemmatize:
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return tokens[0] if tokens else ""

# Text Processing Functions
def preprocess_text(text, remove_stopwords=True, lemmatize=True):
    """Clean and normalize text input"""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+|\#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    
    tokens = word_tokenize(text)
    if remove_stopwords:
        tokens = [word for word in tokens if word.lower() not in stop_words]
    if lemmatize:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return " ".join([word for word in tokens if len(word) > 2])

# Sentiment Analysis Functions
def analyze_sentiment_vader(text, username, remove_stopwords=True, lemmatize=True):
    """Enhanced VADER analysis with custom words"""
    custom = get_custom_sentiment(text, username, remove_stopwords, lemmatize)
    if custom:
        return custom
    
    cleaned = preprocess_text(text, remove_stopwords, lemmatize)
    sentiment = analyzer.polarity_scores(cleaned)
    if sentiment['compound'] >= 0.05:
        return "Positive"
    elif sentiment['compound'] <= -0.05:
        return "Negative"
    return "Neutral"

def analyze_sentiment_textblob(text, username, remove_stopwords=True, lemmatize=True):
    """Enhanced TextBlob analysis with custom words"""
    custom = get_custom_sentiment(text, username, remove_stopwords, lemmatize)
    if custom:
        return custom
    
def detect_emotion(text):
    """Advanced emotion detection system"""
    analysis = TextBlob(text)
    vader = analyzer.polarity_scores(text)
    
    # Emotion detection rules
    if vader['compound'] >= 0.5 and analysis.sentiment.polarity >= 0.3:
        return "Joy"
    elif vader['compound'] <= -0.5 and analysis.sentiment.subjectivity >= 0.5:
        return "Anger"
    elif vader['compound'] <= -0.2 and analysis.sentiment.polarity < 0:
        return "Sadness"
    elif analysis.sentiment.subjectivity >= 0.7:
        return "Surprise"
    elif vader['compound'] >= 0.1:
        return "Positive"
    elif vader['compound'] <= -0.1:
        return "Negative"
    return "Neutral"
    
    cleaned = preprocess_text(text, remove_stopwords, lemmatize)
    analysis = TextBlob(cleaned)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    return "Neutral"

# Reporting Functions
def generate_pdf(df):
    """Generate PDF report from analysis results"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Sentiment Analysis Report", ln=True, align="C")
    
    pdf.cell(200, 10, txt="Sentiment Distribution:", ln=True)
    for sentiment, count in df['sentiment'].value_counts().items():
        pdf.cell(200, 10, txt=f"{sentiment}: {count}", ln=True)
    
    pdf.cell(200, 10, txt="Most Common Words:", ln=True)
    for sentiment in df['sentiment'].unique():
        words = " ".join(df[df['sentiment'] == sentiment]['cleaned_text'])
        common_words = pd.Series(words.split()).value_counts().head(5)
        pdf.cell(200, 10, txt=f"{sentiment} Words:", ln=True)
        for word, count in common_words.items():
            pdf.cell(200, 10, txt=f"- {word}: {count}", ln=True)
    
    return pdf.output(dest="S").encode("latin1")

# Authentication Component
def auth_section():
    """User authentication interface"""
    with st.sidebar:
        if not st.session_state.get('authenticated'):
            st.title("🔐 Authentication")
            auth_mode = st.radio("Choose action:", 
                               ("Login", "Create Account"), 
                               label_visibility="collapsed")
            
            with st.form("auth_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                
                if auth_mode == "Create Account":
                    confirm_password = st.text_input("Confirm Password", type="password")
                
                if st.form_submit_button("Continue →" if auth_mode == "Login" else "Create Account"):
                    if auth_mode == "Login":
                        if authenticate_user(username, password):
                            st.session_state.authenticated = True
                            st.session_state.username = username
                            st.rerun()
                        else:
                            st.error("Invalid credentials")
                    else:
                        if password == confirm_password:
                            add_user(username, password)
                        else:
                            st.error("Passwords do not match")
        else:
            st.success(f"Welcome back, {st.session_state.username}!")
            if st.button("Logout"):
                st.session_state.clear()
                st.rerun()

# Machine Learning Functions
def train_model(X_train, y_train):
    """Train and return classification model"""
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    return vectorizer, model

# Visualization Functions
def enhanced_visualizations(df):
    """Generate advanced analytical visualizations"""
    st.subheader("📈 Sentiment Score Correlation")
    heatmap_data = pd.DataFrame({
        'VADER': df['vader_score'],
        'TextBlob': df['textblob_score']
    }).corr()
    
    fig1 = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='RdBu',
        zmid=0,
        text=heatmap_data.values.round(2),
        hoverinfo="text",
        texttemplate="%{text}",
        colorbar=dict(title="Correlation")
    ))
    fig1.update_layout(
        width=800,
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        title="Correlation Between Sentiment Scores"
    )
    st.plotly_chart(fig1)
    
    st.subheader("🎭 Emotion Distribution Analysis")
    with st.spinner("Analyzing emotional content..."):
        if 'emotion' not in df.columns:
            df['emotion'] = df['text'].progress_apply(detect_emotion)
    
    emotion_counts = df['emotion'].value_counts().reset_index()
    emotion_counts.columns = ['Emotion', 'Count']
    
    fig2 = px.pie(emotion_counts, 
                values='Count', 
                names='Emotion',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel,
                title="Emotional Content Distribution")
    
    fig2.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        pull=[0.1 if i == emotion_counts['Count'].idxmax() else 0 
             for i in range(len(emotion_counts))],
        marker=dict(line=dict(color='#000000', width=0.5)))
    
    fig2.update_layout(
        showlegend=False,
        height=500,
        margin=dict(l=0, r=0, t=60, b=0),
        uniformtext_minsize=12,
        uniformtext_mode='hide'
    )
    st.plotly_chart(fig2)

# Main Application Interface
def main_app():
    """Primary application interface"""
    st.title("📈 Sentiment Analysis Dashboard")
    
    # Real-Time Analysis Section
    with st.expander("🔍 Real-Time Text Analysis", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            live_text = st.text_area("Analyze text instantly:", 
                                   placeholder="Enter any text to analyze...",
                                   height=100)
        with col2:
            add_to_dataset = st.checkbox("Add to dataset", True)
            preprocess_live = st.checkbox("Preprocess text", True)
        
        if live_text:
            analysis = {
                "VADER": analyze_sentiment_vader(live_text, st.session_state.username, preprocess_live, preprocess_live),
                "TextBlob": analyze_sentiment_textblob(live_text, st.session_state.username, preprocess_live, preprocess_live),
                "Raw Text": live_text,
                "Cleaned Text": preprocess_text(live_text, preprocess_live, preprocess_live)
            }
            
            cols = st.columns(3)
            with cols[0]:
                st.metric("VADER Sentiment", analysis["VADER"])
            with cols[1]:
                st.metric("TextBlob Sentiment", analysis["TextBlob"])
            with cols[2]:
                st.write("**Processed Text:**")
                st.caption(analysis["Cleaned Text"][:200] + "...")
            
            if add_to_dataset:
                if 'df' not in st.session_state:
                    st.session_state.df = pd.DataFrame()
                
                new_row = pd.DataFrame([{
                    'text': analysis["Raw Text"],
                    'cleaned_text': analysis["Cleaned Text"],
                    'vader_sentiment': analysis["VADER"],
                    'textblob_sentiment': analysis["TextBlob"]
                }])
                st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
                st.success("Text added to dataset!")

    # File Upload Section
    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], 
                                           help="Requires a 'text' column")
        with col2:
            if st.button("🔄 Reset Session"):
                st.session_state.clear()
                st.rerun()
    
    # Data Processing Pipeline with Encoding Handling
    if uploaded_file or ('df' in st.session_state and not st.session_state.df.empty):
        try:
            if uploaded_file:
                try:
                    # First try UTF-8 encoding
                    st.session_state.df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    # Fallback to Latin-1 encoding
                    try:
                        uploaded_file.seek(0)
                        st.session_state.df = pd.read_csv(uploaded_file, encoding='latin1')
                        st.info("File loaded using Latin-1 encoding. Some characters may not display correctly.")
                    except Exception as e:
                        st.error(f"Error loading file: {str(e)}")
                        st.session_state.df = pd.DataFrame()
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
                    st.session_state.df = pd.DataFrame()
                
                if 'text' not in st.session_state.df.columns:
                    st.error("CSV file must contain a 'text' column")
                    st.session_state.df = pd.DataFrame()
                    return
                    
            df = st.session_state.df
            
            # Sidebar Configuration
            with st.sidebar:
                # Processing Settings
                with st.expander("⚙️ Processing Settings"):
                    remove_stopwords = st.checkbox("Remove stopwords", True)
                    lemmatize = st.checkbox("Lemmatize text", True)
                    model_choice = st.selectbox("Analysis Model", 
                                              ["VADER", "TextBlob", "Machine Learning (Naive Bayes)"])
                    
                    use_custom_labels = False
                    if model_choice == "Machine Learning (Naive Bayes)":
                        use_custom_labels = st.checkbox(
                            "Use custom sentiment labels",
                            help="Requires 'sentiment' column with values: Positive, Negative, Neutral"
                        )

                # Custom Word Labeling UI
                with st.expander("🔖 Custom Word Labels"):
                    st.write("Teach the model specific word meanings:")
                    new_word = st.text_input("Enter word/phrase")
                    new_sentiment = st.selectbox("Assign sentiment", 
                                               ["Positive", "Negative", "Neutral"])
                    
                    if st.button("Add Label"):
                        if new_word.strip():
                            conn = sqlite3.connect('users.db')
                            c = conn.cursor()
                            try:
                                c.execute('INSERT INTO custom_labels VALUES (?, ?, ?)',
                                         (st.session_state.username, new_word.lower(), new_sentiment))
                                conn.commit()
                                st.success("Label added!")
                            except sqlite3.IntegrityError:
                                st.error("This word already has a label")
                            finally:
                                conn.close()
                        else:
                            st.error("Please enter a word/phrase")
                    
                    # Show existing labels
                    conn = sqlite3.connect('users.db')
                    c = conn.cursor()
                    c.execute('SELECT word, sentiment FROM custom_labels WHERE username = ?',
                             (st.session_state.username,))
                    labels = c.fetchall()
                    conn.close()
                    
                    if labels:
                        st.write("Your custom labels:")
                        for word, sentiment in labels:
                            cols = st.columns([4, 1])
                            cols[0].write(f"`{word}` → {sentiment}")
                            if cols[1].button("×", key=f"del_{word}"):
                                conn = sqlite3.connect('users.db')
                                c = conn.cursor()
                                c.execute('DELETE FROM custom_labels WHERE username = ? AND word = ?',
                                         (st.session_state.username, word))
                                conn.commit()
                                conn.close()
                                st.rerun()

            # Text Cleaning
            if 'cleaned_text' not in df.columns:
                df['cleaned_text'] = df['text'].progress_apply(
                    lambda x: preprocess_text(x, remove_stopwords, lemmatize))
            
            # Model Execution
            if model_choice == "VADER":
                df['sentiment'] = df['text'].apply(
                    lambda x: analyze_sentiment_vader(x, st.session_state.username, remove_stopwords, lemmatize))
            elif model_choice == "TextBlob":
                df['sentiment'] = df['text'].apply(
                    lambda x: analyze_sentiment_textblob(x, st.session_state.username, remove_stopwords, lemmatize))
            else:
                if use_custom_labels:
                    if 'sentiment' not in df.columns:
                        st.error("Dataset must contain 'sentiment' column for custom labels")
                        return
                    
                    df['sentiment'] = df['sentiment'].str.strip().str.capitalize()
                    valid_labels = ['Positive', 'Negative', 'Neutral']
                    invalid = df[~df['sentiment'].isin(valid_labels)]
                    
                    if not invalid.empty:
                        st.error(f"Invalid labels: {invalid['sentiment'].unique()}")
                        return
                else:
                    st.info("Using VADER-generated labels for training")
                    df['sentiment'] = df['text'].apply(
                        lambda x: analyze_sentiment_vader(x, st.session_state.username, remove_stopwords, lemmatize))
                
                valid_labels = ['Positive', 'Negative', 'Neutral']
                df = df[df['sentiment'].isin(valid_labels)]
                if df.empty:
                    st.error("No valid sentiment labels found")
                    return
                
                present_sentiments = sorted(df['sentiment'].unique())
                if len(present_sentiments) < 2:
                    st.error("Requires at least 2 sentiment classes")
                    return
                
                sentiment_mapping = {label: idx for idx, label in enumerate(present_sentiments)}
                df['target'] = df['sentiment'].map(sentiment_mapping)
                
                X_train, X_test, y_train, y_test = train_test_split(
                    df['cleaned_text'], df['target'], test_size=0.2, random_state=42
                )
                vectorizer, model = train_model(X_train, y_train)
                df['sentiment'] = df['text'].apply(
                    lambda x: get_custom_sentiment(x, st.session_state.username, remove_stopwords, lemmatize)
                    or list(sentiment_mapping.keys())[int(model.predict(vectorizer.transform(
                        [preprocess_text(x, remove_stopwords, lemmatize)]
                    )[0]))]
                )
                
                # Model Evaluation
                X_test_tfidf = vectorizer.transform(X_test)
                y_pred = model.predict(X_test_tfidf)
                
                st.subheader("📊 Model Performance")
                cols = st.columns(4)
                metrics = {
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, average='weighted'),
                    "Recall": recall_score(y_test, y_pred, average='weighted'),
                    "F1-Score": f1_score(y_test, y_pred, average='weighted')
                }
                for col, (name, value) in zip(cols, metrics.items()):
                    with col:
                        st.metric(name, f"{value:.2%}")
                
                with st.expander("📄 Classification Report"):
                    present_classes = sorted(list(set(y_test) | set(y_pred)))
                    class_names = [list(sentiment_mapping.keys())[i] for i in present_classes]
                    report = classification_report(
                        y_test, y_pred,
                        labels=present_classes,
                        target_names=class_names,
                        zero_division=0
                    )
                    st.code(report)
                
                with st.expander("🔍 Feature Importance"):
                    feature_names = vectorizer.get_feature_names_out()
                    for idx, label in enumerate(present_sentiments):
                        st.write(f"**{label} Indicators:**")
                        features = sorted(zip(model.feature_log_prob_[idx], feature_names), 
                                      reverse=True)[:10]
                        for score, word in features:
                            st.write(f"- {word} ({score:.2f})")

            # Main Interface Tabs
            tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Insights", "📥 Export"])
            
            with tab1:
                st.subheader("Sentiment Distribution")
                
                chart_type = st.selectbox("Choose Visualization Style:", 
                                        ["Bar Chart", "Pie Chart"],
                                        key="chart_type")
                
                sentiment_counts = df['sentiment'].value_counts().reset_index()
                sentiment_counts.columns = ['Sentiment', 'Count']
                
                if chart_type == "Bar Chart":
                    fig = px.bar(
                        sentiment_counts,
                        x='Sentiment',
                        y='Count',
                        color='Sentiment',
                        color_discrete_sequence=px.colors.qualitative.Vivid,
                        template='plotly_white',
                        labels={'Count': 'Number of Texts', 'Sentiment': 'Sentiment Category'},
                        title='Sentiment Distribution - Interactive Bar Chart'
                    )
                    fig.update_layout(
                        xaxis_tickangle=-45,
                        hovermode='x unified',
                        showlegend=False
                    )
                else:
                    fig = px.pie(
                        sentiment_counts,
                        names='Sentiment',
                        values='Count',
                        color_discrete_sequence=px.colors.qualitative.Pastel,
                        hole=0.35,
                        template='plotly_white',
                        labels={'Count': 'Number of Texts'},
                        title='Sentiment Distribution - Interactive Pie Chart'
                    )
                    fig.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        pull=[0.02]*len(sentiment_counts),
                        rotation=45
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Polarity Scores")
                df['vader_score'] = df['cleaned_text'].apply(
                    lambda x: analyzer.polarity_scores(x)['compound'])
                df['textblob_score'] = df['cleaned_text'].apply(
                    lambda x: TextBlob(x).sentiment.polarity)
                
                cols = st.columns(2)
                cols[0].metric("Avg VADER", f"{df['vader_score'].mean():.2f}")
                cols[1].metric("Avg TextBlob", f"{df['textblob_score'].mean():.2f}")
            
            with tab2:
                st.subheader("Word Clouds")
                cols = st.columns(3)
                sentiments = df['sentiment'].unique()
                colors = ['Greens', 'Reds', 'Blues'][:len(sentiments)]
                for col, sentiment, color in zip(cols, sentiments, colors):
                    with col:
                        st.markdown(f"**{sentiment} Words**")
                        text = " ".join(df[df['sentiment'] == sentiment]['cleaned_text'])
                        if text:
                            wc = WordCloud(width=400, height=300,
                                         background_color='white', 
                                         colormap=color).generate(text)
                            st.image(wc.to_array())
                
                enhanced_visualizations(df)
                
                st.subheader("Example Texts")
                selected_sentiment = st.selectbox("Filter by sentiment", sentiments)
                st.dataframe(df[df['sentiment'] == selected_sentiment][['text', 'sentiment']].head(10),
                             height=300)
            
            with tab3:
                st.subheader("Export Results")
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV", csv, "analysis.csv", "text/csv")
                st.download_button("Generate PDF", generate_pdf(df), 
                                  "report.pdf", "application/pdf")

            # Support Information
            with st.expander("📞 Contact Support"):
                st.markdown("""
                **Need help?**
                - Email: i23024766@student.newinti.edu.my
                - Docs: [documentation](https://example.com)
                - GitHub: [repository](https://github.com)
                """)

        except Exception as e:
            st.error(f"Processing error: {str(e)}")
    else:
        with st.expander("📞 Contact Support"):
            st.markdown("""
            **Contact our team:**
            - Support email: i23024766@student.newinti.edu.my
            - Documentation portal: https://docs.sentimentapp.com
            - GitHub repository: https://github.com/sentiment-app
            """)

# Application Control Flow
auth_section()
if st.session_state.get('authenticated'):
    main_app()
else:
    st.info("Please authenticate to access analysis features")
    # Replace with your actual image path
    st.image("C:/Users/yewhe/OneDrive/Pictures/Screenshots/Screenshot 2025-04-15 042445.png")