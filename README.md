# Sentiment Analyzer Pro

Advanced sentiment analysis application with custom word labeling and machine learning integration.

## Features
- Real-time text analysis
- CSV file processing
- Custom word sentiment labeling
- VADER and TextBlob integration
- Machine learning model training
- Interactive visualizations
- PDF report generation
- User authentication system

## Installation
```bash
git clone https://github.com/yourusername/sentiment-analyzer-pro.git
cd sentiment-analyzer-pro
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords wordnet


## Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows

## Install dependencies
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords wordnet

## Launch application
streamlit run app.py

## Usage
1. Create an account or login
2. Upload CSV file with 'text' column or analyze live text
3. Configure processing settings in the sidebar
4. View interactive dashboards and visualizations
5. Export results as CSV or PDF