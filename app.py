import streamlit as st
import pickle
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem.porter import PorterStemmer

# Load models
ps = PorterStemmer()
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Page config
st.set_page_config(page_title="Spam Classifier", page_icon="📩", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #4CAF50;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #AAAAAA;
        margin-bottom: 30px;
    }
    .spam {
        background-color: #ff4b4b;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-size: 24px;
        font-weight: bold;
    }
    .ham {
        background-color: #4CAF50;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">📩 SMS / Email Spam Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect whether a message is Spam or Not</div>', unsafe_allow_html=True)

# Input box
sms = st.text_area("Enter your message below:", placeholder="Type or paste your SMS/email here...")

# Preprocess function (NO NLTK TOKENIZER)
def text_transform(text):
    text = text.lower()

    # Simple tokenization (safe for deployment)
    words = text.split()

    # Remove non-alphanumeric
    words = [word for word in words if word.isalnum()]

    # Remove stopwords + punctuation
    words = [word for word in words if word not in ENGLISH_STOP_WORDS and word not in string.punctuation]

    # Stemming
    words = [ps.stem(word) for word in words]

    return " ".join(words)

# Predict button
if st.button("🔍 Analyze Message"):
    if sms.strip() == "":
        st.warning("Please enter a message first.")
    else:
        transformed_sms = text_transform(sms)
        vector_sms = tfidf.transform([transformed_sms])
        output = model.predict(vector_sms)[0]

        if output == 1:
            st.markdown('<div class="spam">🚨 Spam Message</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="ham">✅ Not Spam</div>', unsafe_allow_html=True)
