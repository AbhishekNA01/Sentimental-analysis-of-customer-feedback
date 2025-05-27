import requests
from bs4 import BeautifulSoup
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from collections import Counter

# Download NLTK resources (only required once)
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Helper Functions
def fetch_reviews(url, pages=1):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    reviews = []
    for page in range(1, pages + 1):
        response = requests.get(f"{url}&pageNumber={page}", headers=headers)
        if response.status_code != 200:
            st.error(f"Failed to fetch page {page}. Status code: {response.status_code}")
            continue

        soup = BeautifulSoup(response.content, "html.parser")
        review_blocks = soup.find_all("div", {"data-hook": "review"})

        for review in review_blocks:
            try:
                title = review.find("a", {"data-hook": "review-title"}).text.strip()
                rating = float(review.find("i", {"data-hook": "review-star-rating"}).text.split()[0])
                text = review.find("span", {"data-hook": "review-body"}).text.strip()
                reviews.append({"Title": title, "Rating": rating, "Review": text})
            except AttributeError:
                continue
    return reviews

def clean_review_text(text):
    # Tokenization and Lemmatization
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    tokens = word_tokenize(text)
    cleaned = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned)

def analyze_sentiments(reviews):
    for review in reviews:
        blob = TextBlob(review["Review"])
        polarity = blob.sentiment.polarity
        if polarity > 0:
            review["Sentiment"] = "Positive"
        elif polarity < 0:
            review["Sentiment"] = "Negative"
        else:
            review["Sentiment"] = "Neutral"
    return reviews

def visualize_sentiments(reviews_df):
    st.subheader("Sentiment Distribution - Bar Graph")
    plt.figure(figsize=(10, 6))
    sns.countplot(data=reviews_df, x="Sentiment", palette="coolwarm")
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    st.pyplot(plt)

    st.subheader("Sentiment Distribution - Pie Chart")
    sentiment_counts = reviews_df["Sentiment"].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(
        sentiment_counts,
        labels=sentiment_counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=sns.color_palette("coolwarm", len(sentiment_counts))
    )
    plt.title("Sentiment Proportion")
    st.pyplot(plt)

# Streamlit App
st.title("Amazon Product Review Sentiment Analyzer")
product_url = st.text_input("Enter Amazon Product URL:")
pages = st.number_input("Number of Pages to Analyze", min_value=1, max_value=10, value=1)

if st.button("Fetch and Analyze"):
    if not product_url:
        st.error("Please enter a valid Amazon product URL.")
    else:
        st.info("Fetching reviews...")
        reviews = fetch_reviews(product_url, pages)

        if not reviews:
            st.warning("No reviews found for the given product.")
        else:
            st.success(f"Fetched {len(reviews)} reviews successfully!")
            reviews_df = pd.DataFrame(reviews)
            reviews_df["Cleaned_Review"] = reviews_df["Review"].apply(clean_review_text)
            reviews_df = pd.DataFrame(analyze_sentiments(reviews))

            st.write("Sample Reviews Data", reviews_df.head())
            visualize_sentiments(reviews_df)
