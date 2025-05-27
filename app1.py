import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import streamlit as st

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

def get_headers():
    return {
        'authority': 'www.amazon.com',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,/;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'accept-language': 'en-US,en;q=0.9,bn;q=0.8',
        'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="102", "Google Chrome";v="102"',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'
    }

def get_reviews_url():
    return 'https://www.amazon.com/Fitbit-Smartwatch-Readiness-Exercise-Tracking/product-reviews/B0B4MWCFV4/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews'

def reviewsHtml(url, len_page):
    headers = get_headers()
    soups = []
    for page_no in range(1, len_page + 1):
        params = {
            'ie': 'UTF8',
            'reviewerType': 'all_reviews',
            'filterByStar': 'critical',
            'pageNumber': page_no,
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        soups.append(soup)
    return soups

def get_reviews_data(html_data):
    data_dicts = []
    boxes = html_data.select('div[data-hook="review"]')
    for box in boxes:
        try:
            name = box.select_one('[class="a-profile-name"]').text.strip()
        except Exception as e:
            name = 'N/A'
        try:
            stars = box.select_one('[data-hook="review-star-rating"]').text.strip().split(' out')[0]
        except Exception as e:
            stars = 'N/A'
        try:
            title = box.select_one('[data-hook="review-title"]').text.strip()
        except Exception as e:
            title = 'N/A'
        try:
            datetime_str = box.select_one('[data-hook="review-date"]').text.strip().split(' on ')[-1]
            date = datetime.strptime(datetime_str, '%B %d, %Y').strftime("%d/%m/%Y")
        except Exception as e:
            date = 'N/A'
        try:
            description = box.select_one('[data-hook="review-body"]').text.strip()
        except Exception as e:
            description = 'N/A'
        data_dict = {
            'Name': name,
            'Stars': stars,
            'Title': title,
            'Date': date,
            'Description': description
        }
        data_dicts.append(data_dict)
    return data_dicts

def process_data(html_datas, len_page):
    reviews = []
    for html_data in html_datas:
        review = get_reviews_data(html_data)
        reviews += review
    df_reviews = pd.DataFrame(reviews)
    return df_reviews

def clean_data(df_reviews):
    df_reviews['Description'] = df_reviews['Description'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
    df_reviews['Description'] = df_reviews['Description'].apply(lambda x: x.lower())
    stop_words = set(stopwords.words('english'))
    df_reviews['Description'] = df_reviews['Description'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in stop_words]))
    lemmatizer = WordNetLemmatizer()
    df_reviews['Description'] = df_reviews['Description'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]))
    df_reviews.to_csv('cleaned_reviews.csv', index=False)
    print("Data processing and cleaning completed.")
    return df_reviews

def analyze_sentiment(description):
    analysis = TextBlob(description)
    sentiment = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity
    
    # Adjust confidence calculation
    if subjectivity == 0:
        confidence = abs(sentiment) * 100  # If completely objective, use sentiment directly
    else:
        confidence = (1 - subjectivity) * 100 + abs(sentiment) * 50  # Adjust based on subjectivity

    if sentiment > 0:
        return 'Positive', confidence
    elif sentiment < 0:
        return 'Negative', confidence
    else:
        return 'Neutral', confidence

def train_data(df_reviews):
    df_reviews[['Sentiment', 'Confidence']] = df_reviews['Description'].apply(analyze_sentiment).apply(pd.Series)
    return df_reviews[['Description', 'Sentiment', 'Confidence']]

def visualize_data(df_reviews):
    st.subheader("Visualized Data:")

    st.subheader("Sentiment Distribution:")
    info_text = '''
                - This visualization represents the distribution of sentiment categories in the reviews.
                - Each bar represents a different sentiment category: Positive, Negative, or Neutral.
                - The size of each bar indicates the proportion of reviews belonging to that sentiment category.
                '''
    with st.expander("💡Info"):
        st.write(info_text)

    sentiment_counts = df_reviews['Sentiment'].value_counts()
    st.bar_chart(sentiment_counts)

    st.subheader("Pie Chart:")
    visualize_pie_chart(df_reviews)

    st.subheader("Histogram:")
    visualize_histogram(df_reviews)

def visualize_pie_chart(df_reviews):
    info_text = '''
        - This chart shows the proportion of positive, negative, and neutral sentiments in the reviews.
        '''
    with st.expander("💡Info"):
        st.write(info_text)

    sentiment_counts = df_reviews['Sentiment'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(
        sentiment_counts, 
        labels=sentiment_counts.index, 
        autopct='%1.1f%%', 
        colors=['#4CAF50', '#FFC107', '#F44336'],  # Green, Yellow, Red for Positive, Neutral, Negative
        startangle=90
    )
    ax.axis('equal')  # Equal aspect ratio ensures that the pie is drawn as a circle.
    st.pyplot(fig)

def visualize_histogram(df_reviews):
    info_text = '''
        - This histogram shows the distribution of confidence scores across positive, negative, and neutral sentiments.
        '''
    with st.expander("💡Info"):
        st.write(info_text)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        data=df_reviews, 
        x='Confidence', 
        hue='Sentiment', 
        bins=20, 
        kde=True, 
        palette={'Positive': '#4CAF50', 'Neutral': '#FFC107', 'Negative': '#F44336'},
        ax=ax
    )
    ax.set_title('Confidence Scores by Sentiment')
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Frequency')

    st.pyplot(fig)


def import_data(file_path):
    """
    Import and dynamically analyze a CSV file, identify key columns, and visualize the dataset.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned and analyzed DataFrame.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Drop duplicate rows if any
        df.drop_duplicates(inplace=True)

        # Fill missing values with a placeholder or appropriate value
        df.fillna("N/A", inplace=True)

        # Dynamically identify review and rating columns
        review_col = None
        rating_col = None

        for col in df.columns:
            if df[col].dtype == 'object' and df[col].str.len().mean() > 10:  # Assuming review text is longer
                review_col = col
            elif df[col].dtype in ['int64', 'float64']:  # Assuming ratings are numeric
                rating_col = col

        if not review_col or not rating_col:
            st.error("Could not find appropriate columns for reviews or ratings.")
            return None
        
        # Perform sentiment analysis on the reviews
        df['Sentiment'], df['Confidence'] = zip(*df[review_col].apply(analyze_sentiment))

        # If a rating column exists, you can also analyze sentiment based on ratings
        if rating_col:
            df['Rating_Sentiment'] = df[rating_col].apply(lambda x: 'Positive' if x >= 4 else 'Negative' if x <= 2 else 'Neutral')
        
        # Save the cleaned data for further use
        df.to_csv('cleaned_reviews_with_sentiment.csv', index=False)
        st.success("Data successfully imported and cleaned.")
        
        return df
    
    except Exception as e:
        st.error(f"Enter the valid data")
        return None

def app():


    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file is not None:
        df = import_data(uploaded_file)
        if df is not None:
            visualize_data(df)

if __name__ == "__main__":
    app()
  