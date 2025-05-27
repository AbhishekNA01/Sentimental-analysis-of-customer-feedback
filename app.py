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
from app1 import app



nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

def get_headers():
    """Return headers for the HTTP requests."""
    return {
        'authority': 'www.amazon.com',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'accept-language': 'en-US,en;q=0.9,bn;q=0.8',
        'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="102", "Google Chrome";v="102"',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'
    }

def get_reviews_url(product_url):
    return product_url


def reviewsHtml(url, len_page):
    """Fetch review pages and parse HTML."""
    headers = get_headers()
    soups = []
    for page_no in range(1, len_page + 1):
        params = {
            'ie': 'UTF8',
            'reviewerType': 'all_reviews',
            'filterByStar': 'critical',
            'pageNumber': page_no,
        }
        # Send request and parse HTML
        response = requests.get(url, headers=headers, params=params)
        soup = BeautifulSoup(response.text, 'html.parser')
        soups.append(soup)
    return soups

def get_reviews_data(html_data):
    """Extract review data from HTML soup."""
    data_dicts = []
    boxes = html_data.select('div[data-hook="review"]')  # Select all review containers
    for box in boxes:
        # Extract review data with error handling
        name = box.select_one('[class="a-profile-name"]').text.strip() if box.select_one('[class="a-profile-name"]') else 'N/A'
        stars = box.select_one('[data-hook="review-star-rating"]').text.strip().split(' out')[0] if box.select_one('[data-hook="review-star-rating"]') else 'N/A'
        title = box.select_one('[data-hook="review-title"]').text.strip() if box.select_one('[data-hook="review-title"]') else 'N/A'
        date = 'N/A'
        if box.select_one('[data-hook="review-date"]'):
            try:
                datetime_str = box.select_one('[data-hook="review-date"]').text.strip().split(' on ')[-1]
                date = datetime.strptime(datetime_str, '%B %d, %Y').strftime("%d/%m/%Y")
            except ValueError:
                pass
        description = box.select_one('[data-hook="review-body"]').text.strip() if box.select_one('[data-hook="review-body"]') else 'N/A'

        data_dicts.append({
            'Name': name,
            'Stars': stars,
            'Title': title,
            'Date': date,
            'Description': description
        })
    return data_dicts

def process_data(html_datas, len_page):
    reviews = []
    for html_data in html_datas:
        reviews += get_reviews_data(html_data)
    return pd.DataFrame(reviews)


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
                - For example, if the "Positive" bar is larger, it means there are more positive reviews compared to negative or neutral ones
    '''
    with st.expander("💡Info"):
        st.write(info_text)

    sentiment_counts = df_reviews['Sentiment'].value_counts()
    st.bar_chart(sentiment_counts)

    st.subheader("Pie Chart:")
    visualize_pie_chart(df_reviews)

    st.subheader("Histogram:")
    visualize_histogram(df_reviews)

    st.subheader("Distribution of Review Length:")
    visualize_review_length_distribution(df_reviews)
            
    st.subheader("Comparison of Sentiment Across Products:")
    compare_sentiment_across_products(df_reviews)

    st.subheader("Keyword Frequency Analysis:")
    all_words = ' '.join(df_reviews['Description'])
    generate_wordcloud_st(all_words)
    

def visualize_pie_chart(df_reviews):
    info_text = '''
        - This chart shows the proportion of positive, negative, and neutral sentiments in the reviews.
        - Each slice size represents the percentage of reviews falling into each sentiment category.
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
        - Each bar height represents the frequency of reviews with specific confidence levels.
    '''
    with st.expander("💡Info"):
        st.write(info_text)

    # Create figure and axes
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

    # Display the plot in Streamlit
    st.pyplot(fig)


def analyze_sentiment_st(description):
    analysis = TextBlob(description)
    sentiment = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity
    
    # Adjust confidence calculation
    if subjectivity == 0:
        confidence = abs(sentiment) * 100  # If completely objective, use sentiment directly
    else:
        confidence = (1 - subjectivity) * 100 + abs(sentiment) * 50  # Adjust based on subjectivity

    # Ensure confidence is not negative
    confidence = max(confidence, 0)

    # Debugging: Print the analysis results
    print(f"Review: {description}")
    print(f"Sentiment: {sentiment}, Subjectivity: {subjectivity}, Confidence: {confidence}")

    if sentiment > 0:
        return 'Positive', confidence
    elif sentiment < 0:
        return 'Negative', confidence
    else:
        return 'Neutral', confidence


def generate_wordcloud_st(words):
    info_text = '''
        - This shows us which words appear most often in the reviews.
        - Think of it as finding the most popular words in a book.
        - The bigger the word in the cloud, the more often it appears in the reviews.
    '''
    with st.expander("💡Info"):
        st.write(info_text)

    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(words)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')

    # Display the plot in Streamlit
    st.pyplot(fig)


def visualize_review_length_distribution(df):
    info_text = '''
        - This visualization shows how review lengths are distributed.
        - Review length refers to the number of words in each review.
        - Frequency indicates how often reviews of certain lengths occur.
    '''
    with st.expander("💡Info"):
        st.write(info_text)

    # Compute review lengths
    df['Review Length'] = df['Description'].apply(lambda x: len(x.split()))

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Review Length'], bins=20, kde=True, color='skyblue', ax=ax)
    ax.set_title('Distribution of Review Length')
    ax.set_xlabel('Review Length')
    ax.set_ylabel('Frequency')

    # Display the plot in Streamlit
    st.pyplot(fig)


def compare_sentiment_across_products(df):
    info_text = '''
        - This visualization compares the sentiment of reviews for different products.
        - Imagine comparing how people feel about various items or services.
        - Each bar on the chart represents the number of positive, negative, and neutral reviews for each product.
        - For example, if you see a tall blue section (positive sentiment) on a bar, it means many reviews for that product are positive.
        - This comparison helps us understand the overall sentiment distribution across different products.
    '''
    with st.expander("💡Info"):
        st.write(info_text)

    # Group data and count sentiments for each product
    sentiment_counts_by_product = df.groupby('Name')['Sentiment'].value_counts().unstack(fill_value=0)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create a stacked bar chart
    sentiment_counts_by_product.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        color={'Positive': '#4CAF50', 'Neutral': '#FFC107', 'Negative': '#F44336'}
    )
    
    # Customize plot
    ax.set_title('Sentiment Comparison Across Products')
    ax.set_xlabel('Product')
    ax.set_ylabel('Number of Reviews')
    ax.legend(title='Sentiment')

    # Display the plot in Streamlit
    st.pyplot(fig)


def visualize_keyword_frequency(df):
    all_words = ' '.join(df['Description'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot()

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

        # Ensure all 'reviewText' or textual data are strings
        if 'reviewText' in df.columns:
            df['reviewText'] = df['reviewText'].astype(str)

        # Dynamically analyze the dataset
        st.write("Dataset Overview:")
        st.dataframe(df.head())

        st.write("Basic Statistics:")
        st.write(df.describe(include='all'))

        # Analyze specific columns
        if 'rating' in df.columns or 'overall' in df.columns:
            rating_col = 'rating' if 'rating' in df.columns else 'overall'
            st.write(f"Distribution of {rating_col.capitalize()}:")
            fig, ax = plt.subplots()
            sns.countplot(data=df, x=rating_col, ax=ax)
            st.pyplot(fig)

        if 'reviewText' in df.columns:
            st.write("Sample Reviews:")
            st.write(df['reviewText'].head(5))

        if 'sentiment' in df.columns:
            df['sentiment'] = df['sentiment'].astype(str)  # Ensure sentiments are strings
            st.write("Sentiment Distribution:")
            fig, ax = plt.subplots()
            sns.countplot(data=df, x='sentiment', ax=ax, palette='coolwarm')
            st.pyplot(fig)

        # Print a success message
        st.success(f"File successfully loaded! {df.shape[0]} rows and {df.shape[1]} columns imported.")

        return df

    except FileNotFoundError:
        st.error("File not found. Please check the file path.")
    except pd.errors.EmptyDataError:
        st.error("The file is empty. Please upload a valid CSV file.")
    except Exception as e:
        st.error(f"Not cntains the required credentials")


def clean_and_store_data(df, csv_filename='cleaned_reviews.csv'):
    # Clean data
    df['Description'] = df['Description'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', str(x)))  # Remove non-alphanumeric characters
    df['Description'] = df['Description'].apply(lambda x: x.lower())  # Convert text to lowercase
    stop_words = set(stopwords.words('english'))
    df['Description'] = df['Description'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in stop_words]))  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    df['Description'] = df['Description'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]))  # Lemmatize words

    # Store cleaned data in a new CSV
    cleaned_csv_path = csv_filename
    df.to_csv(cleaned_csv_path, index=False)
    
    return cleaned_csv_path

def is_valid_amazon_url(url):
    """
    Validate if the given URL is a valid Amazon product URL.
    
    Args:
        url (str): The URL to validate.
    
    Returns:
        bool: True if valid, False otherwise.
    """
    amazon_url_pattern = re.compile(
        r'^(https?://)?(www\.)?(amazon\.com|amazon\.[a-z]{2})/.*$'
    )
    return bool(amazon_url_pattern.match(url))
def main():
    st.title("SentiMart📦: Amazon Sentiment App")

    option = st.sidebar.selectbox("Choose an option", ["Write Review", "Enter Amazon URL", "Import CSV"])
    
    if option == "Import CSV":
        st.header("Import CSV for Analysis")
        app()

    elif option == "Write Review":
        st.header("Write Review for Analysis")

        user_input = st.text_area("Enter your review:")

        if st.button("Analyze"):
            if user_input:
                result, confidence = analyze_sentiment_st(user_input)
                st.subheader("Sentiment Analysis Result:")
                st.write(f"Sentiment: {result}")
                st.write(f"Confidence Score: {confidence}")
            else:
                st.warning("Please enter a review for analysis.")

    elif option == "Enter Amazon URL":
        st.header("Enter Your Favourite Amazon product's URL")

        URL_input = st.text_input("Enter Valid Amazon URL:")
        page_len = st.slider("Select the number of pages to scrape", min_value=1, max_value=10, value=1)

        if st.button("Analyze"):
            if URL_input:
                if is_valid_amazon_url(URL_input):
                    html_datas = reviewsHtml(URL_input, page_len)
                    df_reviews = process_data(html_datas, page_len)

                    # Check if 'Description' column exists before cleaning
                    if 'Description' in df_reviews.columns:
                        df_reviews = clean_data(df_reviews)
                        cleaned_csv_path = clean_and_store_data(df_reviews)

                        df_cleaned = import_data(cleaned_csv_path)
                        df_cleaned[['Sentiment', 'Confidence']] = df_cleaned['Description'].apply(analyze_sentiment_st).apply(pd.Series)

                        st.subheader("Data Preview after Cleaning:")
                        st.write(df_cleaned.head())
                        visualize_data(df_cleaned)
                    else:
                        st.warning("The URL does not contain valid review data. Please enter a valid Amazon product URL.")
                else:
                    st.warning("Please enter a valid Amazon URL.")
            else:
                st.warning("Please enter a URL first!")

if __name__ == "__main__":
    main()