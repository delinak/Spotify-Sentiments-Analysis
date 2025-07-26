# Required packages:
# pip install google-play-scrape transformers plotly-express

import pandas as pd
import numpy as np
import plotly.express as px
import re
import emoji
import string

from google_play_scraper import app, Sort, reviews
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --------------------
# Text cleaning functions
# --------------------
def clean_for_vader(text):
    """Light cleaning for VADER sentiment analysis."""
    return re.sub(r'\s+', ' ', text).strip()

def clean_for_ml(text):
    """Heavy cleaning for classical ML models."""
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)     # normalize whitespace
    return text.strip().lower()

# --------------------
# Sentiment Analysis functions
# --------------------
def apply_sentiment(df, text_column = 'content_vader'):
    analyzer = SentimentIntensityAnalyzer()
    df['sentiment_score'] = df['content_vader'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    df['sentiment_label'] = df['sentiment_score'].apply(
        lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral')
    )
    return df

# --------------------
# Main Data Processing
# --------------------
def main():
    # Collect app reviews
    continuation_token = None
    all_reviews = []

    while len(all_reviews) <= 2000:  
        chunk, continuation_token = reviews(
            'com.spotify.music',
            lang='en',  
            country='us', 
            sort=Sort.NEWEST,  
            count=200, 
            filter_score_with=None 
        )
        all_reviews.extend(chunk)  
        if continuation_token is None:
            break
        
    df = pd.DataFrame(all_reviews)
    
    # Cleaning and preprocessing
    df = df[df['reviewCreatedVersion'].notna()] # filter out rows without version info
    df.dropna(subset=['content'], inplace=True) 
    df.duplicated().sum() 
    df.duplicated(subset=['reviewId']).sum()
    df.drop_duplicates(inplace=True)  
    
    # Fix data types
    df['score'] = df['score'].astype(int)  
    df['thumbsUpCount'] = df['thumbsUpCount'].astype(int)
    df['at'] = pd.to_datetime(df['at'])
    df['repliedAt'] = pd.to_datetime(df['repliedAt'], errors='coerce')

    # Stratifed sampling: top 5 versions with 500+ reviews
    vc = (df['reviewCreatedVersion']).value_counts()
    versions = vc.head(5).index.tolist()  
    strata = []
    for v in versions:
        sub = df[df['reviewCreatedVersion'] == v]
        n = min(500, len(sub))  # ensure we don't sample more than available
        strata.append(sub.sample(n=n, random_state=42))

    sampled_df = pd.concat(strata).reset_index(drop=True)
    
   # Sampled data
    sampled_df.to_csv('spotify_reviews.csv', index=False)

    # Clean sampled data in-place
    sampled_df['content_vader'] = sampled_df['content'].apply(clean_for_vader)
    sampled_df['content_ml'] = sampled_df['content'].apply(clean_for_ml)

    # Optional: Save only cleaned text if needed separately
    sampled_df.to_csv('spotify_reviews_cleaned.csv', index=False)

    # Apply sentiment on the cleaned vader column
    sampled_df = apply_sentiment(sampled_df, 'content_vader')

    # Save full version with sentiment + all metadata
    sampled_df.to_csv('spotify_reviews_sentiment.csv', index=False)
    

# run script
if __name__ == "__main__":
    main()
