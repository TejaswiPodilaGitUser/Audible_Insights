import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler
import pickle

# Text Cleaning: Removes punctuation, numbers, and stopwords.
# TF-IDF Vectorization: Converts text into numerical features.
# Word2Vec Embeddings: Generates word vectors for book descriptions.
# Feature Scaling: Normalizes numerical data like ratings and prices.
# Model Saving: Stores TF-IDF, Word2Vec, and scaler models for reuse.

# 📂 Generated Outputs:
# data/processed/processed_catalog.csv (Processed Catalog dataset)
# data/processed/processed_advanced_features.csv (Processed Advanced Features dataset)
# models/tfidf_vectorizer.pkl (TF-IDF Model)
# models/word2vec_model.bin (Word2Vec Model)
# models/scaler.pkl (Scaler for normalization)

nltk.download("stopwords")

# Load Dataset
def load_data(filepath):
    return pd.read_csv(filepath)

# Text Cleaning Function
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = " ".join([word for word in text.split() if word not in stopwords.words("english")])  # Remove stopwords
    return text

# TF-IDF Vectorization
def compute_tfidf(corpus):
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return tfidf_matrix, vectorizer

# Word2Vec Embedding
def compute_word2vec(corpus, vector_size=100, window=5, min_count=1):
    tokenized_corpus = [text.split() for text in corpus]
    model = Word2Vec(sentences=tokenized_corpus, vector_size=vector_size, window=window, min_count=min_count, workers=4)
    word_vectors = model.wv
    return word_vectors, model

# Compute Average Word2Vec Vector
def get_average_word2vec(text, word_vectors, vector_size=100):
    words = text.split()
    word_vecs = [word_vectors[word] for word in words if word in word_vectors]
    if len(word_vecs) == 0:
        return np.zeros(vector_size)
    return np.mean(word_vecs, axis=0)

# Normalize Numerical Features
def scale_features(df, numerical_cols):
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df, scaler

# Main Feature Engineering Pipeline
def feature_engineering(input_file, output_file):
    df = load_data(input_file)
    
    if 'Description' in df.columns:
        df['Cleaned_Description'] = df['Description'].apply(clean_text)
        
        # TF-IDF Processing
        tfidf_matrix, tfidf_vectorizer = compute_tfidf(df['Cleaned_Description'])
        
        # Word2Vec Processing
        word_vectors, word2vec_model = compute_word2vec(df['Cleaned_Description'])
        df['Word2Vec_Feature'] = df['Cleaned_Description'].apply(lambda x: get_average_word2vec(x, word_vectors))
    
    # Normalize Numerical Features if they exist
    numerical_cols = ['Rating', 'Number of Reviews', 'Price']
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    
    if numerical_cols:
        df, scaler = scale_features(df, numerical_cols)
        pickle.dump(scaler, open("models/scaler.pkl", "wb"))
    
    # Save Processed Features
    df.to_csv(output_file, index=False)
    
    # Save Models
    if 'Cleaned_Description' in df.columns:
        pickle.dump(tfidf_vectorizer, open("models/tfidf_vectorizer.pkl", "wb"))
        word2vec_model.save("models/word2vec_model.bin")
    
    print(f"Feature Engineering Completed and Saved Successfully! ({output_file})")

# Process both datasets
if __name__ == "__main__":
    feature_engineering("data/cleaned/audible_catalog_cleaned.csv", "data/processed/processed_audible_catalog.csv")
    feature_engineering("data/cleaned/audible_catalog_advanced_cleaned.csv", "data/processed/processed_advanced_features.csv")
