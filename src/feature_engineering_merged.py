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
import os

# Download stopwords
nltk.download("stopwords")

# File paths
INPUT_FILE = "data/processed/audible_catalog_processed.csv"
OUTPUT_FILE = "data/processed/processed_audible_catalog_merged.csv"
MODEL_PATH = "models"

# Ensure output directories exist
os.makedirs("data/processed", exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# Load Dataset
def load_data(filepath):
    print("📌 Loading dataset...")
    return pd.read_csv(filepath)

# Handle Missing Data
def clean_missing_data(df):
    print("📌 Cleaning missing data...")
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    df.dropna(thresh=len(df.columns) * 0.7, inplace=True)
    
    return df

# Remove Duplicates
def remove_duplicates(df):
    print("📌 Removing duplicates...")
    df.drop_duplicates(subset=["Book Name", "Author"], inplace=True)
    return df

# Text Cleaning Function
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)
    text = " ".join([word for word in text.split() if word not in stopwords.words("english")])
    return text

# TF-IDF Vectorization
def compute_tfidf(corpus):
    print("📌 Computing TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return tfidf_matrix, vectorizer

# Word2Vec Embedding
def compute_word2vec(corpus, vector_size=100, window=5, min_count=1):
    print("📌 Training Word2Vec model...")
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
    print("📌 Scaling numerical features...")
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df, scaler

# Compute Popularity Score
def compute_popularity_score(df):
    if "Rating" in df.columns and "Number of Reviews" in df.columns:
        print("📌 Computing popularity score...")
        df["Number of Reviews"] = df["Number of Reviews"].apply(lambda x: np.log1p(x))
        df["Popularity Score"] = df["Rating"] * df["Number of Reviews"]
    else:
        print("⚠️ Skipping popularity score: columns missing.")
    return df

# Main Feature Engineering Pipeline
def feature_engineering(input_file, output_file):
    df = load_data(input_file)
    
    df = clean_missing_data(df)
    df = remove_duplicates(df)
    df = compute_popularity_score(df)

    if "Description" in df.columns:
        print("📌 Processing descriptions...")
        df["Cleaned_Description"] = df["Description"].apply(clean_text)

        # TF-IDF
        tfidf_matrix, tfidf_vectorizer = compute_tfidf(df["Cleaned_Description"])

        # Word2Vec
        word_vectors, word2vec_model = compute_word2vec(df["Cleaned_Description"])
        print("📌 Computing average Word2Vec vectors...")
        df["Word2Vec_Feature"] = df["Cleaned_Description"].apply(lambda x: get_average_word2vec(x, word_vectors))

        # Expand Word2Vec list column to individual numeric columns
        word2vec_df = pd.DataFrame(df["Word2Vec_Feature"].to_list(), columns=[f"Word2Vec_{i}" for i in range(100)])
        df = pd.concat([df.drop(columns=["Word2Vec_Feature"]), word2vec_df], axis=1)

        # Save Models
        print("📦 Saving models...")
        pickle.dump(tfidf_vectorizer, open(f"{MODEL_PATH}/tfidf_vectorizer.pkl", "wb"))
        word2vec_model.save(f"{MODEL_PATH}/word2vec_model.bin")
    else:
        print("⚠️ 'Description' column not found. Skipping text feature extraction.")

    # Normalize Numerical Features
    numerical_cols = ["Rating", "Number of Reviews", "Price", "Popularity Score"]
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    
    if numerical_cols:
        df, scaler = scale_features(df, numerical_cols)
        pickle.dump(scaler, open(f"{MODEL_PATH}/scaler.pkl", "wb"))
    else:
        print("⚠️ No numerical columns found for scaling.")

    # Save Final Dataset
    print("💾 Saving processed dataset...")
    df.to_csv(output_file, index=False)

    print(f"✅ Feature Engineering Completed and Saved Successfully! ({output_file})")

# Execute
if __name__ == "__main__":
    feature_engineering(INPUT_FILE, OUTPUT_FILE)
