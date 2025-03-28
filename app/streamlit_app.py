import streamlit as st
import pandas as pd
import pickle
import gensim
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Ensure correct paths for loading models
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_models():
    """ Load saved models for recommendation. """
    try:
        tfidf_vectorizer = pickle.load(open(os.path.join(BASE_DIR, "models/tfidf_vectorizer.pkl"), "rb"))
        word2vec_model = gensim.models.Word2Vec.load(os.path.join(BASE_DIR, "models/word2vec_model.bin"))
        scaler = pickle.load(open(os.path.join(BASE_DIR, "models/scaler.pkl"), "rb"))
        return tfidf_vectorizer, word2vec_model, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

@st.cache_data
def load_data():
    """ Load and validate the book dataset. """
    try:
        df = pd.read_csv(os.path.join(BASE_DIR, "data/processed/audible_catalog_processed.csv"))

        required_columns = ["Book Name", "Author", "Rating_x", "Number of Reviews_x", 
                            "Price_x", "Popularity Score", "Main Genre", "Top Rank", 
                            "Is Free", "Is Audible", "Description"]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing columns in dataset: {missing_columns}")
            st.stop()

        df.fillna("", inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

def main():
    st.title("📚 Audible Insights: Intelligent Book Recommendations")

    df = load_data()
    tfidf_vectorizer, word2vec_model, scaler = load_models()

    # Layout
    col1, spacer, col2 = st.columns([1, 0.2, 2])

    # Filters in col1
    with col1:
        selected_genre = st.selectbox("Select Genre", ["All"] + sorted(df["Main Genre"].dropna().unique().tolist()))
        if selected_genre != "All":
            df = df[df["Main Genre"] == selected_genre].copy()

        selected_author = st.selectbox("Select Author", ["All"] + sorted(df["Author"].dropna().unique().tolist()))
        if selected_author != "All":
            df = df[df["Author"] == selected_author].copy()

        rating_range = st.slider("Select Rating Range", 0.0, 5.0, (0.0, 5.0), 0.1)
        df = df[(df["Rating_x"] >= rating_range[0]) & (df["Rating_x"] <= rating_range[1])].copy()
        
        max_price = df["Price_x"].dropna().max() or 1
        price_range = st.slider("Select Price Range", 0, int(max_price), (0, int(max_price)))
        df = df[(df["Price_x"] >= price_range[0]) & (df["Price_x"] <= price_range[1])].copy()
        
        max_reviews = df["Number of Reviews_x"].dropna().max() or 1
        reviews_range = st.slider("Select Number of Reviews", 0, int(max_reviews), (0, int(max_reviews)))
        df = df[(df["Number of Reviews_x"] >= reviews_range[0]) & (df["Number of Reviews_x"] <= reviews_range[1])].copy()
    
        selected_book = st.selectbox("Select a Book", df["Book Name"].dropna().unique())
        book_details = df[df["Book Name"] == selected_book].iloc[0]
    if df.empty:
        st.warning("No books match the selected filters. Try adjusting the filters.")
        st.stop()

    # Book selection and recommendations in col2
    with col2:

        st.subheader(f"📖 **{book_details['Book Name']}** by {book_details['Author']}")
        st.write(f"⭐ **Rating:** {book_details['Rating_x']} ({book_details['Number of Reviews_x']} reviews)")
        st.write(f"💰 **Price:** ₹{book_details['Price_x']}")
        st.write(f"📌 **Genre:** {book_details['Main Genre']}")
        st.write(f"🏆 **Top Rank:** #{int(book_details['Top Rank']) if not pd.isna(book_details['Top Rank']) else 'N/A'}")
        st.write(f"🎧 **Audible Exclusive:** {'Yes' if book_details['Is Audible'] else 'No'}")
        st.write(f"🆓 **Free:** {'Yes' if book_details['Is Free'] else 'No'}")
        st.write(f"📜 **Description:** {book_details['Description']}")

        @st.cache_data
        def compute_word2vec_similarity(book_desc, df):
            """ Compute cosine similarity using Word2Vec embeddings. """
            def get_avg_word_vector(text):
                words = text.lower().split()
                word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
                return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(word2vec_model.vector_size)

            selected_book_vector = get_avg_word_vector(book_desc).reshape(1, -1)
            book_vectors = np.array([get_avg_word_vector(desc) for desc in df["Description"]])

            if book_vectors.shape[0] > 0 and selected_book_vector.shape[1] == book_vectors.shape[1]:
                similarities = cosine_similarity(selected_book_vector, book_vectors).flatten()
                return similarities
            return np.array([])

        selected_book_desc = book_details["Description"]
        if not selected_book_desc:
            st.error("Error: Selected book description is missing.")
            st.stop()

        similarities = compute_word2vec_similarity(selected_book_desc, df)

        if similarities.size > 0:
            df["Similarity"] = similarities
            recommendations = df.sort_values(by="Similarity", ascending=False).head(6)

            st.subheader("📖 Recommended Books")
            for _, row in recommendations.iterrows():
                st.write(f"**{row['Book Name']}** by {row['Author']} (⭐ {row['Rating_x']})")
        else:
            st.warning("No recommendations found. The book description may not contain recognizable words.")

if __name__ == "__main__":
    main()