import streamlit as st
import pandas as pd
import pickle
import gensim
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load models
def load_models():
    tfidf_vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))
    word2vec_model = gensim.models.Word2Vec.load("models/word2vec_model.bin")
    scaler = pickle.load(open("models/scaler.pkl", "rb"))
    return tfidf_vectorizer, word2vec_model, scaler

@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/audible_catalog_processed.csv")

    # Ensure required columns exist
    required_columns = ["Book Name", "Author", "Rating_x", "Number of Reviews_x", 
                        "Price_x", "Popularity Score", "Main Genre", "Top Rank", 
                        "Is Free", "Is Audible", "Description"]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing columns in dataset: {missing_columns}")
        st.stop()
    
    # Fill NaN values
    df.fillna("", inplace=True)
    
    return df

# Load data
df = load_data()

# Load models
tfidf_vectorizer, word2vec_model, scaler = load_models()

# Streamlit App UI
st.title("📚 Audible Insights: Intelligent Book Recommendations")

# Sidebar Filters
st.sidebar.header("📌 Filter Books")

# Genre Selection
selected_genre = st.sidebar.selectbox("Select Genre", ["All"] + sorted(df["Main Genre"].dropna().unique().tolist()))
if selected_genre != "All":
    df = df[df["Main Genre"] == selected_genre]

# Author Selection
selected_author = st.sidebar.selectbox("Select Author", ["All"] + sorted(df["Author"].dropna().unique().tolist()))
if selected_author != "All":
    df = df[df["Author"] == selected_author]

# Rating Filter
rating_range = st.sidebar.slider("Select Rating Range", min_value=0.0, max_value=5.0, value=(0.0, 5.0), step=0.1)
df = df[(df["Rating_x"] >= rating_range[0]) & (df["Rating_x"] <= rating_range[1])]

# Price Filter (Handle NaN and Edge Cases)
max_price = df["Price_x"].dropna().max()
if pd.isna(max_price) or max_price == 0:
    max_price = 1  # Set default max value to avoid slider error

price_range = st.sidebar.slider("Select Price Range", min_value=0, max_value=int(max_price), value=(0, int(max_price)))
df = df[(df["Price_x"] >= price_range[0]) & (df["Price_x"] <= price_range[1])]

# Number of Reviews Filter (Handle NaN and Edge Cases)
max_reviews = df["Number of Reviews_x"].dropna().max()
if pd.isna(max_reviews) or max_reviews == 0:
    max_reviews = 1  # Set default max value to avoid slider error

reviews_range = st.sidebar.slider("Select Number of Reviews", min_value=0, max_value=int(max_reviews), value=(0, int(max_reviews)))
df = df[(df["Number of Reviews_x"] >= reviews_range[0]) & (df["Number of Reviews_x"] <= reviews_range[1])]

# Book Selection (Moved to Sidebar)
if df.empty:
    st.warning("No books match the selected filters. Try adjusting the filters.")
    st.stop()

selected_book = st.sidebar.selectbox("Select a Book", df["Book Name"].dropna().unique())

# Display Selected Book Details
book_details = df[df["Book Name"] == selected_book].iloc[0]

st.subheader(f"📖 **{book_details['Book Name']}** by {book_details['Author']}")
st.write(f"⭐ **Rating:** {book_details['Rating_x']} ({book_details['Number of Reviews_x']} reviews)")
st.write(f"💰 **Price:** ₹{book_details['Price_x']}")
st.write(f"📌 **Genre:** {book_details['Main Genre']}")
st.write(f"🏆 **Top Rank:** #{int(book_details['Top Rank']) if not pd.isna(book_details['Top Rank']) else 'N/A'}")
st.write(f"🎧 **Audible Exclusive:** {'Yes' if book_details['Is Audible'] else 'No'}")
st.write(f"🆓 **Free:** {'Yes' if book_details['Is Free'] else 'No'}")
st.write(f"📜 **Description:** {book_details['Description']}")

# Function to Compute Word2Vec Similarity
@st.cache_data
def compute_word2vec_similarity(book_desc, df):
    def get_avg_word_vector(text):
        words = text.lower().split()
        word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
        return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(word2vec_model.vector_size)

    # Compute vector for the selected book
    selected_book_vector = get_avg_word_vector(book_desc).reshape(1, -1)

    # Compute vectors for all books
    book_vectors = np.array([get_avg_word_vector(desc) for desc in df["Description"]])

    # Compute similarity
    if book_vectors.shape[0] > 0 and selected_book_vector.shape[1] == book_vectors.shape[1]:
        similarities = cosine_similarity(selected_book_vector, book_vectors).flatten()
        return similarities
    return np.array([])

# Get Book Description Safely
selected_book_desc = book_details["Description"]
if not selected_book_desc:
    st.error("Error: Selected book description is missing.")
    st.stop()

# Compute Similarities
similarities = compute_word2vec_similarity(selected_book_desc, df)

# Show Recommendations
if similarities.size > 0:
    df["Similarity"] = similarities
    recommendations = df.sort_values(by="Similarity", ascending=False).head(6)

    st.subheader("📖 Recommended Books")
    for _, row in recommendations.iterrows():
        st.write(f"**{row['Book Name']}** by {row['Author']} (⭐ {row['Rating_x']})")
else:
    st.warning("No recommendations found. The book description may not contain recognizable words.")
