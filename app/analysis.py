import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

def main():
    # Load Data
    data_path = 'data/processed/audible_catalog_processed.csv'
    df = pd.read_csv(data_path)
    
    # Preprocess Data
    df.dropna(subset=['Genres', 'Author', 'Rating_x', 'Number of Reviews_x', 'Description'], inplace=True)
    df['Rating'] = pd.to_numeric(df['Rating_x'], errors='coerce')
    df['Review_Count'] = pd.to_numeric(df[['Number of Reviews_x', 'Number of Reviews_y']].mean(axis=1), errors='coerce')
    title_col = "Book Name" if "Book Name" in df.columns else "Title"
    
    st.title("📚 Audible Insights: Book Analysis & Recommendations")
    
    # Layout
    col1, spacer, col2 = st.columns([1, 0.2, 2])
    
    with col1:
        option = st.selectbox("Choose an analysis:", [
            "Most Popular Genres",
            "Top Rated Authors",
            "Rating Distribution",
            "Trending Books (Top 5)",
            "Ratings vs Review Counts",
            "Clustered Books",
            "Genre-Based Recommendations",
            "Author Popularity & Ratings",
            "Best Feature Combinations for Recommendations",
            "Personalized Recommendations",
            "Genre Similarity & Recommendations",
            "Hidden Gems: Highly Rated but Less Popular Books",
            "Top 5 Recommendations for New Science Fiction Readers",
            "Thriller Enthusiast: Personalized Recommendations"
        ])
    
    with col2:
        if option == "Most Popular Genres":
            st.subheader("Most Popular Genres")
            genre_counts = df['Genres'].value_counts().head(10)
            fig, ax = plt.subplots()
            sns.barplot(x=genre_counts.values, y=genre_counts.index, ax=ax)
            ax.set_xlabel("Number of Books")
            st.pyplot(fig)
        
        elif option == "Top Rated Authors":
            st.subheader("Authors with Highest Rated Books")
            top_authors = df.groupby('Author')['Rating'].mean().sort_values(ascending=False).head(10).round(2)
            st.dataframe(top_authors, use_container_width=False)
        
        elif option == "Rating Distribution":
            st.subheader("Distribution of Book Ratings")
            fig, ax = plt.subplots(figsize=(4, 2))
            sns.histplot(df['Rating'], bins=15, kde=True, ax=ax)
            ax.set_xlabel("Rating")
            st.pyplot(fig)
        
        elif option == "Trending Books (Top 5)":
            st.subheader("📚 Trending Books - Top 5")
            
            def get_top_books(metric, title):
                top_books = df.nlargest(5, metric)[['Book Name', 'Author', metric]].reset_index(drop=True)
                st.markdown(f"### {title}")
                st.dataframe(top_books, use_container_width=True)

            col1, col2, col3 = st.columns(3)

            with col1:
                get_top_books('Popularity Score', "🔥 Most Popular Books")

            with col2:
                get_top_books('Number of Reviews_x', "📝 Most Reviewed Books")

            with col3:
                get_top_books('Rating_x', "⭐ Highest Rated Books")
        
        elif option == "Ratings vs Review Counts":
            st.subheader("How Ratings Vary with Review Counts")
            fig, ax = plt.subplots(figsize=(4, 2))
            sns.scatterplot(x=df['Review_Count'], y=df['Rating'], alpha=0.5, ax=ax)
            ax.set_xlabel("Review Count")
            st.pyplot(fig)
        
        elif option == "Clustered Books":
            st.subheader("Books Clustered by Description")
            vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
            X = vectorizer.fit_transform(df['Description'])
            kmeans = KMeans(n_clusters=5, random_state=42)
            df['Cluster'] = kmeans.fit_predict(X)
            st.dataframe(df[['Book Name', 'Cluster']].sample(10), use_container_width=False)
        
        elif option == "Genre-Based Recommendations":
            genre = st.selectbox("Select a Genre:", df['Genres'].unique())
            st.write(df[df['Genres'] == genre][['Book Name', 'Author', 'Rating']].head(10))
        
        elif option == "Author Popularity & Ratings":
            st.subheader("Effect of Author Popularity on Ratings")
            author_popularity = df.groupby('Author').agg({'Review_Count': 'sum', 'Rating': 'mean'})
            fig, ax = plt.subplots(figsize=(4, 2))
            sns.scatterplot(x=author_popularity['Review_Count'], y=author_popularity['Rating'], alpha=0.5, ax=ax)
            st.pyplot(fig)
        
        elif option == "Hidden Gems: Highly Rated but Less Popular Books":
            st.subheader("Highly Rated but Less Popular Books")
            hidden_gems = df[df['Review_Count'] < df['Review_Count'].quantile(0.2)].sort_values(by='Rating', ascending=False).head(10)
            st.dataframe(hidden_gems)
        
        elif option == "Top 5 Recommendations for New Science Fiction Readers":
            st.subheader("Recommended Sci-Fi Books")
            sci_fi_books = df[df['Genres'].str.contains('Science Fiction', case=False, na=False)].sort_values(by='Rating', ascending=False).head(5)
            st.dataframe(sci_fi_books)
        
        elif option == "Thriller Enthusiast: Personalized Recommendations":
            st.subheader("Top Thrillers for Thriller Fans")
            thriller_books = df[df['Genres'].str.contains('Thriller', case=False, na=False)].sort_values(by='Rating', ascending=False).head(5)
            st.dataframe(thriller_books)

if __name__ == "__main__":
    main()