import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process, fuzz
import re

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

            st.subheader("🔍 Genre-Based Recommendations")

            # Function to clean genre names properly
            def clean_genre(genre_text):
                if pd.isna(genre_text):
                    return None  # Ignore NaN values
                genre_text = re.sub(r"#\d+", "", genre_text)  # Remove numeric rankings like #1, #32
                genre_text = re.sub(r"\b\d+\s+in\s+", "", genre_text)  # Remove "032 in"
                genre_text = re.sub(r"\(.*?\)", "", genre_text)  # Remove (Books), (Audible Originals)
                genre_text = re.sub(r"#", "", genre_text).strip()  # Remove `#` symbols
                return genre_text if genre_text else None

            # Apply cleaning function
            df['Clean_Genres'] = df['Genres'].apply(clean_genre)

            # Extract unique genres from comma-separated values
            genre_list = df['Clean_Genres'].dropna().str.split(',').explode().str.strip().unique()
            
            # Sort genres and remove duplicates
            unique_genres = sorted(set(genre_list))

            # Display dropdown with cleaned genres
            genre = st.selectbox("Select a Genre:", unique_genres)

            if genre:
                # Ensure exact genre matches, considering comma-separated values
                genre_books = df[df['Clean_Genres'].str.contains(fr'\b{re.escape(genre)}\b', case=False, na=False)]
                
                if not genre_books.empty:
                    st.write(f"### 📚 Books in **{genre}** Genre")
                    st.dataframe(genre_books[[title_col, 'Author', 'Rating_x']].sort_values(by="Rating_x", ascending=False).head(10))
                else:
                    st.warning(f"No books found for the genre: **{genre}**.")




            # Display the top 5 books in the selected genre    
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

        elif option == "Best Feature Combinations for Recommendations":
            st.subheader("Best Feature Combinations for Recommendations")

            # Define potential numeric features
            numeric_features = ["Review_Count", "Price", "Duration", "Popularity Score", "Rating_x"]

            # Check which features exist in the dataset
            available_features = [col for col in numeric_features if col in df.columns]

            if len(available_features) < 2:
                st.error("Not enough numeric features available for analysis.")
            else:
                df_numeric = df[available_features].dropna()

                # Compute correlation matrix
                st.write("### Feature Correlation Heatmap")
                fig, ax = plt.subplots(figsize=(3, 2))
                sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                # Rotate x-axis labels by 30 degrees
                plt.xticks(rotation=30)
                plt.yticks(rotation=0)  # Keep y-axis labels normal
                st.pyplot(fig)

                # Feature Importance using RandomForestRegressor
                from sklearn.ensemble import RandomForestRegressor

                if "Rating_x" in df_numeric.columns:
                    st.write("### Feature Importance for Book Ratings")

                    # Prepare data
                    X = df_numeric.drop(columns=["Rating_x"], errors="ignore")
                    y = df_numeric["Rating_x"]

                    # Train model
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X, y)

                    # Get feature importance
                    feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

                    # Plot feature importance
                    fig, ax = plt.subplots(figsize=(3, 2))
                    sns.barplot(x=feature_importance.values, y=feature_importance.index, palette="viridis", ax=ax)
                    ax.set_xlabel("Importance Score")
                    st.pyplot(fig)

                # Display insights
                st.write("### Insights:")
                st.markdown("""
                - **Highly correlated features with book ratings:**
                  - If a feature has a strong correlation (above **0.5 or below -0.5**), it influences book ratings.
                - **Top 3 features for recommendations:**
                  - Based on feature importance, we prioritize these features when building recommendation models.
                - **Next Steps:**
                  - Use these features to improve personalized book recommendations.
                """)
            


        elif option == "Personalized Recommendations":

            st.subheader("📖 Personalized Recommendations")

            user_input = st.text_input("Enter a book title:")

            if user_input:
                # **Fuzzy matching for best book title**
                matches = process.extract(user_input, df[title_col].dropna(), scorer=fuzz.token_sort_ratio, limit=3)

                if matches and matches[0][1] > 60:  # Ensure confidence is above 60%
                    closest_match = matches[0][0]  # Get best match
                    confidence = matches[0][1]
                    st.write(f"🔍 Closest match: **{closest_match}** (Confidence: {confidence}%)")

                    # **Find the matched book's details**
                    selected_book = df[df[title_col] == closest_match]

                    if not selected_book.empty:
                        # Convert book descriptions into TF-IDF vectors
                        tfidf = TfidfVectorizer(stop_words="english")
                        book_vector = tfidf.fit_transform(df["Description"].fillna(""))

                        # Compute cosine similarity
                        selected_index = selected_book.index[0]
                        similarity_scores = cosine_similarity(book_vector[selected_index], book_vector).flatten()

                        # Recommend top 5 books (excluding the matched book itself)
                        recommended_indices = similarity_scores.argsort()[-6:-1][::-1]
                        recommended_books = df.iloc[recommended_indices][[title_col, "Author", "Genres", "Rating_x"]]

                        st.write("### 📚 Recommended Books:")
                        st.dataframe(recommended_books)
                    else:
                        st.warning("No similar books found in the dataset.")
                else:
                    st.warning("No close match found. Please try a different title.")


if __name__ == "__main__":
    main()