import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process, fuzz
import re
from sklearn.ensemble import RandomForestRegressor

def main():
   # st.set_page_config(page_title="Audible Insights", layout="wide")
    st.title("📚 Audible Insights: Book Analysis & Recommendations")

    # Load data
    data_path = 'data/processed/audible_catalog_processed.csv'
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return

    # Fix column names
    title_col = "Book Name" if "Book Name" in df.columns else "Title"
    df.rename(columns={'Main Genre': 'Genres'}, inplace=True)

    required_cols = ['Genres', 'Author', 'Rating', 'Description', title_col]
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Missing column: {col}")
            return

    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    if 'Number of Reviews_x' in df.columns and 'Number of Reviews_y' in df.columns:
        df['Review_Count'] = pd.to_numeric(df[['Number of Reviews_x', 'Number of Reviews_y']].mean(axis=1), errors='coerce')
    elif 'Number of Reviews' in df.columns:
        df['Review_Count'] = pd.to_numeric(df['Number of Reviews'], errors='coerce')
    else:
        df['Review_Count'] = 0

    df.dropna(subset=['Description', 'Rating', 'Review_Count'], inplace=True)
    df['Author'] = df.get('Author', 'Unknown Author')
    df['Genres'] = df.get('Genres', 'Unknown')

    col1, col2 = st.columns([1, 3])

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
            st.subheader("📊 Most Popular Genres")
            genre_counts = df['Genres'].str.split(',').explode().str.strip().value_counts().head(10)
            fig, ax = plt.subplots()
            sns.barplot(x=genre_counts.values, y=genre_counts.index, ax=ax)
            ax.set_xlabel("Number of Books")
            st.pyplot(fig)

        elif option == "Top Rated Authors":
            st.subheader("🏆 Top Rated Authors")
            top_authors = df.groupby('Author')['Rating'].mean().sort_values(ascending=False).head(10).round(2)
            st.dataframe(top_authors)

        elif option == "Rating Distribution":
            st.subheader("📈 Rating Distribution")
            fig, ax = plt.subplots()
            sns.histplot(df['Rating'], bins=15, kde=True, ax=ax)
            ax.set_xlabel("Rating")
            st.pyplot(fig)

        elif option == "Trending Books (Top 5)":
            st.subheader("🔥 Trending Books")

            def show_top_books(metric, label):
                if metric in df.columns:
                    top_books = df[[title_col, 'Author', metric]].dropna().sort_values(by=metric, ascending=False).head(5)
                    st.markdown(f"#### {label}")
                    st.dataframe(top_books)

            colA, colB, colC = st.columns(3)
            with colA:
                show_top_books('Popularity Score', "Most Popular")
            with colB:
                show_top_books('Review_Count', "Most Reviewed")
            with colC:
                show_top_books('Rating', "Highest Rated")

        elif option == "Ratings vs Review Counts":
            st.subheader("🔍 Ratings vs Review Counts")
            fig, ax = plt.subplots()
            sns.scatterplot(x=df['Review_Count'], y=df['Rating'], alpha=0.6, ax=ax)
            ax.set_xlabel("Review Count")
            st.pyplot(fig)

        elif option == "Clustered Books":
            st.subheader("🧠 Books Clustered by Description")
            vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
            X = vectorizer.fit_transform(df['Description'])
            kmeans = KMeans(n_clusters=5, random_state=42)
            df['Cluster'] = kmeans.fit_predict(X)
            st.dataframe(df[[title_col, 'Cluster']].sample(10))

        elif option == "Genre-Based Recommendations":
            st.subheader("🎯 Genre-Based Recommendations")
            df['Clean_Genres'] = df['Genres'].apply(lambda x: re.sub(r"#\d+|\(.*?\)|#|\d+ in ", "", str(x)).strip())
            genres = df['Clean_Genres'].str.split(',').explode().str.strip().dropna().unique()
            selected_genre = st.selectbox("Select a genre:", sorted(genres))
            filtered = df[df['Clean_Genres'].str.contains(fr'\b{re.escape(selected_genre)}\b', case=False, na=False)]
            st.dataframe(filtered[[title_col, 'Author', 'Rating']].sort_values(by="Rating", ascending=False).head(10))

        elif option == "Author Popularity & Ratings":
            st.subheader("👨‍💻 Author Popularity & Ratings")
            author_stats = df.groupby('Author').agg({'Review_Count': 'sum', 'Rating': 'mean'})
            fig, ax = plt.subplots()
            sns.scatterplot(x=author_stats['Review_Count'], y=author_stats['Rating'], ax=ax, alpha=0.6)
            ax.set_xlabel("Total Reviews")
            ax.set_ylabel("Average Rating")
            st.pyplot(fig)

        elif option == "Best Feature Combinations for Recommendations":
            st.subheader("💡 Important Features for Rating")
            numeric_cols = ["Review_Count", "Price", "Duration", "Popularity Score", "Rating"]
            df_numeric = df[[col for col in numeric_cols if col in df.columns]].dropna()
            if not df_numeric.empty:
                fig, ax = plt.subplots()
                sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)

                if "Rating" in df_numeric.columns:
                    X = df_numeric.drop(columns="Rating")
                    y = df_numeric["Rating"]
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X, y)
                    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
                    fig, ax = plt.subplots()
                    sns.barplot(x=importance.values, y=importance.index, ax=ax)
                    ax.set_xlabel("Feature Importance")
                    st.pyplot(fig)
            else:
                st.warning("Not enough numeric data for feature analysis.")

        elif option == "Personalized Recommendations":
            st.subheader("🧠 Personalized Book Recommendations")
            user_input = st.text_input("Enter a book title:")
            if user_input:
                matches = process.extract(user_input, df[title_col].dropna(), scorer=fuzz.token_sort_ratio, limit=3)
                if matches and matches[0][1] > 60:
                    closest_match = matches[0][0]
                    st.success(f"Closest Match: {closest_match}")
                    selected_book = df[df[title_col] == closest_match]
                    if not selected_book.empty:
                        tfidf = TfidfVectorizer(stop_words="english")
                        tfidf_matrix = tfidf.fit_transform(df['Description'].fillna(""))
                        idx = selected_book.index[0]
                        sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
                        top_indices = sim_scores.argsort()[-6:-1][::-1]
                        st.dataframe(df.iloc[top_indices][[title_col, 'Author', 'Genres', 'Rating']])
                    else:
                        st.warning("Book not found.")
                else:
                    st.warning("No close match found. Try again.")

        elif option == "Hidden Gems: Highly Rated but Less Popular Books":
            st.subheader("💎 Hidden Gems")
            threshold = df['Review_Count'].quantile(0.2)
            hidden_gems = df[df['Review_Count'] < threshold].sort_values(by='Rating', ascending=False).head(10)
            st.dataframe(hidden_gems[[title_col, 'Author', 'Rating', 'Review_Count']])

        elif option == "Top 5 Recommendations for New Science Fiction Readers":
            st.subheader("🚀 Sci-Fi Starter Pack")
            sci_fi = df[df['Genres'].str.contains("Science Fiction", case=False, na=False)]
            st.dataframe(sci_fi.sort_values(by="Rating", ascending=False).head(5))

        elif option == "Thriller Enthusiast: Personalized Recommendations":
            st.subheader("🔪 Top Thrillers")
            thrillers = df[df['Genres'].str.contains("Thriller", case=False, na=False)]
            st.dataframe(thrillers.sort_values(by="Rating", ascending=False).head(5))

if __name__ == "__main__":
    main()
