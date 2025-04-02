import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():


    # Load Data
    df = pd.read_csv("data/processed/audible_catalog_processed.csv")

    # Ensure correct column name
    df["Genre"] = df["Genres"].fillna("Unknown")

    st.title("📊 Exploratory Data Analysis (EDA) - Audible Insights")

    # Function to shorten long labels
    def shorten_label(label, max_length=12):
        return label[:max_length] + "..." if len(label) > max_length else label

    def shorten_labels(df, column, max_length=12):
        df[column] = df[column].astype(str).apply(lambda x: shorten_label(x, max_length))
        return df

    # Convert 'Number of Reviews_x' to numeric
    df["Number of Reviews_x"] = pd.to_numeric(df["Number of Reviews_x"], errors="coerce").fillna(0).astype(int)

    # Bin ratings into ranges
    df["Rating Range"] = pd.cut(
        df["Rating_x"], bins=[0, 1, 2, 3, 4, 5], labels=["0-1", "1-2", "2-3", "3-4", "4-5"], include_lowest=True
    )

    # Aggregate review count per genre
    genre_review_stats = df.groupby("Genre", as_index=False)["Number of Reviews_x"].sum()
    genre_review_stats = shorten_labels(genre_review_stats, "Genre")
    top_5_genres = genre_review_stats.nlargest(5, "Number of Reviews_x")

    # Aggregate review count per author
    author_review_stats = df.groupby("Author", as_index=False)["Number of Reviews_x"].sum()
    top_5_authors = shorten_labels(author_review_stats, "Author").nlargest(5, "Number of Reviews_x")

    # Most popular genre
    most_popular_genre = genre_review_stats.iloc[0]["Genre"] if not genre_review_stats.empty else "Unknown"

    # Top-rated books in the most popular genre
    popular_genre_books = df[df["Genre"] == most_popular_genre].nlargest(5, "Rating_x", "all")[
        ["Book Name", "Rating_x"]
    ]
    popular_genre_books = shorten_labels(popular_genre_books, "Book Name")

    col1, spacer, col2 = st.columns([1, 0.1, 1.2])
    with col1:
        # Display Regression Model Metrics
        st.subheader("📈 Regression Model Performance Metrics")
        regression_metrics = pd.read_csv("results/regression_tuned_metrics.csv")
        st.table(regression_metrics)
    with col2:
        # Display Clustering Model Metrics
        st.subheader("🔍 Clustering Model Evaluation Metrics")
        clustering_metrics = pd.read_csv("results/model_metrics.csv")
        st.table(clustering_metrics)

    # Display top genres and authors
    col1, spacer, col2 = st.columns([1, 0.1, 1.2])
    with col1:
        st.subheader("🏆 Top 5 Genres by Review Count")
        st.table(top_5_genres)
    with col2:
        st.subheader("✍️ Top 5 Authors by Review Count")
        st.table(top_5_authors)

    # Genre and Author Review Count Charts
    col1, spacer, col2 = st.columns([1, 0.1, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x="Number of Reviews_x", y="Genre", data=top_5_genres, palette="viridis", ax=ax)
        ax.set_title("Top 5 Genres by Reviews")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x="Number of Reviews_x", y="Author", data=top_5_authors, palette="magma", ax=ax)
        ax.set_title("Top 5 Authors by Reviews")
        st.pyplot(fig)

    # Ratings Distribution
    col1, spacer, col2 = st.columns([1, 0.1, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df["Rating_x"], bins=20, kde=True, color="blue", ax=ax)
        ax.set_title("Distribution of Ratings")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x=df["Rating Range"], palette="coolwarm", ax=ax)
        ax.set_title("Rating Ranges")
        st.pyplot(fig)

    # Top 5 Books in Most Popular Genre
    if not popular_genre_books.empty:
        col1, spacer, col2 = st.columns([1, 0.1, 1])
        with col1:
            st.subheader(f"📚 Top 5 Books in {most_popular_genre} Genre")
            st.table(popular_genre_books)
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x="Rating_x", y="Book Name", data=popular_genre_books, palette="viridis", ax=ax)
            ax.set_title(f"Top 5 Books in {most_popular_genre} Genre")
            st.pyplot(fig)

    # Review Count Distribution
    col1, spacer, col2 = st.columns([1, 0.1, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df["Number of Reviews_x"], bins=20, kde=True, color="purple", ax=ax)
        ax.set_title("Distribution of Review Counts")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x=df["Number of Reviews_x"], color="red", ax=ax)
        ax.set_title("Review Count Boxplot")
        st.pyplot(fig)

    col1, spacer, col2 = st.columns([1, 0.1, 1])
    with col1:
        # Correlation Heatmap
        st.subheader("📊 Correlation Heatmap")
        correlation_cols = ["Rating_x", "Number of Reviews_x"]
        corr_matrix = df[correlation_cols].corr()

        fig, ax = plt.subplots(figsize=(3, 2))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)

        # Rotate labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center")

        ax.set_title("Correlation between Ratings and Reviews")
        st.pyplot(fig)

if __name__ == "__main__":
    main()
