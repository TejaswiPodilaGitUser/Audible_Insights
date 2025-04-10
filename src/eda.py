import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # st.set_page_config(page_title="Audible Insights - EDA", layout="wide")

    # Load Data
    df = pd.read_csv("data/processed/audible_catalog_processed.csv")
    df.columns = df.columns.str.strip()

    # Handle missing Genres
   # Correct Genre Extraction
    if "Main Genre" in df.columns:
        df["Genre"] = df["Main Genre"].fillna("Unknown")
    else:
        df["Genre"] = "Unknown"


    # Convert Number of Reviews to numeric
    df["Number of Reviews"] = pd.to_numeric(df["Number of Reviews"], errors="coerce").fillna(0).astype(int)

    # Rating Ranges
    df["Rating Range"] = pd.cut(df["Rating"], bins=[0,1,2,3,4,5], labels=["0-1", "1-2", "2-3", "3-4", "4-5"], include_lowest=True)

    st.title("📊 Audible Insights - Exploratory Data Analysis (EDA)")

    # --- Shorten Labels Function ---
    def shorten_label(label, max_length=12):
        return label[:max_length] + "..." if len(label) > max_length else label

    def shorten_labels(df, column, max_length=12):
        df[column] = df[column].astype(str).apply(lambda x: shorten_label(x, max_length))
        return df

    # Use full data
    filtered_df = df.copy()

    # --- Summary Stats ---
    st.subheader("📌 Dataset Summary")
    st.write(filtered_df.describe(include="all"))

    # --- Clustering Model Metrics ---
    st.subheader("🔍 Clustering Model Evaluation Metrics")
    clustering_metrics = pd.read_csv("results/model_metrics.csv")
    st.table(clustering_metrics)

    # --- Top Genres and Authors ---
    genre_review_stats = filtered_df.groupby("Genre", as_index=False)["Number of Reviews"].sum()
    genre_review_stats = shorten_labels(genre_review_stats, "Genre")
    # top_5_genres = genre_review_stats.nlargest(5, "Number of Reviews")
    top_5_genres = genre_review_stats[genre_review_stats["Genre"] != "Unknown"].nlargest(5, "Number of Reviews")


    author_review_stats = filtered_df.groupby("Author", as_index=False)["Number of Reviews"].sum()
    top_5_authors = shorten_labels(author_review_stats, "Author").nlargest(5, "Number of Reviews")

    most_popular_genre = genre_review_stats.iloc[0]["Genre"] if not genre_review_stats.empty else "Unknown"

    popular_genre_books = filtered_df[filtered_df["Genre"] == most_popular_genre].nlargest(5, "Rating")[
        ["Book Name", "Rating"]
    ]
    popular_genre_books = shorten_labels(popular_genre_books, "Book Name")

    # --- Display Tables ---
    col1, _, col2 = st.columns([1, 0.1, 1.2])
    with col1:
        st.subheader("🏆 Top 5 Genres by Review Count")
        st.table(top_5_genres)
    with col2:
        st.subheader("✍️ Top 5 Authors by Review Count")
        st.table(top_5_authors)

    # --- Bar Charts ---
    col1, _, col2 = st.columns([1, 0.1, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x="Number of Reviews", y="Genre", data=top_5_genres, palette="viridis", ax=ax)
        ax.set_title("Top 5 Genres by Reviews")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x="Number of Reviews", y="Author", data=top_5_authors, palette="magma", ax=ax)
        ax.set_title("Top 5 Authors by Reviews")
        st.pyplot(fig)

    # --- Rating Distributions ---
    col1, _, col2 = st.columns([1, 0.1, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(filtered_df["Rating"], bins=20, kde=True, color="blue", ax=ax)
        ax.set_title("Distribution of Ratings")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x=filtered_df["Rating Range"], palette="coolwarm", ax=ax)
        ax.set_title("Rating Ranges")
        st.pyplot(fig)

    # --- Top Books in Most Popular Genre ---
    if not popular_genre_books.empty:
        col1, _, col2 = st.columns([1, 0.1, 1])
        with col1:
            st.subheader(f"📚 Top 5 Books in {most_popular_genre} Genre")
            st.table(popular_genre_books)
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x="Rating", y="Book Name", data=popular_genre_books, palette="viridis", ax=ax)
            ax.set_title(f"Top 5 Books in {most_popular_genre} Genre")
            st.pyplot(fig)

    # --- Review Count Distribution ---
    col1, _, col2 = st.columns([1, 0.1, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(filtered_df["Number of Reviews"], bins=20, kde=True, color="purple", ax=ax)
        ax.set_title("Distribution of Review Counts")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x=filtered_df["Number of Reviews"], color="red", ax=ax)
        ax.set_title("Review Count Boxplot")
        st.pyplot(fig)

    # --- Correlation Heatmap ---
    st.subheader("📊 Correlation Heatmap")
    correlation_cols = ["Rating", "Number of Reviews"]
    corr_matrix = filtered_df[correlation_cols].corr()
    fig, ax = plt.subplots(figsize=(3, 2))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center")
    ax.set_title("Correlation between Ratings and Reviews")
    st.pyplot(fig)

if __name__ == "__main__":
    main()
