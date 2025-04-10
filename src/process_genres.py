import pandas as pd
import os
import re

# File paths
INPUT_FILE = "data/merged/audible_catalog_merged.csv"
OUTPUT_FILE = "data/processed/audible_catalog_processed.csv"

# Ensure output directory exists
os.makedirs("data/processed", exist_ok=True)

# Load dataset
def load_data(filepath):
    df = pd.read_csv(filepath)
    print("🧾 Columns in the dataset:", df.columns.tolist())
    return df

# Identify the genre column dynamically
def get_genre_column(df):
    possible_columns = [col for col in df.columns if "genre" in col.lower() or "rank" in col.lower()]
    for col in possible_columns:
        if df[col].dtype == object:
            return col
    return None

# Extract Rank and Genre properly
def extract_rank_genre(genres_str):
    extracted_data = []
    
    if pd.isna(genres_str):
        return extracted_data
    
    genres = genres_str.split(",")
    
    for genre in genres:
        genre = genre.strip()
        if genre.startswith("#"):
            parts = genre.split(" in ", maxsplit=1)
            if len(parts) == 2:
                rank_text = parts[0].replace("#", "").strip()
                rank_match = re.search(r"\d+", rank_text)
                if rank_match:
                    rank = int(rank_match.group())
                    genre_name = parts[1].strip()
                    is_free = "Free" in rank_text
                    is_audible = "Audible" in genre_name
                    extracted_data.append({
                        "Rank": rank,
                        "Genre": genre_name,
                        "Is Free": is_free,
                        "Is Audible": is_audible
                    })
    return extracted_data

# Process genres and update df
def process_genres(df, genre_col):
    genre_list = []
    main_genre_col, top_rank_col = [], []
    is_free_col, is_audible_col = [], []

    for _, row in df.iterrows():
        book_name = row["Book Name"]
        extracted_genres = extract_rank_genre(row[genre_col])
        
        if extracted_genres:
            extracted_genres.sort(key=lambda x: x["Rank"])
            top_genre = extracted_genres[0]

            main_genre_col.append(top_genre["Genre"])
            top_rank_col.append(top_genre["Rank"])
            is_free_col.append(top_genre["Is Free"])
            is_audible_col.append(top_genre["Is Audible"])

            for item in extracted_genres:
                genre_list.append({
                    "Book Name": book_name,
                    "Rank": item["Rank"],
                    "Genre": item["Genre"],
                    "Is Free": item["Is Free"],
                    "Is Audible": item["Is Audible"]
                })
        else:
            main_genre_col.append(None)
            top_rank_col.append(None)
            is_free_col.append(False)
            is_audible_col.append(False)

    df["Main Genre"] = main_genre_col
    df["Top Rank"] = top_rank_col
    df["Is Free"] = is_free_col
    df["Is Audible"] = is_audible_col

    genre_df = pd.DataFrame(genre_list)
    genre_df.to_csv("data/processed/book_genres.csv", index=False)
    print("✅ Processed genre information saved to: data/processed/book_genres.csv")

    return df

# Save processed data
def save_data(df, filepath):
    df.to_csv(filepath, index=False)

# Main function
def process_and_save():
    print("📌 Loading dataset...")
    df = load_data(INPUT_FILE)

    print("📌 Detecting genre column...")
    genre_col = get_genre_column(df)

    if not genre_col:
        print("❌ 'Genres' column not found in dataset. Skipping genre processing.")
    else:
        print(f"✅ Genre column detected: '{genre_col}'")
        print("📌 Processing genres...")
        df = process_genres(df, genre_col)

    print("📌 Saving final processed dataset...")
    save_data(df, OUTPUT_FILE)
    print(f"✅ Dataset processing complete! File saved at: {OUTPUT_FILE}")

# Run script
if __name__ == "__main__":
    process_and_save()
