import pandas as pd
import os
import re  # Regex for extracting rank

# File paths
INPUT_FILE = "data/merged/audible_catalog_merged.csv"
OUTPUT_FILE = "data/processed/audible_catalog_processed.csv"

# Ensure output directory exists
os.makedirs("data/processed", exist_ok=True)

# Load dataset
def load_data(filepath):
    return pd.read_csv(filepath)

# Extract Rank and Genre properly
def extract_rank_genre(genres_str):
    extracted_data = []
    
    if pd.isna(genres_str):
        return extracted_data
    
    genres = genres_str.split(",")  # Split by comma
    
    for genre in genres:
        genre = genre.strip()
        if genre.startswith("#"):  # Ensure it is ranked
            parts = genre.split(" in ", maxsplit=1)
            
            if len(parts) == 2:
                rank_text = parts[0].replace("#", "").strip()  # Extract rank text
                
                # Extract the first number in the rank (Handles "1 Free" issue)
                rank_match = re.search(r"\d+", rank_text)
                
                if rank_match:
                    rank = int(rank_match.group())  # Convert to integer
                    genre_name = parts[1].strip()  # Extract genre name
                    
                    # Identify if it's a Free or Audible category
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
def process_genres(df):
    genre_list = []
    main_genre_col = []  # Stores top genre for each book
    top_rank_col = []  # Stores the highest rank
    is_free_col = []  # Stores Free status
    is_audible_col = []  # Stores Audible status
    
    for _, row in df.iterrows():
        book_name = row["Book Name"]
        extracted_genres = extract_rank_genre(row["Genres"])
        
        if extracted_genres:
            # Sort genres by rank (lowest rank is best)
            extracted_genres.sort(key=lambda x: x["Rank"])
            top_genre = extracted_genres[0]  # Take the top-ranked genre
            
            # Add columns to df
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
            # If no genres found, add NaN values
            main_genre_col.append(None)
            top_rank_col.append(None)
            is_free_col.append(False)
            is_audible_col.append(False)

    # Add new columns to df
    df["Main Genre"] = main_genre_col
    df["Top Rank"] = top_rank_col
    df["Is Free"] = is_free_col
    df["Is Audible"] = is_audible_col

    # Create a new DataFrame with extracted genre information
    genre_df = pd.DataFrame(genre_list)

    # Save processed genre information separately
    genre_df.to_csv("data/processed/book_genres.csv", index=False)
    print("✅ Processed genre information saved to: data/processed/book_genres.csv")

    return df  # Return updated df

# Save processed data
def save_data(df, filepath):
    df.to_csv(filepath, index=False)

# Main function
def process_and_save():
    print("📌 Loading dataset...")
    df = load_data(INPUT_FILE)

    print("📌 Processing genres...")
    df = process_genres(df)

    print("📌 Saving cleaned dataset...")
    save_data(df, OUTPUT_FILE)

    print(f"✅ Dataset processed successfully! Output saved to: {OUTPUT_FILE}")

# Run script
if __name__ == "__main__":
    process_and_save()
