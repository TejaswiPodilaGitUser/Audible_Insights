import pandas as pd
import os

# Define file paths
CLEANED_DATA_PATH = "data/cleaned"
MERGED_DATA_PATH = "data/merged"

# Ensure merged directory exists
os.makedirs(MERGED_DATA_PATH, exist_ok=True)

def main():
    """Main function to load, merge, and save the cleaned datasets."""
    basic_file = os.path.join(CLEANED_DATA_PATH, "audible_catalog_cleaned.csv")
    advanced_file = os.path.join(CLEANED_DATA_PATH, "audible_catalog_advanced_cleaned.csv")

    # Check file existence
    if not os.path.exists(basic_file) or not os.path.exists(advanced_file):
        print("❌ One or both input files are missing. Please check the paths.")
        return

    # Load datasets
    df_basic = pd.read_csv(basic_file)
    df_advanced = pd.read_csv(advanced_file)

    # Standardize column names (strip whitespace, lowercase optional)
    df_basic.columns = df_basic.columns.str.strip()
    df_advanced.columns = df_advanced.columns.str.strip()

    print("🔍 Preview of column names:")
    print("📘 df_basic:", df_basic.columns.tolist())
    print("📗 df_advanced:", df_advanced.columns.tolist())

    # Warn if key columns are missing
    required_advanced_cols = ["Listening Time (mins)", "Main Genre"]
    for col in required_advanced_cols:
        if col not in df_advanced.columns:
            print(f"⚠️ Warning: '{col}' column not found in df_advanced.")

    # Drop duplicates by book name and author
    df_basic.drop_duplicates(subset=["Book Name", "Author"], inplace=True)
    df_advanced.drop_duplicates(subset=["Book Name", "Author"], inplace=True)

    # Merge on common keys
    df_merged = pd.merge(df_basic, df_advanced, on=["Book Name", "Author"], how="inner")

    # Fill any missing values
    df_merged.fillna("Not Available", inplace=True)

    # Save merged dataset
    merged_file = os.path.join(MERGED_DATA_PATH, "audible_catalog_merged.csv")
    df_merged.to_csv(merged_file, index=False)

    # Summary
    print("\n✅ Merge successful!")
    print(f"📁 Merged file saved at: {merged_file}")
    print(f"📚 Total books after merge: {len(df_merged)}")
    if "Main Genre" in df_merged.columns:
        unique_genres = df_merged["Main Genre"].nunique()
        genre_preview = df_merged["Main Genre"].unique()[:5]
        print(f"🎭 Unique genres: {unique_genres} → {genre_preview} ...")
    else:
        print("⚠️ 'Main Genre' column is missing in the merged dataset.")

# Run
if __name__ == "__main__":
    main()
