import pandas as pd
import os

# Define file paths
CLEANED_DATA_PATH = "data/cleaned"
MERGED_DATA_PATH = "data/merged"

# Ensure merged directory exists
os.makedirs(MERGED_DATA_PATH, exist_ok=True)

def main():
    """Main function to load, clean, merge, and save the dataset."""
    # Load datasets
    basic_file = f"{CLEANED_DATA_PATH}/audible_catalog_cleaned.csv"
    advanced_file = f"{CLEANED_DATA_PATH}/audible_catalog_advanced_cleaned.csv"

    if not os.path.exists(basic_file) or not os.path.exists(advanced_file):
        print("❌ One or both input files are missing. Please check the paths.")
        return

    df_basic = pd.read_csv(basic_file)
    df_advanced = pd.read_csv(advanced_file)

    # Standardize column names
    df_basic.columns = df_basic.columns.str.strip()
    df_advanced.columns = df_advanced.columns.str.strip()

    # Debugging: Print column names before merging
    print("✅ Columns in df_basic:", df_basic.columns.tolist())
    print("✅ Columns in df_advanced:", df_advanced.columns.tolist())

    # Ensure column consistency
    if "Listening Time (mins)" not in df_advanced.columns:
        print("⚠️ Warning: 'Listening Time (mins)' column not found in df_advanced.")

    if "Main Genre" not in df_advanced.columns:
        print("⚠️ Warning: 'Main Genre' column not found in df_advanced.")

    # Remove duplicate entries based on 'Book Name' and 'Author'
    df_basic.drop_duplicates(subset=["Book Name", "Author"], inplace=True)
    df_advanced.drop_duplicates(subset=["Book Name", "Author"], inplace=True)

    # Merge datasets using an inner join (keeping only common books)
    df_merged = pd.merge(df_basic, df_advanced, on=["Book Name", "Author"], how="inner")

    # Rename 'Main Genre' to 'Genres' for consistency
    if "Main Genre" in df_merged.columns:
        df_merged.rename(columns={"Main Genre": "Genres"}, inplace=True)

    # Fill NaN values with "Not Available"
    df_merged.fillna("Not Available", inplace=True)

    # Save the merged dataset
    merged_file = f"{MERGED_DATA_PATH}/audible_catalog_merged.csv"
    df_merged.to_csv(merged_file, index=False)

    print(f"✅ Merged dataset saved at: {merged_file}")
    print(f"📊 Total books after merging: {len(df_merged)}")
    print(f"🔍 Unique genres: {df_merged['Genres'].nunique()}" if "Genres" in df_merged.columns else "⚠️ 'Genres' column missing.")
    
# Run the script
if __name__ == "__main__":
    main()
