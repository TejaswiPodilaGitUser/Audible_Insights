import pandas as pd
import os
import re

# Define file paths
RAW_DATA_PATH = "data/raw"
CLEANED_DATA_PATH = "data/cleaned"

# Ensure cleaned directory exists
os.makedirs(CLEANED_DATA_PATH, exist_ok=True)

# Load datasets
df_basic = pd.read_csv(f"{RAW_DATA_PATH}/Audible_Catlog.csv")
df_advanced = pd.read_csv(f"{RAW_DATA_PATH}/Audible_Catlog_Advanced_Features.csv")

# Standardize column names (strip spaces)
df_basic.columns = df_basic.columns.str.strip()
df_advanced.columns = df_advanced.columns.str.strip()

# Keep only necessary columns
basic_columns = ['Book Name', 'Author', 'Rating', 'Number of Reviews', 'Price']
df_basic = df_basic[[col for col in basic_columns if col in df_basic.columns]]

advanced_columns = ['Book Name', 'Author', 'Description', 'Listening Time', 'Ranks and Genre']
df_advanced = df_advanced[[col for col in advanced_columns if col in df_advanced.columns]]

# Strip whitespace from all string columns (updated: no applymap)
for col in df_basic.select_dtypes(include='object').columns:
    df_basic[col] = df_basic[col].map(lambda x: x.strip() if isinstance(x, str) else x)

for col in df_advanced.select_dtypes(include='object').columns:
    df_advanced[col] = df_advanced[col].map(lambda x: x.strip() if isinstance(x, str) else x)

# ---- PROCESSING df_basic ---- #
affected_columns_basic = 0
affected_values_basic = 0

# Fill missing values for categorical columns
for col in ["Book Name", "Author"]:
    if col in df_basic.columns:
        missing_count = df_basic[col].isna().sum()
        if missing_count > 0:
            df_basic[col] = df_basic[col].fillna(f"Unknown {col}")
            affected_columns_basic += 1
            affected_values_basic += missing_count

# Convert Rating to numeric
if "Rating" in df_basic.columns:
    missing_count = df_basic["Rating"].isna().sum()
    df_basic["Rating"] = pd.to_numeric(df_basic["Rating"], errors="coerce").fillna(df_basic["Rating"].median()).clip(1, 5)
    affected_columns_basic += (missing_count > 0)
    affected_values_basic += missing_count

# Convert Number of Reviews
if "Number of Reviews" in df_basic.columns:
    missing_count = df_basic["Number of Reviews"].isna().sum()
    df_basic["Number of Reviews"] = pd.to_numeric(df_basic["Number of Reviews"], errors="coerce").fillna(0)
    affected_columns_basic += (missing_count > 0)
    affected_values_basic += missing_count

# Handle Price
if "Price" in df_basic.columns:
    missing_count = df_basic["Price"].isna().sum()
    df_basic["Price"] = pd.to_numeric(df_basic["Price"], errors="coerce")
    if missing_count > 0:
        df_basic["Price"] = df_basic["Price"].fillna(df_basic["Price"].median())
        affected_columns_basic += 1
        affected_values_basic += missing_count
    df_basic["Price"] = df_basic["Price"].clip(0, df_basic["Price"].quantile(0.95))

# Remove duplicate rows
df_basic.drop_duplicates(subset=["Book Name", "Author"], inplace=True)

# Save cleaned basic dataset
df_basic.to_csv(f"{CLEANED_DATA_PATH}/audible_catalog_cleaned.csv", index=False)

# ---- PROCESSING df_advanced ---- #
affected_columns_advanced = 0
affected_values_advanced = 0

# Fill missing descriptions
if "Description" in df_advanced.columns:
    missing_count = df_advanced["Description"].isna().sum()
    df_advanced["Description"] = df_advanced["Description"].fillna("No Description Available")

    def clean_text(text):
        """Remove special characters and extra spaces."""
        return re.sub(r'[^a-zA-Z0-9 .,]', '', text).strip()

    df_advanced["Description"] = df_advanced["Description"].apply(clean_text)
    affected_columns_advanced += 1
    affected_values_advanced += missing_count

# Convert Listening Time to minutes
def convert_listening_time(time_str):
    """Convert '1 hour 30 minutes' to total minutes."""
    if pd.isna(time_str) or not isinstance(time_str, str):
        return None
    time_str = time_str.lower()
    hours = minutes = 0
    match_hours = re.search(r'(\d+)\s*hour', time_str)
    match_minutes = re.search(r'(\d+)\s*minute', time_str)
    if match_hours:
        hours = int(match_hours.group(1))
    if match_minutes:
        minutes = int(match_minutes.group(1))
    return hours * 60 + minutes

if "Listening Time" in df_advanced.columns:
    missing_count = df_advanced["Listening Time"].isna().sum()
    df_advanced["Listening Time (mins)"] = df_advanced["Listening Time"].apply(convert_listening_time)
    df_advanced.drop(columns=["Listening Time"], inplace=True)
    if missing_count > 0:
        df_advanced["Listening Time (mins)"] = df_advanced["Listening Time (mins)"].fillna(df_advanced["Listening Time (mins)"].median())
        affected_columns_advanced += 1
        affected_values_advanced += missing_count

# Extract main genre
def extract_main_genre(ranks_genre):
    if pd.isna(ranks_genre) or not isinstance(ranks_genre, str):
        return "Unknown Genre"
    genres = ranks_genre.split(", ")
    return genres[1] if len(genres) > 1 else genres[0]

if "Ranks and Genre" in df_advanced.columns:
    missing_count = df_advanced["Ranks and Genre"].isna().sum()
    df_advanced["Main Genre"] = df_advanced["Ranks and Genre"].apply(extract_main_genre)
    df_advanced.drop(columns=["Ranks and Genre"], inplace=True)
    affected_columns_advanced += (missing_count > 0)
    affected_values_advanced += missing_count

# Remove duplicate rows
df_advanced.drop_duplicates(subset=["Book Name", "Author"], inplace=True)

# Save cleaned advanced dataset
df_advanced.to_csv(f"{CLEANED_DATA_PATH}/audible_catalog_advanced_cleaned.csv", index=False)

# ✅ Summary
print("✅ Data cleaning completed!")
print("📄 Summary:")
print(f"➡️  df_basic:     {affected_columns_basic} columns cleaned, {affected_values_basic} values filled/converted")
print(f"➡️  df_advanced:  {affected_columns_advanced} columns cleaned, {affected_values_advanced} values filled/converted")
