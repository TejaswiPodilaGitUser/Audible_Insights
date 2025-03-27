import pandas as pd
import os

# Define file paths
RAW_DATA_PATH = "data/raw"
CLEANED_DATA_PATH = "data/cleaned"

# Ensure cleaned directory exists
os.makedirs(CLEANED_DATA_PATH, exist_ok=True)

# Load datasets
df_basic = pd.read_csv(f"{RAW_DATA_PATH}/Audible_Catlog.csv")
df_advanced = pd.read_csv(f"{RAW_DATA_PATH}/Audible_Catlog_Advanced_Features.csv")

# Standardize column names (only strip spaces, no other modifications)
df_basic.columns = df_basic.columns.str.strip()
df_advanced.columns = df_advanced.columns.str.strip()

# Debugging: Check available columns
print("✅ Columns in df_basic:", df_basic.columns.tolist())
print("✅ Columns in df_advanced:", df_advanced.columns.tolist())

# ---- PROCESSING df_basic ---- #

# Convert 'Rating' and 'Number of Reviews' to numeric
if "Rating" in df_basic.columns and "Number of Reviews" in df_basic.columns:
    df_basic["Rating"] = pd.to_numeric(df_basic["Rating"], errors="coerce").fillna(df_basic["Rating"].median())
    df_basic["Number of Reviews"] = pd.to_numeric(df_basic["Number of Reviews"], errors="coerce").fillna(0)
    df_basic["Popularity Score"] = (df_basic["Number of Reviews"] * df_basic["Rating"]).apply(lambda x: max(x, 1))
else:
    print("⚠️ Warning: 'Rating' or 'Number of Reviews' column not found in df_basic.")

# Fill missing values
df_basic.ffill(inplace=True)
df_basic.bfill(inplace=True)

# Remove duplicate rows
df_basic.drop_duplicates(inplace=True)

# Save cleaned Audible_Catalog.csv
cleaned_basic_file = f"{CLEANED_DATA_PATH}/audible_catalog_cleaned.csv"
df_basic.to_csv(cleaned_basic_file, index=False)
print(f"✅ Cleaned basic dataset saved at: {cleaned_basic_file}")

# ---- PROCESSING df_advanced ---- #

# Convert 'Listening Time' to minutes
def convert_listening_time(time_str):
    if pd.isna(time_str) or not isinstance(time_str, str):
        return None
    time_parts = time_str.split(" ")
    hours, minutes = 0, 0
    try:
        if "hour" in time_parts:
            hours = int(time_parts[0])
        if "minute" in time_parts:
            minutes = int(time_parts[-2])
    except (ValueError, IndexError):
        return None
    return hours * 60 + minutes

if "Listening Time" in df_advanced.columns:
    df_advanced["Listening Time (mins)"] = df_advanced["Listening Time"].apply(convert_listening_time)
    df_advanced.drop(columns=["Listening Time"], inplace=True)
else:
    print("⚠️ Warning: 'Listening Time' column not found in df_advanced.")

# Extract main genre
def extract_main_genre(ranks_genre):
    if pd.isna(ranks_genre) or not isinstance(ranks_genre, str):
        return None
    genres = ranks_genre.split(", ")
    return genres[1] if len(genres) > 1 else genres[0]

if "Ranks and Genre" in df_advanced.columns:
    df_advanced["Main Genre"] = df_advanced["Ranks and Genre"].apply(extract_main_genre)
    df_advanced.drop(columns=["Ranks and Genre"], inplace=True)
else:
    print("⚠️ Warning: 'Ranks and Genre' column not found in df_advanced.")

# Handle missing values
df_advanced.ffill(inplace=True)
df_advanced.bfill(inplace=True)

# Remove duplicate rows
df_advanced.drop_duplicates(inplace=True)

# Save cleaned Audible_Catlog_Advanced_Features.csv
cleaned_advanced_file = f"{CLEANED_DATA_PATH}/audible_catalog_advanced_cleaned.csv"
df_advanced.to_csv(cleaned_advanced_file, index=False)
print(f"✅ Cleaned advanced dataset saved at: {cleaned_advanced_file}")
