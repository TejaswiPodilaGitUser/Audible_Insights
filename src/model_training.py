import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# File Paths
file_path = "data/processed/processed_audible_catalog_merged.csv"
tfidf_model_path = "models/tfidf_vectorizer.pkl"
cosine_sim_path = "models/cosine_similarity.pkl"
metrics_output_path = "results/regression_model_metrics.csv"

# ✅ Load Dataset
data = pd.read_csv(file_path)

# ✅ Display Available Columns
print("📌 Available Columns in the Dataset:")
print(data.columns.tolist())

# ✅ Fix Column Naming Issues
rename_mapping = {
    "Rating_y": "Ratings",
    "Number of Reviews_y": "Number of Reviews",
    "Price_y": "Price"
}
data.rename(columns=rename_mapping, inplace=True)

# Drop unnecessary columns
data.drop(columns=["Rating_x", "Number of Reviews_x", "Price_x"], errors="ignore", inplace=True)

# ✅ Compute 'Popularity Score' if missing
if "Popularity Score" not in data.columns:
    data["Popularity Score"] = data["Ratings"] * np.log1p(data["Number of Reviews"])

# ✅ Check Required Columns
required_cols = ["Description", "Listening Time (mins)", "Price", "Ratings", "Number of Reviews", "Popularity Score"]
missing_cols = [col for col in required_cols if col not in data.columns]

if missing_cols:
    raise ValueError(f"❌ Missing columns in dataset: {missing_cols}. Check preprocessing.")

# ✅ TF-IDF Feature Extraction for Book Descriptions
vectorizer = TfidfVectorizer(stop_words='english')
book_tfidf_matrix = vectorizer.fit_transform(data['Description'])

# ✅ Save TF-IDF Vectorizer
os.makedirs("models", exist_ok=True)
with open(tfidf_model_path, "wb") as f:
    pickle.dump(vectorizer, f)

# ✅ Compute Cosine Similarity
cosine_sim_matrix = cosine_similarity(book_tfidf_matrix)

# ✅ Save Cosine Similarity Matrix
with open(cosine_sim_path, "wb") as f:
    pickle.dump(cosine_sim_matrix, f)

# ✅ Define Features and Target Variable
selected_features = ["Listening Time (mins)", "Price", "Ratings", "Number of Reviews"]
target_column = "Popularity Score"

X = data[selected_features]
y = data[target_column]

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Define Regression Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror'),
    "LightGBM": LGBMRegressor(n_estimators=100, random_state=42)
}

# ✅ Store Model Performance
model_metrics = []
best_model = None
best_score = float("inf")  # Lower RMSE is better
best_model_name = ""

# ✅ Train and Evaluate Models
for model_name, model in models.items():
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # ✅ Compute Metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # ✅ Save Model
        model_path = f"models/{model_name.replace(' ', '_')}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # ✅ Store Metrics
        model_metrics.append({
            "Model": model_name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R² Score": r2
        })

        # ✅ Update Best Model (Based on RMSE)
        if rmse < best_score:
            best_score = rmse
            best_model = model
            best_model_name = model_name

    except Exception as e:
        print(f"⚠️ Model {model_name} failed: {e}")

# ✅ Save Best Model
if best_model:
    best_model_path = "models/best_regression_model.pkl"
    with open(best_model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"🏆 Best Model: {best_model_name} (RMSE: {best_score:.4f})")

# ✅ Save Model Metrics to CSV
os.makedirs("results", exist_ok=True)
metrics_df = pd.DataFrame(model_metrics)
metrics_df.to_csv(metrics_output_path, index=False)
print(f"📊 Regression Model Metrics saved in '{metrics_output_path}'")

print("✅ Model Training & Evaluation Completed Successfully!")
