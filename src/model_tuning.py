import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ✅ Paths
file_path = "data/processed/processed_audible_catalog_merged.csv"
best_model_path = "models/best_tuned_model.pkl"
scaler_path = "models/scaler.pkl"

# ✅ Load Data
data = pd.read_csv(file_path)

# ✅ Print available columns
print("📌 Available Columns in Dataset:\n", data.columns.tolist())

# ✅ Feature Selection (Check Column Names)
selected_features = ["Listening Time (mins)", "Price_x", "Rating_x", "Number of Reviews_x"]  # Match model_training.py
target_column = "Popularity Score"

# ✅ Check for missing columns
missing_features = [col for col in selected_features if col not in data.columns]
if missing_features:
    raise KeyError(f"Columns not found in dataset: {missing_features}")

X = data[selected_features].copy()
y = data[target_column].values

# ✅ Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Save Scaler
os.makedirs("models", exist_ok=True)
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42)

# ✅ Hyperparameter Grids
param_grid = {
    "Random Forest": {"n_estimators": [100, 200], "max_depth": [10, 20], "min_samples_split": [2, 5]},
    "Gradient Boosting": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
    "XGBoost": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
    "LightGBM": {"n_estimators": [300, 500], "learning_rate": [0.01, 0.05], "max_depth": [5, 7]}
}

# ✅ Model Selection
best_model = None
best_score = float("inf")
best_model_name = None

for model_name, params in param_grid.items():
    print(f"🔍 Tuning {model_name}...")
    
    if model_name == "Random Forest":
        model = RandomForestRegressor(random_state=42)
    elif model_name == "Gradient Boosting":
        model = GradientBoostingRegressor(random_state=42)
    elif model_name == "XGBoost":
        model = XGBRegressor(objective='reg:squarederror', random_state=42)
    elif model_name == "LightGBM":
        model = LGBMRegressor(random_state=42)
    
    grid_search = GridSearchCV(model, param_grid=params, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model_temp = grid_search.best_estimator_
    y_pred = best_model_temp.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"✅ Best RMSE for {model_name}: {rmse:.4f}")
    
    if rmse < best_score:
        best_score = rmse
        best_model = best_model_temp
        best_model_name = model_name

# ✅ Save Best Model
with open(best_model_path, "wb") as f:
    pickle.dump(best_model, f)

print(f"🏆 Best Tuned Model: {best_model_name} (RMSE: {best_score:.4f})")
