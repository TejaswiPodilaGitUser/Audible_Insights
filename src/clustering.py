import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, AgglomerativeClustering, Birch, OPTICS
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, v_measure_score, precision_score, recall_score, mean_squared_error
)

# Load dataset
file_path = "data/processed/processed_audible_catalog_merged.csv"
data = pd.read_csv(file_path)

if 'Description' not in data.columns:
    raise ValueError(f"❌ 'Description' column missing in {file_path}. Check preprocessing.")

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
book_tfidf_matrix = vectorizer.fit_transform(data['Description'])

# Save the TF-IDF vectorizer
with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Clustering Models
models = {
    "KMeans_5": KMeans(n_clusters=5, random_state=42, n_init=10),
    "KMeans_10": KMeans(n_clusters=10, random_state=42, n_init=10),
    "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
    "Spectral_Clustering": SpectralClustering(n_clusters=5, affinity='nearest_neighbors', random_state=42),
    "Agglomerative": AgglomerativeClustering(n_clusters=5),
    "Birch": Birch(n_clusters=5)
}

# Store Model Performance
model_metrics = []
best_model = None
best_score = -1
best_model_name = ""

# Check if Ground Truth Labels Exist
use_ground_truth = "Primary Use" in data.columns
true_labels = data["Primary Use"].factorize()[0] if use_ground_truth else None

# Train and Evaluate Models
for model_name, model in models.items():
    try:
        labels = model.fit_predict(book_tfidf_matrix)

        # Avoid metrics on noise points (-1 in DBSCAN)
        unique_labels = set(labels)
        if -1 in unique_labels and len(unique_labels) == 1:
            silhouette = -1
            davies_bouldin = -1
            calinski_harabasz = -1
        else:
            silhouette = silhouette_score(book_tfidf_matrix, labels)
            davies_bouldin = davies_bouldin_score(book_tfidf_matrix.toarray(), labels)
            calinski_harabasz = calinski_harabasz_score(book_tfidf_matrix.toarray(), labels)

        # Compute ARI & V-Measure only if ground truth labels exist
        ari = adjusted_rand_score(true_labels, labels) if use_ground_truth else "N/A"
        v_measure = v_measure_score(true_labels, labels) if use_ground_truth else "N/A"

        # Compute Precision, Recall (only if ground truth labels exist)
        precision = precision_score(true_labels, labels, average='weighted', zero_division=0) if use_ground_truth else "N/A"
        recall = recall_score(true_labels, labels, average='weighted', zero_division=0) if use_ground_truth else "N/A"

        # Compute RMSE (Treating cluster labels as predictions)
        rmse = np.sqrt(mean_squared_error(true_labels, labels)) if use_ground_truth else "N/A"

        # Save Model
        with open(f"models/{model_name}.pkl", "wb") as f:
            pickle.dump(model, f)

        # Store Metrics
        model_metrics.append({
            "Model": model_name,
            "Silhouette Score": silhouette,
            "Davies-Bouldin Score": davies_bouldin,
            "Calinski-Harabasz Score": calinski_harabasz,
            "Adjusted Rand Index": ari,
            "V-Measure Score": v_measure,
            "Precision": precision,
            "Recall": recall,
            "RMSE": rmse
        })

        # Update Best Model (based on Silhouette Score)
        if silhouette > best_score:
            best_score = silhouette
            best_model = model
            best_model_name = model_name

    except Exception as e:
        print(f"⚠️ Model {model_name} failed: {e}")

# Save Best Model
if best_model:
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    print(f"🏆 Best Model: {best_model_name} (Silhouette Score: {best_score:.4f})")

# Save Model Metrics to CSV
metrics_df = pd.DataFrame(model_metrics)
metrics_df.to_csv("results/model_metrics.csv", index=False)
print("📊 Model Metrics Saved in 'results/model_metrics.csv'")

print("✅ Clustering Completed Successfully!")
