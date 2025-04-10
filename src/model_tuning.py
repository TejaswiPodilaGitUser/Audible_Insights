import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, silhouette_score
import os

# File paths
clustered_data_path = "data/processed/clustered_books.csv"
cosine_sim_path = "models/cosine_similarity.pkl"
tfidf_vectorizer_path = "models/tfidf_vectorizer.pkl"
metrics_output_path = "evaluation/metrics_summary.csv"

def evaluate_models():
    print("📦 Loading dataset and models...")

    # Load clustered data
    data = pd.read_csv(clustered_data_path)

    # Load cosine similarity matrix
    with open(cosine_sim_path, "rb") as f:
        cosine_sim_matrix = pickle.load(f)

    # -------------------------------
    # Evaluate Clustering
    # -------------------------------
    print("🔍 Evaluating KMeans clustering...")

    sil_score = None
    if "Cluster" in data.columns:
        try:
            from sklearn.decomposition import TruncatedSVD

            with open(tfidf_vectorizer_path, "rb") as f:
                tfidf_vectorizer = pickle.load(f)

            tfidf_matrix = tfidf_vectorizer.transform(data["Description"].fillna(""))
            svd = TruncatedSVD(n_components=50, random_state=42)
            tfidf_reduced = svd.fit_transform(tfidf_matrix)
            sil_score = silhouette_score(tfidf_reduced, data["Cluster"])
            print(f"🔍 KMeans Silhouette Score: {sil_score:.4f}")
        except Exception as e:
            print("⚠️ Silhouette Score evaluation failed:", e)
    else:
        print("⚠️ 'Cluster' column not found. Skipping Silhouette Score.")

    # -------------------------------
    # Content-Based RMSE Evaluation
    # -------------------------------
    print("📊 Calculating RMSE for Content-Based Similarity...")

    possible_rating_cols = ['Ratings', 'Rating', 'Rating_y']
    rating_col = next((col for col in possible_rating_cols if col in data.columns), None)

    if not rating_col:
        raise ValueError("❌ No rating column found in dataset. Please check your CSV.")

    true_ratings = data[rating_col].fillna(0).values

    row_sums = np.abs(cosine_sim_matrix).sum(axis=1)
    row_sums[row_sums == 0] = 1e-8

    predicted_ratings = cosine_sim_matrix.dot(true_ratings) / row_sums
    predicted_ratings = np.nan_to_num(predicted_ratings, nan=np.mean(true_ratings))

    rmse_content = np.sqrt(mean_squared_error(true_ratings, predicted_ratings))
    print(f"📊 RMSE for Content-Based Similarity: {rmse_content:.4f}")

    # -------------------------------
    # Hybrid RMSE Evaluation
    # -------------------------------
    print("🔀 Calculating RMSE for Hybrid Recommendations...")

    rmse_hybrid = None
    if "Cluster" in data.columns:
        cluster_means = data.groupby("Cluster")[rating_col].transform("mean")
        hybrid_pred = 0.5 * predicted_ratings + 0.5 * cluster_means.fillna(np.mean(true_ratings)).values
        rmse_hybrid = np.sqrt(mean_squared_error(true_ratings, hybrid_pred))
        print(f"🔀 RMSE for Hybrid Recommendations: {rmse_hybrid:.4f}")
    else:
        print("⚠️ Cluster information not found. Skipping hybrid RMSE.")

    # -------------------------------
    # Precision@5
    # -------------------------------
    print("📏 Calculating Precision@5...")

    def precision_at_k(sim_matrix, true_ratings, k=5):
        precision_scores = []
        for idx, row in enumerate(sim_matrix):
            similar_indices = row.argsort()[::-1][1:k+1]
            relevant = sum(true_ratings[similar_indices] >= 4)
            precision_scores.append(relevant / k)
        return np.mean(precision_scores)

    print("Books with rating >= 4:", (true_ratings >= 4).sum())

    precision5 = precision_at_k(cosine_sim_matrix, true_ratings, k=5)
    print(f"🎯 Precision@5: {precision5:.4f}")

    # -------------------------------
    # Save metrics
    # -------------------------------
    print("💾 Saving evaluation metrics...")

    metrics = {
        "Silhouette Score": round(sil_score, 4) if sil_score is not None else None,
        "RMSE_Content": round(rmse_content, 4),
        "RMSE_Hybrid": round(rmse_hybrid, 4) if rmse_hybrid is not None else None,
        "Precision@5": round(precision5, 4)
    }

    metrics_df = pd.DataFrame([metrics])
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    metrics_df.to_csv(metrics_output_path, index=False)

    print(f"✅ Metrics saved to {metrics_output_path}")

if __name__ == "__main__":
    evaluate_models()
