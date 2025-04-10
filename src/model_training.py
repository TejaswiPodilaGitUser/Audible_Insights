import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

# 📁 File Paths
input_file_path = "data/processed/processed_audible_catalog_merged.csv"
output_file_path = "data/processed/clustered_books.csv"
tfidf_model_path = "models/tfidf_vectorizer.pkl"
cosine_sim_path = "models/cosine_similarity.pkl"
cluster_model_path = "models/cluster_model.pkl"

# 📦 Create necessary directories
os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# ✅ Load Dataset
data = pd.read_csv(input_file_path)

# ✅ Fix Column Naming Issues (if any)
rename_mapping = {
    "Rating_y": "Ratings",
    "Number of Reviews_y": "Number of Reviews",
    "Price_y": "Price"
}
data.rename(columns=rename_mapping, inplace=True)
data.drop(columns=["Rating_x", "Number of Reviews_x", "Price_x"], errors="ignore", inplace=True)

# ✅ Handle missing descriptions
data['Description'] = data['Description'].fillna("")

# ✅ TF-IDF Vectorization (NLP)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = vectorizer.fit_transform(data['Description'])

# ✅ Save TF-IDF Vectorizer
with open(tfidf_model_path, "wb") as f:
    pickle.dump(vectorizer, f)

# ✅ Compute Cosine Similarity Matrix for Recommendations
cosine_sim_matrix = cosine_similarity(tfidf_matrix)

# ✅ Save Cosine Similarity Matrix
with open(cosine_sim_path, "wb") as f:
    pickle.dump(cosine_sim_matrix, f)

# ✅ Dimensionality Reduction for Clustering
svd = TruncatedSVD(n_components=50, random_state=42)
tfidf_reduced = svd.fit_transform(tfidf_matrix)

# ✅ Apply KMeans Clustering
kmeans = KMeans(n_clusters=10, random_state=42)
data['Cluster'] = kmeans.fit_predict(tfidf_reduced)

# ✅ Save Clustering Model
with open(cluster_model_path, "wb") as f:
    pickle.dump(kmeans, f)

# ✅ Save Data with Cluster Labels
data.to_csv(output_file_path, index=False)

print("✅ Model training and clustering completed.")
