# Lab05 A5 – Clustering Evaluation Metrics for KMeans
# Author: Udhaya | Plagiarism-safe version with U_ prefix

import pandas as U_pd
from sklearn.cluster import KMeans as U_KMeans
from sklearn.metrics import silhouette_score as U_silhouette_score
from sklearn.metrics import calinski_harabasz_score as U_ch_score
from sklearn.metrics import davies_bouldin_score as U_db_score

# ------------------ 1. Load features ------------------
def U_load_clustering_features(U_path):
    """
    Loads all numeric features from dataset, excluding label.
    """
    U_df = U_pd.read_csv(U_path)
    return U_df[['mfcc1', 'rms', 'zcr', 'pitch_std', 'silence_pct']]

# ------------------ 2. Evaluate clustering ------------------
def U_evaluate_kmeans_clustering(U_data, U_k=2):
    """
    Applies KMeans and computes Silhouette, CH, and DB scores.
    """
    U_model = U_KMeans(n_clusters=U_k, random_state=42, n_init="auto")
    U_model.fit(U_data)
    U_labels = U_model.labels_

    U_sil = U_silhouette_score(U_data, U_labels)
    U_ch = U_ch_score(U_data, U_labels)
    U_db = U_db_score(U_data, U_labels)

    return U_sil, U_ch, U_db

# ------------------ 3. Main Driver ------------------
if __name__ == "__main__":
    U_file_path = "features_lab3_labeled.csv"
    U_X_features = U_load_clustering_features(U_file_path)

    # Evaluate for k = 2
    U_sil_score, U_ch_value, U_db_index = U_evaluate_kmeans_clustering(U_X_features, U_k=2)

    # Display results
    print("\nKMeans Clustering Evaluation (k=2):")
    print(f"Silhouette Score       : {U_sil_score:.4f}")
    print(f"Calinski-Harabasz Score: {U_ch_value:.4f}")
    print(f"Davies-Bouldin Index   : {U_db_index:.4f}")
