# Lab05 A4 – K-Means Clustering (k=2) on Confidence Feature Dataset
# Author: Udhaya 

import pandas as U_pd
from sklearn.cluster import KMeans as U_KMeans

# ------------------- 1. Load Feature Data (excluding target) -------------------
def U_load_clustering_data(U_file_path):
    """
    Loads the dataset and removes the target column for unsupervised clustering.
    """
    U_df = U_pd.read_csv(U_file_path)
    U_feature_data = U_df[['mfcc1', 'rms', 'zcr', 'pitch_std', 'silence_pct']]  # Only features
    return U_feature_data

# ------------------- 2. Perform K-Means Clustering -------------------
def U_apply_kmeans(U_data, U_k_value=2):
    """
    Applies KMeans clustering on the dataset.
    """
    U_model = U_KMeans(n_clusters=U_k_value, random_state=0, n_init="auto")
    U_model.fit(U_data)
    return U_model.labels_, U_model.cluster_centers_

# ------------------- 3. Main Block -------------------
if __name__ == "__main__":
    U_csv_path = "features_lab3_labeled.csv"  # Update path if needed

    # Load feature data
    U_X_features = U_load_clustering_data(U_csv_path)

    # Run K-Means for k=2
    cluster_labels, cluster_centers = U_apply_kmeans(U_X_features, U_k_value=2)

    # Output
    print("\nKMeans Clustering Completed (k=2)")
    print(f"Cluster Labels:\n{cluster_labels}")
    print(f"\nCluster Centers:\n{cluster_centers}")
