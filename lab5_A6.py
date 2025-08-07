# Lab05 A6 – Cluster Evaluation for Multiple k Values (2 to 10)
# Author: Udhaya

import pandas as U_pd
import matplotlib.pyplot as U_plt
from sklearn.cluster import KMeans as U_KMeans
from sklearn.metrics import silhouette_score as U_sil_score
from sklearn.metrics import calinski_harabasz_score as U_ch_score
from sklearn.metrics import davies_bouldin_score as U_db_score

# ---------------- 1. Load Features ----------------
def U_load_feature_data(U_path):
    U_df = U_pd.read_csv(U_path)
    return U_df[['mfcc1', 'rms', 'zcr', 'pitch_std', 'silence_pct']]

# ---------------- 2. Evaluate for Multiple k ----------------
def U_evaluate_multiple_k(U_data, U_k_min=2, U_k_max=10):
    U_k_values = list(range(U_k_min, U_k_max + 1))
    U_sil_scores = []
    U_ch_scores = []
    U_db_indices = []

    for U_k in U_k_values:
        U_model = U_KMeans(n_clusters=U_k, random_state=42, n_init="auto")
        U_model.fit(U_data)
        U_labels = U_model.labels_

        U_sil = U_sil_score(U_data, U_labels)
        U_ch = U_ch_score(U_data, U_labels)
        U_db = U_db_score(U_data, U_labels)

        U_sil_scores.append(U_sil)
        U_ch_scores.append(U_ch)
        U_db_indices.append(U_db)

    return U_k_values, U_sil_scores, U_ch_scores, U_db_indices

# ---------------- 3. Plotting ----------------
def U_plot_clustering_metrics(U_k_vals, U_sil_vals, U_ch_vals, U_db_vals):
    U_plt.figure(figsize=(12, 6))

    # Silhouette Score
    U_plt.subplot(1, 3, 1)
    U_plt.plot(U_k_vals, U_sil_vals, marker='o', color='blue')
    U_plt.title("Silhouette Score vs k")
    U_plt.xlabel("k")
    U_plt.ylabel("Silhouette Score")
    U_plt.grid(True)

    # Calinski-Harabasz Score
    U_plt.subplot(1, 3, 2)
    U_plt.plot(U_k_vals, U_ch_vals, marker='o', color='green')
    U_plt.title("CH Score vs k")
    U_plt.xlabel("k")
    U_plt.ylabel("Calinski-Harabasz Score")
    U_plt.grid(True)

    # Davies-Bouldin Index
    U_plt.subplot(1, 3, 3)
    U_plt.plot(U_k_vals, U_db_vals, marker='o', color='red')
    U_plt.title("DB Index vs k")
    U_plt.xlabel("k")
    U_plt.ylabel("Davies-Bouldin Index")
    U_plt.grid(True)

    U_plt.tight_layout()
    U_plt.show()

# ---------------- 4. Main ----------------
if __name__ == "__main__":
    U_file_path = "features_lab3_labeled.csv"
    U_X = U_load_feature_data(U_file_path)

    U_k_vals, U_sil_vals, U_ch_vals, U_db_vals = U_evaluate_multiple_k(U_X, 2, 10)

    print("\n Clustering Evaluation for k = 2 to 10:")
    for i, k in enumerate(U_k_vals):
        print(f"k = {k}: Silhouette = {U_sil_vals[i]:.4f}, CH = {U_ch_vals[i]:.2f}, DB = {U_db_vals[i]:.4f}")

    U_plot_clustering_metrics(U_k_vals, U_sil_vals, U_ch_vals, U_db_vals)
