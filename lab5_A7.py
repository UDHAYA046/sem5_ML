# Lab05 A7 – Elbow Plot using Inertia (Distortion)
# Author: Udhaya 

import pandas as U_pd
import matplotlib.pyplot as U_plt
from sklearn.cluster import KMeans as U_KMeans

# ---------------- 1. Load Feature Data ----------------
def U_load_features_for_elbow(U_path):
    U_df = U_pd.read_csv(U_path)
    return U_df[['mfcc1', 'rms', 'zcr', 'pitch_std', 'silence_pct']]

# ---------------- 2. Compute Inertia for Each k ----------------
def U_compute_distortion(U_data, U_k_start=2, U_k_end=10):
    U_k_vals = list(range(U_k_start, U_k_end + 1))
    U_inertias = []

    for U_k in U_k_vals:
        U_model = U_KMeans(n_clusters=U_k, random_state=42, n_init="auto")
        U_model.fit(U_data)
        U_inertias.append(U_model.inertia_)

    return U_k_vals, U_inertias

# ---------------- 3. Plot Elbow ----------------
def U_plot_elbow(U_k_vals, U_inertias):
    U_plt.figure(figsize=(8, 5))
    U_plt.plot(U_k_vals, U_inertias, marker='o', color='purple')
    U_plt.title("Elbow Plot: k vs Inertia")
    U_plt.xlabel("Number of Clusters (k)")
    U_plt.ylabel("Inertia (Distortion)")
    U_plt.grid(True)
    U_plt.tight_layout()
    U_plt.show()

# ---------------- 4. Main Driver ----------------
if __name__ == "__main__":
    U_file_path = "features_lab3_labeled.csv"
    U_X = U_load_features_for_elbow(U_file_path)

    U_k_list, U_distortions = U_compute_distortion(U_X, 2, 10)

    print("\nElbow Plot Inertia Values:")
    for i in range(len(U_k_list)):
        print(f"k = {U_k_list[i]} | Inertia = {U_distortions[i]:.2f}")

    U_plot_elbow(U_k_list, U_distortions)
