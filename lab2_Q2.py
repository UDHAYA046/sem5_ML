import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

def u_compute_similarity_matrices(u_file_path):
    """
    Loads data from the Excel file, performs random sampling (20 rows),
    calculates Jaccard, SMC, and Cosine similarity matrices.
    Returns: all three matrices.
    """

    # Load data
    u_df = pd.read_excel(u_file_path, sheet_name="Purchase data")

    # Select numeric columns
    u_numeric_df = u_df.select_dtypes(include=[np.number])

    # Fill missing values (if any) with column means
    u_numeric_df = u_numeric_df.apply(lambda col: col.fillna(col.mean()), axis=0)

    # Sample 20 rows (or all if less than 20)
    if len(u_numeric_df) < 20:
        print(f"Warning: Only {len(u_numeric_df)} rows available.")
        u_sampled = u_numeric_df.copy()
    else:
        u_sampled = u_numeric_df.sample(n=20, random_state=42).reset_index(drop=True)

    # Normalize for Cosine
    scaler = MinMaxScaler()
    u_normalized = scaler.fit_transform(u_sampled)

    # Binarize using column means for Jaccard & SMC
    u_binary_df = (u_sampled > u_sampled.mean()).astype(int).values  # ✅ convert to ndarray

    # --- Compute Similarities ---
    # Jaccard
    jaccard_sim = 1 - pairwise_distances(u_binary_df, metric='jaccard')

    # Simple Matching Coefficient
    def smc(x, y):
        return np.sum(x == y) / len(x)

    smc_sim = np.array([[smc(x, y) for y in u_binary_df] for x in u_binary_df])

    # Cosine
    cos_sim = cosine_similarity(u_normalized)

    return jaccard_sim, smc_sim, cos_sim

# --- MAIN PROGRAM ---
u_file_path = "Lab Session Data.xlsx"
jaccard_sim, smc_sim, cos_sim = u_compute_similarity_matrices(u_file_path)

# Plot Heatmaps
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.heatmap(jaccard_sim, cmap="Blues")
plt.title("Jaccard Similarity")

plt.subplot(1, 3, 2)
sns.heatmap(smc_sim, cmap="Oranges")
plt.title("Simple Matching Coefficient (SMC)")

plt.subplot(1, 3, 3)
sns.heatmap(cos_sim, cmap="coolwarm")
plt.title("Cosine Similarity")

plt.tight_layout()
plt.show()
