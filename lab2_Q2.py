import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

def u_compute_similarity_matrices(u_file_path):
    """
    Loads numeric purchase data, selects 20 rows randomly,
    and computes Jaccard, SMC, and Cosine similarity matrices.
    Displays all 3 as heatmaps.
    """

    # Load Excel and keep numeric attributes
    df = pd.read_excel(u_file_path, sheet_name="Purchase data")
    numeric_df = df.select_dtypes(include=[np.number])

    # Fill missing values with mean
    numeric_df = numeric_df.apply(lambda col: col.fillna(col.mean()), axis=0)
    
    # Drop any remaining rows with NaNs (important for cosine similarity)
    numeric_df = numeric_df.dropna()

    # Sample 20 rows
    if numeric_df.shape[0] < 20:
        print(f"Only {numeric_df.shape[0]} rows available. Sampling all of them.")
        sample_df = numeric_df.copy()
    else:
        sample_df = numeric_df.sample(n=20, random_state=42).reset_index(drop=True)

    # Normalize values to [0,1] for cosine similarity
    scaler = MinMaxScaler()
    u_normalized = scaler.fit_transform(sample_df)

    # Convert values to binary based on mean threshold for Jaccard/SMC
    binary_df = (sample_df > sample_df.mean()).astype(int)

    # Compute Jaccard Similarity
    jaccard_sim = 1 - pairwise_distances(binary_df, metric='jaccard')

    # Compute SMC Similarity
    def smc(x, y):
        return np.sum(x == y) / len(x)
    smc_sim = np.array([[smc(a, b) for b in binary_df.values] for a in binary_df.values])

    # Compute Cosine Similarity
    cos_sim = cosine_similarity(u_normalized)

    return jaccard_sim, smc_sim, cos_sim

# ==== MAIN EXECUTION ====
u_file_path = "Lab Session Data.xlsx"

# Call function
jaccard_sim, smc_sim, cos_sim = u_compute_similarity_matrices(u_file_path)

# Plot similarity heatmaps
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.heatmap(jaccard_sim, cmap="YlGnBu", cbar=True)
plt.title("Jaccard Similarity")

plt.subplot(1, 3, 2)
sns.heatmap(smc_sim, cmap="Oranges", cbar=True)
plt.title("SMC Similarity")

plt.subplot(1, 3, 3)
sns.heatmap(cos_sim, cmap="coolwarm", cbar=True)
plt.title("Cosine Similarity")

plt.tight_layout()
plt.show()
