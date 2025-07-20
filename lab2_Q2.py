import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler  # ✅ THIS ONE IS MISSING
import matplotlib.pyplot as plt
import seaborn as sns


import matplotlib.pyplot as plt
import seaborn as sns

def u_compute_similarity_matrices(file_path):
    """
    SAMPLES 20 OBSERVATION VECTORS FROM NUMERIC PURCHASE DATA,
    THEN COMPUTES JACCARD, SMC, AND COSINE SIMILARITY MATRICES.
    RETURNS:
    - jaccard_sim
    - smc_sim
    - cos_sim
    """

    # Load and clean data
    df = pd.read_excel(file_path, sheet_name="Purchase data")

    # Select numeric columns and fill NaNs with mean
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df = numeric_df.apply(lambda col: col.fillna(col.mean()), axis=0)

    # ✅ Ensure enough data rows exist before sampling
    num_rows = numeric_df.shape[0]
    if num_rows < 20:
        sample_df = numeric_df.copy()
    else:
        sample_df = numeric_df.sample(n=20, random_state=42).reset_index(drop=True)

    # Normalize for cosine similarity
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(sample_df)

    # Binary encoding based on mean for SMC/Jaccard
    binary_df = (sample_df > sample_df.mean()).astype(int)

    # Jaccard Similarity
    jaccard_sim = 1 - pairwise_distances(binary_df, metric='jaccard')

    # SMC Similarity
    def smc(x, y):
        return np.sum(x == y) / len(x)

    smc_sim = np.array([[smc(a, b) for b in binary_df.values] for a in binary_df.values])

    # Cosine Similarity
    cos_sim = cosine_similarity(normalized)

    return jaccard_sim, smc_sim, cos_sim


if __name__ == "__main__":
    u_file_path = "Lab Session Data.xlsx"

    jaccard_sim, smc_sim, cos_sim = u_compute_similarity_matrices(u_file_path)

    # Plot all 3 similarity matrices
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.heatmap(jaccard_sim, cmap="Blues")
    plt.title("Jaccard Similarity")

    plt.subplot(1, 3, 2)
    sns.heatmap(smc_sim, cmap="Oranges")
    plt.title("SMC Similarity")

    plt.subplot(1, 3, 3)
    sns.heatmap(cos_sim, cmap="coolwarm")
    plt.title("Cosine Similarity")

    plt.tight_layout()
    plt.show()
