import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

def u_compute_similarity_matrices(u_file_path):
    df = pd.read_excel(u_file_path, sheet_name="Purchase data")
    numeric_df = df.select_dtypes(include=[np.number])

    # Fill NaN with mean, then drop remaining NaNs
    numeric_df = numeric_df.apply(lambda col: col.fillna(col.mean()), axis=0)
    numeric_df = numeric_df.dropna()

    # Check if valid rows exist
    if numeric_df.empty:
        print("❌ No valid numeric data found after cleaning. Please check your dataset.")
        return None, None, None

    if numeric_df.shape[0] < 2:
        print(f"❌ Not enough rows to compute similarity. Only {numeric_df.shape[0]} row(s) available.")
        return None, None, None

    # Sample up to 20 rows
    sample_df = numeric_df if numeric_df.shape[0] <= 20 else numeric_df.sample(n=20, random_state=42).reset_index(drop=True)

    # Normalize for cosine
    scaler = MinMaxScaler()
    u_normalized = scaler.fit_transform(sample_df)

    # Binary encoding for Jaccard and SMC
    binary_df = (sample_df > sample_df.mean()).astype(int)

    # Jaccard
    jaccard_sim = 1 - pairwise_distances(binary_df, metric='jaccard')

    # SMC
    def smc(x, y): return np.sum(x == y) / len(x)
    smc_sim = np.array([[smc(a, b) for b in binary_df.values] for a in binary_df.values])

    # Cosine
    cos_sim = cosine_similarity(u_normalized)

    return jaccard_sim, smc_sim, cos_sim

# ==== MAIN EXECUTION ====
u_file_path = "Lab Session Data.xlsx"

jaccard_sim, smc_sim, cos_sim = u_compute_similarity_matrices(u_file_path)

if jaccard_sim is not None:
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
