import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def jaccard_matrix(df):
    # Jaccard Coefficient computation
    n = len(df)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            inter = np.logical_and(df.iloc[i], df.iloc[j]).sum()
            union = np.logical_or(df.iloc[i], df.iloc[j]).sum()
            mat[i, j] = inter / union if union != 0 else 0
    return pd.DataFrame(mat)

def smc_matrix(df):
    # Simple Matching Coefficient computation
    n = len(df)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matches = (df.iloc[i] == df.iloc[j]).sum()
            mat[i, j] = matches / len(df.columns)
    return pd.DataFrame(mat)

def cosine_matrix(df):
    # Cosine Similarity computation
    return pd.DataFrame(cosine_similarity(df))

def u_compute_similarity_visualization():
    # Load or simulate binary data (replace this with your actual DataFrame)
    np.random.seed(0)
    u_df = pd.DataFrame(np.random.choice([0, 1], size=(20, 10)), columns=[f"Attr{i}" for i in range(10)])

    # Compute similarity matrices
    jc = jaccard_matrix(u_df)
    smc = smc_matrix(u_df)
    cos = cosine_matrix(u_df)

    # Plot heatmaps
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    sns.heatmap(jc, cmap='YlGnBu')
    plt.title("Jaccard Coefficient")

    plt.subplot(1, 3, 2)
    sns.heatmap(smc, cmap='YlOrBr')
    plt.title("Simple Matching Coefficient")

    plt.subplot(1, 3, 3)
    sns.heatmap(cos, cmap='Greens')
    plt.title("Cosine Similarity")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    u_compute_similarity_visualization()
