import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Excel data
file_path = "Lab Session Data.xlsx"
df = pd.read_excel(file_path, sheet_name="Purchase data")

# Select numeric columns
numeric_df = df.select_dtypes(include=[np.number])

# Fill missing values with column means to avoid NaN drops
numeric_df = numeric_df.apply(lambda col: col.fillna(col.mean()), axis=0)

# Sample 20 random observation vectors
sample_df = numeric_df.sample(n=20, random_state=42).reset_index(drop=True)

# Normalize for cosine similarity
scaler = MinMaxScaler()
normalized = scaler.fit_transform(sample_df)

# Binary conversion for similarity measures
binary_df = (sample_df > sample_df.mean()).astype(int)

# Jaccard similarity
jaccard_sim = 1 - pairwise_distances(binary_df, metric='jaccard')

# SMC (Simple Matching Coefficient)
def smc(u, v):
    return np.sum(u == v) / len(u)

smc_sim = np.array([[smc(r1, r2) for r2 in binary_df.values] for r1 in binary_df.values])

# Cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
cos_sim = cosine_similarity(normalized)

# Plot similarity heatmaps
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.heatmap(jaccard_sim, cmap="YlGnBu")
plt.title("Jaccard Similarity")

plt.subplot(1, 3, 2)
sns.heatmap(smc_sim, cmap="YlOrBr")
plt.title("SMC Similarity")

plt.subplot(1, 3, 3)
sns.heatmap(cos_sim, cmap="coolwarm")
plt.title("Cosine Similarity")

plt.tight_layout()
plt.show()
