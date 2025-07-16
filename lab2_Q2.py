import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load and clean data
file_path = "Lab Session Data.xlsx"
df = pd.read_excel(file_path, sheet_name="Purchase data")

# Select numeric columns and fill NaNs with mean
numeric_df = df.select_dtypes(include=[np.number])
numeric_df = numeric_df.apply(lambda col: col.fillna(col.mean()), axis=0)

# ✅ Ensure enough data rows exist before sampling
num_rows = numeric_df.shape[0]
if num_rows < 20:
    print(f"Only {num_rows} rows available. Sampling all of them instead of 20.")
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
from sklearn.metrics.pairwise import cosine_similarity
cos_sim = cosine_similarity(normalized)

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
