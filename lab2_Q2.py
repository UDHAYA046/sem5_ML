import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
file_path = "Lab Session Data.xlsx"
df = pd.read_excel(file_path, sheet_name="Purchase data")

# Keep only numeric data and drop rows with missing values
numeric_df = df.select_dtypes(include=[np.number]).dropna()

# Check available rows
if len(numeric_df) < 20:
    print(f" Not enough data rows. Only {len(numeric_df)} rows available after dropping NaNs.")
    exit()

# Randomly pick 20 rows
sample_df = numeric_df.sample(n=20, random_state=42).reset_index(drop=True)

# Normalize for cosine similarity
scaler = MinMaxScaler()
normalized = scaler.fit_transform(sample_df)

# Jaccard similarity: binary conversion
binary_data = (sample_df > sample_df.mean()).astype(int)
jaccard_sim = 1 - pairwise_distances(binary_data, metric='jaccard')

# SMC manually
def smc(u, v):
    return np.sum(u == v) / len(u)

smc_sim = np.array([[smc(r1, r2) for r2 in binary_data.values] for r1 in binary_data.values])

# Cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
cos_sim = cosine_similarity(normalized)

# Plot results
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
