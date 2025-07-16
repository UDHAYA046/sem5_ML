# lab2_Q2.py

import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Excel data (check sheet name carefully)
file_path = "Lab Session Data.xlsx"
df = pd.read_excel(file_path, sheet_name="Purchase data")

# STEP 1: Select numeric attributes and drop rows with NaN for simplicity
numeric_df = df.select_dtypes(include=[np.number]).dropna()

# STEP 2: Randomly select 20 observation vectors
sample_df = numeric_df.sample(n=20, random_state=42).reset_index(drop=True)

# STEP 3: Normalize the data for fair cosine comparison
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(sample_df)

# STEP 4: Compute similarity matrices
# Jaccard Similarity (treat values as binary)
binary_data = (sample_df > sample_df.mean()).astype(int)
jaccard_similarity = 1 - pairwise_distances(binary_data, metric='jaccard')

# Simple Matching Coefficient (SMC)
def smc(u, v):
    return np.sum(u == v) / len(u)

smc_matrix = np.array([[smc(r1, r2) for r2 in binary_data.values] for r1 in binary_data.values])

# Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(normalized_data)

# STEP 5: Plot heatmaps
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.heatmap(jaccard_similarity, annot=False, cmap='viridis')
plt.title("Jaccard Similarity (20 samples)")

plt.subplot(1, 3, 2)
sns.heatmap(smc_matrix, annot=False, cmap='plasma')
plt.title("SMC Similarity (20 samples)")

plt.subplot(1, 3, 3)
sns.heatmap(cosine_sim, annot=False, cmap='coolwarm')
plt.title("Cosine Similarity (20 samples)")

plt.tight_layout()
plt.show()
