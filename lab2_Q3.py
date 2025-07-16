import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
file_path = "Lab Session Data.xlsx"
df = pd.read_excel(file_path, sheet_name="marketing_campaign")

# Drop columns with all NaNs and reset index
df.dropna(how='all', axis=1, inplace=True)
df.reset_index(drop=True, inplace=True)

# Convert categorical columns to numeric
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna('Missing')
        df[col] = LabelEncoder().fit_transform(df[col])
    else:
        df[col] = df[col].fillna(df[col].mean())

# Normalize data for cosine similarity
scaler = MinMaxScaler()
norm_data = scaler.fit_transform(df)

# Binary transformation for Jaccard & SMC (based on mean)
binary_data = (df > df.mean()).astype(int).values

# --- Similarity Calculations ---

# Jaccard Similarity
jaccard_sim = 1 - pairwise_distances(binary_data, metric='jaccard')

# SMC (Simple Matching Coefficient)
def smc_similarity(X):
    n = X.shape[0]
    smc_sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            smc_sim[i][j] = np.sum(X[i] == X[j]) / X.shape[1]
    return smc_sim

smc_sim = smc_similarity(binary_data)

# Cosine Similarity
cos_sim = cosine_similarity(norm_data)

# --- Plotting Heatmaps ---
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
sns.heatmap(jaccard_sim, cmap="Blues", cbar=True)
plt.title("Jaccard Similarity")

plt.subplot(1, 3, 2)
sns.heatmap(smc_sim, cmap="Oranges", cbar=True)
plt.title("SMC Similarity")

plt.subplot(1, 3, 3)
sns.heatmap(cos_sim, cmap="coolwarm", cbar=True)
plt.title("Cosine Similarity")

plt.tight_layout()
plt.show()
