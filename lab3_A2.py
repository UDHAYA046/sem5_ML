# A2. Histogram + Stats for a Feature
# Udhaya Sankari | Roll No: BL.EN.U4CSE23150

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file (update path as needed)
file_path = r"C:\Users\Udhaya\sem5_ML\features_lab3_labeled.csv"
df = pd.read_csv(file_path)

# Choose the feature to analyze (e.g., 'pitch_std')
ud_feature = 'pitch_std'
ud_feature_data = df[ud_feature].dropna()  # Drop any NaN values

# Calculate mean and variance
ud_mean = np.mean(ud_feature_data)
ud_variance = np.var(ud_feature_data)

# Print results
print(f"=== Histogram Analysis for Feature: {ud_feature} ===")
print(f"Mean of {ud_feature}: {ud_mean:.4f}")
print(f"Variance of {ud_feature}: {ud_variance:.4f}")

# Generate histogram
plt.figure(figsize=(8, 6))
plt.hist(ud_feature_data, bins=10, color='skyblue', edgecolor='black')
plt.axvline(ud_mean, color='red', linestyle='--', label=f"Mean = {ud_mean:.2f}")
plt.title(f"Histogram of {ud_feature}")
plt.xlabel(f"{ud_feature} values")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
