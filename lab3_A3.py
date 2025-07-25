# A3. Minkowski Distance between Two Feature Vectors (r = 1 to 10)
# Udhaya Sankari | Roll No: BL.EN.U4CSE23150

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

# Step 1: Load CSV and drop class & filename
file_path = r"C:\Users\Udhaya\sem5_ML\features_lab3_labeled.csv"
df = pd.read_csv(file_path)

# Remove 'filename' and 'class' columns if present
ud_cleaned_df = df.drop(columns=['filename', 'class'], errors='ignore')

# Step 2: Select any two feature vectors (rows)
vec1 = ud_cleaned_df.iloc[0].values
vec2 = ud_cleaned_df.iloc[1].values

# Step 3: Compute Minkowski distance for r = 1 to 10
r_values = range(1, 11)
distances = []

for r in r_values:
    d = distance.minkowski(vec1, vec2, p=r)
    distances.append(d)

# Step 4: Plot the result
plt.figure(figsize=(8, 5))
plt.plot(r_values, distances, marker='o', linestyle='-', color='purple')
plt.title("Minkowski Distance vs r (Feature Vector 1 vs 2)")
plt.xlabel("r value")
plt.ylabel("Minkowski Distance")
plt.grid(True)
plt.xticks(r_values)
plt.tight_layout()
plt.show()

# Print distances if needed
for r, d in zip(r_values, distances):
    print(f"Minkowski Distance (r={r}): {d:.4f}")
