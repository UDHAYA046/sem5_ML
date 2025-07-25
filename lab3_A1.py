# Udhaya Sankari | Lab 3 | A1 – Intraclass Spread and Interclass Distance

import pandas as pd
import numpy as np

# Step 1: Load extracted feature dataset
ud_df_feat = pd.read_csv("features_lab3.csv")

# Step 2: Select two class labels for comparison (e.g., Class 0 and Class 1)
ud_class0_data = ud_df_feat[ud_df_feat['label'] == 0].drop(columns=['filename', 'label'])
ud_class1_data = ud_df_feat[ud_df_feat['label'] == 1].drop(columns=['filename', 'label'])

# Step 3a: Calculate Centroid (Mean Vector) of each class
ud_centroid0 = ud_class0_data.mean(axis=0)
ud_centroid1 = ud_class1_data.mean(axis=0)

# Step 3b: Calculate Intraclass Spread (Standard Deviation) of each class
ud_spread0 = ud_class0_data.std(axis=0)
ud_spread1 = ud_class1_data.std(axis=0)

# Step 3c: Calculate Interclass Distance (Euclidean Distance between centroids)
ud_interclass_distance = np.linalg.norm(ud_centroid0 - ud_centroid1)

# Step 4: Display Results
print("🔹 Centroid for Class 0:\n", ud_centroid0)
print("\n🔹 Centroid for Class 1:\n", ud_centroid1)
print("\n📏 Intraclass Spread (Std Dev) for Class 0:\n", ud_spread0)
print("\n📏 Intraclass Spread (Std Dev) for Class 1:\n", ud_spread1)
print(f"\n📐 Euclidean Distance Between Class Centroids = {ud_interclass_distance:.4f}")
