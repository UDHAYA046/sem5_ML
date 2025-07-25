# Udhaya Sankari | Roll No: BL.EN.U4CSE23150

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the labeled CSV data
ud_file_path = r"C:\Users\Udhaya\sem5_ML\features_lab3_labeled.csv"
ud_df_full = pd.read_csv(ud_file_path)

# Step 2: Drop 'filename' and separate based on numeric class values (e.g., 1 and 2)
ud_df_full = ud_df_full.drop(columns=['filename'])  # Remove filename
ud_class_column = 'class'

# Filter for class 1 and class 2 data
ud_df_1 = ud_df_full[ud_df_full[ud_class_column] == 1].drop(columns=[ud_class_column])
ud_df_2 = ud_df_full[ud_df_full[ud_class_column] == 2].drop(columns=[ud_class_column])

# Step 3: Calculate centroid (mean) and spread (standard deviation)
ud_centroid_1 = ud_df_1.mean(axis=0)
ud_std_1 = ud_df_1.std(axis=0)

ud_centroid_2 = ud_df_2.mean(axis=0)
ud_std_2 = ud_df_2.std(axis=0)

# Step 4: Inter-class Euclidean distance
ud_centroid_distance = np.linalg.norm(ud_centroid_1 - ud_centroid_2)

# Step 5: Display results
print("\n--- Centroid of Class 1 ---\n", ud_centroid_1)
print("\n--- Standard Deviation of Class 1 ---\n", ud_std_1)

print("\n--- Centroid of Class 2 ---\n", ud_centroid_2)
print("\n--- Standard Deviation of Class 2 ---\n", ud_std_2)

print("\n=== Inter-class Euclidean Distance (Class 1 vs Class 2):", round(ud_centroid_distance, 4))
