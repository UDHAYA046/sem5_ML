# Udhaya Sankari | Roll No: 23CSE23012
# Lab 3 – A1: Evaluate Intraclass Spread and Interclass Distance

import pandas as pd
import numpy as np

#  Step 1: Load the extracted feature dataset
ud_input_path = r"C:\Users\Udhaya\sem5_ML\features_lab3.csv"
ud_df_features = pd.read_csv(ud_input_path)

#  Step 2: Select any two class labels to compare (edit this as needed)
ud_class_label_column = 'label'  # make sure this column exists in your CSV
ud_class_A = 0
ud_class_B = 1

#  Step 3: Separate feature vectors of the two classes (drop filename and label)
ud_data_A = ud_df_features[ud_df_features[ud_class_label_column] == ud_class_A].drop(columns=['filename', ud_class_label_column])
ud_data_B = ud_df_features[ud_df_features[ud_class_label_column] == ud_class_B].drop(columns=['filename', ud_class_label_column])

#  Step 4a: Compute the centroid (mean) for each class
ud_centroid_A = ud_data_A.mean(axis=0)
ud_centroid_B = ud_data_B.mean(axis=0)

#  Step 4b: Compute the intraclass spread (standard deviation)
ud_spread_A = ud_data_A.std(axis=0)
ud_spread_B = ud_data_B.std(axis=0)

# Step 4c: Compute the interclass Euclidean distance
ud_interclass_dist = np.linalg.norm(ud_centroid_A - ud_centroid_B)

# Step 5: Print results (in high-resolution friendly format)
print("\n Udhaya Sankari | Lab 3 – A1: Intraclass and Interclass Analysis")
print("\n Centroid Vector for Class", ud_class_A, ":\n", ud_centroid_A.to_string())
print("\n Centroid Vector for Class", ud_class_B, ":\n", ud_centroid_B.to_string())

print("\n Spread (Standard Deviation) for Class", ud_class_A, ":\n", ud_spread_A.to_string())
print("\n Spread (Standard Deviation) for Class", ud_class_B, ":\n", ud_spread_B.to_string())

print(f"\n Euclidean Distance Between Class {ud_class_A} and Class {ud_class_B} Centroids: {ud_interclass_dist:.4f}")
