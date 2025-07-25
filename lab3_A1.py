# Udhaya Sankari | Roll No: BL.EN.U4CSE23150

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define file path and class names
ud_file_path = "Lab Session Data.xlsx"
ud_sheet_name = "Features"
ud_class_label_column = 'class'
ud_class_A = 'A'
ud_class_B = 'B'

# Read the sheet into dataframe
ud_df_features = pd.read_excel(ud_file_path, sheet_name=ud_sheet_name)

# Remove filename column
ud_df_features = ud_df_features.drop(columns=['filename'])

# Split data for each class
ud_data_A = ud_df_features[ud_df_features[ud_class_label_column] == ud_class_A].drop(columns=[ud_class_label_column])
ud_data_B = ud_df_features[ud_df_features[ud_class_label_column] == ud_class_B].drop(columns=[ud_class_label_column])

# Calculate mean (centroid) and standard deviation (spread) for each class
ud_centroid_A = ud_data_A.mean(axis=0)
ud_std_A = ud_data_A.std(axis=0)

ud_centroid_B = ud_data_B.mean(axis=0)
ud_std_B = ud_data_B.std(axis=0)

# Calculate Euclidean distance between centroids
ud_interclass_distance = np.linalg.norm(ud_centroid_A - ud_centroid_B)

# Print calculated values
print("\nCentroid of Class A:\n", ud_centroid_A)
print("\nStandard Deviation of Class A:\n", ud_std_A)

print("\nCentroid of Class B:\n", ud_centroid_B)
print("\nStandard Deviation of Class B:\n", ud_std_B)

print("\nEuclidean Distance between Class A and Class B Centroids: {:.4f}".format(ud_interclass_distance))

# Plotting the mean and std deviation
ud_x_labels = ud_centroid_A.index

x = np.arange(len(ud_x_labels))
width = 0.2

plt.figure(figsize=(10, 6))
plt.bar(x - width, ud_centroid_A, width=width, label='Mean A', color='skyblue')
plt.bar(x, ud_std_A, width=width, label='Std Dev A', color='salmon')
plt.bar(x + width, ud_centroid_B, width=width, label='Mean B', color='lightgreen')

plt.xlabel('Feature')
plt.ylabel('Value')
plt.title('Intra-Class and Inter-Class Feature Comparison')
plt.xticks(x, ud_x_labels, rotation=45)
plt.legend()
plt.tight_layout()

# Save figure with high resolution
plt.savefig('lab3_A1_output1.png', dpi=300)
plt.show()
