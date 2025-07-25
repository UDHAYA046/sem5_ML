# Udhaya Sankari | Roll No: BL.EN.U4CSE23150

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the Excel data
ud_file_path = "features_lab3.xlsx"
ud_sheet_name = "Features"
ud_df_full = pd.read_excel(ud_file_path, sheet_name=ud_sheet_name)

# Step 2: Drop 'filename' and separate class label
ud_df_full = ud_df_full.drop(columns=['filename'])
ud_class_column = 'class'
ud_df_A = ud_df_full[ud_df_full[ud_class_column] == 'A'].drop(columns=[ud_class_column])
ud_df_B = ud_df_full[ud_df_full[ud_class_column] == 'B'].drop(columns=[ud_class_column])

# Step 3: Calculate centroid (mean) and spread (std deviation)
ud_centroid_A = ud_df_A.mean(axis=0)
ud_std_A = ud_df_A.std(axis=0)

ud_centroid_B = ud_df_B.mean(axis=0)
ud_std_B = ud_df_B.std(axis=0)

# Step 4: Inter-class Euclidean distance
ud_centroid_distance = np.linalg.norm(ud_centroid_A - ud_centroid_B)

# Step 5: Display results
print("\n--- Centroid of Class A ---\n", ud_centroid_A)
print("\n--- Standard Deviation of Class A ---\n", ud_std_A)

print("\n--- Centroid of Class B ---\n", ud_centroid_B)
print("\n--- Standard Deviation of Class B ---\n", ud_std_B)

print("\n>>> Euclidean Distance between Class A and Class B centroids: {:.4f}".format(ud_centroid_distance))

# Step 6: Visualization
ud_xlabels = ud_centroid_A.index
x = np.arange(len(ud_xlabels))
bar_width = 0.2

plt.figure(figsize=(10, 6))
plt.bar(x - bar_width, ud_centroid_A, width=bar_width, label='Mean A', color='skyblue')
plt.bar(x, ud_std_A, width=bar_width, label='Std Dev A', color='salmon')
plt.bar(x + bar_width, ud_centroid_B, width=bar_width, label='Mean B', color='lightgreen')

plt.xlabel("Feature", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.title("Intra-Class and Inter-Class Feature Comparison", fontsize=14)
plt.xticks(ticks=x, labels=ud_xlabels, rotation=45)
plt.legend()
plt.tight_layout()

# Step 7: Save plot to specified folder at 300 DPI
output_path = r"C:\Users\Udhaya\sem5_ML\lab3_output_figures\lab3_A1_output1.png"
plt.savefig(output_path, dpi=300)
plt.show()
