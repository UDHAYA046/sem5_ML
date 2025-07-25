# Udhaya Sankari | Roll No: BL.EN.U4CSE23150
# Lab 3 – A1: Intra-class feature spread (mean and standard deviation)

import pandas as pd
import matplotlib.pyplot as plt

# Load features file
ud_file_path = "features_lab3.csv"  # Adjust path if needed
ud_df_features = pd.read_csv(ud_file_path)

# Drop 'filename' since it is not a feature
ud_df_features_clean = ud_df_features.drop(columns=['filename'])

# Calculate mean and standard deviation for each feature
ud_feature_means = ud_df_features_clean.mean()
ud_feature_stds = ud_df_features_clean.std()

# Print the results clearly
print("===== Feature-wise Mean Values =====")
print(ud_feature_means)
print("\n===== Feature-wise Standard Deviations =====")
print(ud_feature_stds)

# Bar Plot: Mean and Standard Deviation
ud_fig, ud_ax = plt.subplots(figsize=(10, 6))
ud_x = range(len(ud_feature_means))

ud_ax.bar([x - 0.2 for x in ud_x], ud_feature_means, width=0.4, label='Mean', color='skyblue')
ud_ax.bar([x + 0.2 for x in ud_x], ud_feature_stds, width=0.4, label='Standard Deviation', color='salmon')

ud_ax.set_xticks(list(ud_x))
ud_ax.set_xticklabels(ud_feature_means.index, rotation=45)
ud_ax.set_title("Intra-Class Feature Spread (Mean vs Std Dev)")
ud_ax.set_ylabel("Value")
ud_ax.legend()
ud_ax.grid(True, linestyle='--', alpha=0.6)

# Save high-resolution plot for print (300 DPI)
plt.tight_layout()
plt.savefig("A1_feature_spread.png", dpi=300)
plt.show()
