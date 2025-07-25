# Udhaya Sankari | Roll No: BL.EN.U4CSE23150

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse

# Step 1: Load your labeled CSV
ud_file_path = r"C:\Users\Udhaya\sem5_ML\features_lab3_labeled.csv"
ud_df = pd.read_csv(ud_file_path)

# Step 2: Separate features and labels
ud_features = ud_df.drop(columns=['filename', 'class'])
ud_labels = ud_df['class']

# Step 3: Split by class
ud_class1_df = ud_features[ud_labels == 1]
ud_class2_df = ud_features[ud_labels == 2]

# Step 4: Calculate centroids and standard deviation (spread)
ud_centroid_1 = ud_class1_df.mean(axis=0)
ud_std_1 = ud_class1_df.std(axis=0)

ud_centroid_2 = ud_class2_df.mean(axis=0)
ud_std_2 = ud_class2_df.std(axis=0)

# Step 5: Interclass Euclidean distance
ud_distance = np.linalg.norm(ud_centroid_1 - ud_centroid_2)

# Step 6: Display metrics
print("\n--- Centroid of Class 1 ---\n", ud_centroid_1)
print("\n--- Standard Deviation of Class 1 ---\n", ud_std_1)
print("\n--- Centroid of Class 2 ---\n", ud_centroid_2)
print("\n--- Standard Deviation of Class 2 ---\n", ud_std_2)
print("\n=== Inter-class Euclidean Distance (Class 1 vs Class 2):", round(ud_distance, 4))

# Step 7: Visualize with PCA
pca = PCA(n_components=2)
ud_pca_features = pca.fit_transform(ud_features)
ud_pca_df = pd.DataFrame(ud_pca_features, columns=['PC1', 'PC2'])
ud_pca_df['class'] = ud_labels.values

# Step 8: Plot
plt.figure(figsize=(8, 6))

# Scatter points
for label, color in zip([1, 2], ['blue', 'red']):
    subset = ud_pca_df[ud_pca_df['class'] == label]
    plt.scatter(subset['PC1'], subset['PC2'], label=f'Class {label}', alpha=0.6, c=color)

# Plot centroids in PCA space
centroid_1_pca = pca.transform([ud_centroid_1])[0]
centroid_2_pca = pca.transform([ud_centroid_2])[0]

plt.scatter(*centroid_1_pca, marker='X', s=200, c='navy', label='Centroid 1')
plt.scatter(*centroid_2_pca, marker='X', s=200, c='darkred', label='Centroid 2')

# Draw interclass line
plt.plot([centroid_1_pca[0], centroid_2_pca[0]], [centroid_1_pca[1], centroid_2_pca[1]],
         c='cyan', linewidth=2.5, label='Interclass Distance')

# Optional: Add intraclass ellipse (1 std)
def draw_ellipse(center, std, color):
    ellipse = Ellipse(xy=center, width=2*std[0], height=2*std[1],
                      edgecolor=color, fc='none', lw=2, linestyle='--')
    plt.gca().add_patch(ellipse)

std_1_pca = pca.transform([ud_centroid_1 + ud_std_1])[0] - centroid_1_pca
std_2_pca = pca.transform([ud_centroid_2 + ud_std_2])[0] - centroid_2_pca
draw_ellipse(centroid_1_pca, std_1_pca, 'blue')
draw_ellipse(centroid_2_pca, std_2_pca, 'red')

plt.title("Intraclass Spread and Interclass Distance (PCA View)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
