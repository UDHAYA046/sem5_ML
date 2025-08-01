import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import os

# Load your dataset
df = pd.read_csv('C:/Users/Udhaya/sem5_ML/features_lab3_labeled.csv')

# Keep only two classes (e.g., 1 and 2) and two features (e.g., mfcc1 and pitch_std)
df_filtered = df[df['class'].isin([1, 2])].copy()
X = df_filtered[['mfcc1', 'pitch_std']].values
y = df_filtered['class'].values

# Encode class labels to 0 and 1
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Set the grid range
x_min, x_max = 0, 10
y_min, y_max = 0, 11
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Output folder to save the images
output_folder = 'C:/Users/Udhaya/sem5_ML/lab4_output_figures'
os.makedirs(output_folder, exist_ok=True)

# Try k = 1, 2, 4
k_values = [1, 2, 4]

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X, y_encoded)

    Z = model.predict(grid_points).reshape(xx.shape)

    plt.figure(figsize=(6, 5), dpi=300)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y_encoded, cmap='coolwarm', edgecolor='k', s=30)
    plt.xlabel('mfcc1')
    plt.ylabel('pitch_std')
    plt.title(f'A6: Decision Region (k={k}) on Project Data')

    plt.savefig(f'{output_folder}/a6_decision_region_k{k}.png', dpi=300, bbox_inches='tight')
    plt.close()
