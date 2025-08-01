import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import os

# A3: Load and preprocess the dataset
def load_and_filter_data(csv_path, classes, features):
    df = pd.read_csv(csv_path)
    df_filtered = df[df['class'].isin(classes)].copy()
    X = df_filtered[features].values
    y = df_filtered['class'].values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return X, y_encoded

# A4: Create grid for decision boundary plotting
def create_decision_grid(x_min, x_max, y_min, y_max, step=0.1):
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))
    return xx, yy, np.c_[xx.ravel(), yy.ravel()]

# A5: Train model, predict and plot decision region
def train_plot_knn(X, y, xx, yy, grid_points, k_value, save_path):
    model = KNeighborsClassifier(n_neighbors=k_value)
    model.fit(X, y)
    Z = model.predict(grid_points).reshape(xx.shape)

    plt.figure(figsize=(6, 5), dpi=300)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k', s=30)
    plt.xlabel("mfcc1")
    plt.ylabel("pitch_std")
    plt.title(f"A6 Decision Boundary (k={k_value})")
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f"{save_path}/a6_decision_region_k{k_value}.png", bbox_inches='tight')
    plt.close()

# ---------------- MAIN PROGRAM ----------------
if __name__ == "__main__":
    # Inputs
    csv_path = "C:/Users/Udhaya/sem5_ML/features_lab3_labeled.csv"
    output_dir = "C:/Users/Udhaya/sem5_ML/lab4_output_figures"
    selected_classes = [1, 2]
    selected_features = ["mfcc1", "pitch_std"]
    k_values = [1, 2, 4]

    # Load data (A3)
    X_data, y_data = load_and_filter_data(csv_path, selected_classes, selected_features)

    # Create grid (A4)
    xx_mesh, yy_mesh, mesh_points = create_decision_grid(x_min=0, x_max=10, y_min=0, y_max=11)

    # Train and plot (A5)
    for k in k_values:
        train_plot_knn(X_data, y_data, xx_mesh, yy_mesh, mesh_points, k, output_dir)
