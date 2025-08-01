# ---------------------------- MODULE IMPORTS ----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import os

# ---------------------------- FUNCTION DEFINITIONS ----------------------------

def U_generate_training_data(U_seed=42):
    """
    Generate 20 synthetic training points (X, Y) with class labels.
    Rule: class 1 if X + Y > 10, else class 0.
    """
    np.random.seed(U_seed)
    U_data = np.random.uniform(1, 10, size=(20, 2))
    U_labels = np.where(U_data.sum(axis=1) > 10, 1, 0)
    U_df = pd.DataFrame(U_data, columns=["X", "Y"])
    U_df["Class"] = U_labels
    return U_df

def U_generate_test_grid(U_min=0, U_max=10, U_step=0.1):
    """
    Generate test grid of 10,000 points between (0, 10).
    """
    U_x_grid, U_y_grid = np.meshgrid(np.arange(U_min, U_max + U_step, U_step),
                                     np.arange(U_min, U_max + U_step, U_step))
    U_test_points = np.c_[U_x_grid.ravel(), U_y_grid.ravel()]
    return U_test_points

def U_knn_classify_and_save(U_train_df, U_test_points, U_k_values, U_save_dir):
    """
    Repeat classification and plot for various k values and save plots.
    """
    os.makedirs(U_save_dir, exist_ok=True)

    for U_k in U_k_values:
        # Train kNN
        U_knn = KNeighborsClassifier(n_neighbors=U_k)
        U_knn.fit(U_train_df[["X", "Y"]], U_train_df["Class"])

        # Predict
        U_y_pred = U_knn.predict(U_test_points)
        U_color_map = {0: 'blue', 1: 'red'}
        U_colors = [U_color_map[label] for label in U_y_pred]

        # Plot
        plt.figure(figsize=(7, 6))
        plt.scatter(U_test_points[:, 0], U_test_points[:, 1], c=U_colors,
                    s=5, alpha=0.5, marker='o')
        plt.xlabel("Feature X")
        plt.ylabel("Feature Y")
        plt.title(f"A5: Decision Region using kNN (k={U_k})")
        plt.grid(True)
        plt.tight_layout()

        # Save
        U_filename = os.path.join(U_save_dir, f"A5_knn_decision_k{U_k}.png")
        plt.savefig(U_filename, dpi=400)
        plt.show()
        plt.close()

# ---------------------------- MAIN PROGRAM ----------------------------

if __name__ == "__main__":
    # Step 1: Generate training data
    U_train_data = U_generate_training_data()

    # Step 2: Generate test grid points
    U_test_data = U_generate_test_grid()

    # Step 3: Define k values
    U_k_list = [2, 4, 5, 6]

    # Step 4: Set output directory
    U_output_folder = r"C:\Users\Udhaya\sem5_ML\lab4_output_figures"

    # Step 5: Run classification and save plots
    U_knn_classify_and_save(U_train_data, U_test_data, U_k_list, U_output_folder)
