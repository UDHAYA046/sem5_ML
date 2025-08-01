# ---------------------------- MODULE IMPORTS ----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# ---------------------------- FUNCTION DEFINITIONS ----------------------------

def U_generate_training_data(U_seed=42):
    """
    Generate 20 synthetic training points (X, Y) with class labels.
    Label rule: class 1 if X + Y > 10, else class 0.
    """
    np.random.seed(U_seed)
    U_data = np.random.uniform(1, 10, size=(20, 2))
    U_labels = np.where(U_data.sum(axis=1) > 10, 1, 0)
    U_df = pd.DataFrame(U_data, columns=["X", "Y"])
    U_df["Class"] = U_labels
    return U_df

def U_generate_test_grid(U_min=0, U_max=10, U_step=0.1):
    """
    Generate a test grid of (X, Y) values ranging from U_min to U_max.
    Returns a 2D mesh and flattened test set for prediction.
    """
    U_x_grid, U_y_grid = np.meshgrid(np.arange(U_min, U_max + U_step, U_step),
                                     np.arange(U_min, U_max + U_step, U_step))
    U_test_points = np.c_[U_x_grid.ravel(), U_y_grid.ravel()]
    return U_test_points, U_x_grid, U_y_grid

def U_classify_and_plot(U_train_df, U_test_points, U_k=3, U_output_file="A4_knn_decision_boundary.png"):
    """
    Train kNN and classify test data, then plot and save the decision boundary.
    """
    # Prepare training data
    U_X_train = U_train_df[["X", "Y"]].values
    U_y_train = U_train_df["Class"].values

    # Train kNN
    U_knn_model = KNeighborsClassifier(n_neighbors=U_k)
    U_knn_model.fit(U_X_train, U_y_train)

    # Predict test data
    U_y_pred = U_knn_model.predict(U_test_points)

    # Color map
    U_color_map = {0: 'blue', 1: 'red'}
    U_colors = [U_color_map[p] for p in U_y_pred]

    # Plotting
    plt.figure(figsize=(7, 6))
    plt.scatter(U_test_points[:, 0], U_test_points[:, 1], c=U_colors, s=5, alpha=0.5, marker='o')
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.title(f"A4: Decision Region using kNN (k={U_k})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(U_output_file, dpi=400)
    plt.show()
    plt.close()

# ---------------------------- MAIN PROGRAM ----------------------------

if __name__ == "__main__":
    # Step 1: Generate training data
    U_train_data = U_generate_training_data()

    # Step 2: Generate test grid data (100 x 100 = 10,000 points)
    U_test_grid_points, _, _ = U_generate_test_grid()

    # Step 3: Train kNN and plot decision region
    U_classify_and_plot(U_train_data, U_test_grid_points, U_k=3,
                        U_output_file="A4_knn_decision_boundary.png")
