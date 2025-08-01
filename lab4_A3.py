# ---------------------------- MODULE IMPORTS ----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------- FUNCTION DEFINITIONS ----------------------------

def U_generate_random_data(U_num_points=20, U_range=(1, 10), U_seed=42):
    """
    Generates random 2D data points and assigns binary class labels based on X+Y > 10 rule.
    """
    np.random.seed(U_seed)
    U_data = np.random.uniform(U_range[0], U_range[1], size=(U_num_points, 2))
    U_labels = np.where(U_data.sum(axis=1) > 10, 1, 0)
    U_df = pd.DataFrame(U_data, columns=["X", "Y"])
    U_df["Class"] = U_labels
    return U_df

def U_plot_training_data(U_df, U_filename="A3_scatter_train_data.png"):
    """
    Plots the training data with color-coded classes and saves the plot.
    """
    U_color_map = {0: 'blue', 1: 'red'}
    U_colors = [U_color_map[label] for label in U_df["Class"]]

    plt.figure(figsize=(6, 5))
    plt.scatter(U_df["X"], U_df["Y"], c=U_colors, s=60, edgecolors='black', marker='x')
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.title("A3: 20 Training Points Colored by Class")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(U_filename, dpi=400)
    plt.show()
    plt.close()

# ---------------------------- MAIN PROGRAM ----------------------------

if __name__ == "__main__":
    # Generate 20 synthetic training points with class labels
    U_synthetic_train = U_generate_random_data(U_num_points=20)

    # Display first few entries (optional)
    print(U_synthetic_train.head())

    # Plot and save
    U_plot_training_data(U_synthetic_train, U_filename="A3_scatter_train_data.png")
