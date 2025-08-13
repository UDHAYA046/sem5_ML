# Lab06 A2 – Gini Index Calculation
# Author: Udhaya

import pandas as U_pd
import numpy as U_np
import matplotlib.pyplot as U_plt

# --- Load and prepare data ---
def U_load_dataset(U_path, U_target_col):
    """
    Load dataset from CSV and return the target column as a Series.
    """
    U_df = U_pd.read_csv(U_path)
    return U_df[U_target_col]

# --- Equal-width binning function (for numeric targets) ---
def U_equal_width_binning(U_col, U_bins=4, U_labels=None):
    """
    Convert a numeric column into equal-width categorical bins.
    Handles empty or constant columns gracefully.
    """
    U_series = U_pd.Series(U_col, dtype=float)
    if U_series.size == 0:
        return U_pd.Series([], dtype="category")
    U_min, U_max = float(U_series.min()), float(U_series.max())
    if U_min == U_max:
        return U_pd.Series(["bin_0"] * len(U_series), index=U_series.index, dtype="category")
    if U_labels is None:
        U_labels = [f"bin_{i}" for i in range(U_bins)]
    U_binned = U_pd.cut(U_series, bins=U_bins, labels=U_labels, include_lowest=True, duplicates="drop")
    return U_binned.astype("category")

# --- Gini Index calculation ---
def U_gini_index(U_y):
    """
    Calculate Gini index for categorical data.
    If numeric, bin before calling this function.
    Formula: Gini = 1 - Σ(p_j)^2
    """
    U_ser = U_pd.Series(U_y).dropna()
    U_n = len(U_ser)
    if U_n == 0:
        return 0.0
    U_counts = U_ser.value_counts().astype(float).values
    U_probs = U_counts / U_n
    return float(1.0 - U_np.sum(U_probs ** 2))

# --- Main Section ---
if __name__ == "__main__":
    # Path to dataset and target column
    U_csv_path = r"C:\Users\Udhaya\sem5_ML\features_lab3_labeled.csv"  # your file path
    U_target_col = "class"  # change if target column name is different

    # Load target column
    U_y_raw = U_load_dataset(U_csv_path, U_target_col)

    # If numeric, apply equal-width binning
    if U_pd.api.types.is_numeric_dtype(U_y_raw):
        print("[Info] Numeric target detected → applying equal-width binning to 4 categories…")
        U_y_cat = U_equal_width_binning(U_y_raw, U_bins=4)
    else:
        U_y_cat = U_y_raw.astype("category")

    # Calculate Gini Index
    U_G = U_gini_index(U_y_cat)

    # Display results
    print("\n[A2] Category counts:\n", U_y_cat.value_counts())
    print("[A2] Gini Index:", U_G)

    # Plot category counts
    U_counts = U_y_cat.value_counts()
    U_plt.figure(figsize=(6, 4))
    U_counts.plot(kind='bar', color='lightgreen', edgecolor='black')
    U_plt.title('Category Counts (Binned Target) - Gini')
    U_plt.xlabel('Bins')
    U_plt.ylabel('Count')
    U_plt.grid(axis='y', linestyle='--', alpha=0.7)
    U_plt.tight_layout()
    U_plt.show()
