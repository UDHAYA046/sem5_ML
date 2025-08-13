# Lab06 A6 – Visualize Decision Tree
# Author: Udhaya

import pandas as U_pd
from sklearn.tree import DecisionTreeClassifier as U_DTC, plot_tree as U_plot_tree
import matplotlib.pyplot as U_plt

# --- Load dataset ---
def U_load_dataset(U_path):
    return U_pd.read_csv(U_path)

# --- Equal-width binning ---
def U_equal_width_binning(U_col, U_bins=4, U_labels=None):
    U_series = U_pd.Series(U_col, dtype=float)
    if U_series.size == 0:
        return U_pd.Series([], dtype="category")
    if float(U_series.min()) == float(U_series.max()):
        return U_pd.Series(["bin_0"] * len(U_series), index=U_series.index, dtype="category")
    if U_labels is None:
        U_labels = [f"bin_{i}" for i in range(U_bins)]
    return U_pd.cut(U_series, bins=U_bins, labels=U_labels, include_lowest=True, duplicates="drop").astype("category")

# --- Equal-frequency binning ---
def U_equal_freq_binning(U_col, U_bins=4, U_labels=None):
    U_series = U_pd.Series(U_col, dtype=float)
    unique_vals = U_series.dropna().unique()
    q = min(U_bins, len(unique_vals))
    if q <= 1:
        return U_pd.Series(["bin_0"] * len(U_series), index=U_series.index, dtype="category")
    if U_labels is None:
        U_labels = [f"bin_{i}" for i in range(q)]
    try:
        return U_pd.qcut(U_series, q, labels=U_labels, duplicates="drop").astype("category")
    except ValueError:
        return U_pd.cut(U_series, q, labels=U_labels, include_lowest=True, duplicates="drop").astype("category")

# --- Main ---
if __name__ == "__main__":
    U_csv_path = r"C:\Users\Udhaya\sem5_ML\features_lab3_labeled.csv"
    U_target_col = "class"  
    U_binning_type = "equal_width"  # or "equal_freq"
    U_bins = 4

    U_df_all = U_load_dataset(U_csv_path)

    # Remove non-predictive columns
    U_df_all = U_df_all.drop(columns=["filename"], errors="ignore")

    # Bin numeric target if needed
    if U_pd.api.types.is_numeric_dtype(U_df_all[U_target_col]):
        if U_binning_type == "equal_width":
            U_df_all[U_target_col] = U_equal_width_binning(U_df_all[U_target_col], U_bins=U_bins)
        else:
            U_df_all[U_target_col] = U_equal_freq_binning(U_df_all[U_target_col], U_bins=U_bins)

    # Bin numeric features if needed
    for feat in U_df_all.columns:
        if feat != U_target_col and U_pd.api.types.is_numeric_dtype(U_df_all[feat]):
            if U_binning_type == "equal_width":
                U_df_all[feat] = U_equal_width_binning(U_df_all[feat], U_bins=U_bins)
            else:
                U_df_all[feat] = U_equal_freq_binning(U_df_all[feat], U_bins=U_bins)

    # Convert categories to numeric codes for sklearn
    U_df_encoded = U_df_all.apply(lambda col: col.cat.codes if col.dtype.name == "category" else col)

    U_X = U_df_encoded.drop(columns=[U_target_col])
    U_y = U_df_encoded[U_target_col]

    # Fit Decision Tree Classifier
    U_model = U_DTC(criterion="entropy", random_state=42)
    U_model.fit(U_X, U_y)

    # Plot the decision tree
    U_plt.figure(figsize=(15, 8))
    U_plot_tree(U_model,
                feature_names=U_X.columns,
                class_names=[str(c) for c in sorted(U_df_all[U_target_col].unique())],
                filled=True,
                rounded=True)
    U_plt.title("Decision Tree Visualization (A6)")
    U_plt.show()
