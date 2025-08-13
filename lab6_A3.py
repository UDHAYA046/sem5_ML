# Lab06 A3 – Root Feature Detection using Information Gain
# Author: Udhaya

import pandas as U_pd
import numpy as U_np
import math as U_math
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

# --- Entropy ---
def U_entropy(U_y):
    U_ser = U_pd.Series(U_y).dropna()
    U_n = len(U_ser)
    if U_n == 0:
        return 0.0
    U_counts = U_ser.value_counts().astype(float).values
    U_probs = U_counts / U_n
    return float(-U_np.sum([p * U_math.log2(p) for p in U_probs if p > 0]))

# --- Information Gain ---
def U_information_gain(U_df, U_feature, U_target):
    U_target_entropy = U_entropy(U_df[U_target])
    U_total = len(U_df)
    U_weighted_entropy = 0.0
    for U_val, U_subset in U_df.groupby(U_feature, observed=False):
        U_weight = len(U_subset) / U_total
        U_weighted_entropy += U_weight * U_entropy(U_subset[U_target])
    return U_target_entropy - U_weighted_entropy

# --- Find root feature ---
def U_find_root_feature(U_df, U_target, U_bins=4):
    U_ignore_cols = [U_target, "filename", "id", "ID"]
    U_features = [col for col in U_df.columns if col not in U_ignore_cols]
    U_best_feat, U_best_ig = None, -1.0
    U_ig_scores = {}
    for feat in U_features:
        if U_pd.api.types.is_numeric_dtype(U_df[feat]):
            U_df[feat] = U_equal_width_binning(U_df[feat], U_bins=U_bins)
        U_ig = U_information_gain(U_df, feat, U_target)
        U_ig_scores[feat] = U_ig
        if U_ig > U_best_ig:
            U_best_ig, U_best_feat = U_ig, feat
    return U_best_feat, U_best_ig, U_ig_scores

# --- Plot IG scores ---
def U_plot_ig_scores(U_ig_scores, U_title):
    U_series = U_pd.Series(U_ig_scores).sort_values(ascending=False)
    U_plt.figure(figsize=(8, 4))
    U_series.plot(kind="bar", color="skyblue", edgecolor="black")
    U_plt.title(U_title)
    U_plt.xlabel("Features")
    U_plt.ylabel("Information Gain")
    U_plt.grid(axis='y', linestyle='--', alpha=0.7)
    U_plt.tight_layout()
    U_plt.show()

# --- Main ---
if __name__ == "__main__":
    U_csv_path = r"C:\Users\Udhaya\sem5_ML\features_lab3_labeled.csv"
    U_target_col = "class"  

    U_df_all = U_load_dataset(U_csv_path)

    if U_pd.api.types.is_numeric_dtype(U_df_all[U_target_col]):
        U_df_all[U_target_col] = U_equal_width_binning(U_df_all[U_target_col], U_bins=4)

    U_root_feat, U_root_ig, U_ig_scores = U_find_root_feature(U_df_all, U_target_col, U_bins=4)

    print(f"[A3] Best root feature: {U_root_feat}")
    print(f"[A3] Information Gain: {U_root_ig}")
    U_plot_ig_scores(U_ig_scores, "A3 – Information Gain for All Features")
