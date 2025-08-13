# Lab06 A5 – Custom Decision Tree Module
# Author: Udhaya

import pandas as U_pd
import numpy as U_np
import math as U_math

# -----------------------------
# Function Section
# -----------------------------

def U_load_dataset(U_path):
    return U_pd.read_csv(U_path)

def U_equal_width_binning(U_col, U_bins=4, U_labels=None):
    U_series = U_pd.Series(U_col, dtype=float)
    if U_series.size == 0:
        return U_pd.Series([], dtype="category")
    if float(U_series.min()) == float(U_series.max()):
        return U_pd.Series(["bin_0"] * len(U_series), index=U_series.index, dtype="category")
    if U_labels is None:
        U_labels = [f"bin_{i}" for i in range(U_bins)]
    return U_pd.cut(U_series, bins=U_bins, labels=U_labels, include_lowest=True, duplicates="drop").astype("category")

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

def U_entropy(U_y):
    U_ser = U_pd.Series(U_y).dropna()
    U_n = len(U_ser)
    if U_n == 0:
        return 0.0
    U_counts = U_ser.value_counts().astype(float).values
    U_probs = U_counts / U_n
    return float(-U_np.sum([p * U_math.log2(p) for p in U_probs if p > 0]))

def U_information_gain(U_df, U_feature, U_target):
    U_target_entropy = U_entropy(U_df[U_target])
    U_total = len(U_df)
    U_weighted_entropy = 0.0
    for U_val, U_subset in U_df.groupby(U_feature, observed=False):
        U_weight = len(U_subset) / U_total
        U_weighted_entropy += U_weight * U_entropy(U_subset[U_target])
    return U_target_entropy - U_weighted_entropy

def U_find_best_feature(U_df, U_target, U_binning_type="equal_width", U_bins=4):
    U_ignore_cols = [U_target, "filename", "id", "ID"]
    U_features = [col for col in U_df.columns if col not in U_ignore_cols]
    U_best_feat, U_best_ig = None, -1.0
    for feat in U_features:
        if U_pd.api.types.is_numeric_dtype(U_df[feat]):
            if U_binning_type == "equal_width":
                U_df[feat] = U_equal_width_binning(U_df[feat], U_bins=U_bins)
            elif U_binning_type == "equal_freq":
                U_df[feat] = U_equal_freq_binning(U_df[feat], U_bins=U_bins)
        U_ig = U_information_gain(U_df, feat, U_target)
        if U_ig > U_best_ig:
            U_best_ig, U_best_feat = U_ig, feat
    return U_best_feat

class U_TreeNode:
    def __init__(self, U_feature=None, U_children=None, U_label=None):
        self.U_feature = U_feature
        self.U_children = U_children if U_children else {}
        self.U_label = U_label

def U_build_tree(U_df, U_target, U_binning_type="equal_width", U_bins=4, U_depth=0, U_max_depth=None):
    if len(U_df[U_target].unique()) == 1:
        return U_TreeNode(U_label=U_df[U_target].iloc[0])
    if len(U_df.columns) == 1 or (U_max_depth is not None and U_depth >= U_max_depth):
        return U_TreeNode(U_label=U_df[U_target].mode()[0])

    U_best_feat = U_find_best_feature(U_df.copy(), U_target, U_binning_type, U_bins)
    if U_best_feat is None:
        return U_TreeNode(U_label=U_df[U_target].mode()[0])

    if U_pd.api.types.is_numeric_dtype(U_df[U_best_feat]):
        if U_binning_type == "equal_width":
            U_df[U_best_feat] = U_equal_width_binning(U_df[U_best_feat], U_bins=U_bins)
        elif U_binning_type == "equal_freq":
            U_df[U_best_feat] = U_equal_freq_binning(U_df[U_best_feat], U_bins=U_bins)

    U_node = U_TreeNode(U_feature=U_best_feat)

    for U_val, U_subset in U_df.groupby(U_best_feat, observed=False):
        if U_subset.empty:
            U_node.U_children[U_val] = U_TreeNode(U_label=U_df[U_target].mode()[0])
        else:
            U_remaining = U_subset.drop(columns=[U_best_feat])
            U_node.U_children[U_val] = U_build_tree(U_remaining, U_target, U_binning_type, U_bins, U_depth+1, U_max_depth)

    return U_node

def U_print_tree(U_node, U_prefix="", U_is_last=True):
    """
    Pretty print the custom Decision Tree with hierarchy lines.
    """
    connector = "└──" if U_is_last else "├──"

    if U_node.U_label is not None:
        print(f"{U_prefix}{connector} [Leaf] {U_node.U_label}")
    else:
        print(f"{U_prefix}{connector} [Feature] {U_node.U_feature}")
        child_prefix = U_prefix + ("    " if U_is_last else "│   ")
        total_children = len(U_node.U_children)
        for idx, (U_val, U_child) in enumerate(U_node.U_children.items()):
            is_last_child = (idx == total_children - 1)
            print(f"{child_prefix}{'└──' if is_last_child else '├──'} (Value: {U_val})")
            U_print_tree(U_child, child_prefix + ("    " if is_last_child else "│   "), True)

# -----------------------------
# Main Section
# -----------------------------
if __name__ == "__main__":
    U_csv_path = r"C:\Users\Udhaya\sem5_ML\features_lab3_labeled.csv"
    U_target_col = "class"  
    U_binning_type = "equal_width"  # or "equal_freq"
    U_bins = 4
    U_max_depth = None  # set to limit depth

    U_df_all = U_load_dataset(U_csv_path)

    if U_pd.api.types.is_numeric_dtype(U_df_all[U_target_col]):
        if U_binning_type == "equal_width":
            U_df_all[U_target_col] = U_equal_width_binning(U_df_all[U_target_col], U_bins=U_bins)
        else:
            U_df_all[U_target_col] = U_equal_freq_binning(U_df_all[U_target_col], U_bins=U_bins)

    U_tree_root = U_build_tree(U_df_all, U_target_col, U_binning_type, U_bins, U_max_depth=U_max_depth)

    print("[A5] Decision Tree Structure:")
    U_print_tree(U_tree_root)
