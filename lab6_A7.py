# Lab06 A7 – Decision Boundary for Two Features (Decision Tree)
# Author: Udhaya

import pandas as U_pd
import numpy as U_np
import matplotlib.pyplot as U_plt
from sklearn.tree import DecisionTreeClassifier as U_DTC

# -----------------------------
# Function Section
# -----------------------------

def U_load_dataset(U_path):
    return U_pd.read_csv(U_path)

def U_equal_width_binning(U_col, U_bins=4, U_labels=None):
    """Equal-width binning for numeric target when needed (A1 rule)."""
    U_series = U_pd.Series(U_col, dtype=float)
    if U_series.size == 0:
        return U_pd.Series([], dtype="category")
    if float(U_series.min()) == float(U_series.max()):
        return U_pd.Series(["bin_0"] * len(U_series), index=U_series.index, dtype="category")
    if U_labels is None:
        U_labels = [f"bin_{i}" for i in range(U_bins)]
    return U_pd.cut(U_series, bins=U_bins, labels=U_labels, include_lowest=True, duplicates="drop").astype("category")

def U_equal_freq_binning(U_col, U_bins=4, U_labels=None):
    """Equal-frequency (quantile) binning for numeric target when needed (A4 rule)."""
    U_series = U_pd.Series(U_col, dtype=float)
    uniq = U_series.dropna().unique()
    q = min(U_bins, len(uniq))
    if q <= 1:
        return U_pd.Series(["bin_0"] * len(U_series), index=U_series.index, dtype="category")
    if U_labels is None:
        U_labels = [f"bin_{i}" for i in range(q)]
    try:
        return U_pd.qcut(U_series, q, labels=U_labels, duplicates="drop").astype("category")
    except ValueError:
        return U_pd.cut(U_series, q, labels=U_labels, include_lowest=True, duplicates="drop").astype("category")

def U_prepare_two_features(
    U_df,
    U_feat_x,
    U_feat_y,
    U_target,
    U_target_binning_type="equal_width",
    U_bins=4,
):
    """
    Prepare X (two columns) and y for boundary plotting.
    - Drops non-predictive columns like 'filename'
    - Bins target if it is numeric (per A1/A4)
    - Drops rows with NaNs in the selected columns
    Returns: X_2 (DataFrame with two columns), y (Series), target_classes (list[str])
    """
    U_df = U_df.drop(columns=["filename"], errors="ignore")

    # Handle target (must be categorical for classification)
    if U_pd.api.types.is_numeric_dtype(U_df[U_target]):
        if U_target_binning_type == "equal_freq":
            U_df[U_target] = U_equal_freq_binning(U_df[U_target], U_bins=U_bins)
        else:
            U_df[U_target] = U_equal_width_binning(U_df[U_target], U_bins=U_bins)
    else:
        U_df[U_target] = U_df[U_target].astype("category")

    # Keep only the two selected features + target; remove rows with NaN
    U_keep = [U_feat_x, U_feat_y, U_target]
    U_df_small = U_df[U_keep].dropna(axis=0).copy()

    # If any chosen feature is categorical, encode to numeric codes (DT can handle numeric thresholds)
    for f in [U_feat_x, U_feat_y]:
        if U_df_small[f].dtype.name == "category":
            U_df_small[f] = U_df_small[f].cat.codes

    # Final X and y
    U_X2 = U_df_small[[U_feat_x, U_feat_y]]
    U_y = U_df_small[U_target].cat.codes if U_df_small[U_target].dtype.name == "category" else U_df_small[U_target]
    U_classes = list(map(str, U_df[U_target].cat.categories)) if U_df[U_target].dtype.name == "category" else sorted(map(str, U_df_small[U_target].unique()))

    return U_X2, U_y, U_classes

def U_plot_decision_boundary(U_model, U_X2, U_y, U_feat_x, U_feat_y, U_title="A7 – Decision Boundary (Decision Tree)"):
    """
    Plot a 2D decision boundary for a fitted sklearn classifier.
    """
    # Grid range with margin
    x_min, x_max = U_X2[U_feat_x].min(), U_X2[U_feat_x].max()
    y_min, y_max = U_X2[U_feat_y].min(), U_X2[U_feat_y].max()
    dx, dy = (x_max - x_min) * 0.05, (y_max - y_min) * 0.05
    x_min, x_max = x_min - dx, x_max + dx
    y_min, y_max = y_min - dy, y_max + dy

    U_xx, U_yy = U_np.meshgrid(
        U_np.linspace(x_min, x_max, 400),
        U_np.linspace(y_min, y_max, 400)
    )
    U_grid = U_pd.DataFrame({U_feat_x: U_xx.ravel(), U_feat_y: U_yy.ravel()})
    U_Z = U_model.predict(U_grid[[U_feat_x, U_feat_y]]).reshape(U_xx.shape)

    # Plot
    U_plt.figure(figsize=(7, 6))
    U_plt.contourf(U_xx, U_yy, U_Z, alpha=0.3)
    # scatter points by class
    U_y_ser = U_pd.Series(U_y, index=U_X2.index)
    for cls in sorted(U_y_ser.unique()):
        U_mask = (U_y_ser == cls)
        U_plt.scatter(U_X2.loc[U_mask, U_feat_x], U_X2.loc[U_mask, U_feat_y], edgecolor="k", label=str(cls), s=45)
    U_plt.xlabel(U_feat_x)
    U_plt.ylabel(U_feat_y)
    U_plt.title(U_title)
    U_plt.legend(title="Class", loc="best")
    U_plt.tight_layout()
    U_plt.show()

# -----------------------------
# Main Section
# -----------------------------
if __name__ == "__main__":
    # --- Set your paths/columns here ---
    U_csv_path = r"C:\Users\Udhaya\sem5_ML\features_lab3_labeled.csv"
    U_target_col = "class"          # your label column
    U_feat_x = "mfcc1"              # pick any two meaningful features (numeric preferred)
    U_feat_y = "rms"

    # Target binning if numeric
    U_target_binning_type = "equal_width"   # or "equal_freq"
    U_bins = 4

    # Decision Tree hyperparams for a clean boundary
    U_max_depth = 4                 # keep it small so the regions are visible
    U_random_state = 42

    # --- Load & prepare ---
    U_df_all = U_load_dataset(U_csv_path)
    U_X2, U_y, U_class_names = U_prepare_two_features(
        U_df_all, U_feat_x, U_feat_y, U_target_col,
        U_target_binning_type=U_target_binning_type, U_bins=U_bins
    )

    # --- Fit a Decision Tree on the two features ---
    U_model = U_DTC(criterion="entropy", max_depth=U_max_depth, random_state=U_random_state)
    U_model.fit(U_X2[[U_feat_x, U_feat_y]], U_y)

    # --- Plot decision boundary ---
    U_plot_decision_boundary(U_model, U_X2, U_y, U_feat_x, U_feat_y,
                             U_title=f"A7 – Decision Boundary: {U_feat_x} vs {U_feat_y}")
