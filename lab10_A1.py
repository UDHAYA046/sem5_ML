# ============================================================
# Lab 10 – A1: Feature Correlation Analysis + Heatmap
# Author: S. Udhaya Sankari
# Rules followed: functions have no prints; only main prints/plots
# ============================================================

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple

# ---------------------- Functions (no prints) ----------------------

def U_load_table(file_path: str) -> pd.DataFrame:
    """Load CSV or Excel based on extension. Raises if path missing."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p}")
    if p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p)  # requires openpyxl for .xlsx
    return pd.read_csv(p)

def U_prepare_features(df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
    """Return numeric feature frame (drops target column if provided)."""
    if target_col is not None and target_col in df.columns:
        df = df.drop(columns=[target_col])
    return df.select_dtypes(include=[np.number])

def U_corr_matrix(df_features: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson correlation (numeric_only)."""
    return df_features.corr(numeric_only=True, method="pearson")

def U_upper_triangle_mask(corr: pd.DataFrame) -> np.ndarray:
    """Boolean mask for plotting only the upper triangle of a symmetric matrix."""
    return np.triu(np.ones_like(corr, dtype=bool))

def U_strong_pairs(corr: pd.DataFrame, threshold: float = 0.80) -> pd.DataFrame:
    """Return dataframe of strongly correlated feature pairs |r| >= threshold, no self/dups."""
    pairs = (
        corr.stack()
            .reset_index()
            .rename(columns={"level_0": "Feature_1", "level_1": "Feature_2", 0: "Correlation"})
    )
    # Keep only one of (i,j)/(j,i) and drop self-pairs
    keep = pairs["Feature_1"] < pairs["Feature_2"]
    pairs = pairs[keep]
    return pairs.loc[pairs["Correlation"].abs() >= threshold].sort_values("Correlation", ascending=False)

def U_save_heatmap(
    corr: pd.DataFrame,
    out_path: str,
    title: str = "Feature Correlation Heatmap (Upper Triangle)"
) -> None:
    """Save an upper-triangle heatmap image to out_path."""
    mask = U_upper_triangle_mask(corr)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr, mask=mask, cmap="coolwarm", annot=False, square=True,
        linewidths=0.5, cbar_kws={"shrink": 0.8}
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# --------------------------- Main (prints/plots only) ---------------------------

if __name__ == "__main__":
    # >>> EDIT target column name if you have labels to exclude from correlation
    U_DATA_PATH  = r"C:\Users\Udhaya\sem5_ML\features_lab3_labeled.csv"
    U_TARGET_COL = None   # e.g., "label" or "confidence_label" if present
    U_OUT_PNG    = str(Path(U_DATA_PATH).with_suffix("")) + "_corr_heatmap.png"

    # 1) Load data
    df = U_load_table(U_DATA_PATH)
    print(f"Loaded: {U_DATA_PATH} | Shape: {df.shape}")

    # 2) Prepare numeric features (drop target if provided)
    df_feat = U_prepare_features(df, target_col=U_TARGET_COL)
    print(f"Numeric features shape: {df_feat.shape} | Columns: {len(df_feat.columns)}")

    # 3) Correlation
    corr = U_corr_matrix(df_feat)

    # 4) Plot (interactive) – optional in scripts; keep for notebooks
    mask = U_upper_triangle_mask(corr)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}
    )
    plt.title("Feature Correlation Heatmap (Upper Triangle)", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.show()

    # 5) Save PNG for report/GitHub
    U_save_heatmap(corr, U_OUT_PNG)
    print(f"Saved heatmap image → {U_OUT_PNG}")

    # 6) Strongly correlated pairs for your write-up
    strong = U_strong_pairs(corr, threshold=0.80)
    if strong.empty:
        print("No feature pairs with |r| ≥ 0.80.")
    else:
        print("\nStrongly correlated feature pairs (|r| ≥ 0.80):")
        print(strong.to_string(index=False))
