# ============================================================
# A1. Feature Correlation Analysis with Heatmap
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def U_feature_correlation_heatmap(data: pd.DataFrame, title: str = "Feature Correlation Heatmap") -> pd.DataFrame:
    """
    Performs correlation analysis and returns the correlation matrix.
    Does not print or plot internally (per lab instructions).
    """
    corr_matrix = data.corr(numeric_only=True)
    return corr_matrix

# ---------------- Main Section -----------------
if __name__ == "__main__":
    # Load your dataset (replace path and target column)
    U_DATA_PATH = r"/path/to/your_dataset.csv"
    U_TARGET_COL = "target"

    df = pd.read_csv(U_DATA_PATH)

    # Drop target for correlation analysis (only features)
    df_features = df.drop(columns=[U_TARGET_COL])

    # Compute correlation matrix
    corr_matrix = U_feature_correlation_heatmap(df_features)

    # --- Plot heatmap (outside function) ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True,
                cbar_kws={"shrink": 0.8}, linewidths=0.5)
    plt.title("Feature Correlation Heatmap", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.show()

    # Optional: print top correlated pairs
    print("\nTop correlated feature pairs (|r| > 0.8):")
    high_corr = (
        corr_matrix.unstack()
        .sort_values(ascending=False)
        .drop_duplicates()
        .reset_index()
    )
    high_corr.columns = ["Feature_1", "Feature_2", "Correlation"]
    print(high_corr[(abs(high_corr["Correlation"]) > 0.8) & 
                    (high_corr["Feature_1"] != high_corr["Feature_2"])])
