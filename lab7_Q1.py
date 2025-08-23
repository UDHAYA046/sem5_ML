# U_Lab07_O1_SHAP.py
# O1: SHAP explainability for RandomForest and LogisticRegression

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ------------------- CONFIG -------------------
U_DATA_PATH = r"C:\Users\Udhaya\sem5_ML\features_lab3_labeled.csv"
U_OUT_DIR   = r"C:\Users\Udhaya\sem5_ML\lab7_output_figures"
U_TARGET    = "class"
U_TEST_SIZE = 0.30
U_SEED      = 42
# ----------------------------------------------


# ----------------- FUNCTIONS (no prints) -----------------
def U_load_numeric_xy(csv_path: str, target_col: str):
    """Load CSV, keep numeric features only; return X, y, feature_names."""
    df = pd.read_csv(csv_path)
    y = df[target_col]
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).copy()
    # basic NaN handling (mean imputation)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean(numeric_only=True))
    return X, y, X.columns.tolist()

def U_filter_rare_classes(X: pd.DataFrame, y: pd.Series, min_count: int = 2):
    """Keep only labels with at least min_count samples (for stratify)."""
    counts = y.value_counts()
    valid = counts[counts >= min_count].index
    mask = y.isin(valid)
    return X[mask], y[mask]

def U_train_models(Xtr, ytr):
    """Train RF and LogisticRegression; return fitted models."""
    rf = RandomForestClassifier(random_state=U_SEED)
    rf.fit(Xtr, ytr)
    log = LogisticRegression(max_iter=1000, solver="liblinear", random_state=U_SEED)
    log.fit(Xtr, ytr)
    return rf, log

def U_shap_importance(shap_values, X_test: pd.DataFrame):
    """
    Compute mean |SHAP| per feature from shap_values (handles list for tree multiclass).
    Returns pandas Series aligned with X_test.columns.
    """
    if isinstance(shap_values, list):  # tree explainer (one array per class)
        # aggregate absolute contributions across classes
        vals = np.mean([np.abs(v) for v in shap_values], axis=0)
    else:
        vals = shap_values
    imp = np.mean(np.abs(vals), axis=0).ravel()
    imp = pd.Series(imp[:len(X_test.columns)], index=X_test.columns)
    return imp.sort_values(ascending=False), vals

def U_save_bar(series: pd.Series, title: str, out_path: str, top_k: int = 15):
    """Save a horizontal bar plot of top-k importances."""
    plt.figure(figsize=(8, 5))
    series.head(top_k).iloc[::-1].plot(kind="barh")
    plt.title(title)
    plt.xlabel("mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path

def U_save_shap_summary(vals, X_test: pd.DataFrame, title: str, out_path_bar: str, out_path_bees: str):
    """Save SHAP summary (bar + beeswarm)."""
    # Bar (SHAP's own bar)
    shap.summary_plot(vals, X_test, plot_type="bar", show=False)
    plt.title(title + " — bar")
    plt.tight_layout()
    plt.savefig(out_path_bar, dpi=200)
    plt.close()

    # Beeswarm
    shap.summary_plot(vals, X_test, show=False)
    plt.title(title + " — beeswarm")
    plt.tight_layout()
    plt.savefig(out_path_bees, dpi=200)
    plt.close()
    return out_path_bar, out_path_bees
# ---------------------------------------------------------


# --------------------------- MAIN ------------------------
if __name__ == "__main__":
    os.makedirs(U_OUT_DIR, exist_ok=True)

    # 1) Load numeric features and target
    U_X, U_y, U_feat_names = U_load_numeric_xy(U_DATA_PATH, U_TARGET)

    # 2) Filter ultra-rare labels and split (stratify if possible)
    U_X, U_y = U_filter_rare_classes(U_X, U_y, min_count=2)
    try:
        U_Xtr, U_Xte, U_ytr, U_yte = train_test_split(U_X, U_y, test_size=U_TEST_SIZE,
                                                      random_state=U_SEED, stratify=U_y)
    except ValueError:
        U_Xtr, U_Xte, U_ytr, U_yte = train_test_split(U_X, U_y, test_size=U_TEST_SIZE,
                                                      random_state=U_SEED)

    # 3) Train models
    U_rf, U_log = U_train_models(U_Xtr, U_ytr)

    # 4) SHAP for RandomForest
    U_rf_expl = shap.TreeExplainer(U_rf)
    U_rf_vals = U_rf_expl.shap_values(U_Xte)
    U_rf_imp, U_rf_vals_arr = U_shap_importance(U_rf_vals, U_Xte)
    U_save_bar(U_rf_imp, "RandomForest SHAP importance",
               os.path.join(U_OUT_DIR, "Lab07_O1_RF_SHAP_topbar.png"))
    U_save_shap_summary(U_rf_vals_arr, U_Xte,
                        "RandomForest", 
                        os.path.join(U_OUT_DIR, "Lab07_O1_RF_SHAP_summary_bar.png"),
                        os.path.join(U_OUT_DIR, "Lab07_O1_RF_SHAP_summary_bees.png"))

    # 5) SHAP for Logistic Regression
    U_log_expl = shap.LinearExplainer(U_log, U_Xtr, feature_perturbation="interventional")
    U_log_vals = U_log_expl.shap_values(U_Xte)
    U_log_imp, U_log_vals_arr = U_shap_importance(U_log_vals, U_Xte)
    U_save_bar(U_log_imp, "Logistic Regression SHAP importance",
               os.path.join(U_OUT_DIR, "Lab07_O1_LOG_SHAP_topbar.png"))
    U_save_shap_summary(U_log_vals_arr, U_Xte,
                        "Logistic Regression", 
                        os.path.join(U_OUT_DIR, "Lab07_O1_LOG_SHAP_summary_bar.png"),
                        os.path.join(U_OUT_DIR, "Lab07_O1_LOG_SHAP_summary_bees.png"))

    # 6) Save comparison table
    U_cmp = pd.concat([U_rf_imp.rename("RF_SHAP"), U_log_imp.rename("LOG_SHAP")], axis=1)
    U_cmp.to_csv(os.path.join(U_OUT_DIR, "Lab07_O1_SHAP_comparison.csv"))
    print("Saved SHAP figures and comparison CSV to:", U_OUT_DIR)
